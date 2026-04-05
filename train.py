"""
ADME multi-task regression training script.
Uses Chemprop MPNN + Lightning for SMILES → PXR property prediction.

Usage:
    uv run train.py

Edit the configuration sections below to tune the run. No CLI flags needed.
The agent may freely modify TARGET_COLS, EXTRA_FEATURES_FN, and all
hyperparameters. Do not modify prepare.py.
"""

import os

os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"
os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"

import time
import math
from dataclasses import dataclass, asdict

import numpy as np
import torch
import lightning.pytorch as pl
from lightning.pytorch.callbacks import Callback, StochasticWeightAveraging

from prepare import TIME_BUDGET, AVAILABLE_TARGET_COLS, make_dataloader, evaluate_regression
# ---------------------------------------------------------------------------
# Target columns — pick any subset of AVAILABLE_TARGET_COLS
# ---------------------------------------------------------------------------
# fmt: off
TARGET_COLS = [
    "pEC50",
    "Emax.vs.pos.ctrl_estimate (dimensionless)",
    "Emax.vs.pos.ctrl_ci.upper (dimensionless)",
]
# fmt: on

# ---------------------------------------------------------------------------
# Featurizer — define extra molecule-level features for MoleculeDatapoint.x_d
# ---------------------------------------------------------------------------
# Set to None to use no extra features (default chemprop graph features only).
# To add features, define a function with signature:
#   (smiles: str) -> np.ndarray | None
# Return None to skip a molecule; return a 1-D float32 array otherwise.
# Example using RDKit 2D descriptors:
#
#   from rdkit import Chem
#   from rdkit.Chem import Descriptors
#   def rdkit_descriptors(smiles: str):
#       mol = Chem.MolFromSmiles(smiles)
#       if mol is None:
#           return None
#       return np.array([v for _, v in Descriptors.descList], dtype=np.float32)
#
EXTRA_FEATURES_FN = None

# ---------------------------------------------------------------------------
# Gasteiger charges as per-atom vertex descriptors (V_d)
# ---------------------------------------------------------------------------
USE_GASTEIGER_VD = False  # set to True to add Gasteiger charges as V_d
USE_V1_FEATURIZER = False  # use chemprop v1 atom featurizer (133-dim vs default 72-dim)

# ---------------------------------------------------------------------------
# Hyperparameters (edit these directly — no CLI flags needed)
# ---------------------------------------------------------------------------

# Message passing
DEPTH = 3               # number of bond message-passing steps
HIDDEN_SIZE = 300       # hidden dimension in message passing layers
DROPOUT = 0.0           # dropout applied in both MP and FFN

# Feed-forward network (predictor)
FFN_NUM_LAYERS = 2      # number of FFN layers after aggregation
FFN_HIDDEN_SIZE = 300   # hidden dimension in FFN

# Training schedule (Noam / warm-up cosine used by chemprop MPNN)
BATCH_SIZE = 64         # molecules per mini-batch
WARMUP_EPOCHS = 2       # epochs of LR warm-up
INIT_LR = 1e-4          # starting learning rate
MAX_LR = 8e-4           # peak learning rate
FINAL_LR = 1e-4         # final learning rate after decay

# Misc
BATCH_NORM = True       # apply batch normalisation on aggregated fingerprint
NUM_WORKERS = 15         # dataloader workers (>0 is faster on Linux)
SEED = 42

# Task weights — relative loss weight per output task (same order as TARGET_COLS)
# Default [1,1,1] treats all tasks equally. Upweight pEC50 to prioritise it.
TASK_WEIGHTS = [3.0, 1.0, 1.0]

# Ensemble: train multiple models with different seeds and average predictions
ENSEMBLE_SEEDS = [SEED]           # single model (set to [42, 0] etc. for ensemble)
BUDGET_PER_MODEL = TIME_BUDGET    # wall-clock seconds per ensemble member

# ---------------------------------------------------------------------------
# Model config (read-only after build — logged at startup)
# ---------------------------------------------------------------------------

N_TASKS = len(TARGET_COLS)   # derived from TARGET_COLS above


@dataclass
class MPNNConfig:
    depth: int = DEPTH
    hidden_size: int = HIDDEN_SIZE
    ffn_num_layers: int = FFN_NUM_LAYERS
    ffn_hidden_size: int = FFN_HIDDEN_SIZE if FFN_HIDDEN_SIZE else HIDDEN_SIZE
    dropout: float = DROPOUT
    n_tasks: int = N_TASKS
    batch_norm: bool = BATCH_NORM
    warmup_epochs: int = WARMUP_EPOCHS
    init_lr: float = INIT_LR
    max_lr: float = MAX_LR
    final_lr: float = FINAL_LR


# ---------------------------------------------------------------------------
# TimeBudget callback — stops Lightning training when wall-clock time is up
# ---------------------------------------------------------------------------

class TimeBudgetCallback(Callback):
    """Stop training once `budget_seconds` of *active* training time have elapsed."""

    def __init__(self, budget_seconds: float):
        self.budget = budget_seconds
        self._start: float | None = None
        self._elapsed: float = 0.0
        # exposed so the main script can read it after training
        self.total_training_seconds: float = 0.0

    def on_train_start(self, trainer, pl_module):
        self._start = time.time()

    def on_train_epoch_end(self, trainer, pl_module):
        if self._start is None:
            return
        now = time.time()
        self._elapsed = now - self._start
        self.total_training_seconds = self._elapsed
        remaining = max(0.0, self.budget - self._elapsed)
        epoch = trainer.current_epoch
        print(
            f"  [TimeBudget] epoch {epoch} done | "
            f"elapsed: {self._elapsed:.1f}s | remaining: {remaining:.1f}s"
        )
        if self._elapsed >= self.budget:
            print("  [TimeBudget] budget exhausted — stopping training.")
            trainer.should_stop = True


class BestValLossCallback(Callback):
    """Restores the model weights from the epoch with the lowest val_loss at end of training."""

    def __init__(self):
        self.best_val_loss = float('inf')
        self._best_state: dict | None = None

    def on_validation_epoch_end(self, trainer, pl_module):
        val_loss = trainer.callback_metrics.get('val_loss', float('inf'))
        if float(val_loss) < self.best_val_loss:
            self.best_val_loss = float(val_loss)
            self._best_state = {k: v.clone() for k, v in pl_module.state_dict().items()}

    def on_fit_end(self, trainer, pl_module):
        if self._best_state is not None:
            pl_module.load_state_dict(self._best_state)
            print(f"  [BestValLoss] Restored weights from best val_loss={self.best_val_loss:.6f}")


class TopKAvgCallback(Callback):
    """Averages weights from the K epochs with lowest val_loss at end of training.

    Weight averaging over the best K checkpoints is a form of ensembling in
    weight space that can reduce variance without additional compute at inference.
    """

    def __init__(self, k: int = 5):
        self.k = k
        self._snapshots: list[tuple[float, dict]] = []  # (val_loss, state_dict)

    def on_validation_epoch_end(self, trainer, pl_module):
        val_loss = float(trainer.callback_metrics.get('val_loss', float('inf')))
        state = {key: val.clone() for key, val in pl_module.state_dict().items()}
        self._snapshots.append((val_loss, state))
        # Keep only the K best (lowest val_loss)
        self._snapshots.sort(key=lambda x: x[0])
        if len(self._snapshots) > self.k:
            self._snapshots.pop()  # remove worst

    def on_fit_end(self, trainer, pl_module):
        if not self._snapshots:
            return
        # Average weights from top-K snapshots
        avg_state = {}
        for key in self._snapshots[0][1]:
            tensors = [snap[1][key].float() for snap in self._snapshots]
            avg_state[key] = torch.stack(tensors).mean(0).to(self._snapshots[0][1][key].dtype)
        pl_module.load_state_dict(avg_state)
        best_losses = [f"{s[0]:.4f}" for s in self._snapshots]
        print(f"  [TopKAvg] Averaged weights from K={len(self._snapshots)} best epochs: {best_losses}")


class EMACallback(Callback):
    """Exponential moving average (EMA) of model weights for smoother optimization.

    Maintains a running EMA of parameters updated after each training batch.
    At the end of training, loads EMA weights for evaluation — often more
    stable than the final noisy SGD iterate.
    """

    def __init__(self, decay: float = 0.999):
        self.decay = decay
        self._ema_state: dict | None = None

    def on_train_batch_end(self, trainer, pl_module, *args, **kwargs):
        sd = pl_module.state_dict()
        if self._ema_state is None:
            self._ema_state = {k: v.clone() for k, v in sd.items()}
        else:
            for key in self._ema_state:
                self._ema_state[key] = (
                    self.decay * self._ema_state[key] + (1.0 - self.decay) * sd[key]
                )

    def on_fit_end(self, trainer, pl_module):
        if self._ema_state is not None:
            pl_module.load_state_dict(self._ema_state)
            print(f"  [EMA] Loaded EMA weights (decay={self.decay})")


FINE_TUNE_SECS = 45  # last N seconds of training devoted to pEC50-only fine-tuning


class TwoStageMixin:
    """Mixin: train multi-task for most of budget, then fine-tune pEC50-only.

    For the last FINE_TUNE_SECS seconds, only the pEC50 loss (column 0) is
    backpropagated. This lets the model specialize on the primary metric after
    the auxiliary tasks have shaped the shared representation.
    """

    _ft_mode: bool = False
    _ft_training_start: float = None

    def on_train_start(self):
        super().on_train_start()
        self._ft_training_start = time.time()

    def training_step(self, batch, batch_idx):
        # Switch to fine-tune mode when approaching the time budget
        if not self._ft_mode and self._ft_training_start is not None:
            elapsed = time.time() - self._ft_training_start
            if elapsed >= TIME_BUDGET - FINE_TUNE_SECS:
                self._ft_mode = True
                print(f"\n[Stage 2] pEC50-only fine-tuning starts at t={elapsed:.1f}s")

        batch_size = self.get_batch_size(batch)
        bmg, V_d, X_d, targets, weights, lt_mask, gt_mask = batch
        preds = self(bmg, V_d, X_d)
        mask = targets.isfinite()
        if self._ft_mode:
            mask[:, 1:] = False  # Only backprop pEC50 loss during fine-tuning
        targets = targets.nan_to_num(nan=0.0)
        l = self.criterion(preds, targets, mask, weights, lt_mask, gt_mask)
        self.log("train_loss", self.criterion, batch_size=batch_size, prog_bar=True, on_epoch=True)
        return l



def build_model(config: MPNNConfig, output_transform=None, n_extra_features: int = 0, n_atom_descriptors: int = 0, d_v: int = 72, task_weights=None):
    """Construct a chemprop MPNN for multi-task regression."""
    from chemprop.models import MPNN
    from chemprop.nn import (
        AtomMessagePassing,
        BondMessagePassing,
        NormAggregation,
        RegressionFFN,
        metrics as cp_metrics,
    )

    mp = AtomMessagePassing(
        depth=config.depth,
        d_v=d_v,
        d_h=config.hidden_size,
        dropout=config.dropout,
        d_vd=n_atom_descriptors if n_atom_descriptors > 0 else None,
    )

    agg = NormAggregation(norm=25.0)  # drug-like molecules ~30-40 atoms, default 100 is too high

    # mp.output_dim = hidden_size + n_atom_descriptors when d_vd is set
    ffn_input_dim = mp.output_dim + n_extra_features

    ffn_kwargs = dict(
        n_tasks=config.n_tasks,
        input_dim=ffn_input_dim,
        hidden_dim=config.ffn_hidden_size,
        n_layers=config.ffn_num_layers,
        dropout=config.dropout,
    )
    if output_transform is not None:
        ffn_kwargs["output_transform"] = output_transform
    if task_weights is not None:
        ffn_kwargs["task_weights"] = torch.tensor(task_weights, dtype=torch.float32)

    ffn = RegressionFFN(**ffn_kwargs)

    model = MPNN(
        message_passing=mp,
        agg=agg,
        predictor=ffn,
        batch_norm=config.batch_norm,
        warmup_epochs=config.warmup_epochs,
        init_lr=config.init_lr,
        max_lr=config.max_lr,
        final_lr=config.final_lr,
        metrics=[cp_metrics.RMSE(), cp_metrics.MAE()],
    )
    return model


# ---------------------------------------------------------------------------
# Setup
# ---------------------------------------------------------------------------

t_start = time.time()
torch.set_float32_matmul_precision("high")

config = MPNNConfig()
print(f"MPNNConfig: {asdict(config)}")
print(f"Target columns ({N_TASKS}): {TARGET_COLS}")
print(f"Extra features: {EXTRA_FEATURES_FN.__name__ if EXTRA_FEATURES_FN is not None else 'None'}")
print(f"Time budget: {TIME_BUDGET}s")

# ---- Dataloaders -----------------------------------------------------------
print("\nLoading data…")
train_loader = make_dataloader("train",      target_cols=TARGET_COLS, batch_size=BATCH_SIZE, num_workers=NUM_WORKERS, extra_features_fn=EXTRA_FEATURES_FN)
val_loader   = make_dataloader("validation", target_cols=TARGET_COLS, batch_size=BATCH_SIZE, num_workers=NUM_WORKERS, extra_features_fn=EXTRA_FEATURES_FN)
test_loader  = make_dataloader("test",       target_cols=TARGET_COLS, batch_size=BATCH_SIZE, num_workers=NUM_WORKERS, extra_features_fn=EXTRA_FEATURES_FN)

train_dset = train_loader.dataset
val_dset   = val_loader.dataset
test_dset  = test_loader.dataset

# ---- Target scaling (fit on train, apply to val + test; inverse baked into model) --
from chemprop.nn.transforms import UnscaleTransform

output_scaler = train_dset.normalize_targets()
val_dset.normalize_targets(output_scaler)
test_dset.normalize_targets(output_scaler)
output_transform = UnscaleTransform.from_standard_scaler(output_scaler)

print(
    f"Train molecules: {len(train_dset):,} | "
    f"Val molecules:   {len(val_dset):,} | "
    f"Test molecules:  {len(test_dset):,}"
)

# ---- Gasteiger charges as per-atom vertex descriptors ----------------------
_n_atom_desc = 0
if USE_GASTEIGER_VD:
    from rdkit.Chem import AllChem

    def _add_gasteiger_vd(dset):
        for dp in dset.data:
            if dp.mol is None:
                continue
            mol = dp.mol
            AllChem.ComputeGasteigerCharges(mol)
            charges = np.array(
                [[mol.GetAtomWithIdx(i).GetDoubleProp("_GasteigerCharge")]
                 for i in range(mol.GetNumAtoms())],
                dtype=np.float32,
            )
            # Replace NaN/inf with 0
            charges = np.nan_to_num(charges, nan=0.0, posinf=0.0, neginf=0.0)
            dp.V_d = charges
        # Update V_ds via setter to bypass the stale cached _V_ds property
        dset.V_ds = [dp.V_d for dp in dset.data]

    _add_gasteiger_vd(train_dset)
    _add_gasteiger_vd(val_dset)
    _add_gasteiger_vd(test_dset)
    _n_atom_desc = 1
    print("Added Gasteiger charges as 1-dim V_d per atom")

# ---- V1 atom featurizer (133-dim) ------------------------------------------
_d_v = 72  # default atom feature dimension
if USE_V1_FEATURIZER:
    from chemprop.featurizers.atom import MultiHotAtomFeaturizer
    from chemprop.featurizers import SimpleMoleculeMolGraphFeaturizer as _SMF
    _v1_feat = _SMF(atom_featurizer=MultiHotAtomFeaturizer.v1())
    _d_v = _v1_feat.atom_fdim
    for dset in [train_dset, val_dset, test_dset]:
        dset.featurizer = _v1_feat
        dset._init_cache()
    print(f"Switched to v1 atom featurizer (d_v={_d_v})")

# ---- Model (build once to log param count; rebuild per seed inside loop) ---
_n_extra = 0
if EXTRA_FEATURES_FN is not None:
    _test_feat = EXTRA_FEATURES_FN("C")
    _n_extra = len(_test_feat) if _test_feat is not None else 0
_sample_model = build_model(config, output_transform=output_transform, n_extra_features=_n_extra, n_atom_descriptors=_n_atom_desc, d_v=_d_v, task_weights=TASK_WEIGHTS)
n_params = sum(p.numel() for p in _sample_model.parameters() if p.requires_grad)
print(f"Trainable parameters per model: {n_params:,}")
del _sample_model

# ---------------------------------------------------------------------------
# Training — ensemble loop
# ---------------------------------------------------------------------------

# Pick accelerator automatically
accelerator = "gpu" if torch.cuda.is_available() else "cpu"
devices = 1
MAX_EPOCHS = 500

trained_models = []
total_training_time = 0.0
t_train_start = time.time()

for _ens_idx, _ens_seed in enumerate(ENSEMBLE_SEEDS):
    print(f"\n[Ensemble {_ens_idx+1}/{len(ENSEMBLE_SEEDS)}, SEED={_ens_seed}]")
    torch.manual_seed(_ens_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(_ens_seed)

    _model = build_model(config, output_transform=output_transform, n_extra_features=_n_extra, n_atom_descriptors=_n_atom_desc, d_v=_d_v, task_weights=TASK_WEIGHTS)

    _time_cb = TimeBudgetCallback(budget_seconds=BUDGET_PER_MODEL)
    _best_val_cb = BestValLossCallback()

    _trainer = pl.Trainer(
        accelerator=accelerator,
        devices=devices,
        max_epochs=MAX_EPOCHS,
        logger=True,
        enable_checkpointing=False,
        enable_progress_bar=True,
        callbacks=[_time_cb, _best_val_cb],
    )

    print(f"  Starting training (accelerator={accelerator}, budget={BUDGET_PER_MODEL}s)…")
    _trainer.fit(_model, train_loader, val_loader)

    total_training_time += _time_cb.total_training_seconds
    _num_epochs = _trainer.current_epoch + 1
    print(f"  Done: {_num_epochs} epochs, {_time_cb.total_training_seconds:.1f}s")
    trained_models.append(_model)

# ---------------------------------------------------------------------------
# Ensemble evaluation helpers
# ---------------------------------------------------------------------------

def _ensemble_preds(models, loader):
    """Return averaged unscaled predictions [N, T] from all ensemble models."""
    device = next(models[0].parameters()).device
    per_model = []
    for m in models:
        m.eval()
        batches = []
        with torch.no_grad():
            for batch in loader:
                bmg, V_d, X_d, *_ = batch
                if device.type == "cuda":
                    bmg = bmg.to(device)
                batches.append(m(bmg, V_d, X_d).cpu().numpy())
        per_model.append(np.concatenate(batches, axis=0))
    return np.mean(per_model, axis=0)


def ensemble_evaluate(models, loader, target_cols):
    """Evaluate an ensemble of models; replicates evaluate_regression metric."""
    preds = _ensemble_preds(models, loader)
    all_tgt = []
    for batch in loader:
        tgt = batch[3] if isinstance(batch, (list, tuple)) else batch.Y
        all_tgt.append(tgt.numpy())
    targets = np.concatenate(all_tgt, axis=0)

    per_task = {}
    rmses = []
    for i, col in enumerate(target_cols):
        p = preds[:, i]
        t = targets[:, i]
        mask = np.isfinite(t) & np.isfinite(p)
        if mask.sum() == 0:
            per_task[col] = float("nan")
        else:
            rmse = float(np.sqrt(np.mean((p[mask] - t[mask]) ** 2)))
            per_task[col] = rmse
            rmses.append(rmse)
    return {"mean_rmse": float(np.mean(rmses)) if rmses else float("nan"), "per_task": per_task}


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

print("\nRunning final evaluation on validation set…")
val_results  = ensemble_evaluate(trained_models, val_loader,  TARGET_COLS)

print("Running final evaluation on test set…")
test_results = ensemble_evaluate(trained_models, test_loader, TARGET_COLS)

# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------

t_end = time.time()
startup_time = t_train_start - t_start
peak_vram_mb = (
    torch.cuda.max_memory_allocated() / 1024 / 1024
    if torch.cuda.is_available()
    else 0.0
)

print("\n--- Results ---")
print(f"val_mean_rmse:      {val_results['mean_rmse']:.6f}")
print("val_per_task_rmse:")
for task, rmse in val_results["per_task"].items():
    tag = f"{rmse:.6f}" if not math.isnan(rmse) else "  n/a (no labels)"
    print(f"  {task:<40s}: {tag}")
print(f"\ntest_mean_rmse:     {test_results['mean_rmse']:.6f}")
print("test_per_task_rmse:")
for task, rmse in test_results["per_task"].items():
    tag = f"{rmse:.6f}" if not math.isnan(rmse) else "  n/a (no labels)"
    print(f"  {task:<40s}: {tag}")
print(f"\ntraining_seconds:   {total_training_time:.1f}")
print(f"total_seconds:      {t_end - t_start:.1f}")
print(f"startup_seconds:    {startup_time:.1f}")
print(f"peak_vram_mb:       {peak_vram_mb:.1f}")
print(f"ensemble_members:   {len(trained_models)}")
print(f"ensemble_seeds:     {ENSEMBLE_SEEDS}")
print(f"budget_per_model:   {BUDGET_PER_MODEL}s")
print(f"train_molecules:    {len(train_dset):,}")
print(f"val_molecules:      {len(val_dset):,}")
print(f"test_molecules:     {len(test_dset):,}")
print(f"n_tasks:            {N_TASKS}")
print(f"target_cols:        {TARGET_COLS}")
print(f"extra_features:     {EXTRA_FEATURES_FN.__name__ if EXTRA_FEATURES_FN is not None else 'None'}")
print(f"depth:              {DEPTH}")
print(f"hidden_size:        {HIDDEN_SIZE}")
print(f"ffn_num_layers:     {FFN_NUM_LAYERS}")
print(f"batch_size:         {BATCH_SIZE}")
print(f"max_lr:             {MAX_LR}")

#write to log file
with open('run.log', 'w') as f:
    f.write(f"val_mean_rmse:      {val_results['mean_rmse']:.6f}\n")
    f.write("val_per_task_rmse:\n")
    for task, rmse in val_results["per_task"].items():
        tag = f"{rmse:.6f}" if not math.isnan(rmse) else "  n/a (no labels)"
        f.write(f"  {task:<40s}: {tag}\n")
    f.write(f"\ntest_mean_rmse:     {test_results['mean_rmse']:.6f}\n")
    f.write("test_per_task_rmse:\n")
    for task, rmse in test_results["per_task"].items():
        tag = f"{rmse:.6f}" if not math.isnan(rmse) else "  n/a (no labels)"
        f.write(f"  {task:<40s}: {tag}\n")
    f.write(f"\ntraining_seconds:   {total_training_time:.1f}\n")
    f.write(f"total_seconds:      {t_end - t_start:.1f}\n")
    f.write(f"startup_seconds:    {startup_time:.1f}\n")
    f.write(f"peak_vram_mb:       {peak_vram_mb:.1f}\n")
    f.write(f"ensemble_members:   {len(trained_models)}\n")
    f.write(f"ensemble_seeds:     {ENSEMBLE_SEEDS}\n")
    f.write(f"train_molecules:    {len(train_dset):,}\n")
    f.write(f"val_molecules:      {len(val_dset):,}\n")
    f.write(f"test_molecules:     {len(test_dset):,}\n")
    f.write(f"n_tasks:            {N_TASKS}\n")
    f.write(f"target_cols:        {TARGET_COLS}\n")
    f.write(f"extra_features:     {EXTRA_FEATURES_FN.__name__ if EXTRA_FEATURES_FN is not None else 'None'}\n")
    f.write(f"depth:              {DEPTH}\n")
    f.write(f"hidden_size:        {HIDDEN_SIZE}\n")
    f.write(f"ffn_num_layers:     {FFN_NUM_LAYERS}\n")
    f.write(f"batch_size:         {BATCH_SIZE}\n")
    f.write(f"max_lr:             {MAX_LR}\n")
