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
    "Emax_estimate (log2FC vs. baseline)",
    "Emax.vs.pos.ctrl_estimate (dimensionless)",
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
MAX_LR = 2e-3           # peak learning rate - larger batch allows higher LR
FINAL_LR = 1e-4         # final learning rate after decay

# Misc
BATCH_NORM = True       # apply batch normalisation on aggregated fingerprint
NUM_WORKERS = 15         # dataloader workers (>0 is faster on Linux)
SEED = 42

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


# ---------------------------------------------------------------------------
# Build model
# ---------------------------------------------------------------------------

def build_model(config: MPNNConfig, output_transform=None, n_extra_features: int = 0):
    """Construct a chemprop MPNN for multi-task regression."""
    from chemprop.models import MPNN
    from chemprop.nn import (
        BondMessagePassing,
        NormAggregation,
        RegressionFFN,
        metrics as cp_metrics,
    )

    mp = BondMessagePassing(
        depth=config.depth,
        d_h=config.hidden_size,
        dropout=config.dropout,
    )

    agg = NormAggregation(norm=25.0)  # drug-like molecules ~30-40 atoms, default 100 is too high

    ffn_kwargs = dict(
        n_tasks=config.n_tasks,
        input_dim=config.hidden_size + n_extra_features,
        hidden_dim=config.ffn_hidden_size,
        n_layers=config.ffn_num_layers,
        dropout=config.dropout,
    )
    if output_transform is not None:
        ffn_kwargs["output_transform"] = output_transform

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
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed(SEED)
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

# ---- Model -----------------------------------------------------------------
_n_extra = 0
if EXTRA_FEATURES_FN is not None:
    _test_feat = EXTRA_FEATURES_FN("C")
    _n_extra = len(_test_feat) if _test_feat is not None else 0
model = build_model(config, output_transform=output_transform, n_extra_features=_n_extra)

n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"Trainable parameters: {n_params:,}")

# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

time_callback = TimeBudgetCallback(budget_seconds=TIME_BUDGET)

# Pick accelerator automatically
accelerator = "gpu" if torch.cuda.is_available() else "cpu"
devices = 1

# Estimate a generous upper bound on epochs; TimeBudgetCallback will stop early
# Chemprop default is ~max_epochs=50 – we set a large ceiling and rely on time.
MAX_EPOCHS = 500

trainer = pl.Trainer(
    accelerator=accelerator,
    devices=devices,
    max_epochs=MAX_EPOCHS,
    logger=True,
    enable_checkpointing=False,
    enable_progress_bar=True,
    callbacks=[time_callback],
)

print(f"\nStarting training (accelerator={accelerator}, max wall-clock={TIME_BUDGET}s)…\n")
t_train_start = time.time()

trainer.fit(model, train_loader, val_loader)

total_training_time = time_callback.total_training_seconds
num_epochs = trainer.current_epoch + 1

# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

model.eval()
print("\nRunning final evaluation on validation set…")
val_results  = evaluate_regression(model, val_loader,  target_cols=TARGET_COLS, batch_size=BATCH_SIZE)

print("Running final evaluation on test set…")
test_results = evaluate_regression(model, test_loader, target_cols=TARGET_COLS, batch_size=BATCH_SIZE)

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
print(f"num_epochs:         {num_epochs}")
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
    f.write(f"num_epochs:         {num_epochs}\n")
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
