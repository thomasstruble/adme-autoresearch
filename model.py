"""
PXR Activation Model — best configuration (exp223).

Provides two entry points:
  uv run model.py train            Train NUM_RUNS times, keep the best by val pEC50.
                                   Saves the best model to model.pt.
  uv run model.py predict input.csv [output.csv]
                                   Run inference on a CSV with a 'smiles' column.
                                   Writes predictions alongside original columns.
                                   output.csv defaults to input_predictions.csv.

Model config (exp223 best):
  DEPTH=5, HIDDEN_SIZE=300, FFN_NUM_LAYERS=2, FFN_HIDDEN_SIZE=300
  BATCH_SIZE=64, MAX_LR=8e-4, TASK_WEIGHTS=[2,1,1], SEED=42
  BestValLossCallback active, NUM_RUNS=3 (seeds 42, 43, 44 — best saved)
"""

import os
import sys
import time
import math

os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"
os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"

import numpy as np
import torch
import lightning.pytorch as pl
from lightning.pytorch.callbacks import Callback
from dataclasses import dataclass

# ---------------------------------------------------------------------------
# Configuration — best model (exp223)
# ---------------------------------------------------------------------------

TARGET_COLS = [
    "pEC50",
    "Emax.vs.pos.ctrl_estimate (dimensionless)",
    "Emax.vs.pos.ctrl_ci.upper (dimensionless)",
]

DEPTH           = 5
HIDDEN_SIZE     = 300
FFN_NUM_LAYERS  = 2
FFN_HIDDEN_SIZE = 300
DROPOUT         = 0.0
BATCH_NORM      = True
BATCH_SIZE      = 64
WARMUP_EPOCHS   = 2
INIT_LR         = 1e-4
MAX_LR          = 8e-4
FINAL_LR        = 1e-4
TASK_WEIGHTS    = [2.0, 1.0, 1.0]
SEED            = 42
NUM_WORKERS     = 4           # conservative default for portability
TIME_BUDGET     = 150         # seconds (matches prepare.py)
NUM_RUNS        = 3           # train this many times, keep the run with best val pEC50 RMSE

MODEL_PATH = "model.pt"       # where the trained model bundle is saved/loaded

N_TASKS = len(TARGET_COLS)


# ---------------------------------------------------------------------------
# Callbacks
# ---------------------------------------------------------------------------

class TimeBudgetCallback(Callback):
    def __init__(self, budget_seconds: float):
        self.budget = budget_seconds
        self._start: float | None = None
        self.total_training_seconds: float = 0.0

    def on_train_start(self, trainer, pl_module):
        self._start = time.time()

    def on_train_epoch_end(self, trainer, pl_module):
        if self._start is None:
            return
        elapsed = time.time() - self._start
        self.total_training_seconds = elapsed
        remaining = max(0.0, self.budget - elapsed)
        print(
            f"  [TimeBudget] epoch {trainer.current_epoch} | "
            f"elapsed: {elapsed:.1f}s | remaining: {remaining:.1f}s"
        )
        if elapsed >= self.budget:
            print("  [TimeBudget] budget exhausted — stopping.")
            trainer.should_stop = True


class BestValLossCallback(Callback):
    def __init__(self):
        self.best_val_loss = float("inf")
        self._best_state: dict | None = None

    def on_validation_epoch_end(self, trainer, pl_module):
        val_loss = trainer.callback_metrics.get("val_loss", float("inf"))
        if float(val_loss) < self.best_val_loss:
            self.best_val_loss = float(val_loss)
            self._best_state = {k: v.clone() for k, v in pl_module.state_dict().items()}

    def on_fit_end(self, trainer, pl_module):
        if self._best_state is not None:
            pl_module.load_state_dict(self._best_state)
            print(f"  [BestValLoss] Restored best val_loss={self.best_val_loss:.6f}")


# ---------------------------------------------------------------------------
# Model builder
# ---------------------------------------------------------------------------

@dataclass
class MPNNConfig:
    depth: int = DEPTH
    hidden_size: int = HIDDEN_SIZE
    ffn_num_layers: int = FFN_NUM_LAYERS
    ffn_hidden_size: int = FFN_HIDDEN_SIZE
    dropout: float = DROPOUT
    n_tasks: int = N_TASKS
    batch_norm: bool = BATCH_NORM
    warmup_epochs: int = WARMUP_EPOCHS
    init_lr: float = INIT_LR
    max_lr: float = MAX_LR
    final_lr: float = FINAL_LR


def build_model(config: MPNNConfig, output_transform=None, task_weights=None):
    from chemprop.models import MPNN
    from chemprop.nn import AtomMessagePassing, NormAggregation, RegressionFFN, metrics as cp_metrics

    mp = AtomMessagePassing(
        depth=config.depth,
        d_h=config.hidden_size,
        dropout=config.dropout,
    )
    agg = NormAggregation(norm=25.0)

    ffn_kwargs = dict(
        n_tasks=config.n_tasks,
        input_dim=mp.output_dim,
        hidden_dim=config.ffn_hidden_size,
        n_layers=config.ffn_num_layers,
        dropout=config.dropout,
    )
    if output_transform is not None:
        ffn_kwargs["output_transform"] = output_transform
    if task_weights is not None:
        ffn_kwargs["task_weights"] = torch.tensor(task_weights, dtype=torch.float32)

    ffn = RegressionFFN(**ffn_kwargs)

    return MPNN(
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


# ---------------------------------------------------------------------------
# Train
# ---------------------------------------------------------------------------

def train():
    from prepare import make_dataloader, evaluate_regression
    from chemprop.nn.transforms import UnscaleTransform

    torch.set_float32_matmul_precision("high")

    print("Loading data…")
    train_loader = make_dataloader("train",      target_cols=TARGET_COLS, batch_size=BATCH_SIZE, num_workers=NUM_WORKERS)
    val_loader   = make_dataloader("validation", target_cols=TARGET_COLS, batch_size=BATCH_SIZE, num_workers=NUM_WORKERS)
    test_loader  = make_dataloader("test",       target_cols=TARGET_COLS, batch_size=BATCH_SIZE, num_workers=NUM_WORKERS)

    train_dset = train_loader.dataset
    val_dset   = val_loader.dataset
    test_dset  = test_loader.dataset

    output_scaler = train_dset.normalize_targets()
    val_dset.normalize_targets(output_scaler)
    test_dset.normalize_targets(output_scaler)
    output_transform = UnscaleTransform.from_standard_scaler(output_scaler)

    config = MPNNConfig()
    _sample = build_model(config, output_transform=output_transform, task_weights=TASK_WEIGHTS)
    n_params = sum(p.numel() for p in _sample.parameters() if p.requires_grad)
    del _sample
    print(f"Parameters: {n_params:,}")

    accelerator = "gpu" if torch.cuda.is_available() else "cpu"

    best_val_pecso  = float("inf")
    best_bundle     = None
    best_val_res    = None
    best_test_res   = None

    for run_idx in range(NUM_RUNS):
        run_seed = SEED + run_idx
        print(f"\n[Run {run_idx+1}/{NUM_RUNS}, seed={run_seed}] Training for {TIME_BUDGET}s…")
        torch.manual_seed(run_seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(run_seed)

        model = build_model(config, output_transform=output_transform, task_weights=TASK_WEIGHTS)

        time_cb     = TimeBudgetCallback(budget_seconds=TIME_BUDGET)
        best_val_cb = BestValLossCallback()

        trainer = pl.Trainer(
            accelerator=accelerator,
            devices=1,
            max_epochs=500,
            logger=False,
            enable_checkpointing=False,
            enable_progress_bar=True,
            callbacks=[time_cb, best_val_cb],
        )

        t0 = time.time()
        trainer.fit(model, train_loader, val_loader)
        elapsed = time.time() - t0
        print(f"  Done in {elapsed:.1f}s ({trainer.current_epoch+1} epochs)")

        model.eval()
        val_res  = evaluate_regression(model, val_loader,  target_cols=TARGET_COLS, batch_size=BATCH_SIZE)
        test_res = evaluate_regression(model, test_loader, target_cols=TARGET_COLS, batch_size=BATCH_SIZE)

        val_pec50 = val_res["per_task"].get("pEC50", float("inf"))
        print(f"  val pEC50={val_pec50:.6f}  test pEC50={test_res['per_task'].get('pEC50', float('nan')):.6f}")

        if val_pec50 < best_val_pecso:
            best_val_pecso = val_pec50
            best_val_res   = val_res
            best_test_res  = test_res
            best_bundle = {
                "model_state_dict": {k: v.clone() for k, v in model.state_dict().items()},
                "output_scaler_mean": output_scaler.mean_,
                "output_scaler_scale": output_scaler.scale_,
                "target_cols": TARGET_COLS,
                "config": config,
                "task_weights": TASK_WEIGHTS,
                "seed": run_seed,
                "val_results": val_res,
                "test_results": test_res,
            }
            print(f"  *** New best (run {run_idx+1}, seed={run_seed}) ***")

    print("\n--- Best run results ---")
    print(f"val_mean_rmse:  {best_val_res['mean_rmse']:.6f}")
    for col, rmse in best_val_res["per_task"].items():
        print(f"  {col}: {rmse:.6f}" if not math.isnan(rmse) else f"  {col}: n/a")
    print(f"test_mean_rmse: {best_test_res['mean_rmse']:.6f}")
    for col, rmse in best_test_res["per_task"].items():
        print(f"  {col}: {rmse:.6f}" if not math.isnan(rmse) else f"  {col}: n/a")

    torch.save(best_bundle, MODEL_PATH)
    print(f"\nBest model saved to {MODEL_PATH} (seed={best_bundle['seed']})")


# ---------------------------------------------------------------------------
# Predict
# ---------------------------------------------------------------------------

def predict(input_csv: str, output_csv: str | None = None):
    import pandas as pd
    from chemprop.data import MoleculeDatapoint, MoleculeDataset
    from chemprop.data.collate import collate_batch
    from torch.utils.data import DataLoader
    from chemprop.nn.transforms import UnscaleTransform
    from sklearn.preprocessing import StandardScaler

    if not os.path.exists(MODEL_PATH):
        print(f"Model file '{MODEL_PATH}' not found. Run: uv run model.py train")
        sys.exit(1)

    print(f"Loading model from {MODEL_PATH}…")
    bundle = torch.load(MODEL_PATH, map_location="cpu", weights_only=False)

    config: MPNNConfig = bundle["config"]
    target_cols: list[str] = bundle["target_cols"]
    task_weights = bundle.get("task_weights", [1.0] * len(target_cols))

    # Rebuild scaler + output transform
    scaler = StandardScaler()
    scaler.mean_  = bundle["output_scaler_mean"]
    scaler.scale_ = bundle["output_scaler_scale"]
    output_transform = UnscaleTransform.from_standard_scaler(scaler)

    model = build_model(config, output_transform=output_transform, task_weights=task_weights)
    model.load_state_dict(bundle["model_state_dict"])
    model.eval()

    print(f"Reading {input_csv}…")
    df = pd.read_csv(input_csv)

    smiles_col = None
    for candidate in ("smiles", "SMILES", "Smiles", "canonical_smiles"):
        if candidate in df.columns:
            smiles_col = candidate
            break
    if smiles_col is None:
        print(f"Error: no SMILES column found. Columns: {list(df.columns)}")
        sys.exit(1)

    smiles_list = df[smiles_col].tolist()
    print(f"Running inference on {len(smiles_list)} molecules…")

    # Build dataset (no targets needed for inference)
    from chemprop.featurizers import SimpleMoleculeMolGraphFeaturizer
    featurizer = SimpleMoleculeMolGraphFeaturizer()
    data_points = [MoleculeDatapoint.from_smi(smi) for smi in smiles_list]
    dset = MoleculeDataset(data_points, featurizer=featurizer)

    loader = DataLoader(
        dset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0,
        collate_fn=collate_batch,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    all_preds = []
    with torch.no_grad():
        for batch in loader:
            bmg, V_d, X_d, *_ = batch
            bmg.to(device)  # mutates in-place; BatchMolGraph.to() returns None
            preds = model(bmg, V_d, X_d)
            all_preds.append(preds.cpu().numpy())

    preds_np = np.concatenate(all_preds, axis=0)  # [N, n_tasks]

    # Add prediction columns
    for i, col in enumerate(target_cols):
        df[f"pred_{col}"] = preds_np[:, i]

    if output_csv is None:
        base = os.path.splitext(input_csv)[0]
        output_csv = f"{base}_predictions.csv"

    df.to_csv(output_csv, index=False)
    print(f"Predictions written to {output_csv}")
    print(f"Output columns: {[f'pred_{c}' for c in target_cols]}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    if len(sys.argv) < 2 or sys.argv[1] not in ("train", "predict"):
        print(__doc__)
        sys.exit(0)

    if sys.argv[1] == "train":
        train()

    elif sys.argv[1] == "predict":
        if len(sys.argv) < 3:
            print("Usage: uv run model.py predict input.csv [output.csv]")
            sys.exit(1)
        input_csv  = sys.argv[2]
        output_csv = sys.argv[3] if len(sys.argv) > 3 else None
        predict(input_csv, output_csv)
