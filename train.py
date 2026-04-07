"""
ADME multi-task regression training script.
Uses Chemprop MPNN + Lightning for SMILES → 9 ADME property prediction.

Usage:
    uv run train.py

Edit the constants below to tune the run. No CLI flags needed.
"""

import os

os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"
os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"

import shutil
import tempfile
import time
import math
from dataclasses import dataclass, asdict

import numpy as np
import torch
import lightning.pytorch as pl
from lightning.pytorch.callbacks import Callback, ModelCheckpoint

from prepare import TIME_BUDGET, make_dataloader, evaluate_regression

# ---------------------------------------------------------------------------
# Hyperparameters (edit these directly — no CLI flags needed)
# ---------------------------------------------------------------------------

# Message passing
DEPTH = 3               # number of bond message-passing steps
HIDDEN_SIZE = 300       # hidden dimension in message passing layers
DROPOUT = 0.0           # dropout applied in both MP and FFN

# Feed-forward network (predictor)
FFN_NUM_LAYERS = 2      # number of FFN layers after aggregation
FFN_HIDDEN_SIZE = 300   # hidden dimension in FFN (None → same as HIDDEN_SIZE)

# Training schedule (Noam / warm-up cosine used by chemprop MPNN)
BATCH_SIZE = 64         # molecules per mini-batch
WARMUP_EPOCHS = 2       # epochs of LR warm-up
INIT_LR = 1e-4          # starting learning rate
MAX_LR = 1e-3           # peak learning rate
FINAL_LR = 1e-4         # final learning rate after decay

# Misc
BATCH_NORM = True       # apply batch normalisation on aggregated fingerprint
NUM_WORKERS = 15         # dataloader workers (>0 is faster on Linux)
SEED = 42

# columns to choose from
"""
['Molecule Name', 'SMILES', 'OCNT Batch', 'pEC50',
       'pEC50_ci.lower (-log10(molarity))',
       'pEC50_ci.upper (-log10(molarity))',
       'Emax_estimate (log2FC vs. baseline)',
       'Emax_ci.lower (log2FC vs. baseline)',
       'Emax_ci.upper (log2FC vs. baseline)',
       'Emax.vs.pos.ctrl_estimate (dimensionless)',
       'Emax.vs.pos.ctrl_ci.lower (dimensionless)',
       'Emax.vs.pos.ctrl_ci.upper (dimensionless)',
       'pEC50_std.error (-log10(molarity))',
       'Emax_std.error (log2FC vs. baseline)',
       'Emax.vs.pos.ctrl_std.error (dimensionless)', 'Split',
       'counter_OCNT Batch', 'counter_pEC50',
       'counter_pEC50_ci.lower (-log10(molarity))',
       'counter_pEC50_ci.upper (-log10(molarity))',
       'counter_Emax_estimate (log2FC vs. baseline)',
       'counter_Emax_ci.lower (log2FC vs. baseline)',
       'counter_Emax_ci.upper (log2FC vs. baseline)',
       'counter_Emax.vs.pos.ctrl_estimate (dimensionless)',
       'counter_Emax.vs.pos.ctrl_ci.lower (dimensionless)',
       'counter_Emax.vs.pos.ctrl_ci.upper (dimensionless)',
       'counter_pEC50_std.error (-log10(molarity))',
       'counter_Emax_std.error (log2FC vs. baseline)',
       'counter_Emax.vs.pos.ctrl_std.error (dimensionless)']
"""

# be NaN for compounds not screened in the PXR-null cell line.
TARGET_COLS = [
    "pEC50",
    "Emax_estimate (log2FC vs. baseline)",
    "counter_pEC50",
    "counter_Emax_estimate (log2FC vs. baseline)",
]

# ---------------------------------------------------------------------------
# Model config (read-only after build — logged at startup)
# ---------------------------------------------------------------------------

N_TASKS = len(TARGET_COLS)   # 2 PXR regression targets (pEC50, Emax)


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


# ---------------------------------------------------------------------------
# Build model
# ---------------------------------------------------------------------------

def build_model(config: MPNNConfig, output_transform=None):
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

    agg = NormAggregation()

    ffn_kwargs = dict(
        n_tasks=config.n_tasks,
        input_dim=config.hidden_size,
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
print(f"Time budget: {TIME_BUDGET}s")

# ---- Dataloaders -----------------------------------------------------------
print("\nLoading data…")
train_loader = make_dataloader("train", batch_size=BATCH_SIZE, num_workers=NUM_WORKERS)
val_loader   = make_dataloader("validation", batch_size=BATCH_SIZE, num_workers=NUM_WORKERS)
test_loader  = make_dataloader("test", batch_size=BATCH_SIZE, num_workers=NUM_WORKERS)

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
model = build_model(config, output_transform=output_transform)

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

ckpt_dir = tempfile.mkdtemp(prefix="pxr_ckpt_")
checkpoint_callback = ModelCheckpoint(
    dirpath=ckpt_dir,
    filename="best",
    monitor="val_loss",
    mode="min",
    save_top_k=1,
    verbose=False,
)

trainer = pl.Trainer(
    accelerator=accelerator,
    devices=devices,
    max_epochs=MAX_EPOCHS,
    logger=True,
    enable_checkpointing=True,    # required when ModelCheckpoint is in callbacks
    enable_progress_bar=True,
    callbacks=[time_callback, checkpoint_callback],
)

print(f"\nStarting training (accelerator={accelerator}, max wall-clock={TIME_BUDGET}s)…\n")
t_train_start = time.time()

trainer.fit(model, train_loader, val_loader)

total_training_time = time_callback.total_training_seconds
num_epochs = trainer.current_epoch + 1

# ---------------------------------------------------------------------------
# Load best checkpoint before evaluation
# ---------------------------------------------------------------------------

best_ckpt = checkpoint_callback.best_model_path
if best_ckpt:
    print(f"\nLoading best checkpoint (epoch {checkpoint_callback.best_model_score:.6f} val_loss): {best_ckpt}")
    best_state = torch.load(best_ckpt, map_location="cuda" if torch.cuda.is_available() else "cpu", weights_only=False)
    model.load_state_dict(best_state["state_dict"])
else:
    print("\nNo checkpoint saved — using final model weights.")

# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

model.eval()
print("\nRunning final evaluation on validation set…")
val_results  = evaluate_regression(model, val_loader,  batch_size=BATCH_SIZE)

print("Running final evaluation on test set…")
test_results = evaluate_regression(model, test_loader, batch_size=BATCH_SIZE)

shutil.rmtree(ckpt_dir, ignore_errors=True)

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

METRIC_KEYS = ["rmse", "mae", "rae", "r2", "spearman", "kendall"]


def _fmt(v):
    return f"{v:.6f}" if not math.isnan(v) else "n/a"


def _print_results(label: str, results: dict):
    print(f"\n--- {label} ---")
    print(f"  mean_rmse: {_fmt(results['mean_rmse'])}")
    header = f"  {'task':<42s}" + "".join(f"  {m:>10s}" for m in METRIC_KEYS)
    print(header)
    print("  " + "-" * (42 + 12 * len(METRIC_KEYS)))
    for task, metrics in results["per_task"].items():
        row = f"  {task:<42s}" + "".join(f"  {_fmt(metrics[m]):>10s}" for m in METRIC_KEYS)
        print(row)


def _write_results(f, label: str, results: dict):
    f.write(f"\n--- {label} ---\n")
    f.write(f"  mean_rmse: {_fmt(results['mean_rmse'])}\n")
    header = f"  {'task':<42s}" + "".join(f"  {m:>10s}" for m in METRIC_KEYS)
    f.write(header + "\n")
    f.write("  " + "-" * (42 + 12 * len(METRIC_KEYS)) + "\n")
    for task, metrics in results["per_task"].items():
        row = f"  {task:<42s}" + "".join(f"  {_fmt(metrics[m]):>10s}" for m in METRIC_KEYS)
        f.write(row + "\n")


_print_results("Validation", val_results)
_print_results("Test", test_results)

print(f"\ntraining_seconds:   {total_training_time:.1f}")
print(f"total_seconds:      {t_end - t_start:.1f}")
print(f"startup_seconds:    {startup_time:.1f}")
print(f"peak_vram_mb:       {peak_vram_mb:.1f}")
print(f"num_epochs:         {num_epochs}")
print(f"train_molecules:    {len(train_dset):,}")
print(f"val_molecules:      {len(val_dset):,}")
print(f"test_molecules:     {len(test_dset):,}")
print(f"n_tasks:            {N_TASKS}")
print(f"depth:              {DEPTH}")
print(f"hidden_size:        {HIDDEN_SIZE}")
print(f"ffn_num_layers:     {FFN_NUM_LAYERS}")
print(f"batch_size:         {BATCH_SIZE}")
print(f"max_lr:             {MAX_LR}")

with open("run.log", "w") as f:
    _write_results(f, "Validation", val_results)
    _write_results(f, "Test", test_results)
    f.write(f"\ntraining_seconds:   {total_training_time:.1f}\n")
    f.write(f"total_seconds:      {t_end - t_start:.1f}\n")
    f.write(f"startup_seconds:    {startup_time:.1f}\n")
    f.write(f"peak_vram_mb:       {peak_vram_mb:.1f}\n")
    f.write(f"num_epochs:         {num_epochs}\n")
    f.write(f"train_molecules:    {len(train_dset):,}\n")
    f.write(f"val_molecules:      {len(val_dset):,}\n")
    f.write(f"test_molecules:     {len(test_dset):,}\n")
    f.write(f"n_tasks:            {N_TASKS}\n")
    f.write(f"depth:              {DEPTH}\n")
    f.write(f"hidden_size:        {HIDDEN_SIZE}\n")
    f.write(f"ffn_num_layers:     {FFN_NUM_LAYERS}\n")
    f.write(f"batch_size:         {BATCH_SIZE}\n")
    f.write(f"max_lr:             {MAX_LR}\n")
