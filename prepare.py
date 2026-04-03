import os

import numpy as np
import pandas as pd
import torch
import glob

from datasets import load_dataset, load_from_disk


# ---------------------------------------------------------------------------
# Constants (fixed, do not modify)
# ---------------------------------------------------------------------------

TIME_BUDGET = 150        # training time budget in seconds (5 minutes)

# Regression target columns present in the dataset
TARGET_COLS = [
    "pEC50",
    "Emax_estimate (log2FC vs. baseline)",
]

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

CACHE_DIR = os.path.join(os.path.expanduser("~"), ".cache", "autoresearch")
DATA_DIR = os.path.join(CACHE_DIR, "data")
BASE_REPO = "openadmet/pxr-challenge-train-test"

VAL_FRACTION  = 0.1   # fraction of training data held out for validation
TEST_FRACTION = 0.1   # fraction of training data held out for local test
SPLIT_SEED = 42


# ---------------------------------------------------------------------------
# Data download
# ---------------------------------------------------------------------------
def download_data():
    """
    Download the PXR challenge training data from Hugging Face, split into
    train/validation/test (80/10/10), and save all three to disk.

    The official HuggingFace test set is blinded (no labels), so both a
    held-out validation split and a local test split are carved from the
    training data instead.
    """
    full_train = load_dataset(BASE_REPO, split="train", cache_dir=CACHE_DIR)

    # First carve off 20 % (val + test), leaving 80 % for training
    split1 = full_train.train_test_split(
        test_size=VAL_FRACTION + TEST_FRACTION,
        seed=SPLIT_SEED,
        shuffle=True,
    )
    train_ds   = split1["train"]
    remainder  = split1["test"]

    # Split the 20 % remainder evenly into val (10 %) and test (10 %)
    split2 = remainder.train_test_split(
        test_size=0.5,
        seed=SPLIT_SEED,
        shuffle=True,
    )
    val_ds  = split2["train"]
    test_ds = split2["test"]

    train_ds.save_to_disk(os.path.join(DATA_DIR, "train"))
    val_ds.save_to_disk(os.path.join(DATA_DIR, "validation"))
    test_ds.save_to_disk(os.path.join(DATA_DIR, "test"))

    print(f"Saved train split      ({len(train_ds):,} rows) → {os.path.join(DATA_DIR, 'train')}")
    print(f"Saved validation split ({len(val_ds):,} rows)  → {os.path.join(DATA_DIR, 'validation')}")
    print(f"Saved test split       ({len(test_ds):,} rows)  → {os.path.join(DATA_DIR, 'test')}")


# ---------------------------------------------------------------------------
# Runtime utilities (imported by train.py)
# ---------------------------------------------------------------------------
def make_dataloader(split: str = "train", batch_size: int = 64, num_workers: int = 10):
    """
    Build a chemprop-compatible DataLoader for the requested dataset split.

    The function loads the HuggingFace dataset from disk (written by
    ``download_data``), converts it to a list of ``MoleculeDatapoint`` objects
    using the SMILES column as input and ``TARGET_COLS`` as regression targets,
    and wraps everything in a ``MoleculeDataset`` + ``build_dataloader``.

    Missing target values (NaN / null) are kept as ``np.nan`` in the ``y``
    array; during training, chemprop's loss functions automatically mask them
    out so they don't contribute to the gradient.

    Parameters
    ----------
    split : str
        One of ``"train"``, ``"validation"``, or ``"test"`` (saved by
        ``download_data``).
    batch_size : int
        Number of molecules per mini-batch.
    num_workers : int
        Worker processes for parallel graph featurization.  Set > 0 on Linux
        for faster loading; keep at 0 on Windows / macOS to avoid hangs.

    Returns
    -------
    torch.utils.data.DataLoader
        A chemprop ``DataLoader`` ready to be passed to a Lightning ``Trainer``
        or iterated directly.
    """
    from chemprop.data import MoleculeDatapoint, MoleculeDataset
    from chemprop.data.dataloader import build_dataloader

    split_path = os.path.join(DATA_DIR, split)
    hf_dataset = load_from_disk(split_path)
    df = hf_dataset.to_pandas()

    # # Some numeric columns may arrive as strings with thousand-separators
    # # (e.g. "1,351.1").  Normalise them to plain floats.
    # for col in TARGET_COLS:
    #     if col in df.columns:
    #         df[col] = (
    #             df[col]
    #             .astype(str)
    #             .str.replace(",", "", regex=False)
    #             .replace({"nan": np.nan, "None": np.nan, "null": np.nan})
    #             .astype(float)
    #         )

    datapoints: list[MoleculeDatapoint] = []
    for _, row in df.iterrows():
        smi = row.get("SMILES", None)
        if not isinstance(smi, str) or not smi.strip():
            continue

        # Build a float32 target vector; NaN where the label is missing
        y = np.array(
            [
                float(row[col]) if col in df.columns and pd.notna(row[col]) else np.nan
                for col in TARGET_COLS
            ],
            dtype=np.float32,
        )

        try:
            dp = MoleculeDatapoint.from_smi(
                smi.strip(),
                y,
                name=str(row.get("Molecule Name", smi)),
            )
            datapoints.append(dp)
        except Exception:
            # Skip molecules that RDKit cannot parse
            continue

    dataset = MoleculeDataset(datapoints)

    # Training data is shuffled; validation / test data is not
    shuffle = split == "train"

    loader = build_dataloader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
    )
    return loader


# ---------------------------------------------------------------------------
# Evaluation (DO NOT CHANGE — this is the fixed metric)
# ---------------------------------------------------------------------------

@torch.no_grad()
def evaluate_regression(model, val_loader, batch_size: int = 64):
    """
    Evaluate a chemprop MPNN on a validation DataLoader using per-task RMSE.

    Missing ground-truth labels (NaN) are masked out per task before computing
    RMSE, so tasks with sparse labels are still evaluated fairly.  The returned
    ``"mean_rmse"`` is the mean RMSE across all tasks that have *at least one*
    non-missing label in the split — this is the primary scalar metric used to
    compare runs.

    .. note::
        If target scaling (``StandardScaler``) was applied during training, the
        ``model`` should include the inverse transform in its
        ``output_transform`` so that predictions are returned in the original
        units.  Otherwise RMSE values will be on the scaled scale.

    Parameters
    ----------
    model : chemprop.models.MPNN
        A trained (or partially trained) chemprop MPNN model.
    val_loader : torch.utils.data.DataLoader
        DataLoader produced by ``make_dataloader`` for the validation or test
        split (``shuffle=False``).
    batch_size : int
        Unused — kept for API compatibility; batch size is fixed on the loader.

    Returns
    -------
    dict with keys:
        ``"rmse_per_task"``  ``list[float]`` RMSE for each target in
                            ``TARGET_COLS``; ``float('nan')`` if no labels.
        ``"mean_rmse"``     ``float`` mean RMSE across tasks with valid labels.
        ``"per_task"``      ``dict[str, float]`` mapping target name → RMSE.
    """
    import lightning.pytorch as pl

    device = next(model.parameters()).device
    accelerator = "gpu" if device.type == "cuda" else "cpu"

    trainer = pl.Trainer(
        logger=False,
        enable_checkpointing=False,
        enable_progress_bar=False,
        accelerator=accelerator,
        devices=1,
    )

    # trainer.predict returns a list of per-batch prediction tensors [B, T]
    raw_preds = trainer.predict(model, val_loader)
    preds = torch.cat(raw_preds, dim=0).cpu().numpy()   # [N, T]

    # Collect ground-truth targets directly from the DataLoader batches.
    # TrainingBatch namedtuple: (bmg, V_d, X_d, Y, w, lt_mask, gt_mask)
    all_targets = []
    for batch in val_loader:
        Y = batch.Y if hasattr(batch, "Y") else batch[3]
        all_targets.append(Y.numpy() if isinstance(Y, torch.Tensor) else np.array(Y))
    targets = np.concatenate(all_targets, axis=0)       # [N, T]

    # Per-task RMSE, ignoring NaN ground-truth entries
    n_tasks = targets.shape[1]
    rmse_per_task: list[float] = []
    for t in range(n_tasks):
        gt = targets[:, t]
        pr = preds[:, t] if preds.ndim > 1 else preds
        mask = ~np.isnan(gt)
        if mask.sum() == 0:
            rmse_per_task.append(float("nan"))
        else:
            rmse = float(np.sqrt(np.mean((gt[mask] - pr[mask]) ** 2)))
            rmse_per_task.append(rmse)

    valid_rmses = [r for r in rmse_per_task if not np.isnan(r)]
    mean_rmse = float(np.mean(valid_rmses)) if valid_rmses else float("nan")

    per_task = {col: rmse_per_task[i] for i, col in enumerate(TARGET_COLS)}

    return {
        "rmse_per_task": rmse_per_task,
        "mean_rmse": mean_rmse,
        "per_task": per_task,
    }


if __name__ == "__main__":
    download_data()
