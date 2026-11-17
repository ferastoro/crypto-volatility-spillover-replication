#!/usr/bin/env python3
"""Self-contained Kaggle GPU robustness script: 20-origin rolling refit of Stage-2 neural models.

Refits TFT (NOSPILL, SPILL) and NBEATSx (NOSPILL, SPILL) every 20 OOS origins
(17 refits at origins r in {0, 20, ..., 320}) on the most recent 1,082-row rolling
window (915 train + 167 validation) ending at origin r's cutoff date, then freezes
the 5-seed ensemble to forecast origins r .. min(r + 19, 334) over horizons h = 1..10.

Matches kaggle_3a3b_v2.ipynb / Stage-2 (3a) hyperparameters,
causal SPILL features (SHA-256 verified, zero ffill), seeds, losses, scalers, and output schema.

Usage:
  Kaggle GPU:   python refit20_kaggle.py
  Local check:  python refit20_kaggle.py --dry-run
"""

from __future__ import annotations

import argparse
from copy import deepcopy
import gc
import hashlib
import os
from pathlib import Path
import random
import subprocess
import sys
import time
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

# ==============================================================================
# FIXED PROTOCOL & HYPERPARAMETERS (matched verbatim to Stage-2 / 3a v2)
# ==============================================================================
VARIANT_NAME = "Refit20"
SEEDS: List[int] = [42, 7, 13, 21, 99]
H: int = 10
WARMUP_DISCARD: int = 250
N_TRAIN: int = 915
N_VAL: int = 167
WINDOW_SIZE: int = N_TRAIN + N_VAL  # 1,082 rows
N_TEST_CUTOFFS: int = 335
REFIT_STEP: int = 20

MAX_STEPS: int = 1000
VAL_CHECK_STEPS: int = 25
EARLY_STOP_PATIENCE: int = 5
SCALER_TYPE: str = "standard"

EXPECTED_SPILL_SHA256 = "224728802dea162577c638d57a06775574769143b6ce2af1c965debcb4a3cafd"

BEST_HP: Dict[str, Dict[str, Any]] = {
    "TFT": {
        "input_size": 50,
        "learning_rate": 0.001,
        "hidden_size": 64,
        "loss_name": "MSE",
    },
    "NBEATSx": {
        "input_size": 20,
        "learning_rate": 0.0003,
        "mlp_units": [[256, 256], [256, 256], [256, 256]],
        "loss_name": "MAE",
    },
}

NOSPILL_COLS: List[str] = [
    "ret_saham_nvda",
    "ret_saham_msft",
    "ret_saham_tsla",
    "ret_saham_googl",
    "ret_saham_meta",
    "ret_crypto_btc_usd",
    "ret_crypto_eth_usd",
    "crypto_btc_usd_weekend_return",
    "gap_days",
    "is_monday",
    "is_ihsg_holiday",
]

SPILL_ONLY_9: List[str] = [
    "tvp_tsi",
    "net_spill_saham_nvda",
    "net_spill_saham_msft",
    "net_spill_saham_tsla",
    "net_spill_saham_googl",
    "net_spill_saham_meta",
    "net_spill_crypto_btc_usd",
    "net_spill_crypto_eth_usd",
    "spillover_ai_to_btcidr",
]

SPILL_COLS: List[str] = NOSPILL_COLS + SPILL_ONLY_9

CONFIGS: List[Tuple[str, str]] = [
    ("TFT", "NOSPILL"),
    ("TFT", "SPILL"),
    ("NBEATSx", "NOSPILL"),
    ("NBEATSx", "SPILL"),
]

# Exact predictions_ensemble.csv schema + refit provenance columns
ENSEMBLE_BASE_COLUMNS: List[str] = [
    "model",
    "feature_set",
    "variant",
    "origin_idx",
    "cutoff_date",
    "target_date",
    "horizon",
    "y_true",
    "y_pred",
]
PRED_COLUMNS: List[str] = ENSEMBLE_BASE_COLUMNS + [
    "refit_id",
    "refit_origin_date",
    "window_start_date",
    "window_end_date",
]

FIT_LOG_COLUMNS: List[str] = [
    "refit_id",
    "refit_origin_date",
    "window_start_date",
    "window_end_date",
    "model",
    "feature_set",
    "variant",
    "seed",
    "steps_run",
    "best_step",
    "best_val_loss",
    "fit_time_sec",
    "weights_restored",
    "pred_std",
    "collapsed",
]

LEGACY_TVP_FILENAMES = {
    "tvp_var_dynamic_features_causal.csv",
    "tvp_var_dynamic_features.csv",
}


# ==============================================================================
# INPUT DISCOVERY & DATA LOADING (strict causal merge, SHA-256 verified)
# ==============================================================================
def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def resolve_input_files(
    dl_path_arg: Optional[str] = None,
    spill_path_arg: Optional[str] = None,
) -> Tuple[Path, Path]:
    """Locate dataset_deep_learning.csv and spill_features_main_us1677_causal.csv."""
    resolved = Path(__file__).resolve()
    repo_root = resolved.parents[2] if len(resolved.parents) > 2 else resolved.parent

    def _find_one(explicit: Optional[str], preferred: List[Path], filename: str) -> Path:
        if explicit:
            p = Path(explicit).expanduser().resolve()
            if p.name in LEGACY_TVP_FILENAMES:
                raise RuntimeError(f"ABORT: legacy TVP file forbidden: {p}")
            if not p.is_file():
                raise FileNotFoundError(f"Input file not found: {p}")
            return p

        for cand in preferred:
            if cand.is_file():
                return cand

        kaggle_root = Path("/kaggle/input")
        if kaggle_root.exists():
            matches = sorted(kaggle_root.rglob(filename))
            if matches:
                return matches[0]

        raise FileNotFoundError(
            f"Could not find {filename}. Checked {[str(c) for c in preferred]} "
            "and recursive /kaggle/input/."
        )

    dl_preferred = [
        Path(os.environ.get("KAGGLE_INPUT_ROOT", ".")) / "data" / "clean" / "dataset_deep_learning.csv",
        Path("/kaggle/input/ai-stocks-crypto-and-ihsg/data_clean/dataset_deep_learning.csv"),
        repo_root / "data" / "clean" / "dataset_deep_learning.csv",
    ]
    spill_preferred = [
        Path(os.environ.get("KAGGLE_INPUT_ROOT", ".")) / "revision" / "outputs" / "spill_features_main_us1677_causal.csv",
        Path("/kaggle/input/spillover-v5-inputs/spill_features_main_us1677_causal.csv"),
        repo_root / "revision" / "outputs" / "spill_features_main_us1677_causal.csv",
    ]

    dl_path = _find_one(dl_path_arg, dl_preferred, "dataset_deep_learning.csv")
    spill_path = _find_one(spill_path_arg, spill_preferred, "spill_features_main_us1677_causal.csv")

    if spill_path.name in LEGACY_TVP_FILENAMES:
        raise RuntimeError(f"ABORT: legacy TVP file forbidden: {spill_path}")

    actual_sha = sha256_file(spill_path)
    if actual_sha != EXPECTED_SPILL_SHA256:
        raise RuntimeError(
            f"SHA-256 mismatch for {spill_path}:\n"
            f"  expected: {EXPECTED_SPILL_SHA256}\n"
            f"  actual:   {actual_sha}"
        )

    return dl_path, spill_path


def load_post_warmup_df(dl_path: Path, spill_path: Path) -> pd.DataFrame:
    """Build causal 1,426-row post-warmup dataframe matching 3a v2 identically."""
    dl_df = pd.read_csv(dl_path)
    spill_df = pd.read_csv(spill_path)

    if "Date" not in dl_df.columns or "Date" not in spill_df.columns:
        raise KeyError("Both input CSVs must contain a 'Date' column.")
    if "ret_crypto_btc_idr_indodax" not in dl_df.columns:
        raise KeyError("dataset_deep_learning.csv missing 'ret_crypto_btc_idr_indodax'.")

    missing_nospill = [c for c in NOSPILL_COLS if c not in dl_df.columns]
    if missing_nospill:
        raise KeyError(f"Missing NOSPILL columns in dataset_deep_learning.csv: {missing_nospill}")

    missing_spill = [c for c in SPILL_ONLY_9 if c not in spill_df.columns]
    if missing_spill:
        raise KeyError(f"Missing SPILL columns in spill_features_main_us1677_causal.csv: {missing_spill}")

    if "net_spill_crypto_btc_idr_indodax" in SPILL_COLS:
        raise AssertionError("Leakage check failed: target spillover column in SPILL_COLS")

    # Strict inner merge (zero forward-fill) matching kaggle_3a3b_v2.ipynb
    spill_take = ["Date"] + [c for c in SPILL_ONLY_9 if c not in dl_df.columns]
    df = pd.merge(dl_df, spill_df[spill_take], on="Date", how="inner")
    df["Date"] = df["Date"].astype(str)
    df = df.sort_values("Date").reset_index(drop=True)

    if len(df) != 1676:
        raise RuntimeError(f"Expected 1676 inner-merged rows, got {len(df)}")

    # Exact float64 squared-return target and contiguous daily synthetic ds index (matching 3a v2)
    df["y"] = np.square(df["ret_crypto_btc_idr_indodax"].astype(np.float64))
    df["unique_id"] = "BTC_IDR"
    df["ds"] = pd.date_range("2020-01-01", periods=len(df), freq="D")

    df_post = df.iloc[WARMUP_DISCARD:].copy().reset_index(drop=True)
    if len(df_post) != 1426:
        raise RuntimeError(f"Expected 1426 post-warmup rows, got {len(df_post)}")

    if df_post[SPILL_COLS].isna().any().any():
        na_cols = df_post[SPILL_COLS].columns[df_post[SPILL_COLS].isna().any()].tolist()
        raise RuntimeError(f"Unexpected NaNs in post-warmup features (no ffill allowed): {na_cols}")

    expected_origins = len(df_post) - WINDOW_SIZE - H + 1
    if expected_origins != N_TEST_CUTOFFS:
        raise RuntimeError(
            f"Expected {N_TEST_CUTOFFS} OOS origins from {len(df_post)} rows, got {expected_origins}"
        )

    return df_post


# ==============================================================================
# ROLLING 20-ORIGIN REFIT SCHEDULE & CAUSAL VERIFICATION
# ==============================================================================
def build_refit_schedule(df_post: pd.DataFrame) -> List[Dict[str, Any]]:
    """Construct all 17 rolling refit windows and assert zero look-ahead leakage."""
    windows: List[Dict[str, Any]] = []
    covered_origins: List[int] = []

    for refit_id, r in enumerate(range(0, N_TEST_CUTOFFS, REFIT_STEP)):
        b_start = r
        b_end = min(r + REFIT_STEP, N_TEST_CUTOFFS)
        n_win_blk = b_end - b_start
        test_size_blk = H + (n_win_blk - 1)

        fit_slice = slice(r, r + WINDOW_SIZE)
        train_slice = slice(r, r + N_TRAIN)
        val_slice = slice(r + N_TRAIN, r + WINDOW_SIZE)
        all_slice = slice(r, r + WINDOW_SIZE + test_size_blk)

        fit_df = df_post.iloc[fit_slice]
        train_df = df_post.iloc[train_slice]
        val_df = df_post.iloc[val_slice]
        all_df = df_post.iloc[all_slice]

        window_start_date = str(fit_df["Date"].iloc[0])
        window_end_date = str(fit_df["Date"].iloc[-1])
        train_start_date = str(train_df["Date"].iloc[0])
        train_end_date = str(train_df["Date"].iloc[-1])
        val_start_date = str(val_df["Date"].iloc[0])
        val_end_date = str(val_df["Date"].iloc[-1])
        refit_origin_date = str(df_post["Date"].iloc[(WINDOW_SIZE - 1) + r])

        origin_indices = list(range(b_start, b_end))
        origin_cutoffs = [str(df_post["Date"].iloc[(WINDOW_SIZE - 1) + o]) for o in origin_indices]
        first_cutoff = origin_cutoffs[0]
        last_cutoff = origin_cutoffs[-1]
        last_target_h10 = str(df_post["Date"].iloc[(WINDOW_SIZE - 1) + (b_end - 1) + H])

        # Strict assertions
        assert len(fit_df) == WINDOW_SIZE, f"Expected {WINDOW_SIZE} fit rows, got {len(fit_df)}"
        assert len(train_df) == N_TRAIN, f"Expected {N_TRAIN} train rows, got {len(train_df)}"
        assert len(val_df) == N_VAL, f"Expected {N_VAL} val rows, got {len(val_df)}"
        assert len(all_df) == WINDOW_SIZE + test_size_blk
        assert window_end_date <= refit_origin_date, (
            f"Leakage in refit {refit_id}: window_end_date={window_end_date} > {refit_origin_date}"
        )
        assert window_end_date == refit_origin_date == first_cutoff
        for o, c_date in zip(origin_indices, origin_cutoffs):
            assert window_end_date <= c_date, (
                f"Origin {o} ({c_date}) uses future training data ending {window_end_date}"
            )
        assert str(all_df["Date"].iloc[-1]) == last_target_h10

        covered_origins.extend(origin_indices)
        windows.append(
            {
                "refit_id": refit_id,
                "r": r,
                "b_start": b_start,
                "b_end": b_end,
                "n_win_blk": n_win_blk,
                "test_size_blk": test_size_blk,
                "fit_slice": fit_slice,
                "all_slice": all_slice,
                "train_rows": len(train_df),
                "val_rows": len(val_df),
                "train_start_date": train_start_date,
                "train_end_date": train_end_date,
                "val_start_date": val_start_date,
                "val_end_date": val_end_date,
                "window_start_date": window_start_date,
                "window_end_date": window_end_date,
                "refit_origin_date": refit_origin_date,
                "first_cutoff": first_cutoff,
                "last_cutoff": last_cutoff,
                "last_target_h10": last_target_h10,
                "origin_indices": origin_indices,
            }
        )

    assert len(windows) == 17, f"Expected 17 refit windows, got {len(windows)}"
    assert covered_origins == list(range(N_TEST_CUTOFFS)), "Refit origins do not partition 0..334"

    # Verify refit_id=0 reproduces the original 3a 915 + 167 window identically
    w0 = windows[0]
    assert (w0["train_start_date"], w0["train_end_date"]) == ("2020-12-31", "2024-08-21"), w0
    assert (w0["val_start_date"], w0["val_end_date"]) == ("2024-08-22", "2025-04-23"), w0
    assert w0["refit_origin_date"] == "2025-04-23"
    assert windows[-1]["last_cutoff"] == "2026-08-21"
    assert windows[-1]["last_target_h10"] == "2026-09-04"

    return windows


def print_dry_run_report(windows: List[Dict[str, Any]], dl_path: Path, spill_path: Path) -> None:
    print("=" * 110)
    print("DRY-RUN SELF-TEST: 20-Origin Rolling Refit Schedule (No Training Executed)")
    print("=" * 110)
    print(f"dataset_deep_learning.csv:          {dl_path}")
    print(f"spill_features_main_us1677_causal:  {spill_path} (SHA-256 OK)")
    print(f"Configs ({len(CONFIGS)}): {', '.join(f'{m}-{fs}' for m, fs in CONFIGS)} | Seeds ({len(SEEDS)}): {SEEDS}")
    print(f"Total neural fits: {len(windows)} refits x {len(CONFIGS)} configs x {len(SEEDS)} seeds = {len(windows)*len(CONFIGS)*len(SEEDS)} fits")
    print(f"Predictions schema ({len(PRED_COLUMNS)} cols): {PRED_COLUMNS}")
    print("-" * 110)
    hdr = (
        f"{'id':>2} | {'window_start':>10} -> {'window_end':>10} | "
        f"{'tr':>3}+{'va':>3} | {'origins':>9} | {'first_cut':>10} -> {'last_cut':>10} | {'n_win':>5}"
    )
    print(hdr)
    print("-" * 110)
    for w in windows:
        orig_rng = f"{w['b_start']:3d}..{w['b_end']-1:<3d}"
        print(
            f"{w['refit_id']:2d} | {w['window_start_date']:>10} -> {w['window_end_date']:>10} | "
            f"{w['train_rows']:3d}+{w['val_rows']:3d} | {orig_rng:>9} | "
            f"{w['first_cutoff']:>10} -> {w['last_cutoff']:>10} | {w['n_win_blk']:5d}"
        )
    print("-" * 110)
    w0 = windows[0]
    print(
        f"[PASS] refit_id=0 matches original 3a window: "
        f"train={w0['train_start_date']}..{w0['train_end_date']} ({w0['train_rows']}), "
        f"val={w0['val_start_date']}..{w0['val_end_date']} ({w0['val_rows']})"
    )
    print("[PASS] All 17 windows satisfy window_end_date <= refit_origin_date <= forecast_cutoff_date.")
    print("[PASS] Zero forecast origin uses a model trained on later data; 335 origins (3,350 rows/cfg) covered.")


# ==============================================================================
# TRAINING & KAGGLE RUNTIME UTILITIES (only imported outside --dry-run)
# ==============================================================================
def ensure_neuralforecast_installed() -> None:
    """Install pinned neuralforecast==3.3.0 on Kaggle if missing."""
    try:
        import neuralforecast  # noqa: F401
        if getattr(neuralforecast, "__version__", None) == "3.3.0":
            return
    except ImportError:
        pass
    print("[setup] Installing neuralforecast==3.3.0 ...")
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "neuralforecast==3.3.0"])


def set_all_seeds(seed: int) -> None:
    import pytorch_lightning as pl
    import torch

    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    pl.seed_everything(seed, workers=True)


def run_training(
    df_post: pd.DataFrame,
    windows: List[Dict[str, Any]],
    output_dir: Path,
    skip_pip: bool = False,
) -> None:
    if not skip_pip:
        ensure_neuralforecast_installed()

    import neuralforecast
    import pytorch_lightning as pl
    import torch
    from neuralforecast import NeuralForecast
    from neuralforecast.losses.pytorch import MAE, MSE
    from neuralforecast.models import NBEATSx, TFT

    assert neuralforecast.__version__ == "3.3.0", (
        f"Expected neuralforecast==3.3.0, found {neuralforecast.__version__}"
    )

    print(f"[env] torch={torch.__version__} cuda={torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"[env] gpu={torch.cuda.get_device_name(0)}")

    def _weights_checksum(state_dict: Dict[str, torch.Tensor]) -> str:
        h = hashlib.md5()
        for k, v in sorted(state_dict.items()):
            h.update(k.encode("utf-8"))
            h.update(v.detach().cpu().numpy().tobytes())
        return h.hexdigest()

    class RestoreBestWeightsCallback(pl.callbacks.Callback):
        """Restore best ptl/val_loss state_dict at train end and record steps/losses."""

        def __init__(self, monitor: str = "ptl/val_loss") -> None:
            super().__init__()
            self.monitor = monitor
            self.best_val_loss: float = float("inf")
            self.best_step: Optional[int] = None
            self.best_state_dict: Optional[Dict[str, torch.Tensor]] = None
            self.best_checksum: Optional[str] = None
            self.actual_final_step: int = 0
            self.weights_restored: bool = False

        def on_validation_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
            if getattr(trainer, "sanity_checking", False) or trainer.global_step == 0:
                return
            val = trainer.callback_metrics.get(self.monitor)
            if val is not None:
                val_f = float(val.detach().cpu().item() if torch.is_tensor(val) else val)
            else:
                traj = getattr(pl_module, "valid_trajectories", None) or []
                if not traj:
                    return
                val_f = float(traj[-1][1])

            if np.isfinite(val_f) and val_f < self.best_val_loss:
                self.best_val_loss = val_f
                self.best_step = int(trainer.global_step)
                self.best_state_dict = {
                    k: v.detach().cpu().clone() for k, v in pl_module.state_dict().items()
                }
                self.best_checksum = _weights_checksum(self.best_state_dict)

        def on_train_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
            self.actual_final_step = int(trainer.global_step)
            if self.best_state_dict is not None:
                pl_module.load_state_dict(
                    {k: v.to(pl_module.device) for k, v in self.best_state_dict.items()}
                )
                self.weights_restored = True

    device_cfg = (
        {"accelerator": "cuda", "devices": 1}
        if torch.cuda.is_available()
        else {"accelerator": "cpu", "devices": 1}
    )

    def build_model(model_name: str, fcols: List[str], seed: int, cb: RestoreBestWeightsCallback):
        common_kwargs = dict(
            h=H,
            hist_exog_list=fcols,
            max_steps=MAX_STEPS,
            val_check_steps=VAL_CHECK_STEPS,
            early_stop_patience_steps=EARLY_STOP_PATIENCE,
            scaler_type=SCALER_TYPE,
            random_seed=seed,
            enable_progress_bar=False,
            logger=False,
            enable_checkpointing=False,
            callbacks=[cb],
            **device_cfg,
        )
        if model_name == "TFT":
            hp = BEST_HP["TFT"]
            return TFT(
                input_size=hp["input_size"],
                learning_rate=hp["learning_rate"],
                hidden_size=hp["hidden_size"],
                loss=MSE(),
                **common_kwargs,
            )
        if model_name == "NBEATSx":
            hp = BEST_HP["NBEATSx"]
            return NBEATSx(
                input_size=hp["input_size"],
                learning_rate=hp["learning_rate"],
                mlp_units=deepcopy(hp["mlp_units"]),
                loss=MAE(),
                **common_kwargs,
            )
        raise ValueError(f"Unknown model: {model_name}")

    def align_cv_predictions(
        cv_df: pd.DataFrame,
        model_col: str,
        all_src: pd.DataFrame,
        window: Dict[str, Any],
        model_name: str,
        feature_set: str,
    ) -> pd.DataFrame:
        ds_to_date = dict(zip(all_src["ds"], all_src["Date"]))
        date_to_y64 = dict(zip(all_src["Date"], all_src["y"].astype(np.float64)))

        cols_take = ["cutoff", "ds", model_col] + (["y"] if "y" in cv_df.columns else [])
        out = cv_df[cols_take].copy()
        out["cutoff_date"] = out["cutoff"].map(ds_to_date)
        out["target_date"] = out["ds"].map(ds_to_date)
        if out["cutoff_date"].isna().any() or out["target_date"].isna().any():
            raise RuntimeError("Unmapped cutoff/target ds in cross_validation output.")

        out = out.sort_values(["cutoff_date", "target_date"]).reset_index(drop=True)
        out["horizon"] = out.groupby("cutoff_date").cumcount() + 1

        # Map each cutoff_date strictly to expected origin_idx in [b_start, b_end)
        expected_cutoffs = [
            str(df_post["Date"].iloc[(WINDOW_SIZE - 1) + o])
            for o in window["origin_indices"]
        ]
        cutoff_to_origin = dict(zip(expected_cutoffs, window["origin_indices"]))
        out["origin_idx"] = out["cutoff_date"].map(cutoff_to_origin)
        if out["origin_idx"].isna().any():
            raise RuntimeError("cross_validation cutoffs do not match refit block origins.")
        out["origin_idx"] = out["origin_idx"].astype(int)

        # Float64 ground-truth precision check (matches 3c/3e verification)
        y_f64 = out["target_date"].map(date_to_y64).astype(np.float64)
        if "y" in out.columns:
            max_diff = float(np.max(np.abs(out["y"].to_numpy(dtype=np.float64) - y_f64.to_numpy())))
            assert np.allclose(out["y"].to_numpy(), y_f64.to_numpy(), rtol=1e-6, atol=1e-12), max_diff

        out["model"] = model_name
        out["feature_set"] = feature_set
        out["variant"] = VARIANT_NAME
        out["y_true"] = y_f64
        out["y_pred"] = out[model_col].astype(np.float64)
        out["refit_id"] = int(window["refit_id"])
        out["refit_origin_date"] = window["refit_origin_date"]
        out["window_start_date"] = window["window_start_date"]
        out["window_end_date"] = window["window_end_date"]

        return out[PRED_COLUMNS]

    output_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir = output_dir / "checkpoints_refit20"
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    ens_path = output_dir / "predictions_refit20.csv"
    log_path = output_dir / "refit20_fit_logs.csv"

    # Load any existing incremental outputs to resume cleanly
    if ens_path.is_file():
        ens_existing = pd.read_csv(ens_path)
    else:
        ens_existing = pd.DataFrame(columns=PRED_COLUMNS)

    if log_path.is_file():
        log_existing = pd.read_csv(log_path)
    else:
        log_existing = pd.DataFrame(columns=FIT_LOG_COLUMNS)

    completed_refits = set()
    if len(ens_existing) > 0 and len(log_existing) > 0:
        for w in windows:
            rid = w["refit_id"]
            expected_rows = len(CONFIGS) * w["n_win_blk"] * H
            sub = ens_existing[ens_existing["refit_id"] == rid]
            log_sub = log_existing[log_existing["refit_id"] == rid]
            if len(sub) == expected_rows and len(log_sub) == len(CONFIGS) * len(SEEDS):
                completed_refits.add(rid)

    if completed_refits:
        print(f"[resume] Skipping already-completed refit_ids: {sorted(completed_refits)}")

    t_global_start = time.time()

    for w in windows:
        rid = w["refit_id"]
        if rid in completed_refits:
            continue

        t_refit_start = time.time()
        print(
            f"\n[refit {rid:2d}/16] origins {w['b_start']}..{w['b_end']-1} "
            f"| fit={w['window_start_date']}..{w['window_end_date']}"
        )

        fit_src = df_post.iloc[w["fit_slice"]].copy()
        all_src = df_post.iloc[w["all_slice"]].copy()

        refit_ens_parts: List[pd.DataFrame] = []
        refit_log_rows: List[Dict[str, Any]] = []

        def _prep_nf(src_df: pd.DataFrame, fcols_list: List[str]) -> pd.DataFrame:
            out_nf = pd.DataFrame(
                {
                    "unique_id": src_df["unique_id"].values,
                    "ds": pd.to_datetime(src_df["ds"].values),
                    "y": src_df["y"].to_numpy(dtype=np.float64).astype(np.float32),
                }
            )
            for col in sorted(fcols_list):
                out_nf[col] = src_df[col].to_numpy(dtype=np.float32)
            return out_nf

        for model_name, feature_set in CONFIGS:
            fcols = SPILL_COLS if feature_set == "SPILL" else NOSPILL_COLS
            fit_nf = _prep_nf(fit_src, fcols)
            all_nf = _prep_nf(all_src, fcols)

            seed_preds: List[pd.DataFrame] = []
            for seed in SEEDS:
                seed_ckpt = ckpt_dir / f"refit{rid:02d}_{model_name}_{feature_set}_s{seed}.csv"
                log_ckpt = ckpt_dir / f"refit{rid:02d}_{model_name}_{feature_set}_s{seed}_log.csv"

                if seed_ckpt.is_file() and log_ckpt.is_file():
                    df_seed = pd.read_csv(seed_ckpt)
                    expected_seed_rows = w["n_win_blk"] * H
                    if len(df_seed) == expected_seed_rows:
                        seed_preds.append(df_seed)
                        refit_log_rows.extend(pd.read_csv(log_ckpt).to_dict("records"))
                        print(f"  [ckpt] {model_name}-{feature_set} s={seed} loaded")
                        continue

                set_all_seeds(seed)
                cb = RestoreBestWeightsCallback(monitor="ptl/val_loss")
                model = build_model(model_name, fcols, seed, cb)
                nf = NeuralForecast(models=[model], freq="D")

                t0 = time.time()
                nf.fit(df=fit_nf, val_size=N_VAL)

                fitted_model = nf.models[0]
                active_cb = cb
                for c in (fitted_model.trainer_kwargs.get("callbacks") or []):
                    if isinstance(c, RestoreBestWeightsCallback) and c.actual_final_step > 0:
                        active_cb = c
                        break
                chk_post_fit = _weights_checksum(fitted_model.state_dict())
                if active_cb.best_checksum is not None:
                    assert chk_post_fit == active_cb.best_checksum, "Restored weights checksum mismatch!"
                    assert active_cb.weights_restored, "Best weights were not restored!"

                cv_df = nf.cross_validation(
                    df=all_nf,
                    n_windows=w["n_win_blk"],
                    step_size=1,
                    use_fitted=True,
                )
                elapsed = round(time.time() - t0, 2)
                chk_post_cv = _weights_checksum(fitted_model.state_dict())
                assert chk_post_cv == chk_post_fit, "Model weights mutated during cross_validation!"

                best_step = active_cb.best_step
                best_val = active_cb.best_val_loss if np.isfinite(active_cb.best_val_loss) else None
                steps_run = active_cb.actual_final_step
                if (best_step is None or best_val is None) and getattr(fitted_model, "valid_trajectories", None):
                    traj = fitted_model.valid_trajectories
                    best_pair = min(traj, key=lambda x: x[1])
                    best_step = int(best_pair[0])
                    best_val = float(best_pair[1])
                    if steps_run == 0:
                        steps_run = int(traj[-1][0])

                model_col = model_name
                df_seed = align_cv_predictions(cv_df, model_col, all_src, w, model_name, feature_set)
                pred_std = float(df_seed["y_pred"].std())
                collapsed = bool(pred_std <= 1e-12 or not np.isfinite(pred_std))

                log_row = {
                    "refit_id": int(rid),
                    "refit_origin_date": w["refit_origin_date"],
                    "window_start_date": w["window_start_date"],
                    "window_end_date": w["window_end_date"],
                    "model": model_name,
                    "feature_set": feature_set,
                    "variant": VARIANT_NAME,
                    "seed": int(seed),
                    "steps_run": int(steps_run),
                    "best_step": int(best_step) if best_step is not None else -1,
                    "best_val_loss": float(best_val) if best_val is not None else float("nan"),
                    "fit_time_sec": elapsed,
                    "weights_restored": bool(active_cb.weights_restored),
                    "pred_std": pred_std,
                    "collapsed": collapsed,
                }

                df_seed.to_csv(seed_ckpt, index=False)
                pd.DataFrame([log_row]).to_csv(log_ckpt, index=False)

                seed_preds.append(df_seed)
                refit_log_rows.append(log_row)
                print(
                    f"  [done] {model_name:7s}-{feature_set:7s} s={seed:2d} "
                    f"steps={steps_run:4d} best={log_row['best_step']:4d} "
                    f"val={log_row['best_val_loss']:.3e} t={elapsed:.1f}s "
                    f"restored={active_cb.weights_restored}"
                )
                del nf, model, fitted_model
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

            # Average predictions across the 5 seeds
            cat_seeds = pd.concat(seed_preds, ignore_index=True)
            group_cols = [
                "model",
                "feature_set",
                "variant",
                "origin_idx",
                "cutoff_date",
                "target_date",
                "horizon",
                "y_true",
                "refit_id",
                "refit_origin_date",
                "window_start_date",
                "window_end_date",
            ]
            ens_cfg = cat_seeds.groupby(group_cols, as_index=False)["y_pred"].mean()
            ens_cfg = ens_cfg[PRED_COLUMNS].sort_values(["origin_idx", "horizon"]).reset_index(drop=True)
            refit_ens_parts.append(ens_cfg)

        # Atomic per-refit update to main output CSVs
        refit_ens_df = pd.concat(refit_ens_parts, ignore_index=True)
        ens_existing = pd.concat(
            [ens_existing[ens_existing["refit_id"] != rid], refit_ens_df],
            ignore_index=True,
        ).sort_values(["refit_id", "model", "feature_set", "origin_idx", "horizon"]).reset_index(drop=True)
        ens_tmp = ens_path.with_suffix(".csv.tmp")
        ens_existing[PRED_COLUMNS].to_csv(ens_tmp, index=False)
        ens_tmp.replace(ens_path)

        refit_log_df = pd.DataFrame(refit_log_rows, columns=FIT_LOG_COLUMNS)
        log_existing = pd.concat(
            [log_existing[log_existing["refit_id"] != rid], refit_log_df],
            ignore_index=True,
        ).sort_values(["refit_id", "model", "feature_set", "seed"]).reset_index(drop=True)
        log_tmp = log_path.with_suffix(".csv.tmp")
        log_existing[FIT_LOG_COLUMNS].to_csv(log_tmp, index=False)
        log_tmp.replace(log_path)

        refit_sec = round(time.time() - t_refit_start, 1)
        total_min = round((time.time() - t_global_start) / 60.0, 2)
        print(f"[saved] refit {rid:2d} committed -> {ens_path} ({len(ens_existing)} rows) | refit={refit_sec}s | elapsed={total_min}m")

    # Final integrity verification
    assert len(ens_existing) == len(CONFIGS) * N_TEST_CUTOFFS * H, (
        f"Expected {len(CONFIGS) * N_TEST_CUTOFFS * H} final rows, got {len(ens_existing)}"
    )
    print(f"\n[COMPLETE] {len(ens_existing)} ensemble rows written to {ens_path}")
    print(f"[COMPLETE] {len(log_existing)} fit log rows written to {log_path}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="20-origin rolling refit Kaggle robustness script (Stage-2 TFT & NBEATSx)."
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate data, split dates, all 17 windows, and causal assertions on CPU without training.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="/kaggle/working" if Path("/kaggle/working").exists() else str(Path(__file__).resolve().parent / "outputs"),
        help="Directory for predictions_refit20.csv and refit20_fit_logs.csv (default: /kaggle/working on Kaggle).",
    )
    parser.add_argument(
        "--dl-path",
        type=str,
        default=None,
        help="Optional explicit path to dataset_deep_learning.csv.",
    )
    parser.add_argument(
        "--spill-path",
        type=str,
        default=None,
        help="Optional explicit path to spill_features_main_us1677_causal.csv.",
    )
    parser.add_argument(
        "--skip-pip",
        action="store_true",
        help="Skip automatic pip install neuralforecast==3.3.0 check.",
    )
    args = parser.parse_args()

    dl_path, spill_path = resolve_input_files(args.dl_path, args.spill_path)
    df_post = load_post_warmup_df(dl_path, spill_path)
    windows = build_refit_schedule(df_post)

    if args.dry_run:
        print_dry_run_report(windows, dl_path, spill_path)
        return

    run_training(df_post, windows, Path(args.output_dir), skip_pip=args.skip_pip)


if __name__ == "__main__":
    main()
