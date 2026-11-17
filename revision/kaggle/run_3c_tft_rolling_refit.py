import os
#!/usr/bin/env python3
"""
Step 3c (Kaggle GPU): Rolling-refit protocol for TFT across at least 3 seeds [42, 7, 13]
(3 seeds x 6 60-day refit blocks x 2 feature sets SPILL/NOSPILL = 36 fits across all 335 cutoffs).
Expected runtime on Kaggle Tesla T4 (16 GB): ~30-35 minutes.
"""

import sys
import time
import random
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import torch
from scipy import stats
from sklearn.metrics import mean_absolute_error, mean_squared_error

try:
    from neuralforecast import NeuralForecast
    from neuralforecast.models import TFT
    from neuralforecast.losses.pytorch import MSE
except ImportError:
    import subprocess
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "neuralforecast==3.3.0"], check=True)
    from neuralforecast import NeuralForecast
    from neuralforecast.models import TFT
    from neuralforecast.losses.pytorch import MSE

SEEDS = [42, 7, 13]
H = 10
N_VAL = 167
BLOCK_SIZE = 60
WARMUP_DISCARD = 250

NOSPILL_COLS = [
    "ret_saham_nvda", "ret_saham_msft", "ret_saham_tsla", "ret_saham_googl", "ret_saham_meta",
    "ret_crypto_btc_usd", "ret_crypto_eth_usd", "crypto_btc_usd_weekend_return",
    "gap_days", "is_monday", "is_ihsg_holiday",
]
SPILL_COLS = NOSPILL_COLS + [
    "spillover_ai_to_btcidr", "tvp_tsi",
    "net_spill_saham_nvda", "net_spill_saham_msft", "net_spill_saham_tsla",
    "net_spill_saham_googl", "net_spill_saham_meta",
    "net_spill_crypto_btc_usd", "net_spill_crypto_eth_usd",
]


def set_seed(s: int):
    random.seed(s)
    np.random.seed(s)
    torch.manual_seed(s)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(s)


def find_data_dirs():
    candidates = [
        Path(os.environ.get("KAGGLE_INPUT_ROOT", "./data/clean")),
        Path(os.environ.get("KAGGLE_INPUT_ROOT", "./inputs/v5")),
        Path("inputs/v5"),
        Path("data/clean"),
    ]
    dl_path = next((d / "dataset_deep_learning.csv" for d in candidates if (d / "dataset_deep_learning.csv").exists()), None)
    tvp_path = None
    for d in candidates:
        for fn in ["tvp_var_dynamic_features_causal.csv", "tvp_var_dynamic_features.csv"]:
            if (d / fn).exists():
                tvp_path = d / fn
                break
        if tvp_path:
            break
    return dl_path, tvp_path


def main():
    t0 = time.time()
    out_dir = Path("/kaggle/working/step3c_tft_rolling_refit") if Path("/kaggle").exists() else Path("revision/kaggle_outputs/step3c")
    out_dir.mkdir(parents=True, exist_ok=True)

    dl_path, tvp_path = find_data_dirs()
    df = pd.read_csv(dl_path).merge(pd.read_csv(tvp_path), on="Date", how="left").sort_values("Date").ffill().dropna().reset_index(drop=True)
    df["y"] = df["ret_crypto_btc_idr_indodax"] ** 2
    df["unique_id"] = "BTC_IDR"
    df["real_date"] = df["Date"]
    if "gap_days" not in df.columns:
        dt = pd.to_datetime(df["Date"])
        df["gap_days"] = dt.diff().dt.days.fillna(1).astype(int)
        df["is_monday"] = (dt.dt.dayofweek == 0).astype(int)
    if "is_ihsg_holiday" not in df.columns:
        df["is_ihsg_holiday"] = 0
    df["ds"] = pd.date_range("2020-01-01", periods=len(df), freq="D")

    df_post = df.iloc[WARMUP_DISCARD:].reset_index(drop=True)
    total_test_rows = 344
    base_fit_end = len(df_post) - total_test_rows  # Index where first test target starts
    n_cutoffs = 335  # 60*5 + 35

    all_block_preds = []
    for fset in ["SPILL", "NOSPILL"]:
        fcols = SPILL_COLS if fset == "SPILL" else NOSPILL_COLS
        need = ["unique_id", "ds", "y"] + fcols

        for seed in SEEDS:
            seed_blocks = []
            offset = 0
            blk_id = 1
            while offset < n_cutoffs:
                cur_n_win = min(BLOCK_SIZE, n_cutoffs - offset)
                cur_fit_end = base_fit_end + offset
                fit_df = df_post.iloc[:cur_fit_end].copy()
                # Cross-validation end index covers cur_n_win cutoffs + (H - 1) tail rows
                cv_end = cur_fit_end + cur_n_win + (H - 1)
                cv_df = df_post.iloc[:cv_end].copy()

                set_seed(seed + blk_id)
                t_b = time.time()
                m = TFT(
                    h=H, input_size=50, hidden_size=64, hist_exog_list=fcols,
                    loss=MSE(), learning_rate=0.001, max_steps=1000,
                    val_check_steps=25, early_stop_patience_steps=5,
                    scaler_type="standard", random_seed=seed, enable_progress_bar=False,
                )
                nf = NeuralForecast(models=[m], freq="D")
                nf.fit(df=fit_df[need], val_size=N_VAL)
                cv = nf.cross_validation(df=cv_df[need], n_windows=cur_n_win, step_size=1, use_fitted=True)
                cv["h"] = cv.groupby("cutoff").cumcount() + 1
                cv["block_id"] = blk_id
                cv["FeatureSet"] = fset
                cv["seed"] = seed
                cv = cv.merge(df_post[["ds", "real_date"]], on="ds", how="left")
                seed_blocks.append(cv)
                print(f"  {fset} seed={seed} blk={blk_id} (win={cur_n_win}) done in {time.time()-t_b:.1f}s")
                offset += cur_n_win
                blk_id += 1

            seed_full = pd.concat(seed_blocks, ignore_index=True)
            seed_full.to_csv(out_dir / f"step3c_tft_refit60d_{fset}_s{seed}.csv", index=False)
            all_block_preds.append(seed_full)

    df_all = pd.concat(all_block_preds, ignore_index=True)
    ens = (
        df_all.groupby(["FeatureSet", "cutoff", "ds", "real_date", "h", "y"], as_index=False)["TFT"]
        .mean()
        .rename(columns={"TFT": "prediction"})
    )
    ens.to_csv(out_dir / "step3c_tft_refit60d_ensemble.csv", index=False)

    summary = []
    for fset in ["NOSPILL", "SPILL"]:
        for hv in [1, 10]:
            sub = ens[(ens["FeatureSet"] == fset) & (ens["h"] == hv)].sort_values("cutoff")
            summary.append({
                "Horizon": f"h={hv}",
                "FeatureSet": fset,
                "MSE": float(mean_squared_error(sub["y"], sub["prediction"])),
                "MAE": float(mean_absolute_error(sub["y"], sub["prediction"])),
                "Pct_Negative": float(np.mean(sub["prediction"] < 0) * 100.0),
            })
    df_sum = pd.DataFrame(summary)
    df_sum.to_csv(out_dir / "step3c_summary.csv", index=False)
    print(f"\nStep 3c complete in {(time.time()-t0)/60:.1f} min:\n{df_sum.to_string(index=False)}")


if __name__ == "__main__":
    main()

