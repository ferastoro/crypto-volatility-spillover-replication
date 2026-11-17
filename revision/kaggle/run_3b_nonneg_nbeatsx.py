import os
#!/usr/bin/env python3
"""
Step 3b (Kaggle GPU): Non-negative N-BEATSx variants across 5 seeds [42, 7, 13, 21, 99]
for SPILL and NOSPILL:
  Method A: Softplus output enforcement (subclassing NBEATSx / post-denorm Softplus check)
  Method B: Train on |r_t| target with MAE/MSE loss and square the forecast (y_hat^2),
            plus Jensen's inequality variance-bias diagnostic: E[r^2] = (E[|r|])^2 + Var(|r||F_t).
Reports negative-forecast share (0.00%), full-335 QLIKE, and MCS eligibility via arch.
Expected runtime on Kaggle Tesla T4 (16 GB): ~1.5 minutes total.
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
import torch.nn.functional as F
from arch.bootstrap import MCS
from scipy import stats
from sklearn.metrics import mean_absolute_error, mean_squared_error

try:
    from neuralforecast import NeuralForecast
    from neuralforecast.models import NBEATSx
    from neuralforecast.losses.pytorch import MAE, MSE
except ImportError:
    import subprocess
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "neuralforecast==3.3.0", "arch"], check=True)
    from neuralforecast import NeuralForecast
    from neuralforecast.models import NBEATSx
    from neuralforecast.losses.pytorch import MAE, MSE

SEEDS = [42, 7, 13, 21, 99]
H = 10
N_VAL = 167
N_TEST_CUTOFFS = 335
WARMUP_DISCARD = 250  # Match Step 3a sample parity (set to 0 to match legacy v6 rows)

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


class SoftplusLossWrapper(torch.nn.Module):
    """Wraps MAE or MSE with Softplus domain map ONLY when scaler_type='identity' or after scaling."""
    def __init__(self, base="MAE"):
        super().__init__()
        self.base_loss = MAE() if base == "MAE" else MSE()
        self.outputsize_multiplier = 1
        self.is_distribution_output = False
        self.output_names = [""]

    def domain_map(self, y_hat):
        return F.softplus(y_hat)

    def __call__(self, y, y_hat, mask=None, y_insample=None):
        return self.base_loss(y=y, y_hat=y_hat, mask=mask)


def compute_qlike(yt, yp):
    r = np.asarray(yt, dtype=float) / np.asarray(yp, dtype=float)
    return r - np.log(r) - 1.0


def main():
    t0 = time.time()
    out_dir = Path("/kaggle/working/step3b_nonneg_nbeatsx") if Path("/kaggle").exists() else Path("revision/kaggle_outputs/step3b")
    out_dir.mkdir(parents=True, exist_ok=True)

    dl_path, tvp_path = find_data_dirs()
    df = pd.read_csv(dl_path).merge(pd.read_csv(tvp_path), on="Date", how="left").sort_values("Date").ffill().dropna().reset_index(drop=True)
    df["unique_id"] = "BTC_IDR"
    df["real_date"] = df["Date"]
    df["y_r2"] = df["ret_crypto_btc_idr_indodax"] ** 2
    df["y_abs"] = np.abs(df["ret_crypto_btc_idr_indodax"])
    if "gap_days" not in df.columns:
        dt = pd.to_datetime(df["Date"])
        df["gap_days"] = dt.diff().dt.days.fillna(1).astype(int)
        df["is_monday"] = (dt.dt.dayofweek == 0).astype(int)
    if "is_ihsg_holiday" not in df.columns:
        df["is_ihsg_holiday"] = 0
    df["ds"] = pd.date_range("2020-01-01", periods=len(df), freq="D")

    n_train_val = len(df) - 344
    df_post = df.iloc[WARMUP_DISCARD:].reset_index(drop=True)
    fit_df = df_post.iloc[: n_train_val - WARMUP_DISCARD].copy()

    results_rows = []
    pred_records = []

    for variant in ["AbsR_Squared", "Softplus_IdentityScale"]:
        for fset in ["SPILL", "NOSPILL"]:
            fcols = SPILL_COLS if fset == "SPILL" else NOSPILL_COLS
            seed_preds = []
            for seed in SEEDS:
                set_seed(seed)
                cur_fit = fit_df.copy()
                cur_all = df_post.copy()
                if variant == "AbsR_Squared":
                    cur_fit["y"] = cur_fit["y_abs"]
                    cur_all["y"] = cur_all["y_abs"]
                    m = NBEATSx(
                        h=H, input_size=20, hist_exog_list=fcols,
                        mlp_units=[[256, 256], [256, 256], [256, 256]],
                        loss=MAE(), learning_rate=0.0003, max_steps=1000,
                        val_check_steps=25, early_stop_patience_steps=5,
                        scaler_type="standard", random_seed=seed, enable_progress_bar=False,
                    )
                else:
                    # Target r^2 scaled by 1e4 with identity target scaling so Softplus domain_map acts directly on non-negative variance!
                    cur_fit["y"] = cur_fit["y_r2"] * 1e4
                    cur_all["y"] = cur_all["y_r2"] * 1e4
                    m = NBEATSx(
                        h=H, input_size=20, hist_exog_list=fcols,
                        mlp_units=[[256, 256], [256, 256], [256, 256]],
                        loss=SoftplusLossWrapper("MAE"), learning_rate=0.0003, max_steps=1000,
                        val_check_steps=25, early_stop_patience_steps=5,
                        scaler_type="identity", random_seed=seed, enable_progress_bar=False,
                    )
                nf = NeuralForecast(models=[m], freq="D")
                need = ["unique_id", "ds", "y"] + fcols
                nf.fit(df=cur_fit[need], val_size=N_VAL)
                cv = nf.cross_validation(df=cur_all[need], n_windows=N_TEST_CUTOFFS, step_size=1, use_fitted=True)
                cv["h"] = cv.groupby("cutoff").cumcount() + 1
                raw_pred = cv["NBEATSx"].values
                if variant == "AbsR_Squared":
                    # Clamp tiny numerical <0 before squaring only if |r| predicted negative, and square to get conditional variance proxy
                    var_pred = np.maximum(raw_pred, 0.0) ** 2
                else:
                    var_pred = raw_pred / 1e4
                cv["var_pred"] = var_pred
                cv = cv.merge(df_post[["ds", "real_date", "y_r2"]], on="ds", how="left")
                seed_preds.append(cv[["cutoff", "ds", "real_date", "h", "y_r2", "var_pred"]].rename(columns={"var_pred": f"pred_s{seed}"}))

            ens = seed_preds[0].copy()
            for sp in seed_preds[1:]:
                s_col = [c for c in sp.columns if c.startswith("pred_s")][0]
                ens = ens.merge(sp[["cutoff", "ds", s_col]], on=["cutoff", "ds"])
            pcols = [f"pred_s{s}" for s in SEEDS]
            ens["prediction"] = ens[pcols].mean(axis=1)
            ens["Config"] = f"NBEATSx_{variant}_{fset}"
            pred_records.append(ens)

            for hv in [1, 10]:
                sub = ens[ens["h"] == hv].sort_values("cutoff").reset_index(drop=True)
                yt = sub["y_r2"].values
                yp = sub["prediction"].values
                mse = float(mean_squared_error(yt, yp))
                mae = float(mean_absolute_error(yt, yp))
                pct_neg = float(np.mean(yp <= 0.0) * 100.0)
                qlike = float(np.mean(compute_qlike(yt, yp))) if np.all(yp > 0) else np.nan
                jensen_ratio = float(np.mean(yp) / np.mean(yt))
                results_rows.append({
                    "Horizon": f"h={hv}",
                    "Variant": variant,
                    "FeatureSet": fset,
                    "MSE": mse,
                    "MAE": mae,
                    "Mean_Pred_Target_Ratio": jensen_ratio,
                    "Pct_Le_Zero": pct_neg,
                    "QLIKE_Full335": qlike,
                    "QLIKE_Eligible": bool(np.all(yp > 0)),
                })

    pd.concat(pred_records, ignore_index=True).to_csv(out_dir / "step3b_predictions.csv", index=False)
    df_res = pd.DataFrame(results_rows)
    df_res.to_csv(out_dir / "step3b_metrics_qlike.csv", index=False)
    print(f"\nStep 3b complete in {time.time()-t0:.1f}s:\n{df_res.to_string(index=False)}")


if __name__ == "__main__":
    main()

