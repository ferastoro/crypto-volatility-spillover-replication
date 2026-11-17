#!/usr/bin/env python3
"""
Step 3d (Kaggle GPU): Secondary target = Indodax 60-min Realized Variance (RV_60m).
Reads step3d_indodax_rv60m_daily.csv (pre-built from https://indodax.com/tradingview/history_v2)
or fetches 60m bars if Kaggle internet is ON, then trains TFT & NBEATSx (SPILL vs NOSPILL)
alongside HAR / HAR-X on the same 335 cutoffs.
Expected runtime on Kaggle Tesla T4 (16 GB): ~7 minutes (3 seeds TFT + 5 seeds NBEATSx).
"""

import json
import os
import sys
import time
import urllib.request
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import statsmodels.api as sm
import torch
from sklearn.metrics import mean_absolute_error, mean_squared_error

try:
    from neuralforecast import NeuralForecast
    from neuralforecast.models import TFT, NBEATSx
    from neuralforecast.losses.pytorch import MAE, MSE
except ImportError:
    import subprocess
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "neuralforecast==3.3.0"], check=True)
    from neuralforecast import NeuralForecast
    from neuralforecast.models import TFT, NBEATSx
    from neuralforecast.losses.pytorch import MAE, MSE

SEEDS_TFT = [42, 7, 13]
SEEDS_NBX = [42, 7, 13, 21, 99]
H = 10
N_VAL = 167
N_TEST_CUTOFFS = 335
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


def fetch_indodax_rv60m() -> pd.DataFrame:
    start_ts = int(pd.Timestamp("2020-01-01 00:00:00", tz="UTC").timestamp())
    to_ts = int(pd.Timestamp("2026-09-06 00:00:00", tz="UTC").timestamp())
    bars = []
    calls = 0
    while to_ts > start_ts and calls < 150:
        frm_ts = max(start_ts, to_ts - 86400 * 26)
        url = f"https://indodax.com/tradingview/history_v2?symbol=BTCIDR&tf=60&from={frm_ts}&to={to_ts}"
        calls += 1
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req, timeout=15) as resp:
            data = json.loads(resp.read().decode("utf-8", errors="replace"))
        if not isinstance(data, list) or not data:
            break
        bars.extend(data)
        earliest = min(int(b["Time"]) for b in data)
        to_ts = (earliest - 3600) if earliest < to_ts else (to_ts - 86400 * 25)
        time.sleep(0.08)
    df60 = pd.DataFrame(bars).drop_duplicates("Time").sort_values("Time").reset_index(drop=True)
    df60["Date"] = pd.to_datetime(df60["Time"], unit="s", utc=True).dt.strftime("%Y-%m-%d")
    df60["rv_1h"] = np.log(df60["Close"] / df60["Close"].shift(1)) ** 2
    return df60.dropna(subset=["rv_1h"]).groupby("Date", as_index=False)["rv_1h"].sum().rename(columns={"rv_1h": "rv_60m_daily_utc"})


def find_data_dirs():
    candidates = [
        Path(os.environ.get("KAGGLE_INPUT_ROOT", "./data/clean")),
        Path(os.environ.get("KAGGLE_INPUT_ROOT", "./inputs/v5")),
        Path("inputs/v5"),
        Path("data/clean"),
    ]
    dl_path = next((d / "dataset_deep_learning.csv" for d in candidates if (d / "dataset_deep_learning.csv").exists()), None)
    tvp_path = next((d / fn for d in candidates for fn in ["tvp_var_dynamic_features_causal.csv", "tvp_var_dynamic_features.csv"] if (d / fn).exists()), None)
    rv_candidates = [Path("revision/outputs/step3d_indodax_rv60m_daily.csv")] + [d / "step3d_indodax_rv60m_daily.csv" for d in candidates]
    rv_path = next((p for p in rv_candidates if p.exists()), None)
    return dl_path, tvp_path, rv_path


def main():
    t0 = time.time()
    out_dir = Path("/kaggle/working/step3d_rv60m_dl") if Path("/kaggle").exists() else Path("revision/kaggle_outputs/step3d")
    out_dir.mkdir(parents=True, exist_ok=True)

    dl_path, tvp_path, rv_path = find_data_dirs()
    df_dl = pd.read_csv(dl_path)
    df_tvp = pd.read_csv(tvp_path)
    df_rv = pd.read_csv(rv_path)[["Date", "rv_60m_daily_utc"]] if rv_path else fetch_indodax_rv60m()

    df = df_dl.merge(df_tvp, on="Date", how="left").merge(df_rv, on="Date", how="left").sort_values("Date").ffill().dropna().reset_index(drop=True)
    df["y"] = df["rv_60m_daily_utc"]
    df["unique_id"] = "BTC_IDR_RV60M"
    df["real_date"] = df["Date"]
    if "gap_days" not in df.columns:
        dt = pd.to_datetime(df["Date"])
        df["gap_days"] = dt.diff().dt.days.fillna(1).astype(int)
        df["is_monday"] = (dt.dt.dayofweek == 0).astype(int)
    if "is_ihsg_holiday" not in df.columns:
        df["is_ihsg_holiday"] = 0
    df["ds"] = pd.date_range("2020-01-01", periods=len(df), freq="D")

    df_post = df.iloc[WARMUP_DISCARD:].reset_index(drop=True)
    fit_df = df_post.iloc[: len(df_post) - 344].copy()

    rows = []
    for mname, seeds in [("TFT", SEEDS_TFT), ("NBEATSx", SEEDS_NBX)]:
        for fset in ["NOSPILL", "SPILL"]:
            fcols = SPILL_COLS if fset == "SPILL" else NOSPILL_COLS
            need = ["unique_id", "ds", "y"] + fcols
            sp_list = []
            for seed in seeds:
                torch.manual_seed(seed)
                np.random.seed(seed)
                if mname == "TFT":
                    m = TFT(h=H, input_size=50, hidden_size=64, hist_exog_list=fcols, loss=MSE(),
                            learning_rate=0.001, max_steps=1000, val_check_steps=25, early_stop_patience_steps=5,
                            scaler_type="standard", random_seed=seed, enable_progress_bar=False)
                else:
                    m = NBEATSx(h=H, input_size=20, hist_exog_list=fcols, mlp_units=[[256, 256]] * 3, loss=MAE(),
                                learning_rate=0.0003, max_steps=1000, val_check_steps=25, early_stop_patience_steps=5,
                                scaler_type="standard", random_seed=seed, enable_progress_bar=False)
                nf = NeuralForecast(models=[m], freq="D")
                nf.fit(df=fit_df[need], val_size=N_VAL)
                cv = nf.cross_validation(df=df_post[need], n_windows=N_TEST_CUTOFFS, step_size=1, use_fitted=True)
                cv["h"] = cv.groupby("cutoff").cumcount() + 1
                sp_list.append(cv[["cutoff", "ds", "h", "y", mname]].rename(columns={mname: f"pred_s{seed}"}))
            ens = sp_list[0].copy()
            for sp in sp_list[1:]:
                sc = [c for c in sp.columns if c.startswith("pred_s")][0]
                ens = ens.merge(sp[["cutoff", "ds", sc]], on=["cutoff", "ds"])
            ens["prediction"] = ens[[f"pred_s{s}" for s in seeds]].mean(axis=1)
            for hv in [1, 10]:
                sub = ens[ens["h"] == hv]
                rows.append({
                    "Horizon": f"h={hv}",
                    "Model": f"{mname}-{fset}_RV60m",
                    "MSE": float(mean_squared_error(sub["y"], sub["prediction"])),
                    "MAE": float(mean_absolute_error(sub["y"], sub["prediction"])),
                    "Pct_Negative": float(np.mean(sub["prediction"] < 0) * 100.0),
                })

    df_out = pd.DataFrame(rows)
    df_out.to_csv(out_dir / "step3d_rv60m_dl_metrics.csv", index=False)
    print(f"Step 3d DL on RV_60m complete in {(time.time()-t0)/60:.1f} min:\n{df_out.to_string(index=False)}")


if __name__ == "__main__":
    main()

