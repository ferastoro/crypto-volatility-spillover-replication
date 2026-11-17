#!/usr/bin/env python3
"""
Step 3a (Kaggle GPU): Retrain TFT and NBEATSx (SPILL & NOSPILL) excluding the
first 250 warm-up observations from training (fair identical row indices for both),
across 5 seeds [42, 7, 13, 21, 99].
Input path: ${KAGGLE_INPUT_ROOT}/data_clean
            (plus inputs/v5/tvp_var_dynamic_features_causal.csv or Stage-1 causal features)
Expected runtime on Kaggle Tesla T4 (16 GB): ~6.5 minutes total.
"""

import os
import sys
import json
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
    from neuralforecast.models import TFT, NBEATSx
    from neuralforecast.losses.pytorch import MAE, MSE
except ImportError:
    import subprocess
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "neuralforecast==3.3.0", "arch"], check=True)
    from neuralforecast import NeuralForecast
    from neuralforecast.models import TFT, NBEATSx
    from neuralforecast.losses.pytorch import MAE, MSE

from pytorch_lightning.callbacks import EarlyStopping
import pytorch_lightning as pl

SEEDS = [42, 7, 13, 21, 99]
H = 10
WARMUP_DISCARD = 250
MAX_STEPS = 1000
VAL_CHECK_STEPS = 25
PATIENCE = 5
N_VAL = 167
N_TEST_CUTOFFS = 335

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
BEST_HP = {
    "TFT": {"input_size": 50, "learning_rate": 0.001, "hidden_size": 64, "loss_name": "MSE"},
    "NBEATSx": {"input_size": 20, "learning_rate": 0.0003, "mlp_units": [[256, 256], [256, 256], [256, 256]], "loss_name": "MAE"},
}


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
    dl_path, tvp_path = None, None
    for d in candidates:
        if (d / "dataset_deep_learning.csv").exists():
            dl_path = d / "dataset_deep_learning.csv"
            break
    for d in candidates:
        for fn in ["tvp_var_dynamic_features_causal.csv", "tvp_var_dynamic_features.csv"]:
            if (d / fn).exists():
                tvp_path = d / fn
                break
        if tvp_path is not None:
            break
    if dl_path is None or tvp_path is None:
        raise FileNotFoundError(f"Could not find dataset_deep_learning.csv or tvp_var features in {candidates}")
    return dl_path, tvp_path


class RestoreBestCallback(pl.Callback):
    def __init__(self):
        super().__init__()
        self.best_score = None
        self.best_step = None
        self.best_state = None

    def on_validation_end(self, trainer, pl_module):
        if getattr(trainer, "sanity_checking", False) or trainer.global_step == 0:
            return
        v = trainer.callback_metrics.get("ptl/val_loss", None)
        if v is not None and (self.best_score is None or float(v) < self.best_score):
            self.best_score = float(v)
            self.best_step = int(trainer.global_step)
            self.best_state = {k: val.detach().cpu().clone() for k, val in pl_module.state_dict().items()}

    def on_train_end(self, trainer, pl_module):
        if self.best_state is not None:
            pl_module.load_state_dict(self.best_state)


def build_model(mname, fcols, seed):
    hp = BEST_HP[mname]
    loss = MAE() if hp["loss_name"] == "MAE" else MSE()
    cb = RestoreBestCallback()
    kw = {
        "h": H,
        "input_size": hp["input_size"],
        "hist_exog_list": fcols,
        "max_steps": MAX_STEPS,
        "val_check_steps": VAL_CHECK_STEPS,
        "early_stop_patience_steps": PATIENCE,
        "loss": loss,
        "random_seed": seed,
        "scaler_type": "standard",
        "learning_rate": hp["learning_rate"],
    }
    if mname == "TFT":
        kw["hidden_size"] = hp["hidden_size"]
        m = TFT(**kw)
    else:
        kw["mlp_units"] = hp["mlp_units"]
        m = NBEATSx(**kw)
    m.trainer_kwargs["callbacks"] = list(m.trainer_kwargs.get("callbacks", [])) + [cb]
    m.trainer_kwargs["enable_progress_bar"] = False
    return m, cb


def dm_hln(l1, l2, h):
    d = np.asarray(l1 - l2, dtype=float)
    n = len(d)
    md = float(np.mean(d))
    lag = max(0, h - 1)
    dc = d - md
    v = float(np.dot(dc, dc) / n)
    for l in range(1, lag + 1):
        v += 2.0 * (1.0 - l / (lag + 1.0)) * float(np.dot(dc[l:], dc[:-l]) / n)
    v = max(v, 1e-24)
    stat = float(md / np.sqrt(v / n))
    if h > 1:
        stat *= float(np.sqrt((n + 1.0 - 2.0 * h + (h / n) * (h - 1.0)) / n))
    p = float(2.0 * (1.0 - stats.t.cdf(abs(stat), df=n - 1)))
    return stat, p


def main():
    t0 = time.time()
    out_dir = Path("/kaggle/working/step3a_warmup_excluded") if Path("/kaggle").exists() else Path("revision/kaggle_outputs/step3a")
    out_dir.mkdir(parents=True, exist_ok=True)

    dl_path, tvp_path = find_data_dirs()
    df_dl = pd.read_csv(dl_path)
    df_tvp = pd.read_csv(tvp_path)
    df = df_dl.merge(df_tvp, on="Date", how="left").sort_values("Date").ffill().dropna().reset_index(drop=True)
    # Prepare target & missing calendar dummies identically to v6
    df["y"] = df["ret_crypto_btc_idr_indodax"] ** 2
    df["unique_id"] = "BTC_IDR"
    df["real_date"] = df["Date"]
    if "gap_days" not in df.columns:
        dt = pd.to_datetime(df["Date"])
        df["gap_days"] = dt.diff().dt.days.fillna(1).astype(int)
        df["is_monday"] = (dt.dt.dayofweek == 0).astype(int)
    if "is_ihsg_holiday" not in df.columns:
        df["is_ihsg_holiday"] = 0

    # Synthetic daily ds for NeuralForecast
    df["ds"] = pd.date_range("2020-01-01", periods=len(df), freq="D")

    # Discard first 250 warm-up observations from TRAINING ONLY while preserving exact val (167) and OOS test (344 rows / 335 cutoffs)!
    # Total len(df)=1676 -> train_val=1332, test=344. After discarding [0:250], fit_df has rows [250:1332] (1082 rows: 915 train + 167 val),
    # while cv_df has rows [250:] (1426 rows), so cross_validation(n_windows=335) evaluates on the exact same 335 test cutoffs!
    n_test_rows = 344
    n_train_val_orig = len(df) - n_test_rows
    df_post = df.iloc[WARMUP_DISCARD:].reset_index(drop=True)
    fit_df = df_post.iloc[: n_train_val_orig - WARMUP_DISCARD].copy()
    print(f"Step 3a Dataset: total={len(df)}, after removing {WARMUP_DISCARD} warmup fit rows={len(fit_df)} "
          f"(train={len(fit_df)-N_VAL}, val={N_VAL}), test_cutoffs={N_TEST_CUTOFFS}")

    configs = [("TFT", "SPILL"), ("TFT", "NOSPILL"), ("NBEATSx", "SPILL"), ("NBEATSx", "NOSPILL")]
    ens_records = []
    metric_rows = []

    for mname, fset in configs:
        fcols = SPILL_COLS if fset == "SPILL" else NOSPILL_COLS
        need_cols = ["unique_id", "ds", "y"] + fcols
        seed_preds = []
        for seed in SEEDS:
            set_seed(seed)
            t_run = time.time()
            model, cb = build_model(mname, fcols, seed)
            nf = NeuralForecast(models=[model], freq="D")
            nf.fit(df=fit_df[need_cols], val_size=N_VAL)
            cv = nf.cross_validation(df=df_post[need_cols], n_windows=N_TEST_CUTOFFS, step_size=1, use_fitted=True)
            cv = cv.merge(df_post[["ds", "real_date"]], on="ds", how="left")
            cv["h"] = cv.groupby("cutoff").cumcount() + 1
            cv.to_csv(out_dir / f"v6p_nowarmup_{mname}_{fset}_s{seed}.csv", index=False)
            seed_preds.append(cv[["cutoff", "ds", "real_date", "h", "y", mname]].rename(columns={mname: f"pred_s{seed}"}))
            print(f"  {mname}-{fset} seed={seed}: best_step={cb.best_step}, time={time.time()-t_run:.1f}s")

        ens_df = seed_preds[0].copy()
        for sp in seed_preds[1:]:
            ens_df = ens_df.merge(sp[["cutoff", "ds", f"pred_s{sp.columns[-1].replace('pred_s', '')}"]], on=["cutoff", "ds"])
        pcols = [f"pred_s{s}" for s in SEEDS]
        ens_df["prediction"] = ens_df[pcols].mean(axis=1)
        ens_df["Config"] = f"{mname}_{fset}_NoWarmup250"
        ens_records.append(ens_df)

        for hv in [1, 10]:
            sub = ens_df[ens_df["h"] == hv].sort_values("cutoff").reset_index(drop=True)
            mse = float(mean_squared_error(sub["y"], sub["prediction"]))
            mae = float(mean_absolute_error(sub["y"], sub["prediction"]))
            pct_neg = float(np.mean(sub["prediction"] < 0) * 100.0)
            metric_rows.append({"Horizon": f"h={hv}", "Model": mname, "FeatureSet": fset, "MSE": mse, "MAE": mae, "Pct_Negative": pct_neg})

    all_ens = pd.concat(ens_records, ignore_index=True)
    all_ens.to_csv(out_dir / "step3a_ensemble_predictions.csv", index=False)
    df_met = pd.DataFrame(metric_rows)
    df_met.to_csv(out_dir / "step3a_metrics_summary.csv", index=False)

    # SPILL vs NOSPILL HLN-DM test
    dm_rows = []
    for mname in ["TFT", "NBEATSx"]:
        for hv in [1, 10]:
            sp = all_ens[(all_ens["Config"] == f"{mname}_SPILL_NoWarmup250") & (all_ens["h"] == hv)].sort_values("cutoff")
            ns = all_ens[(all_ens["Config"] == f"{mname}_NOSPILL_NoWarmup250") & (all_ens["h"] == hv)].sort_values("cutoff")
            se_sp = (sp["y"].values - sp["prediction"].values) ** 2
            se_ns = (ns["y"].values - ns["prediction"].values) ** 2
            stat, p = dm_hln(se_sp, se_ns, hv)
            dm_rows.append({"Horizon": f"h={hv}", "Model": mname, "HLN_DM_SE": stat, "p_raw": p})
    pd.DataFrame(dm_rows).to_csv(out_dir / "step3a_dm_spill_vs_nospill.csv", index=False)
    print(f"\nStep 3a complete in {(time.time()-t0)/60:.1f} min. Metrics:\n{df_met.to_string(index=False)}")


if __name__ == "__main__":
    main()

