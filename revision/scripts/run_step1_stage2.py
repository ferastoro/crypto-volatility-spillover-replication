#!/usr/bin/env python3
"""
Step 1 Stage-2 Revisions (Items 1g, 1h, 1i)
- 1i: Correlation matrix, VIF, and exact linear recoverability proof for 9 Stage-2 SPILL features
- 1h: New econometric baselines on identical 335 cutoffs:
      GARCH(1,1)-t, EGARCH(1,1)-t (Student-t innovations via arch),
      parsimonious HAR-X (NOSPILL, SPILL={TSI, AI->BTC/IDR}, NOSPILL+SPILL)
- 1g: Complete model-set evaluation (SE, AE, QLIKE), N-BEATSx raw vs clipped,
      Feb 5-6 outlier impact (MSE + ranks), HLN Diebold-Mariano tests (explicit loss & Bonferroni family table),
      Model Confidence Set (MCS), and exploratory v6/v7 variants appendix CSVs.
All outputs saved to revision/outputs/*.csv.
"""

import json
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm
from arch import arch_model
from arch.bootstrap import MCS
from scipy import stats
from sklearn.metrics import mean_absolute_error, mean_squared_error
from statsmodels.stats.outliers_influence import variance_inflation_factor

warnings.filterwarnings("ignore")

BASE_DIR = Path(__file__).resolve().parent.parent.parent
CLEAN_DIR = BASE_DIR / "data" / "clean"
INPUTS_V5 = BASE_DIR / "inputs" / "v5"
RESULTS_V6 = BASE_DIR / "results" / "v6"
RESULTS_V7 = BASE_DIR / "results" / "v7-tuned"
OUT_DIR = BASE_DIR / "revision" / "outputs"
OUT_DIR.mkdir(parents=True, exist_ok=True)

ROLL_WINDOW = 1174
FEB_OUTLIERS = {"2026-02-05", "2026-02-06"}


def compute_qlike(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
    yt = np.asarray(y_true, dtype=float)
    yp = np.asarray(y_pred, dtype=float)
    ratio = yt / yp
    return ratio - np.log(ratio) - 1.0


def dm_hln_test(loss_m: np.ndarray, loss_b: np.ndarray, h: int) -> dict:
    """
    Diebold-Mariano (1995) with Newey-West/Bartlett HAC weights and
    Harvey-Leybourne-Newbold (1997) Student-t (df=n-1) finite-sample adjustment.
    Negative stat means `loss_m < loss_b` (candidate model beats benchmark).
    """
    mask = ~(np.isnan(loss_m) | np.isnan(loss_b) | np.isinf(loss_m) | np.isinf(loss_b))
    d = np.asarray(loss_m[mask] - loss_b[mask], dtype=float)
    n = len(d)
    md = float(np.mean(d))
    lag = max(0, h - 1)
    d_c = d - md
    gamma0 = float(np.dot(d_c, d_c) / n)
    var_sum = gamma0
    for l in range(1, lag + 1):
        w = 1.0 - l / (lag + 1.0)
        cov_l = float(np.dot(d_c[l:], d_c[:-l]) / n)
        var_sum += 2.0 * w * cov_l
    if var_sum <= 0:
        var_sum = max(gamma0, 1e-24)
    stat_raw = md / np.sqrt(var_sum / n)
    if h > 1:
        hln = np.sqrt((n + 1.0 - 2.0 * h + (h / n) * (h - 1.0)) / n)
        stat = float(stat_raw * hln)
    else:
        stat = float(stat_raw)
    p_t = float(2.0 * (1.0 - stats.t.cdf(abs(stat), df=n - 1)))
    p_norm = float(2.0 * (1.0 - stats.norm.cdf(abs(stat))))
    return {"N": n, "md": md, "stat_hln": stat, "p_raw_t": p_t, "p_raw_norm": p_norm}


def run_step1i_feature_check(dl: pd.DataFrame, tvp_c: pd.DataFrame):
    print("\n--- Running Step 1i: SPILL Feature Correlation, VIF & Recoverability ---")
    m = dl[["Date"]].merge(tvp_c, on="Date", how="inner").sort_values("Date").reset_index(drop=True)
    assert len(m) == 1676, f"Expected 1676 US-aligned SPILL rows without ffill, got {len(m)}"
    assert not m.isna().any().any(), "Unexpected NaNs in Stage-2 SPILL features!"
    spill9 = [
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
    net7 = [c for c in spill9 if c.startswith("net_spill_")]
    if "net_spill_crypto_btc_idr_indodax" in m.columns:
        rec_err = np.abs(m[net7].sum(axis=1) + m["net_spill_crypto_btc_idr_indodax"])
    else:
        import sys
        if str(Path(__file__).resolve().parent) not in sys.path:
            sys.path.insert(0, str(Path(__file__).resolve().parent))
        from run_step1_stage1 import ASSETS_8, WARMUP_OBS, build_aligned_panel, estimate_tvp_var_gfevd, make_proxy_matrices
        panel = build_aligned_panel(ASSETS_8, join_ihsg=False)
        lbls = [a[1] for a in ASSETS_8]
        y_lr2 = make_proxy_matrices(panel, lbls)["log_r2"]
        res_s1 = estimate_tvp_var_gfevd(y_lr2, p=1, warmup=WARMUP_OBS, causal_sigma0=True)
        btci_i = lbls.index("BTC/IDR")
        G = res_s1["gfevd_all"]
        btci_net = (G[:, :, btci_i].sum(axis=1) - G[:, btci_i, :].sum(axis=1)) * 100.0
        rec_err = np.abs(m[net7].sum(axis=1).values + btci_net)

    # Compute across full merged Stage-2 sample (N=1676) AND train-only first 1174 effective rows
    corr = m[spill9].corr()
    corr.to_csv(OUT_DIR / "step1i_spill_corr_matrix.csv")

    X = sm.add_constant(m[spill9].values)
    X_post250 = sm.add_constant(m[spill9].iloc[250:].values)
    vif_rows = []
    cond_full = float(np.linalg.cond(X))
    cond_std = float(np.linalg.cond(sm.add_constant((m[spill9] - m[spill9].mean()) / m[spill9].std()).values))
    for i, col in enumerate(spill9, start=1):
        vif_full = float(variance_inflation_factor(X, i))
        vif_post = float(variance_inflation_factor(X_post250, i))
        vif_rows.append({
            "Feature": col,
            "Mean_N1676": float(m[col].mean()),
            "SD_N1676": float(m[col].std(ddof=1)),
            "VIF_Full_N1676": vif_full,
            "VIF_PostWarmup_N1426": vif_post,
            "Cond_Num_Unstd": cond_full,
            "Cond_Num_Std": cond_std,
            "Max_TargetNet_Recovery_Error": float(rec_err.max()),
        })
    df_vif = pd.DataFrame(vif_rows)
    df_vif.to_csv(OUT_DIR / "step1i_spill_vif.csv", index=False)
    print(df_vif[["Feature", "VIF_Full_N1676", "VIF_PostWarmup_N1426", "Max_TargetNet_Recovery_Error"]].round(4).to_string(index=False))


def estimate_garch_t_baselines(dl: pd.DataFrame, cutoffs_df: pd.DataFrame) -> pd.DataFrame:
    print("\n--- Estimating GARCH(1,1)-t and EGARCH(1,1)-t (roll=1174, 335 cutoffs) ---")
    t0 = time.time()
    r = dl["ret_crypto_btc_idr_indodax"].values
    dates = dl["Date"].values

    rows = []
    conv_fail_g = 0
    conv_fail_e = 0
    unique_cutoffs = cutoffs_df[["cutoff", "origin_idx"]].drop_duplicates().sort_values("origin_idx")

    for k, (_, crow) in enumerate(unique_cutoffs.iterrows()):
        c_str = crow["cutoff"]
        o_idx = int(crow["origin_idx"])
        win = r[o_idx - ROLL_WINDOW + 1 : o_idx + 1] * 100.0

        # GARCH(1,1)-t
        gm = arch_model(win, mean="Constant", vol="GARCH", p=1, q=1, dist="StudentsT").fit(disp="off", show_warning=False)
        if gm.convergence_flag != 0:
            gm = arch_model(win, mean="Constant", vol="GARCH", p=1, q=1, dist="StudentsT").fit(
                disp="off", show_warning=False, options={"maxiter": 1000}
            )
            if gm.convergence_flag != 0:
                conv_fail_g += 1
        fc_g = gm.forecast(horizon=10, reindex=False).variance.values[-1] / 10000.0

        # EGARCH(1,1)-t
        em = arch_model(win, mean="Constant", vol="EGARCH", p=1, o=1, q=1, dist="StudentsT").fit(disp="off", show_warning=False)
        if em.convergence_flag != 0:
            em = arch_model(win, mean="Constant", vol="EGARCH", p=1, o=1, q=1, dist="StudentsT").fit(
                disp="off", show_warning=False, options={"maxiter": 1000}
            )
            if em.convergence_flag != 0:
                conv_fail_e += 1
        fc_e = em.forecast(horizon=10, method="simulation", simulations=3000, random_state=42 + k, reindex=False).variance.values[-1] / 10000.0

        # Map exact target real_dates h=1..10
        sub_c = cutoffs_df[cutoffs_df["cutoff"] == c_str].sort_values("h")
        for _, srow in sub_c.iterrows():
            hv = int(srow["h"])
            rows.append({
                "cutoff": c_str,
                "real_date": srow["real_date"],
                "h": hv,
                "GARCH11_t": float(fc_g[hv - 1]),
                "EGARCH_t": float(fc_e[hv - 1]),
                "garch_t_conv": int(gm.convergence_flag),
                "egarch_t_conv": int(em.convergence_flag),
            })

    print(f"GARCH-t & EGARCH-t done in {time.time() - t0:.1f}s (non-zero conv flags: GARCH-t={conv_fail_g}, EGARCH-t={conv_fail_e})")
    return pd.DataFrame(rows)


def estimate_parsimonious_har_x(dl: pd.DataFrame, tvp_c: pd.DataFrame, cutoffs_df: pd.DataFrame) -> pd.DataFrame:
    """
    Fair parsimonious HAR-X test (RQ3):
      HAR:            [r2_1, r2_5, r2_22]
      HAR_X_NOSPILL:  [r2_1, r2_5, r2_22, r2_btcusd, r2_ai_mean]         (+2 raw market vol controls)
      HAR_X_SPILL:    [r2_1, r2_5, r2_22, tvp_tsi, spillover_ai_to_btcidr] (+2 strictly post-warm-up spillover features)
      HAR_X_FULL:     [r2_1, r2_5, r2_22, r2_btcusd, r2_ai_mean, tvp_tsi, spillover_ai_to_btcidr] (nested, post-warm-up)
    Univariate HAR and HAR_X_NOSPILL use the full rolling 1,174-day window; HAR_X_SPILL and HAR_X_FULL
    exclude the first 250 TVP-VAR burn-in steps (train_s >= post-250 date 2021-01-26) so spillover covariates
    are strictly post-warm-up across every origin.
    """
    print("\n--- Estimating Parsimonious HAR & HAR-X variants (h=1..10, 335 cutoffs; HAR-X-SPILL/FULL post-250 warmup) ---")
    tvp_sorted = tvp_c.sort_values("Date").reset_index(drop=True)
    warmup_cutoff_date = str(tvp_sorted.loc[250, "Date"])
    df = dl.merge(tvp_sorted[["Date", "tvp_tsi", "spillover_ai_to_btcidr"]], on="Date", how="left").sort_values("Date").reset_index(drop=True)
    dl_idx_post250 = int(df[df["Date"] >= warmup_cutoff_date].index[0])
    assert not df.iloc[dl_idx_post250:][["tvp_tsi", "spillover_ai_to_btcidr"]].isna().any().any(), "Unexpected NaNs in post-warmup HAR-X SPILL covariates!"
    df["r2_1"] = df["ret_crypto_btc_idr_indodax"] ** 2
    df["r2_5"] = df["r2_1"].rolling(5, min_periods=1).mean()
    df["r2_22"] = df["r2_1"].rolling(22, min_periods=1).mean()
    df["r2_btcusd"] = df["ret_crypto_btc_usd"] ** 2
    ai_ret_cols = ["ret_saham_nvda", "ret_saham_msft", "ret_saham_tsla", "ret_saham_googl", "ret_saham_meta"]
    df["r2_ai_mean"] = (df[ai_ret_cols] ** 2).mean(axis=1)

    specs = {
        "HAR_recheck": ["r2_1", "r2_5", "r2_22"],
        "HAR_X_NOSPILL": ["r2_1", "r2_5", "r2_22", "r2_btcusd", "r2_ai_mean"],
        "HAR_X_SPILL": ["r2_1", "r2_5", "r2_22", "tvp_tsi", "spillover_ai_to_btcidr"],
        "HAR_X_FULL": ["r2_1", "r2_5", "r2_22", "r2_btcusd", "r2_ai_mean", "tvp_tsi", "spillover_ai_to_btcidr"],
    }

    unique_cutoffs = cutoffs_df[["cutoff", "origin_idx"]].drop_duplicates().sort_values("origin_idx")
    out_rows = []
    for _, crow in unique_cutoffs.iterrows():
        c_str = crow["cutoff"]
        o_idx = int(crow["origin_idx"])
        sub_c = cutoffs_df[cutoffs_df["cutoff"] == c_str].sort_values("h")
        for _, srow in sub_c.iterrows():
            hv = int(srow["h"])
            train_s = np.arange(o_idx - hv - ROLL_WINDOW + 1, o_idx - hv + 1)
            train_s = train_s[train_s >= 0]
            rec = {"cutoff": c_str, "real_date": srow["real_date"], "h": hv}
            for mname, flist in specs.items():
                ts_use = train_s[train_s >= dl_idx_post250] if mname in ("HAR_X_SPILL", "HAR_X_FULL") else train_s
                X_tr = df.loc[ts_use, flist].values
                Y_tr = df.loc[ts_use + hv, "r2_1"].values
                valid = ~(np.isnan(X_tr).any(axis=1) | np.isnan(Y_tr))
                X_tr_c = sm.add_constant(X_tr[valid], has_constant="add")
                fit = sm.OLS(Y_tr[valid], X_tr_c).fit()
                x_te = np.insert(df.loc[o_idx, flist].values, 0, 1.0)
                rec[mname] = float(fit.predict(x_te)[0])
            out_rows.append(rec)

    res_df = pd.DataFrame(out_rows)
    return res_df


def eval_metrics_row(y: np.ndarray, yhat: np.ndarray, y_hexp: np.ndarray, real_dates: np.ndarray) -> dict:
    mse = float(mean_squared_error(y, yhat))
    mae = float(mean_absolute_error(y, yhat))
    rmse = float(np.sqrt(mse))
    mse_hexp = float(mean_squared_error(y, y_hexp))
    r2_oos = float(1.0 - mse / mse_hexp)
    mask_no = ~np.isin(real_dates, list(FEB_OUTLIERS))
    mse_no = float(mean_squared_error(y[mask_no], yhat[mask_no]))
    mae_no = float(mean_absolute_error(y[mask_no], yhat[mask_no]))
    out_share = float((1.0 - np.sum((y[~mask_no] - yhat[~mask_no]) ** 2) / np.sum((y - yhat) ** 2)) if False else
                      (np.sum((y[~mask_no] - yhat[~mask_no]) ** 2) / np.sum((y - yhat) ** 2) * 100.0))
    pct_neg = float(np.mean(yhat < 0.0) * 100.0)
    pct_le0 = float(np.mean(yhat <= 0.0) * 100.0)
    qlike_full = float(np.mean(compute_qlike(y, yhat))) if np.all(yhat > 0) else np.nan
    return {
        "N": int(len(y)),
        "MSE": mse,
        "MAE": mae,
        "RMSE": rmse,
        "R2_OOS_HistExp": r2_oos,
        "Bias_MeanErr": float(np.mean(yhat - y)),
        "Mean_Forecast": float(np.mean(yhat)),
        "Mean_Target": float(np.mean(y)),
        "Forecast_Target_Ratio": float(np.mean(yhat) / np.mean(y)),
        "Pct_Negative": pct_neg,
        "Pct_Le_Zero": pct_le0,
        "MSE_No_Feb5_6": mse_no,
        "MAE_No_Feb5_6": mae_no,
        "Feb5_6_Error_Share_pct": out_share,
        "QLIKE_Full335": qlike_full,
    }


def main():
    t0 = time.time()
    print("=== Step 1 Stage-2 Re-runs ===")

    dl = pd.read_csv(CLEAN_DIR / "dataset_deep_learning.csv")
    dl["r2"] = dl["ret_crypto_btc_idr_indodax"] ** 2
    tvp_c = pd.read_csv(OUT_DIR / "spill_features_main_us1677_causal.csv")

    # 1i first
    run_step1i_feature_check(dl, tvp_c)

    # Master aligned cutoff grid (335 origins x 10 horizons = 3,350)
    ens = pd.read_csv(RESULTS_V6 / "v6_ensemble_predictions.csv")
    ens["h"] = ens.groupby(["Config", "cutoff"]).cumcount() + 1
    g4 = pd.read_csv(INPUTS_V5 / "GARCH_Predictions_V4.csv")
    g4["h"] = g4.groupby("cutoff").cumcount() + 1
    har1 = pd.read_csv(INPUTS_V5 / "har_preds_T0_h1_v2.csv").sort_values("Date").reset_index(drop=True)
    har10 = pd.read_csv(INPUTS_V5 / "har_preds_T0_h10_v2.csv").sort_values("Date").reset_index(drop=True)

    sub_ref = ens[ens["Config"] == "TFT_SPILL_T0"].sort_values(["cutoff", "h"]).reset_index(drop=True)
    date_to_idx = {d: i for i, d in enumerate(dl["Date"].values)}
    h1_origins = sub_ref[sub_ref["h"] == 1].set_index("cutoff")["real_date"].map(lambda d: date_to_idx[d] - 1)
    sub_ref["origin_idx"] = sub_ref["cutoff"].map(h1_origins)

    # 1h: New baselines (reuse cached GARCH-t/EGARCH-t predictions if complete; always recompute fast HAR-X)
    pred_cache = OUT_DIR / "step1h_new_baselines_predictions.csv"
    har_x_df = estimate_parsimonious_har_x(dl, tvp_c, sub_ref[["cutoff", "real_date", "h", "origin_idx"]])
    if pred_cache.exists() and len(pd.read_csv(pred_cache)) == 3350:
        print(f"\n--- Reusing GARCH-t/EGARCH-t from {pred_cache.name} and updating HAR-X columns ---")
        garch_cols = ["cutoff", "real_date", "h", "GARCH11_t", "EGARCH_t", "garch_t_conv", "egarch_t_conv"]
        garch_t_df = pd.read_csv(pred_cache)[garch_cols]
    else:
        garch_t_df = estimate_garch_t_baselines(dl, sub_ref[["cutoff", "real_date", "h", "origin_idx"]])
    merged_new = garch_t_df.merge(har_x_df, on=["cutoff", "real_date", "h"])
    merged_new.to_csv(pred_cache, index=False)

    # Verify HAR_recheck matches inputs/v5/har_preds_T0_h1_v2 & h10_v2
    for hv, har_file in [(1, har1), (10, har10)]:
        sub_hx = merged_new[merged_new["h"] == hv].sort_values("real_date").reset_index(drop=True)
        diff = np.max(np.abs(sub_hx["HAR_recheck"].values - har_file["HAR"].values))
        print(f"Verify HAR_recheck vs inputs/v5 h={hv}: max diff = {diff:.2e}")

    # Assemble all predictions for h=1 and h=10 on identical sorted real_date grids
    v7_ens = pd.read_csv(RESULTS_V7 / "v7_ensemble_predictions.csv")
    v7_ens["cutoff"] = v7_ens["cutoff"].astype(str)
    v7_ens["real_date"] = v7_ens["real_date"].astype(str)
    v7_ens["h"] = v7_ens.groupby(["cutoff", "Config"]).cumcount() + 1

    all_preds = {}
    v7_own_y = {}
    for hv, har_file in [(1, har1), (10, har10)]:
        d = {"real_date": har_file["Date"].values, "y": har_file["True"].values}
        d["Hist-Exp"] = har_file["Hist_Exp"].values
        d["Roll-20"] = har_file["Roll_20"].values
        d["HAR"] = har_file["HAR"].values
        d["HAR-14_NOSPILL_v2"] = har_file["HAR_NOSPILL"].values
        d["HAR-23_SPILL_v2"] = har_file["HAR_SPILL"].values
        d["HAR-11_VOLX_v2"] = har_file["HAR_VOLX"].values

        g_sub = g4[g4["h"] == hv].sort_values("real_date").reset_index(drop=True)
        assert (g_sub["real_date"].values == d["real_date"]).all()
        d["GARCH(1,1)-N"] = g_sub["GARCH11"].values
        d["EGARCH(1,1)-N"] = g_sub["EGARCH"].values

        new_sub = merged_new[merged_new["h"] == hv].sort_values("real_date").reset_index(drop=True)
        assert (new_sub["real_date"].values == d["real_date"]).all()
        d["GARCH(1,1)-t"] = new_sub["GARCH11_t"].values
        d["EGARCH(1,1)-t"] = new_sub["EGARCH_t"].values
        d["HAR-X-NOSPILL"] = new_sub["HAR_X_NOSPILL"].values
        d["HAR-X-SPILL"] = new_sub["HAR_X_SPILL"].values
        d["HAR-X-FULL"] = new_sub["HAR_X_FULL"].values

        for cfg, alias in [
            ("TFT_NOSPILL_T0", "TFT-NOSPILL"),
            ("TFT_SPILL_T0", "TFT-SPILL"),
            ("NBEATSx_NOSPILL_T0", "N-BEATSx-NOSPILL_raw"),
            ("NBEATSx_SPILL_T0", "N-BEATSx-SPILL_raw"),
        ]:
            v6_s = ens[(ens["Config"] == cfg) & (ens["h"] == hv)].sort_values("real_date").reset_index(drop=True)
            assert (v6_s["real_date"].values == d["real_date"]).all()
            d[alias] = v6_s["prediction"].values
            if "N-BEATSx" in alias:
                d[alias.replace("_raw", "_clipped")] = np.maximum(v6_s["prediction"].values, 0.0)

        v7_own_y[hv] = {}
        for cfg in v7_ens["Config"].unique():
            v7_s = v7_ens[(v7_ens["Config"] == cfg) & (v7_ens["h"] == hv)].sort_values(["cutoff", "real_date"]).reset_index(drop=True)
            assert (v7_s["real_date"].values == har1["Date"].values).all()
            d[cfg] = v7_s["prediction"].values
            v7_own_y[hv][cfg] = v7_s["y"].values

        all_preds[hv] = d

    # -------------------------------------------------------------------------
    # Evaluate Main Model Sets (9-model Table 3 + new baselines = 13 raw + clipped)
    # -------------------------------------------------------------------------
    primary_models = [
        ("Hist-Exp", "Table3_Main_9", "Econometric/Naive"),
        ("Roll-20", "Table3_Main_9", "Econometric/Naive"),
        ("GARCH(1,1)-N", "Table3_Main_9", "Econometric/GARCH"),
        ("EGARCH(1,1)-N", "Table3_Main_9", "Econometric/GARCH"),
        ("GARCH(1,1)-t", "Step1h_New_Baseline", "Econometric/GARCH"),
        ("EGARCH(1,1)-t", "Step1h_New_Baseline", "Econometric/GARCH"),
        ("HAR", "Table3_Main_9", "Econometric/HAR"),
        ("HAR-X-NOSPILL", "Step1h_New_Baseline", "Econometric/HAR-X"),
        ("HAR-X-SPILL", "Step1h_New_Baseline", "Econometric/HAR-X"),
        ("HAR-X-FULL", "Step1h_New_Baseline_Nested", "Econometric/HAR-X"),
        ("TFT-NOSPILL", "Table3_Main_9", "DL/TFT"),
        ("TFT-SPILL", "Table3_Main_9", "DL/TFT"),
        ("N-BEATSx-NOSPILL_raw", "Table3_Main_9", "DL/N-BEATSx"),
        ("N-BEATSx-SPILL_raw", "Table3_Main_9", "DL/N-BEATSx"),
        ("N-BEATSx-NOSPILL_clipped", "PostHoc_Clipped_Check", "DL/N-BEATSx"),
        ("N-BEATSx-SPILL_clipped", "PostHoc_Clipped_Check", "DL/N-BEATSx"),
    ]

    metrics_rows = []
    for hv in [1, 10]:
        d = all_preds[hv]
        y, y_hexp, rd = d["y"], d["Hist-Exp"], d["real_date"]
        for mname, mset, grp in primary_models:
            res = eval_metrics_row(y, d[mname], y_hexp, rd)
            res.update({"Horizon": f"h={hv}", "Model": mname, "Model_Set": mset, "Family": grp})
            metrics_rows.append(res)

        # Common strictly positive mask across the primary models that have ANY <= 0 predictions
        pos_elig = ["Hist-Exp", "Roll-20", "GARCH(1,1)-N", "EGARCH(1,1)-N", "GARCH(1,1)-t", "EGARCH(1,1)-t",
                    "HAR", "HAR-X-NOSPILL", "HAR-X-SPILL", "HAR-X-FULL", "TFT-NOSPILL", "TFT-SPILL"]
        strictly_pos_models = [m for m in pos_elig if np.all(d[m] > 0)]
        print(f"h={hv} strictly positive across all 335 dates: {strictly_pos_models}")
        print(f"h={hv} any <=0 among HAR-X: " + ", ".join(f"{m}:{(d[m]<=0).sum()}" for m in ["HAR", "HAR-X-NOSPILL", "HAR-X-SPILL", "HAR-X-FULL"]))

    df_metrics = pd.DataFrame(metrics_rows)
    df_metrics.to_csv(OUT_DIR / "step1g_main_metrics.csv", index=False)

    # QLIKE detailed comparison table (Full-sample 335 strictly positive models vs common positive subset for all 14 raw models)
    qlike_rows = []
    q_models = ["Hist-Exp", "Roll-20", "GARCH(1,1)-N", "EGARCH(1,1)-N", "GARCH(1,1)-t", "EGARCH(1,1)-t",
                "HAR", "HAR-X-NOSPILL", "HAR-X-SPILL", "HAR-X-FULL", "TFT-NOSPILL", "TFT-SPILL",
                "N-BEATSx-NOSPILL_raw", "N-BEATSx-SPILL_raw"]
    for hv in [1, 10]:
        d = all_preds[hv]
        y = d["y"]
        common_mask = np.ones(len(y), dtype=bool)
        for m in q_models:
            common_mask &= (d[m] > 0)
        # Also 7-model strict positive common mask (Table 4 canonical QLIKE set)
        t4_models = ["Hist-Exp", "Roll-20", "GARCH(1,1)-N", "EGARCH(1,1)-N", "HAR", "TFT-NOSPILL", "TFT-SPILL"]
        t4_and_t_models = t4_models + ["GARCH(1,1)-t", "EGARCH(1,1)-t"]
        for m in q_models:
            yp = d[m]
            is_full_pos = bool(np.all(yp > 0))
            ql_full = float(np.mean(compute_qlike(y, yp))) if is_full_pos else np.nan
            ql_comm = float(np.mean(compute_qlike(y[common_mask], yp[common_mask])))
            ql_floor1e6 = float(np.mean(compute_qlike(y, np.maximum(yp, 1e-6))))
            ql_floor1e8 = float(np.mean(compute_qlike(y, np.maximum(yp, 1e-8))))
            qlike_rows.append({
                "Horizon": f"h={hv}",
                "Model": m,
                "In_Table4_7Model_Set": m in t4_models,
                "In_9Model_FullPos_Set": m in t4_and_t_models and is_full_pos,
                "N_Positive": int(np.sum(yp > 0)),
                "QLIKE_Full335": ql_full,
                "N_Common_Positive": int(np.sum(common_mask)),
                "QLIKE_Common_Positive": ql_comm,
                "QLIKE_Floor_1e6": ql_floor1e6,
                "QLIKE_Floor_1e8": ql_floor1e8,
            })
    pd.DataFrame(qlike_rows).to_csv(OUT_DIR / "step1g_qlike_table.csv", index=False)

    # Outlier MSE and Ranks Table (Step 1g: "one table of MSE and ranks for all models with and without 5–6 Feb 2026")
    # Evaluate on the canonical 14 raw models (Table 3 9 + GARCH-t, EGARCH-t, HAR-X-NOSPILL, HAR-X-SPILL, HAR-X-FULL)
    rank_models = ["Hist-Exp", "Roll-20", "GARCH(1,1)-N", "EGARCH(1,1)-N", "GARCH(1,1)-t", "EGARCH(1,1)-t",
                   "HAR", "HAR-X-NOSPILL", "HAR-X-SPILL", "HAR-X-FULL", "TFT-NOSPILL", "TFT-SPILL",
                   "N-BEATSx-NOSPILL_raw", "N-BEATSx-SPILL_raw"]
    rank_sub = df_metrics[df_metrics["Model"].isin(rank_models)].copy()
    rank_rows = []
    for hv in ["h=1", "h=10"]:
        s = rank_sub[rank_sub["Horizon"] == hv].copy()
        s["Rank_Full335"] = s["MSE"].rank(method="min").astype(int)
        s["Rank_No_Feb5_6"] = s["MSE_No_Feb5_6"].rank(method="min").astype(int)
        s["Rank_Shift"] = s["Rank_No_Feb5_6"] - s["Rank_Full335"]
        s["Pct_MSE_Drop"] = (s["MSE"] - s["MSE_No_Feb5_6"]) / s["MSE"] * 100.0
        for _, rrow in s.sort_values("Rank_Full335").iterrows():
            rank_rows.append({
                "Horizon": hv,
                "Model": rrow["Model"],
                "MSE_Full335": rrow["MSE"],
                "Rank_Full335": int(rrow["Rank_Full335"]),
                "MSE_No_Feb5_6": rrow["MSE_No_Feb5_6"],
                "Rank_No_Feb5_6": int(rrow["Rank_No_Feb5_6"]),
                "Rank_Shift": int(rrow["Rank_Shift"]),
                "Feb5_6_Error_Share_pct": rrow["Feb5_6_Error_Share_pct"],
                "Pct_MSE_Drop": rrow["Feb5_6_Error_Share_pct"] if False else rrow["Pct_MSE_Drop"],
            })
    pd.DataFrame(rank_rows).to_csv(OUT_DIR / "step1g_outlier_ranks.csv", index=False)

    # -------------------------------------------------------------------------
    # Diebold-Mariano Tests (SE, AE, QLIKE) + Documented Bonferroni Families (1h)
    # -------------------------------------------------------------------------
    dm_rows = []
    # Families:
    # 1) Table3_vs_HAR_Main (8 non-HAR models in Table 3 x 2 horizons -> m = 16; OR + 5 new baselines -> 13 models x 2 horizons = m = 26)
    # Wait: compute BOTH per-horizon AND pooled-2-horizon Bonferroni AND specific coherent test families!
    family_defs = [
        {"Family_ID": "F1_Main13_vs_HAR_MSE", "Loss": "SE", "m": 26, "Description": "13 non-HAR main models vs HAR benchmark across h=1 and h=10 (13 x 2 = 26 tests)"},
        {"Family_ID": "F2_Table3_8_vs_HAR_MSE", "Loss": "SE", "m": 16, "Description": "Original 8 non-HAR models in Table 3 vs HAR benchmark across h=1 and h=10 (8 x 2 = 16 tests)"},
        {"Family_ID": "F3_DL4_vs_Bench3_MSE", "Loss": "SE", "m": 24, "Description": "4 DL configs vs 3 primary Table-3 benchmarks (Hist-Exp, GARCH(1,1)-N, HAR) across h=1,10 (4 x 3 x 2 = 24 tests; legacy v6 used m=12 per horizon/family)"},
        {"Family_ID": "F4_RQ3_SPILL_vs_NOSPILL_MSE", "Loss": "SE", "m": 6, "Description": "Direct SPILL vs NOSPILL ablation for TFT, N-BEATSx, HAR-X across h=1 and h=10 (3 x 2 = 6 tests; DL-only = 4 tests)"},
        {"Family_ID": "F5_RQ3_HARX_FairTest_MSE", "Loss": "SE", "m": 6, "Description": "Parsimonious HAR-X tests: HAR-X-SPILL vs HAR, HAR-X-SPILL vs HAR-X-NOSPILL, HAR-X-FULL vs HAR-X-NOSPILL across h=1,10 (3 x 2 = 6 tests)"},
        {"Family_ID": "F6_FullPos9_vs_HAR_QLIKE", "Loss": "QLIKE", "m": 16, "Description": "Strictly positive econometric/TFT models (8 non-HAR models) vs HAR under QLIKE across h=1,10 (8 x 2 = 16 tests)"},
        {"Family_ID": "F7_RQ3_SPILL_vs_NOSPILL_MAE", "Loss": "AE", "m": 6, "Description": "Direct SPILL vs NOSPILL ablation under AE loss for TFT, N-BEATSx, HAR-X across h=1,10 (3 x 2 = 6 tests)"},
    ]
    pd.DataFrame(family_defs).to_csv(OUT_DIR / "step1h_bonferroni_families.csv", index=False)

    def run_dm_pair(hv, m_cand, m_bench, loss_type, family_id, m_bonf, exclude_outliers=False):
        d = all_preds[hv]
        y, yc, yb, rd = d["y"], d[m_cand], d[m_bench], d["real_date"]
        mask = np.ones(len(y), dtype=bool)
        if exclude_outliers:
            mask &= ~np.isin(rd, list(FEB_OUTLIERS))
        if loss_type == "QLIKE":
            mask &= (yc > 0) & (yb > 0)
        y_m = y[mask]
        yc_m = yc[mask]
        yb_m = yb[mask]
        if loss_type == "SE":
            l1, l2 = (y_m - yc_m) ** 2, (y_m - yb_m) ** 2
        elif loss_type == "AE":
            l1, l2 = np.abs(y_m - yc_m), np.abs(y_m - yb_m)
        elif loss_type == "QLIKE":
            l1, l2 = compute_qlike(y_m, yc_m), compute_qlike(y_m, yb_m)
        else:
            raise ValueError(loss_type)
        r = dm_hln_test(l1, l2, hv)
        p_bonf = min(1.0, r["p_raw_t"] * m_bonf)
        dm_rows.append({
            "Family_ID": family_id,
            "Sample": "No_Feb5_6" if exclude_outliers else "Full_335",
            "Horizon": f"h={hv}",
            "Loss": loss_type,
            "Candidate": m_cand,
            "Benchmark": m_bench,
            "N": r["N"],
            "Mean_Loss_Diff": r["md"],
            "HLN_Stat": r["stat_hln"],
            "p_raw_t": r["p_raw_t"],
            "p_raw_norm": r["p_raw_norm"],
            "m_Bonferroni": m_bonf,
            "p_Bonf_t": p_bonf,
            "Sig_Bonf_05": bool(p_bonf < 0.05),
        })

    non_har_13 = ["Hist-Exp", "Roll-20", "GARCH(1,1)-N", "EGARCH(1,1)-N", "GARCH(1,1)-t", "EGARCH(1,1)-t",
                  "HAR-X-NOSPILL", "HAR-X-SPILL", "HAR-X-FULL", "TFT-NOSPILL", "TFT-SPILL",
                  "N-BEATSx-NOSPILL_raw", "N-BEATSx-SPILL_raw"]
    t3_non_har = ["Hist-Exp", "Roll-20", "GARCH(1,1)-N", "EGARCH(1,1)-N", "TFT-NOSPILL", "TFT-SPILL",
                  "N-BEATSx-NOSPILL_raw", "N-BEATSx-SPILL_raw"]
    qlike_non_har = ["Hist-Exp", "Roll-20", "GARCH(1,1)-N", "EGARCH(1,1)-N", "GARCH(1,1)-t", "EGARCH(1,1)-t",
                     "TFT-NOSPILL", "TFT-SPILL"]

    for excl in [False, True]:
        for hv in [1, 10]:
            for mc in non_har_13:
                m_b = 16 if mc in t3_non_har else 26
                fid = "F1_Main13_vs_HAR_MSE"
                run_dm_pair(hv, mc, "HAR", "SE", fid, 26, excl)
            # Legacy Table3 family m=16
            for mc in t3_non_har:
                run_dm_pair(hv, mc, "HAR", "SE", "F2_Table3_8_vs_HAR_MSE", 16, excl)
            # DL vs primary benchmarks (Hist-Exp, GARCH(1,1)-N, HAR, GARCH(1,1)-t)
            for mc in ["TFT-NOSPILL", "TFT-SPILL", "N-BEATSx-NOSPILL_raw", "N-BEATSx-SPILL_raw"]:
                for mb in ["Hist-Exp", "GARCH(1,1)-N", "GARCH(1,1)-t"]:
                    run_dm_pair(hv, mc, mb, "SE", "F3_DL4_vs_Bench3_MSE", 24, excl)
            # RQ3 SPILL vs NOSPILL (SE and AE)
            for mc, mb in [("TFT-SPILL", "TFT-NOSPILL"), ("N-BEATSx-SPILL_raw", "N-BEATSx-NOSPILL_raw"), ("HAR-X-SPILL", "HAR-X-NOSPILL")]:
                run_dm_pair(hv, mc, mb, "SE", "F4_RQ3_SPILL_vs_NOSPILL_MSE", 6, excl)
                run_dm_pair(hv, mc, mb, "AE", "F7_RQ3_SPILL_vs_NOSPILL_MAE", 6, excl)
            # HAR-X direct RQ3 tests
            for mc, mb in [("HAR-X-SPILL", "HAR"), ("HAR-X-SPILL", "HAR-X-NOSPILL"), ("HAR-X-FULL", "HAR-X-NOSPILL")]:
                run_dm_pair(hv, mc, mb, "SE", "F5_RQ3_HARX_FairTest_MSE", 6, excl)
            # QLIKE family
            for mc in qlike_non_har:
                run_dm_pair(hv, mc, "HAR", "QLIKE", "F6_FullPos9_vs_HAR_QLIKE", 16, excl)
            run_dm_pair(hv, "TFT-SPILL", "TFT-NOSPILL", "QLIKE", "F6_QLIKE_TFT_Ablation", 2, excl)

    df_dm = pd.DataFrame(dm_rows)
    df_dm.to_csv(OUT_DIR / "step1h_dm_tests.csv", index=False)

    # -------------------------------------------------------------------------
    # Model Confidence Set (MCS) under MSE and QLIKE (with arch.bootstrap.MCS)
    # -------------------------------------------------------------------------
    mcs_rows = []
    mcs_mse_models = ["Hist-Exp", "Roll-20", "GARCH(1,1)-N", "EGARCH(1,1)-N", "GARCH(1,1)-t", "EGARCH(1,1)-t",
                      "HAR", "HAR-X-NOSPILL", "HAR-X-SPILL", "TFT-NOSPILL", "TFT-SPILL",
                      "N-BEATSx-NOSPILL_raw", "N-BEATSx-SPILL_raw"]
    mcs_ql_models_base = ["Hist-Exp", "Roll-20", "GARCH(1,1)-N", "EGARCH(1,1)-N", "GARCH(1,1)-t", "EGARCH(1,1)-t",
                          "HAR", "TFT-NOSPILL", "TFT-SPILL"]

    for hv in [1, 10]:
        d = all_preds[hv]
        y = d["y"]
        # 1) All 13 raw models under SE
        loss_mse = pd.DataFrame({m: (y - d[m]) ** 2 for m in mcs_mse_models})
        mcs_mse = MCS(loss_mse, size=0.10, reps=5000, block_size=10, method="R", seed=42)
        mcs_mse.compute()
        pvals_mse = mcs_mse.pvalues
        inc_mse = set(mcs_mse.included)

        # 2) Strictly positive models under QLIKE (include HAR-X if strictly positive, check dynamically!)
        q_set = list(mcs_ql_models_base)
        for hx in ["HAR-X-NOSPILL", "HAR-X-SPILL"]:
            if np.all(d[hx] > 0):
                q_set.append(hx)
        loss_ql = pd.DataFrame({m: compute_qlike(y, d[m]) for m in q_set})
        mcs_ql = MCS(loss_ql, size=0.10, reps=5000, block_size=10, method="R", seed=42)
        mcs_ql.compute()
        pvals_ql = mcs_ql.pvalues
        inc_ql = set(mcs_ql.included)

        for m in mcs_mse_models:
            mcs_rows.append({
                "Horizon": f"h={hv}",
                "Model": m,
                "MCS_pval_MSE": float(pvals_mse.loc[m, "Pvalue"]),
                "In_M90_MSE": bool(m in inc_mse),
                "MCS_pval_QLIKE_pos": float(pvals_ql.loc[m, "Pvalue"]) if m in q_set else np.nan,
                "In_M90_QLIKE_pos": bool(m in inc_ql) if m in q_set else False,
                "QLIKE_Eligible_Full335": m in q_set,
            })

    df_mcs = pd.DataFrame(mcs_rows)
    df_mcs.to_csv(OUT_DIR / "step1h_mcs_results.csv", index=False)

    # -------------------------------------------------------------------------
    # Exploratory Variants Appendix CSV (Step 1g: v6 16-config grid + v7 grid + v7 OOS runs)
    # -------------------------------------------------------------------------
    v6_tune = pd.read_csv(RESULTS_V6 / "v6_tuning_log.csv")
    v6_tune["hyper_dict"] = v6_tune["hyperparameters"].apply(json.loads)
    v6_tune["input_size"] = v6_tune["hyper_dict"].apply(lambda x: x["input_size"])
    v6_tune["lr"] = v6_tune["hyper_dict"].apply(lambda x: x["learning_rate"])
    v6_grid_sum = v6_tune.groupby(["model", "config_id", "input_size", "lr", "loss_train"]).agg(
        val_mse_h1_mean=("val_mse_h1", "mean"),
        val_mse_h10_mean=("val_mse_h10", "mean"),
        val_mse_avg_mean=("val_mse_avg", "mean"),
        fit_seconds_mean=("fit_seconds", "mean"),
    ).reset_index()
    v6_grid_sum.to_csv(OUT_DIR / "step1g_v6_16config_tuning.csv", index=False)

    v7_sum = pd.read_csv(RESULTS_V7 / "v7_evaluation_metrics_summary.csv")
    v7_dm = pd.read_csv(RESULTS_V7 / "v7_diebold_mariano_tests.csv")
    v7_only_models = [c for c in v7_ens["Config"].unique()]
    app_rows = []
    for hv in [1, 10]:
        d = all_preds[hv]
        y, y_hexp, rd = d["y"], d["Hist-Exp"], d["real_date"]
        har_se = (y - d["HAR"]) ** 2
        for m in v7_only_models:
            res = eval_metrics_row(y, d[m], y_hexp, rd)
            mse_own = float(mean_squared_error(v7_own_y[hv][m], d[m]))
            r_dm = dm_hln_test((y - d[m]) ** 2, har_se, hv)
            app_rows.append({
                "Horizon": f"h={hv}",
                "Variant": m,
                "Training_Loss": "MSE" if "MSE" in m else "MAE",
                "MSE": res["MSE"],
                "MSE_OwnStoredY": mse_own,
                "MAE": res["MAE"],
                "R2_OOS_HistExp": res["R2_OOS_HistExp"],
                "Mean_Forecast": res["Mean_Forecast"],
                "Forecast_Target_Ratio": res["Forecast_Target_Ratio"],
                "Pct_Negative": res["Pct_Negative"],
                "MSE_No_Feb5_6": res["MSE_No_Feb5_6"],
                "Feb5_6_Error_Share_pct": res["Feb5_6_Error_Share_pct"],
                "DM_vs_HAR_HLN": r_dm["stat_hln"],
                "DM_vs_HAR_p_raw_t": r_dm["p_raw_t"],
                "DM_vs_HAR_p_bonf16": min(1.0, r_dm["p_raw_t"] * 16),
            })
    pd.DataFrame(app_rows).to_csv(OUT_DIR / "step1g_exploratory_appendix.csv", index=False)

    print("\n=== Step 1g Main Metrics Summary ===")
    print(df_metrics[["Horizon", "Model", "MSE", "MAE", "R2_OOS_HistExp", "Pct_Negative", "MSE_No_Feb5_6"]].to_string(index=False))
    print("\n=== Step 1h RQ3 / HAR-X DM Tests (SE, Full_335) ===")
    print(df_dm[(df_dm["Sample"] == "Full_335") & (df_dm["Family_ID"].isin(["F4_RQ3_SPILL_vs_NOSPILL_MSE", "F5_RQ3_HARX_FairTest_MSE"]))][
        ["Horizon", "Candidate", "Benchmark", "Mean_Loss_Diff", "HLN_Stat", "p_raw_t", "p_Bonf_t"]
    ].to_string(index=False))
    print(f"\nAll Step 1 Stage-2 outputs written to {OUT_DIR} in {time.time() - t0:.1f}s.")


if __name__ == "__main__":
    main()

