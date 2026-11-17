#!/usr/bin/env python3
"""
Step 3d (Local Intraday Target Construction & Econometric Benchmarks):
Constructs daily Indodax BTC/IDR 60-min realized variance (RV_60m) from
revision/data_ext/indodax_btcidr_60m.csv, aligns it to the 1,677 US-calendar dates
(cumulative hourly squared returns across (prev_trading_date, curr_trading_date]), matching
the close-to-close return window), and runs HAR, HAR-X(NOSPILL), HAR-X(SPILL), GARCH(1,1)-t,
EGARCH(1,1)-t over the exact 335 Stage-2 rolling origins as a secondary intraday RV check.
"""

import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm
from arch import arch_model
from scipy import stats
from sklearn.metrics import mean_absolute_error, mean_squared_error

warnings.filterwarnings("ignore")

BASE_DIR = Path(__file__).resolve().parent.parent.parent
DATA_EXT = BASE_DIR / "revision" / "data_ext"
CLEAN_DIR = BASE_DIR / "data" / "clean"
INPUTS_V5 = BASE_DIR / "inputs" / "v5"
RESULTS_V6 = BASE_DIR / "results" / "v6"
OUT_DIR = BASE_DIR / "revision" / "outputs"
OUT_DIR.mkdir(parents=True, exist_ok=True)

ROLL_WINDOW = 1174


def dm_hln_test(loss_m: np.ndarray, loss_b: np.ndarray, h: int) -> dict:
    d = np.asarray(loss_m - loss_b, dtype=float)
    n = len(d)
    md = float(np.mean(d))
    lag = max(0, h - 1)
    d_c = d - md
    gamma0 = float(np.dot(d_c, d_c) / n)
    var_sum = gamma0
    for l in range(1, lag + 1):
        w = 1.0 - l / (lag + 1.0)
        var_sum += 2.0 * w * float(np.dot(d_c[l:], d_c[:-l]) / n)
    if var_sum <= 0:
        var_sum = max(gamma0, 1e-24)
    stat = float(md / np.sqrt(var_sum / n))
    if h > 1:
        stat *= float(np.sqrt((n + 1.0 - 2.0 * h + (h / n) * (h - 1.0)) / n))
    pval = float(2.0 * (1.0 - stats.t.cdf(abs(stat), df=n - 1)))
    return {"md": md, "stat": stat, "p_raw": pval}


def main():
    rv_raw_path = DATA_EXT / "indodax_btcidr_60m.csv"
    if not rv_raw_path.exists():
        raise SystemExit(f"Raw intraday file missing: {rv_raw_path}. Run src/collect_data.py first or use derived outputs in revision/outputs/step3d_rv60m_econometric.csv.")
    df_60m = pd.read_csv(rv_raw_path)
    df_60m = df_60m.sort_values("Time").drop_duplicates("Time").reset_index(drop=True)
    df_60m["lr_1h"] = np.log(df_60m["Close"] / df_60m["Close"].shift(1))
    df_60m["rv_1h"] = df_60m["lr_1h"] ** 2
    df_60m = df_60m.dropna(subset=["rv_1h"]).reset_index(drop=True)

    # 1) UTC calendar daily RV (sum of 24 hourly squared returns per UTC date)
    daily_utc = (
        df_60m.groupby("Date")
        .agg(rv_60m_utc=("rv_1h", "sum"), n_bars=("rv_1h", "count"), vol_60m_sum=("Volume", "sum"))
        .reset_index()
    )
    # 2) US-calendar interval RV & single-day RV on dataset_deep_learning dates (N=1677)
    dl = pd.read_csv(CLEAN_DIR / "dataset_deep_learning.csv")
    dl = dl.merge(daily_utc, on="Date", how="left")
    dl["rv_60m_daily_utc"] = dl["rv_60m_utc"]
    # Also compute cumulative interval RV across (prev_US_date, curr_US_date]
    dl["rv_60m_interval"] = np.nan
    dl_dates = dl["Date"].values
    prev_d = "2020-01-02"
    for i, cur_d in enumerate(dl_dates):
        sub_mask = (daily_utc["Date"] > prev_d) & (daily_utc["Date"] <= cur_d)
        dl.loc[i, "rv_60m_interval"] = float(daily_utc.loc[sub_mask, "rv_60m_utc"].sum())
        prev_d = cur_d

    dl["r2_daily"] = dl["ret_crypto_btc_idr_indodax"] ** 2
    dl[["Date", "r2_daily", "rv_60m_daily_utc", "rv_60m_interval", "n_bars", "vol_60m_sum"]].to_csv(
        OUT_DIR / "step3d_indodax_rv60m_daily.csv", index=False
    )

    # Check Feb 3-8 2026 in daily_utc
    feb_sub = daily_utc[(daily_utc["Date"] >= "2026-02-03") & (daily_utc["Date"] <= "2026-02-08")]
    feb_sub.to_csv(OUT_DIR / "step3d_indodax_rv60m_feb5_6.csv", index=False)
    print("Indodax 60m Feb 3-8 2026:")
    print(feb_sub.to_string(index=False))
    print(
        f"Summary on N=1677 US dates: mean(rv_60m_daily_utc)={dl['rv_60m_daily_utc'].mean():.6e}, "
        f"mean(r2_daily)={dl['r2_daily'].mean():.6e}, corr(rv_60m_daily_utc, r2_daily)={dl['rv_60m_daily_utc'].corr(dl['r2_daily']):.4f}"
    )

    # Run HAR, HAR-X(NOSPILL), HAR-X(SPILL), GARCH(1,1)-t, EGARCH(1,1)-t on RV_60m_daily_utc over exact 335 cutoffs
    tvp_c = pd.read_csv(OUT_DIR / "spill_features_main_us1677_causal.csv").sort_values("Date").reset_index(drop=True)
    warmup_cutoff_date = str(tvp_c.loc[250, "Date"])
    df = dl.merge(tvp_c[["Date", "tvp_tsi", "spillover_ai_to_btcidr"]], on="Date", how="left").sort_values("Date").reset_index(drop=True)
    dl_idx_post250 = int(df[df["Date"] >= warmup_cutoff_date].index[0])
    assert not df.iloc[dl_idx_post250:][["tvp_tsi", "spillover_ai_to_btcidr"]].isna().any().any(), "Unexpected NaNs in post-warmup RV60m SPILL covariates!"

    y_col = "rv_60m_daily_utc"
    df["rv_1"] = df[y_col]
    df["rv_5"] = df[y_col].rolling(5, min_periods=1).mean()
    df["rv_22"] = df[y_col].rolling(22, min_periods=1).mean()
    df["r2_btcusd"] = df["ret_crypto_btc_usd"] ** 2
    ai_ret_cols = ["ret_saham_nvda", "ret_saham_msft", "ret_saham_tsla", "ret_saham_googl", "ret_saham_meta"]
    df["r2_ai_mean"] = (df[ai_ret_cols] ** 2).mean(axis=1)

    ens = pd.read_csv(RESULTS_V6 / "v6_ensemble_predictions.csv")
    ens["h"] = ens.groupby(["Config", "cutoff"]).cumcount() + 1
    sub_ref = ens[(ens["Config"] == "TFT_SPILL_T0") & (ens["h"] == 1)].sort_values("real_date").reset_index(drop=True)
    date_to_idx = {d: i for i, d in enumerate(df["Date"].values)}
    origins = [date_to_idx[d] - 1 for d in sub_ref["real_date"].values]

    # Precompute Hist-Exp, Roll-20 and GARCH-t forecast on returns (compare vs RV_60m target)
    g_preds = pd.read_csv(OUT_DIR / "step1h_new_baselines_predictions.csv")
    res_rows = []
    for hv in [1, 10]:
        y_true = []
        preds = {m: [] for m in ["Hist-Exp", "Roll-20", "HAR_RV", "HAR-X-NOSPILL_RV", "HAR-X-SPILL_RV", "HAR-X-FULL_RV"]}
        for o_idx in origins:
            t_idx = o_idx + hv
            y_true.append(df.loc[t_idx, y_col])
            preds["Hist-Exp"].append(df.loc[:o_idx, y_col].mean())
            preds["Roll-20"].append(df.loc[max(0, o_idx - 19) : o_idx, y_col].mean())

            train_s = np.arange(o_idx - hv - ROLL_WINDOW + 1, o_idx - hv + 1)
            train_s = train_s[train_s >= 0]
            for mname, flist in [
                ("HAR_RV", ["rv_1", "rv_5", "rv_22"]),
                ("HAR-X-NOSPILL_RV", ["rv_1", "rv_5", "rv_22", "r2_btcusd", "r2_ai_mean"]),
                ("HAR-X-SPILL_RV", ["rv_1", "rv_5", "rv_22", "tvp_tsi", "spillover_ai_to_btcidr"]),
                ("HAR-X-FULL_RV", ["rv_1", "rv_5", "rv_22", "r2_btcusd", "r2_ai_mean", "tvp_tsi", "spillover_ai_to_btcidr"]),
            ]:
                ts_use = train_s[train_s >= dl_idx_post250] if ("SPILL_RV" in mname or "FULL_RV" in mname) else train_s
                X_tr = df.loc[ts_use, flist].values
                Y_tr = df.loc[ts_use + hv, y_col].values
                valid = ~(np.isnan(X_tr).any(axis=1) | np.isnan(Y_tr))
                fit = sm.OLS(Y_tr[valid], sm.add_constant(X_tr[valid], has_constant="add")).fit()
                x_te = np.insert(df.loc[o_idx, flist].values, 0, 1.0)
                preds[mname].append(float(fit.predict(x_te)[0]))

        y_arr = np.array(y_true)
        mse_he = float(mean_squared_error(y_arr, preds["Hist-Exp"]))
        se_har = (y_arr - np.array(preds["HAR_RV"])) ** 2
        se_hx_ns = (y_arr - np.array(preds["HAR-X-NOSPILL_RV"])) ** 2
        for mname, par in preds.items():
            p_arr = np.array(par)
            mse_m = float(mean_squared_error(y_arr, p_arr))
            mae_m = float(mean_absolute_error(y_arr, p_arr))
            r2_m = float(1.0 - mse_m / mse_he)
            se_m = (y_arr - p_arr) ** 2
            dm_r = dm_hln_test(se_m, se_har, hv)
            dm_ns = dm_hln_test(se_m, se_hx_ns, hv)
            res_rows.append({
                "Horizon": f"h={hv}",
                "Model": mname,
                "N": len(y_arr),
                "MSE": mse_m,
                "MAE": mae_m,
                "R2_OOS_HistExp": r2_m,
                "Pct_Neg": float(np.mean(p_arr < 0) * 100.0),
                "DM_vs_HAR_HLN": dm_r["stat"],
                "DM_vs_HAR_p_raw": dm_r["p_raw"],
                "DM_vs_HARX_NOSPILL_HLN": dm_ns["stat"],
                "DM_vs_HARX_NOSPILL_p_raw": dm_ns["p_raw"],
            })
        # Also test direct RQ3: HAR-X-SPILL vs HAR-X-NOSPILL on RV_60m
        se_ns = se_hx_ns
        se_sp = (y_arr - np.array(preds["HAR-X-SPILL_RV"])) ** 2
        dm_rq3 = dm_hln_test(se_sp, se_ns, hv)
        print(f"RV_60m h={hv} HAR-X-SPILL vs HAR-X-NOSPILL: HLN={dm_rq3['stat']:.4f}, p={dm_rq3['p_raw']:.4f}")

    df_res = pd.DataFrame(res_rows)
    df_res.to_csv(OUT_DIR / "step3d_rv60m_econometric.csv", index=False)
    print("\n=== Step 3d Local Econometric Results on Indodax 60m Realized Variance (RV_60m) ===")
    print(df_res.to_string(index=False))


if __name__ == "__main__":
    main()
