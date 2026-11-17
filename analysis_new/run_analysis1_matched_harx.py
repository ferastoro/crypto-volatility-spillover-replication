#!/usr/bin/env python3
"""
Analysis 1: Matched-Sample HAR-X Ablation (RQ3)
Re-estimates HAR, HAR-X-NOSPILL, HAR-X-SPILL, and HAR-X-FULL at all 335 OOS origins
(h=1 and h=10) on one common rolling window of max 1,082 post-warm-up rows (ts >= dl_idx_post250).
"""
from pathlib import Path
import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy import stats
from sklearn.metrics import mean_absolute_error, mean_squared_error

BASE_DIR = Path(__file__).resolve().parent.parent
CLEAN_DIR = BASE_DIR / "data" / "clean"
INPUTS_V5 = BASE_DIR / "inputs" / "v5"
REV_OUT = BASE_DIR / "revision" / "outputs"
KAGGLE_3A3B = BASE_DIR / "revision" / "kaggle_outputs" / "outputs_3a3b_v2"

OUT_DIR = Path(__file__).resolve().parent.parent / "results_new"
OUT_DIR.mkdir(parents=True, exist_ok=True)

ROLL_WINDOW = 1082
FEB_OUTLIERS = {"2026-02-05", "2026-02-06"}


def compute_qlike(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
    yt = np.asarray(y_true, dtype=float)
    yp = np.asarray(y_pred, dtype=float)
    ratio = yt / yp
    return ratio - np.log(ratio) - 1.0


def dm_hln_test(loss_m: np.ndarray, loss_b: np.ndarray, h: int) -> dict:
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
    return {"N": n, "md": md, "stat_hln": stat, "p_raw_t": p_t}


def main():
    dl = pd.read_csv(CLEAN_DIR / "dataset_deep_learning.csv").sort_values("Date").reset_index(drop=True)
    tvp_c = pd.read_csv(REV_OUT / "spill_features_main_us1677_causal.csv").sort_values("Date").reset_index(drop=True)
    warmup_cutoff_date = str(tvp_c.loc[250, "Date"])
    df = dl.merge(tvp_c[["Date", "tvp_tsi", "spillover_ai_to_btcidr"]], on="Date", how="left").sort_values("Date").reset_index(drop=True)
    dl_idx_post250 = int(df[df["Date"] >= warmup_cutoff_date].index[0])
    assert not df.iloc[dl_idx_post250:][["tvp_tsi", "spillover_ai_to_btcidr"]].isna().any().any()

    df["r2_1"] = df["ret_crypto_btc_idr_indodax"] ** 2
    df["r2_5"] = df["r2_1"].rolling(5, min_periods=1).mean()
    df["r2_22"] = df["r2_1"].rolling(22, min_periods=1).mean()
    df["r2_btcusd"] = df["ret_crypto_btc_usd"] ** 2
    ai_ret_cols = ["ret_saham_nvda", "ret_saham_msft", "ret_saham_tsla", "ret_saham_googl", "ret_saham_meta"]
    df["r2_ai_mean"] = (df[ai_ret_cols] ** 2).mean(axis=1)

    ens = pd.read_csv(KAGGLE_3A3B / "predictions_ensemble.csv")
    ref = ens[(ens["model"] == "TFT") & (ens["feature_set"] == "NOSPILL") & (ens["variant"] == "NoWarmup250")].copy()
    date_to_idx = {d: i for i, d in enumerate(df["Date"].values)}
    ref["origin_idx"] = ref["target_date"].map(lambda d: date_to_idx[d] - 1 if ref is not None else -1)
    # Wait: origin_idx is fixed per cutoff_date, NOT target_date - 1 (that's only for h=1)!
    # Use cutoff_date directly:
    ref["origin_idx"] = ref["cutoff_date"].map(date_to_idx)
    cutoffs_df = ref[["cutoff_date", "target_date", "horizon", "origin_idx", "y_true"]].rename(
        columns={"cutoff_date": "cutoff", "target_date": "real_date", "horizon": "h"}
    )
    cutoffs_df = cutoffs_df[cutoffs_df["h"].isin([1, 10])].sort_values(["origin_idx", "h"]).reset_index(drop=True)

    # Hist-Exp from inputs/v5
    har1 = pd.read_csv(INPUTS_V5 / "har_preds_T0_h1_v2.csv")[["Date", "Hist_Exp"]].assign(h=1)
    har10 = pd.read_csv(INPUTS_V5 / "har_preds_T0_h10_v2.csv")[["Date", "Hist_Exp"]].assign(h=10)
    hexp_df = pd.concat([har1, har10], ignore_index=True).rename(columns={"Date": "real_date"})

    specs = {
        "HAR": ["r2_1", "r2_5", "r2_22"],
        "HAR_X_NOSPILL": ["r2_1", "r2_5", "r2_22", "r2_btcusd", "r2_ai_mean"],
        "HAR_X_SPILL": ["r2_1", "r2_5", "r2_22", "tvp_tsi", "spillover_ai_to_btcidr"],
        "HAR_X_FULL": ["r2_1", "r2_5", "r2_22", "r2_btcusd", "r2_ai_mean", "tvp_tsi", "spillover_ai_to_btcidr"],
    }

    rows = []
    n_train_check = {h: {m: [] for m in specs} for h in [1, 10]}
    for _, srow in cutoffs_df.iterrows():
        o_idx = int(srow["origin_idx"])
        hv = int(srow["h"])
        train_s = np.arange(o_idx - hv - ROLL_WINDOW + 1, o_idx - hv + 1)
        ts_use = train_s[train_s >= dl_idx_post250]
        rec = {
            "cutoff": srow["cutoff"],
            "real_date": srow["real_date"],
            "h": hv,
            "y_true": float(srow["y_true"]),
            "n_train": int(len(ts_use)),
        }
        for mname, flist in specs.items():
            X_tr = df.loc[ts_use, flist].values
            Y_tr = df.loc[ts_use + hv, "r2_1"].values
            valid = ~(np.isnan(X_tr).any(axis=1) | np.isnan(Y_tr))
            n_train_check[hv][mname].append(int(valid.sum()))
            X_tr_c = sm.add_constant(X_tr[valid], has_constant="add")
            fit = sm.OLS(Y_tr[valid], X_tr_c).fit()
            x_te = np.insert(df.loc[o_idx, flist].values, 0, 1.0)
            rec[mname] = float(fit.predict(x_te)[0])
        rows.append(rec)

    preds_df = pd.DataFrame(rows).merge(hexp_df, on=["real_date", "h"], how="left")
    assert not preds_df["Hist_Exp"].isna().any()
    preds_df.to_csv(OUT_DIR / "matched_harx_preds.csv", index=False)

    print("=== Sanity Check: Training Row Counts ===")
    for hv in [1, 10]:
        for mname in specs:
            arr = np.array(n_train_check[hv][mname])
            print(f"  h={hv} {mname}: min={arr.min()}, max={arr.max()}")
        assert n_train_check[hv]["HAR_X_NOSPILL"] == n_train_check[hv]["HAR_X_SPILL"]
        assert n_train_check[hv]["HAR"] == n_train_check[hv]["HAR_X_FULL"]

    # Metrics
    metric_rows = []
    for hv in [1, 10]:
        sub = preds_df[preds_df["h"] == hv].sort_values("real_date").reset_index(drop=True)
        y = sub["y_true"].values
        y_hexp = sub["Hist_Exp"].values
        mse_hexp = mean_squared_error(y, y_hexp)
        # Common positive subset across all 4 matched models
        pos_mask = np.ones(len(sub), dtype=bool)
        for mname in specs:
            pos_mask &= (sub[mname].values > 0)
        print(f"h={hv} common-positive subset N = {pos_mask.sum()} / {len(sub)}")

        for mname in specs:
            yhat = sub[mname].values
            mse = mean_squared_error(y, yhat)
            mae = mean_absolute_error(y, yhat)
            r2_oos = 1.0 - mse / mse_hexp
            neg_share = float(np.mean(yhat < 0.0) * 100.0)
            qlike_full = float(np.mean(compute_qlike(y, yhat))) if np.all(yhat > 0) else np.nan
            qlike_pos = float(np.mean(compute_qlike(y[pos_mask], yhat[pos_mask])))
            metric_rows.append({
                "Horizon": f"h={hv}",
                "Model": mname,
                "N": len(sub),
                "MSE": mse,
                "MAE": mae,
                "R2_OOS": r2_oos,
                "Neg_Share_pct": neg_share,
                "QLIKE_Full335": qlike_full,
                "QLIKE_CommonPos": qlike_pos,
                "N_CommonPos": int(pos_mask.sum()),
            })

    # DM tests (m=6 per loss: 3 comparisons x 2 horizons)
    pairs = [
        ("HAR_X_SPILL", "HAR_X_NOSPILL", "SPILL vs NOSPILL"),
        ("HAR_X_FULL", "HAR_X_NOSPILL", "FULL vs NOSPILL"),
        ("HAR_X_SPILL", "HAR", "SPILL vs HAR"),
    ]
    dm_rows = []
    for sample_name, excl_feb in [("Full_335", False), ("Excl_Feb5_6", True)]:
        for loss_name in ["SE", "AE", "QLIKE"]:
            for hv in [1, 10]:
                sub = preds_df[preds_df["h"] == hv].sort_values("real_date").reset_index(drop=True)
                if excl_feb:
                    sub = sub[~sub["real_date"].isin(FEB_OUTLIERS)].reset_index(drop=True)
                y = sub["y_true"].values
                for cand, bench, label in pairs:
                    yc = sub[cand].values
                    yb = sub[bench].values
                    if loss_name == "SE":
                        lc, lb = (y - yc) ** 2, (y - yb) ** 2
                    elif loss_name == "AE":
                        lc, lb = np.abs(y - yc), np.abs(y - yb)
                    else:
                        mask = (yc > 0) & (yb > 0)
                        lc = compute_qlike(y[mask], yc[mask])
                        lb = compute_qlike(y[mask], yb[mask])
                    res = dm_hln_test(lc, lb, hv)
                    dm_rows.append({
                        "Sample": sample_name,
                        "Loss": loss_name,
                        "Horizon": f"h={hv}",
                        "Comparison": label,
                        "Cand": cand,
                        "Bench": bench,
                        "N": res["N"],
                        "md": res["md"],
                        "stat_hln": res["stat_hln"],
                        "p_raw_t": res["p_raw_t"],
                        "p_bonf_m6": min(1.0, 6.0 * res["p_raw_t"]),
                    })

    metrics_df = pd.DataFrame(metric_rows)
    dm_df = pd.DataFrame(dm_rows)
    # Save combined summary to matched_harx.csv (metrics + dm in separate sections or tidy long table; save metrics and dm cleanly)
    metrics_df.to_csv(OUT_DIR / "matched_harx.csv", index=False)
    dm_df.to_csv(OUT_DIR / "matched_harx_dm.csv", index=False)

    print("\n=== Matched-Sample Metrics ===")
    print(metrics_df.to_string(index=False))
    print("\n=== Matched-Sample DM Tests (m=6 per loss) ===")
    print(dm_df.to_string(index=False))


if __name__ == "__main__":
    main()

