#!/usr/bin/env python3
"""
Analysis 2: Confidence Intervals for Loss Differentials and Minimum Detectable Effect (RQ2/RQ3)
"""
from pathlib import Path
import numpy as np
import pandas as pd
from scipy import stats

BASE_DIR = Path(__file__).resolve().parent.parent
REV_OUT = BASE_DIR / "revision" / "outputs"
KAGGLE_3A3B = BASE_DIR / "revision" / "kaggle_outputs" / "outputs_3a3b_v2"
OUT_DIR = Path(__file__).resolve().parent.parent / "results_new"
OUT_DIR.mkdir(parents=True, exist_ok=True)

FEB_OUTLIERS = {"2026-02-05", "2026-02-06"}


def compute_qlike(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
    yt = np.asarray(y_true, dtype=float)
    yp = np.asarray(y_pred, dtype=float)
    ratio = yt / yp
    return ratio - np.log(ratio) - 1.0


def loss_diff_stats(loss_c: np.ndarray, loss_b: np.ndarray, h: int) -> dict:
    mask = ~(np.isnan(loss_c) | np.isnan(loss_b) | np.isinf(loss_c) | np.isinf(loss_b))
    lc = np.asarray(loss_c[mask], dtype=float)
    lb = np.asarray(loss_b[mask], dtype=float)
    d = lc - lb
    n = len(d)
    d_bar = float(np.mean(d))
    bench_mean = float(np.mean(lb))

    lag = max(0, h - 1)
    d_c = d - d_bar
    gamma0 = float(np.dot(d_c, d_c) / n)
    var_sum = gamma0
    for l in range(1, lag + 1):
        w = 1.0 - l / (lag + 1.0)
        cov_l = float(np.dot(d_c[l:], d_c[:-l]) / n)
        var_sum += 2.0 * w * cov_l
    if var_sum <= 0:
        var_sum = max(gamma0, 1e-24)

    se_hac = float(np.sqrt(var_sum / n))
    hln_factor = float(np.sqrt((n + 1.0 - 2.0 * h + (h / n) * (h - 1.0)) / n)) if h > 1 else 1.0
    se_corr = float(se_hac / hln_factor)
    dm_stat = float(d_bar / se_corr)

    t_crit = float(stats.t.ppf(0.975, df=n - 1))
    ci_lo = d_bar - t_crit * se_corr
    ci_hi = d_bar + t_crit * se_corr
    mde = 2.80 * se_corr  # (1.96 + 0.84) * se_corr

    return {
        "N": n,
        "bench_mean": bench_mean,
        "d_bar": d_bar,
        "se_hac": se_hac,
        "hln_factor": hln_factor,
        "se_corr": se_corr,
        "dm_stat": dm_stat,
        "ci_lo": ci_lo,
        "ci_hi": ci_hi,
        "mde": mde,
        "d_bar_pct": 100.0 * d_bar / bench_mean,
        "ci_lo_pct": 100.0 * ci_lo / bench_mean,
        "ci_hi_pct": 100.0 * ci_hi / bench_mean,
        "mde_pct": 100.0 * mde / bench_mean,
    }


def main():
    pred_nn = pd.read_csv(KAGGLE_3A3B / "predictions_ensemble.csv")
    pred_h = pd.read_csv(REV_OUT / "step1h_new_baselines_predictions.csv")
    pred_m = pd.read_csv(OUT_DIR / "matched_harx_preds.csv")

    # Build unified table indexed by (real_date, h)
    y_ref = (
        pred_nn[(pred_nn["model"] == "TFT") & (pred_nn["feature_set"] == "NOSPILL") & (pred_nn["variant"] == "NoWarmup250")]
        [["target_date", "horizon", "y_true"]]
        .rename(columns={"target_date": "real_date", "horizon": "h"})
    )

    def get_nn(m, fs):
        sub = pred_nn[(pred_nn["model"] == m) & (pred_nn["feature_set"] == fs) & (pred_nn["variant"] == "NoWarmup250")]
        return sub[["target_date", "horizon", "y_pred"]].rename(columns={"target_date": "real_date", "horizon": "h", "y_pred": f"{m}_{fs}"})

    df = y_ref.merge(get_nn("TFT", "NOSPILL"), on=["real_date", "h"])
    df = df.merge(get_nn("TFT", "SPILL"), on=["real_date", "h"])
    df = df.merge(get_nn("NBEATSx", "NOSPILL"), on=["real_date", "h"])
    df = df.merge(get_nn("NBEATSx", "SPILL"), on=["real_date", "h"])
    df["NBEATSx_NOSPILL_clip"] = np.maximum(0.0, df["NBEATSx_NOSPILL"])
    df["NBEATSx_SPILL_clip"] = np.maximum(0.0, df["NBEATSx_SPILL"])

    df = df.merge(
        pred_h[["real_date", "h", "HAR_recheck", "HAR_X_NOSPILL", "HAR_X_SPILL"]].rename(
            columns={"HAR_recheck": "HAR_orig", "HAR_X_NOSPILL": "HAR_X_NOSPILL_orig", "HAR_X_SPILL": "HAR_X_SPILL_orig"}
        ),
        on=["real_date", "h"],
    )
    df = df.merge(
        pred_m[["real_date", "h", "HAR_X_NOSPILL", "HAR_X_SPILL"]].rename(
            columns={"HAR_X_NOSPILL": "HAR_X_NOSPILL_matched", "HAR_X_SPILL": "HAR_X_SPILL_matched"}
        ),
        on=["real_date", "h"],
    )
    assert len(df) == 670

    specs = [
        ("TFT-SPILL vs. TFT-NOSPILL", "TFT_SPILL", "TFT_NOSPILL", ["SE", "QLIKE"]),
        ("N-BEATSx-SPILL vs. N-BEATSx-NOSPILL (raw)", "NBEATSx_SPILL", "NBEATSx_NOSPILL", ["SE", "AE"]),
        ("N-BEATSx-SPILL vs. N-BEATSx-NOSPILL (clipped)", "NBEATSx_SPILL_clip", "NBEATSx_NOSPILL_clip", ["SE"]),
        ("HAR-X-SPILL vs. HAR-X-NOSPILL (orig. 1,174)", "HAR_X_SPILL_orig", "HAR_X_NOSPILL_orig", ["SE", "AE", "QLIKE"]),
        ("HAR-X-SPILL vs. HAR-X-NOSPILL (matched 1,082)", "HAR_X_SPILL_matched", "HAR_X_NOSPILL_matched", ["SE", "AE", "QLIKE"]),
        ("TFT-NOSPILL vs. HAR", "TFT_NOSPILL", "HAR_orig", ["SE", "QLIKE"]),
        ("TFT-SPILL vs. HAR", "TFT_SPILL", "HAR_orig", ["SE", "QLIKE"]),
    ]

    out_rows = []
    for sample_name, excl_feb in [("Full_335", False), ("Excl_Feb5_6", True)]:
        for label, c_col, b_col, losses in specs:
            for loss_name in losses:
                for hv in [1, 10]:
                    sub = df[df["h"] == hv].sort_values("real_date").reset_index(drop=True)
                    if excl_feb:
                        sub = sub[~sub["real_date"].isin(FEB_OUTLIERS)].reset_index(drop=True)
                    y = sub["y_true"].values
                    yc = sub[c_col].values
                    yb = sub[b_col].values
                    if loss_name == "SE":
                        lc, lb = (y - yc) ** 2, (y - yb) ** 2
                    elif loss_name == "AE":
                        lc, lb = np.abs(y - yc), np.abs(y - yb)
                    else:
                        mask = (yc > 0) & (yb > 0)
                        lc = compute_qlike(y[mask], yc[mask])
                        lb = compute_qlike(y[mask], yb[mask])
                    st = loss_diff_stats(lc, lb, hv)
                    out_rows.append({
                        "Sample": sample_name,
                        "Comparison": label,
                        "Loss": loss_name,
                        "Horizon": f"h={hv}",
                        **st,
                    })

    res_df = pd.DataFrame(out_rows)
    res_df.to_csv(OUT_DIR / "loss_diff_ci.csv", index=False)

    # Verify Table 8 DM statistics match
    print("=== DM Stat Check vs Table 8 (Full_335) ===")
    checks = [
        ("TFT-SPILL vs. TFT-NOSPILL", "SE", "h=1", -0.509),
        ("TFT-SPILL vs. TFT-NOSPILL", "SE", "h=10", +0.188),
        ("N-BEATSx-SPILL vs. N-BEATSx-NOSPILL (raw)", "SE", "h=1", -0.476),
        ("N-BEATSx-SPILL vs. N-BEATSx-NOSPILL (raw)", "SE", "h=10", +1.216),
        ("N-BEATSx-SPILL vs. N-BEATSx-NOSPILL (raw)", "AE", "h=1", +0.697),
        ("N-BEATSx-SPILL vs. N-BEATSx-NOSPILL (raw)", "AE", "h=10", +1.867),
        ("HAR-X-SPILL vs. HAR-X-NOSPILL (orig. 1,174)", "SE", "h=1", +0.745),
        ("HAR-X-SPILL vs. HAR-X-NOSPILL (orig. 1,174)", "SE", "h=10", -0.695),
        ("HAR-X-SPILL vs. HAR-X-NOSPILL (orig. 1,174)", "AE", "h=1", -4.060),
        ("HAR-X-SPILL vs. HAR-X-NOSPILL (orig. 1,174)", "AE", "h=10", -2.460),
        ("TFT-SPILL vs. HAR", "SE", "h=1", +1.129),
        ("TFT-SPILL vs. HAR", "SE", "h=10", -0.059),
        ("TFT-SPILL vs. HAR", "QLIKE", "h=1", -0.363),
        ("TFT-SPILL vs. HAR", "QLIKE", "h=10", -0.706),
    ]
    for comp, loss, hor, expected in checks:
        row = res_df[(res_df["Sample"] == "Full_335") & (res_df["Comparison"] == comp) & (res_df["Loss"] == loss) & (res_df["Horizon"] == hor)]
        got = float(row["dm_stat"].iloc[0])
        diff = abs(got - expected)
        print(f"  {comp} | {loss} | {hor}: got={got:+.3f}, Table8={expected:+.3f}, diff={diff:.4f}")

    print("\n=== Full_335 Summary (% of benchmark mean loss) ===")
    f335 = res_df[res_df["Sample"] == "Full_335"][["Comparison", "Loss", "Horizon", "dm_stat", "d_bar_pct", "ci_lo_pct", "ci_hi_pct", "mde_pct"]]
    print(f335.to_string(index=False))


if __name__ == "__main__":
    main()
