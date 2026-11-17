#!/usr/bin/env python3
"""
Stationary block-bootstrap confidence intervals for Table 10 loss differentials.
Reuses data loading and loss-series construction from run_analysis2_loss_diff_ci.py.
"""
import time
from pathlib import Path
import numpy as np
import pandas as pd

from run_analysis2_loss_diff_ci import (
    BASE_DIR,
    FEB_OUTLIERS,
    KAGGLE_3A3B,
    OUT_DIR,
    REV_OUT,
    compute_qlike,
    loss_diff_stats,
)

SEED = 20260101
B_RESAMPLES = 10_000
BLOCK_LENGTHS = (10, 5, 20)


def stationary_bootstrap_mean_ci(
    d: np.ndarray,
    bench_mean: float,
    block_length: int,
    n_boot: int = B_RESAMPLES,
    seed: int = SEED,
) -> tuple[float, float]:
    """
    Politis & Romano (1994) stationary bootstrap 95% percentile CI for mean(d),
    expressed as % of bench_mean. Uses a fixed Generator seed per (series, L).
    """
    n = len(d)
    p_new = 1.0 / float(block_length)
    rng = np.random.default_rng(seed)

    # Generate bootstrap index matrix of shape (n_boot, n)
    starts = rng.integers(0, n, size=(n_boot, n), dtype=np.int64)
    new_block = rng.random(size=(n_boot, n)) < p_new

    idx = np.empty((n_boot, n), dtype=np.int64)
    idx[:, 0] = starts[:, 0]
    for t in range(1, n):
        idx[:, t] = np.where(new_block[:, t], starts[:, t], (idx[:, t - 1] + 1) % n)

    boot_means_pct = 100.0 * d[idx].mean(axis=1) / bench_mean
    lo, hi = np.percentile(boot_means_pct, [2.5, 97.5])
    return float(lo), float(hi)


def main():
    t0 = time.perf_counter()

    pred_nn = pd.read_csv(KAGGLE_3A3B / "predictions_ensemble.csv")
    pred_h = pd.read_csv(REV_OUT / "step1h_new_baselines_predictions.csv")
    pred_m = pd.read_csv(OUT_DIR / "matched_harx_preds.csv")

    y_ref = (
        pred_nn[
            (pred_nn["model"] == "TFT")
            & (pred_nn["feature_set"] == "NOSPILL")
            & (pred_nn["variant"] == "NoWarmup250")
        ][["target_date", "horizon", "y_true"]]
        .rename(columns={"target_date": "real_date", "horizon": "h"})
    )

    def get_nn(m, fs):
        sub = pred_nn[
            (pred_nn["model"] == m)
            & (pred_nn["feature_set"] == fs)
            & (pred_nn["variant"] == "NoWarmup250")
        ]
        return sub[["target_date", "horizon", "y_pred"]].rename(
            columns={"target_date": "real_date", "horizon": "h", "y_pred": f"{m}_{fs}"}
        )

    df = y_ref.merge(get_nn("TFT", "NOSPILL"), on=["real_date", "h"])
    df = df.merge(get_nn("TFT", "SPILL"), on=["real_date", "h"])
    df = df.merge(get_nn("NBEATSx", "NOSPILL"), on=["real_date", "h"])
    df = df.merge(get_nn("NBEATSx", "SPILL"), on=["real_date", "h"])
    df["NBEATSx_NOSPILL_clip"] = np.maximum(0.0, df["NBEATSx_NOSPILL"])
    df["NBEATSx_SPILL_clip"] = np.maximum(0.0, df["NBEATSx_SPILL"])

    df = df.merge(
        pred_h[["real_date", "h", "HAR_recheck", "HAR_X_NOSPILL", "HAR_X_SPILL"]].rename(
            columns={
                "HAR_recheck": "HAR_orig",
                "HAR_X_NOSPILL": "HAR_X_NOSPILL_orig",
                "HAR_X_SPILL": "HAR_X_SPILL_orig",
            }
        ),
        on=["real_date", "h"],
    )
    df = df.merge(
        pred_m[["real_date", "h", "HAR_X_NOSPILL", "HAR_X_SPILL"]].rename(
            columns={
                "HAR_X_NOSPILL": "HAR_X_NOSPILL_matched",
                "HAR_X_SPILL": "HAR_X_SPILL_matched",
            }
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

    # Reference existing loss_diff_ci.csv to verify exact HLN reproduction
    ref_df = pd.read_csv(OUT_DIR / "loss_diff_ci.csv")

    rows = []
    max_diff_dbar = 0.0
    max_diff_lo = 0.0
    max_diff_hi = 0.0

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
                    ref_row = ref_df[
                        (ref_df["Sample"] == sample_name)
                        & (ref_df["Comparison"] == label)
                        & (ref_df["Loss"] == loss_name)
                        & (ref_df["Horizon"] == f"h={hv}")
                    ]
                    assert len(ref_row) == 1
                    max_diff_dbar = max(max_diff_dbar, abs(st["d_bar_pct"] - float(ref_row["d_bar_pct"].iloc[0])))
                    max_diff_lo = max(max_diff_lo, abs(st["ci_lo_pct"] - float(ref_row["ci_lo_pct"].iloc[0])))
                    max_diff_hi = max(max_diff_hi, abs(st["ci_hi_pct"] - float(ref_row["ci_hi_pct"].iloc[0])))

                    # Stationary bootstrap on d_t = lc - lb in time order
                    valid = ~(np.isnan(lc) | np.isnan(lb) | np.isinf(lc) | np.isinf(lb))
                    d = np.asarray(lc[valid] - lb[valid], dtype=float)
                    bench_mean = st["bench_mean"]

                    boot_cis = {}
                    for blk in BLOCK_LENGTHS:
                        boot_cis[blk] = stationary_bootstrap_mean_ci(
                            d=d, bench_mean=bench_mean, block_length=blk, n_boot=B_RESAMPLES, seed=SEED
                        )

                    lo_hln, hi_hln = st["ci_lo_pct"], st["ci_hi_pct"]
                    lo_10, hi_10 = boot_cis[10]
                    lo_5, hi_5 = boot_cis[5]
                    lo_20, hi_20 = boot_cis[20]

                    z_excl_hln = bool((lo_hln > 0.0) or (hi_hln < 0.0))
                    z_excl_b10 = bool((lo_10 > 0.0) or (hi_10 < 0.0))
                    disagree = bool(z_excl_hln != z_excl_b10)

                    rows.append({
                        "comparison": label,
                        "loss": loss_name,
                        "h": hv,
                        "sample": sample_name,
                        "d_bar_pct": st["d_bar_pct"],
                        "hln_ci_low": lo_hln,
                        "hln_ci_high": hi_hln,
                        "boot_ci_low_L10": lo_10,
                        "boot_ci_high_L10": hi_10,
                        "boot_ci_low_L5": lo_5,
                        "boot_ci_high_L5": hi_5,
                        "boot_ci_low_L20": lo_20,
                        "boot_ci_high_L20": hi_20,
                        "zero_excluded_hln": z_excl_hln,
                        "zero_excluded_boot_L10": z_excl_b10,
                        "disagreement_flag": disagree,
                    })

    out_df = pd.DataFrame(rows)
    out_path = OUT_DIR / "bootstrap_loss_diff_ci.csv"
    out_df.to_csv(out_path, index=False)

    elapsed = time.perf_counter() - t0
    print(f"Saved {len(out_df)} rows to {out_path} in {elapsed:.2f}s (seed={SEED}, B={B_RESAMPLES})")
    print(
        f"HLN max abs diff vs results_new/loss_diff_ci.csv across all {len(out_df)} rows: "
        f"d_bar_pct={max_diff_dbar:.3e}, ci_lo_pct={max_diff_lo:.3e}, ci_hi_pct={max_diff_hi:.3e}"
    )

    # Check printed precision against Table 10 representative rows
    print("\n--- Printed precision spot-check (>= 5 rows) ---")
    for _, r in out_df.head(6).iterrows():
        print(
            f"  {r['sample']} | {r['comparison']} | {r['loss']} h={r['h']}: "
            f"d_bar={r['d_bar_pct']:+.2f}%, HLN=[{r['hln_ci_low']:+.2f}, {r['hln_ci_high']:+.2f}]"
        )

    print("\n--- Disagreement rows (HLN vs Bootstrap L=10) ---")
    dis = out_df[out_df["disagreement_flag"]]
    if len(dis) == 0:
        print("  NONE")
    else:
        print(
            dis[
                [
                    "sample",
                    "comparison",
                    "loss",
                    "h",
                    "d_bar_pct",
                    "hln_ci_low",
                    "hln_ci_high",
                    "boot_ci_low_L10",
                    "boot_ci_high_L10",
                    "zero_excluded_hln",
                    "zero_excluded_boot_L10",
                ]
            ].to_string(index=False)
        )


if __name__ == "__main__":
    main()

