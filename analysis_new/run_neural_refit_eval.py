#!/usr/bin/env python3
"""Evaluate 20-origin rolling-refit Stage-2 neural models vs. fixed training.

Steps implemented:
1. Sanity checks on predictions_refit20.csv and refit20_fit_logs.csv.
2. Point metrics (MSE, MAE, RMSE, R2_OOS vs Hist-Exp, Neg %, QLIKE full &
   common-positive) at h = 1 and h = 10 for Refit20 and Fixed (NoWarmup250).
3. HLN-corrected DM tests (Bartlett HAC L = h - 1, Student-t df = n - 1)
   with exploratory Bonferroni family "neural refit robustness" and N = 333
   (excl. 2026-02-05 and 2026-02-06) sensitivity.
4. Contradiction checks against paper conclusions.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from scipy import stats

ROOT = Path(__file__).resolve().parent.parent
REFIT_PREDS_PATH = ROOT / "results" / "refit20_kaggle" / "predictions_refit20.csv"
REFIT_LOGS_PATH = ROOT / "results" / "refit20_kaggle" / "refit20_fit_logs.csv"
FIXED_PREDS_PATH = ROOT / "revision" / "kaggle_outputs" / "outputs_3a3b_v2" / "predictions_ensemble.csv"
HAR_H1_PATH = ROOT / "inputs" / "v5" / "har_preds_T0_h1_v2.csv"
HAR_H10_PATH = ROOT / "inputs" / "v5" / "har_preds_T0_h10_v2.csv"
DL_DATA_PATH = ROOT / "inputs" / "v5" / "dataset_deep_learning.csv"
SPILL_DATA_PATH = ROOT / "revision" / "outputs" / "spill_features_main_us1677_causal.csv"

OUT_METRICS = ROOT / "results_new" / "neural_refit_metrics.csv"
OUT_DM = ROOT / "results_new" / "neural_refit_dm.csv"

FEB_OUTLIERS = {"2026-02-05", "2026-02-06"}
SEEDS_EXPECTED = [7, 13, 21, 42, 99]
MODELS_4 = [
    ("TFT", "NOSPILL"),
    ("TFT", "SPILL"),
    ("NBEATSx", "NOSPILL"),
    ("NBEATSx", "SPILL"),
]
DISPLAY_NAMES = {
    ("TFT", "NOSPILL"): "TFT-NOSPILL",
    ("TFT", "SPILL"): "TFT-SPILL",
    ("NBEATSx", "NOSPILL"): "N-BEATSx-NOSPILL (raw)",
    ("NBEATSx", "SPILL"): "N-BEATSx-SPILL (raw)",
}
SHORT_NAMES = {
    ("TFT", "NOSPILL"): "TFT-NOSPILL",
    ("TFT", "SPILL"): "TFT-SPILL",
    ("NBEATSx", "NOSPILL"): "N-BEATSx-NOSPILL",
    ("NBEATSx", "SPILL"): "N-BEATSx-SPILL",
}


def compute_qlike(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
    """Patton (2011) QLIKE loss element-wise."""
    ratio = y_true / y_pred
    return ratio - np.log(ratio) - 1.0


def dm_hln_test(loss_cand: np.ndarray, loss_bench: np.ndarray, h: int) -> Dict[str, float]:
    """Two-sided Diebold-Mariano test with Bartlett HAC and HLN Student-t correction.

    d_t = L(cand)_t - L(bench)_t; negative statistic favors candidate.
    """
    d = np.asarray(loss_cand, dtype=float) - np.asarray(loss_bench, dtype=float)
    n = len(d)
    md = float(np.mean(d))
    dc = d - md

    lag = max(0, h - 1)
    gamma0 = float(np.dot(dc, dc) / n)
    var_lr = gamma0
    for l in range(1, lag + 1):
        w = 1.0 - l / (lag + 1.0)
        cov = float(np.dot(dc[l:], dc[:-l]) / n)
        var_lr += 2.0 * w * cov

    if var_lr <= 0.0:
        se = float(np.sqrt(max(gamma0, 1e-30) / n))
    else:
        se = float(np.sqrt(var_lr / n))

    stat = md / se if se > 0.0 else 0.0
    if h > 1:
        hln = float(np.sqrt((n + 1.0 - 2.0 * h + (h / n) * (h - 1.0)) / n))
        stat *= hln

    p_t = float(2.0 * (1.0 - stats.t.cdf(abs(stat), df=n - 1)))
    return {
        "n": n,
        "mean_diff": md,
        "stat_hln": float(stat),
        "p_raw": p_t,
    }


def step1_sanity_checks(
    refit: pd.DataFrame, logs: pd.DataFrame, fixed_raw: pd.DataFrame
) -> Tuple[bool, pd.DataFrame]:
    """Run STEP 1 sanity checks; return (passed, check_1b_df)."""
    print("=" * 80)
    print("STEP 1: SANITY CHECKS")
    print("=" * 80)
    print("Input files:")
    print(f"  Refit predictions : {REFIT_PREDS_PATH}")
    print(f"  Refit fit logs    : {REFIT_LOGS_PATH}")
    print(f"  Fixed predictions : {FIXED_PREDS_PATH}")
    print(f"  HAR baselines     : {HAR_H1_PATH}, {HAR_H10_PATH}")

    # 1a: 17 refit_ids, 335 origins x 10 horizons x 4 variants, no NaNs/duplicates
    refit_ids_pred = sorted(refit["refit_id"].unique().tolist())
    refit_ids_log = sorted(logs["refit_id"].unique().tolist())
    assert refit_ids_pred == list(range(17)), f"Expected 0..16 refit_ids, got {refit_ids_pred}"
    assert refit_ids_log == list(range(17)), f"Expected 0..16 log refit_ids, got {refit_ids_log}"
    assert len(refit) == 4 * 335 * 10, f"Expected 13400 rows, got {len(refit)}"
    assert not refit[["y_true", "y_pred"]].isna().any().any(), "NaNs found in predictions_refit20.csv"

    dups = refit.duplicated(subset=["model", "feature_set", "origin_idx", "horizon"]).sum()
    assert dups == 0, f"Duplicate rows found: {dups}"

    for m, fs in MODELS_4:
        sub = refit[(refit["model"] == m) & (refit["feature_set"] == fs)]
        assert len(sub) == 3350, f"{m}-{fs} count {len(sub)} != 3350"
        assert sorted(sub["origin_idx"].unique().tolist()) == list(range(335))
        assert sorted(sub["horizon"].unique().tolist()) == list(range(1, 11))

    # Check origin assignment per refit_id
    for r in range(17):
        expected_origins = list(range(20 * r, min(20 * r + 20, 335)))
        got_origins = sorted(refit[refit["refit_id"] == r]["origin_idx"].unique().tolist())
        assert got_origins == expected_origins, f"Refit {r} origins mismatch"

    print(
        f"[PASS 1a] 17 refit_ids (0..16), 4 variants x 335 origins x 10 horizons = {len(refit)} rows, "
        "0 NaNs, 0 duplicates, exact 20-origin block assignment."
    )

    # 1b: Compare first 20 origins (refit_id == 0) against predictions_ensemble.csv (NoWarmup250)
    fixed = fixed_raw[fixed_raw["variant"] == "NoWarmup250"].copy()
    assert len(fixed) == 13400, f"Expected 13400 fixed rows, got {len(fixed)}"

    merged_all = refit.merge(
        fixed,
        on=["model", "feature_set", "origin_idx", "horizon"],
        suffixes=("_refit", "_fixed"),
    )
    assert len(merged_all) == 13400
    max_ytrue_diff = np.max(np.abs(merged_all["y_true_refit"] - merged_all["y_true_fixed"]))
    assert max_ytrue_diff < 1e-15, f"y_true mismatch between refit and fixed: {max_ytrue_diff}"
    assert (merged_all["cutoff_date_refit"] == merged_all["cutoff_date_fixed"]).all()
    assert (merged_all["target_date_refit"] == merged_all["target_date_fixed"]).all()

    r0 = merged_all[merged_all["refit_id"] == 0].copy()
    assert set(r0["origin_idx"].unique()) == set(range(20))

    print("\n[CHECK 1b] Refit 0 (origins 0..19, horizons 1..10, n=200 per variant) vs Fixed NoWarmup250:")
    chk_rows = []
    max_abs_mse_rel = 0.0
    for m, fs in MODELS_4:
        sub = r0[(r0["model"] == m) & (r0["feature_set"] == fs)]
        y = sub["y_true_refit"].to_numpy()
        pr = sub["y_pred_refit"].to_numpy()
        pf = sub["y_pred_fixed"].to_numpy()
        mse_r = float(np.mean((y - pr) ** 2))
        mse_f = float(np.mean((y - pf) ** 2))
        mae_r = float(np.mean(np.abs(y - pr)))
        mae_f = float(np.mean(np.abs(y - pf)))
        mean_r = float(np.mean(pr))
        mean_f = float(np.mean(pf))
        rel_mse = (mse_r - mse_f) / mse_f
        rel_mae = (mae_r - mae_f) / mae_f
        rel_mean = (mean_r - mean_f) / mean_f
        max_abs_mse_rel = max(max_abs_mse_rel, abs(rel_mse))
        chk_rows.append(
            {
                "model": f"{m}-{fs}",
                "mse_refit_1e6": mse_r * 1e6,
                "mse_fixed_1e6": mse_f * 1e6,
                "rel_diff_mse_pct": rel_mse * 100.0,
                "mae_refit_1e3": mae_r * 1e3,
                "mae_fixed_1e3": mae_f * 1e3,
                "rel_diff_mae_pct": rel_mae * 100.0,
                "mean_pred_refit_1e3": mean_r * 1e3,
                "mean_pred_fixed_1e3": mean_f * 1e3,
                "rel_diff_mean_pct": rel_mean * 100.0,
            }
        )
        print(
            f"  {m}-{fs:7s}: MSE refit={mse_r*1e6:.4f} fixed={mse_f*1e6:.4f} ({rel_mse*100:+.4f}%), "
            f"MAE refit={mae_r*1e3:.4f} fixed={mae_f*1e3:.4f} ({rel_mae*100:+.4f}%), "
            f"Mean refit={mean_r*1e3:.4f} fixed={mean_f*1e3:.4f} ({rel_mean*100:+.4f}%)"
        )

    if max_abs_mse_rel > 0.10:
        print(f"[FAIL 1b] Max |MSE rel diff| = {max_abs_mse_rel*100:.2f}% > 10% threshold. STOPPING.")
        return False, pd.DataFrame(chk_rows)
    print(f"[PASS 1b] Max |MSE rel diff| = {max_abs_mse_rel*100:.4f}% <= 10.00%.")

    # 1c: Verify log completeness and exact 915 train + 167 val = 1,082 rows via post-warmup calendar
    assert len(logs) == 17 * 4 * 5, f"Expected 340 log rows, got {len(logs)}"
    assert not logs["collapsed"].any(), f"Collapsed runs found: {logs[logs['collapsed']]}"
    assert logs["weights_restored"].all(), "Some fits did not restore best weights"
    for (m, fs), grp in logs.groupby(["model", "feature_set"]):
        assert grp["refit_id"].nunique() == 17
        assert sorted(grp["seed"].unique().tolist()) == SEEDS_EXPECTED

    dl = pd.read_csv(DL_DATA_PATH)
    spill = pd.read_csv(SPILL_DATA_PATH)
    cal = pd.merge(dl[["Date"]], spill[["Date"]], on="Date", how="inner")
    cal["Date"] = cal["Date"].astype(str)
    cal = cal.sort_values("Date").reset_index(drop=True)
    assert len(cal) == 1676, f"Merged calendar length {len(cal)} != 1676"
    post_dates = cal.iloc[250:]["Date"].tolist()
    assert len(post_dates) == 1426

    date_to_pos = {d: i for i, d in enumerate(post_dates)}
    unique_windows = logs[["refit_id", "refit_origin_date", "window_start_date", "window_end_date"]].drop_duplicates()
    assert len(unique_windows) == 17
    for _, row in unique_windows.sort_values("refit_id").iterrows():
        ws = str(row["window_start_date"])
        we = str(row["window_end_date"])
        rod = str(row["refit_origin_date"])
        assert we == rod, f"window_end_date {we} != refit_origin_date {rod}"
        n_rows = date_to_pos[we] - date_to_pos[ws] + 1
        assert n_rows == 1082, f"Refit {row['refit_id']} window length {n_rows} != 1082"

    earliest_start = str(unique_windows["window_start_date"].min())
    latest_end = str(unique_windows["window_end_date"].max())
    total_fit_min = float(logs["fit_time_sec"].sum() / 60.0)
    print(
        f"[PASS 1c] 340 fits in log (17 refits x 4 variants x 5 seeds), 0 collapsed, 340 weights_restored=True.\n"
        f"          Each window spans exact 1,082 post-warm-up calendar rows (915 train + 167 validation; "
        f"window_end_date == refit_origin_date).\n"
        f"          Earliest window start: {earliest_start} | Latest window end: {latest_end} | "
        f"Total GPU fit time: {total_fit_min:.2f} min."
    )
    for (m, fs), grp in logs.groupby(["model", "feature_set"]):
        print(
            f"          {m}-{fs:7s}: mean fit {grp['fit_time_sec'].mean():.2f}s, "
            f"best_step min/med/max={grp['best_step'].min()}/{grp['best_step'].median():.0f}/{grp['best_step'].max()}, "
            f"steps_run min/max={grp['steps_run'].min()}/{grp['steps_run'].max()}"
        )

    return True, pd.DataFrame(chk_rows)


def load_horizon_panel(refit: pd.DataFrame, fixed_raw: pd.DataFrame, h: int) -> pd.DataFrame:
    """Align y_true, Hist_Exp, HAR, fixed NoWarmup250, and Refit20 predictions for horizon h."""
    har_path = HAR_H1_PATH if h == 1 else HAR_H10_PATH
    har_df = pd.read_csv(har_path).rename(columns={"Date": "target_date", "True": "y_true_har"})

    panel = har_df[["target_date", "y_true_har", "Hist_Exp", "HAR"]].copy()
    panel["target_date"] = panel["target_date"].astype(str)

    fixed = fixed_raw[(fixed_raw["variant"] == "NoWarmup250") & (fixed_raw["horizon"] == h)].copy()
    refit_h = refit[refit["horizon"] == h].copy()

    for m, fs in MODELS_4:
        col_f = f"{m}_{fs}_fixed"
        col_r = f"{m}_{fs}_refit"
        sf = fixed[(fixed["model"] == m) & (fixed["feature_set"] == fs)][
            ["origin_idx", "cutoff_date", "target_date", "y_true", "y_pred"]
        ].rename(columns={"y_pred": col_f, "y_true": "y_true_dl"})
        sr = refit_h[(refit_h["model"] == m) & (refit_h["feature_set"] == fs)][
            ["origin_idx", "target_date", "y_pred"]
        ].rename(columns={"y_pred": col_r})

        if "origin_idx" not in panel.columns:
            panel = panel.merge(sf, on="target_date", how="inner")
            assert len(panel) == 335, f"h={h} panel merge length {len(panel)} != 335"
            assert np.max(np.abs(panel["y_true_har"] - panel["y_true_dl"])) < 1e-15
        else:
            panel = panel.merge(sf[["target_date", col_f]], on="target_date", how="inner")

        panel = panel.merge(sr[["target_date", col_r]], on="target_date", how="inner")

    assert len(panel) == 335
    panel = panel.sort_values("origin_idx").reset_index(drop=True)
    panel["is_feb_outlier"] = panel["target_date"].isin(FEB_OUTLIERS)
    return panel


def main() -> int:
    refit = pd.read_csv(REFIT_PREDS_PATH)
    logs = pd.read_csv(REFIT_LOGS_PATH)
    fixed_raw = pd.read_csv(FIXED_PREDS_PATH)

    ok, _ = step1_sanity_checks(refit, logs, fixed_raw)
    if not ok:
        return 2

    panels = {h: load_horizon_panel(refit, fixed_raw, h) for h in [1, 10]}

    # STEP 2a: Evaluation metrics
    print("\n" + "=" * 80)
    print("STEP 2a: REFIT vs FIXED METRICS (h = 1 and h = 10, N = 335)")
    print("=" * 80)
    metric_rows: List[Dict[str, object]] = []

    for h in [1, 10]:
        p = panels[h]
        y = p["y_true_har"].to_numpy()
        y_he = p["Hist_Exp"].to_numpy()
        mse_he = float(np.mean((y - y_he) ** 2))

        # Common-positive subset across all 8 neural series (fixed + refit) and y > 0
        neural_cols = []
        for m, fs in MODELS_4:
            neural_cols.extend([f"{m}_{fs}_fixed", f"{m}_{fs}_refit"])
        common_mask_neural = y > 0
        for c in neural_cols:
            common_mask_neural &= p[c].to_numpy() > 0
        n_common_neural = int(np.sum(common_mask_neural))

        # Also check positive count for TFT variants individually
        print(f"\n--- Horizon h = {h} (Hist-Exp MSE = {mse_he*1e6:.3f}e-6; "
              f"8-neural common-positive N = {n_common_neural}/335) ---")

        # Reference rows: HAR and Hist-Exp MSE for convenience / ordering verification
        for ref_name, ref_col in [("Hist-Exp", "Hist_Exp"), ("HAR (OLS)", "HAR")]:
            yp = p[ref_col].to_numpy()
            mse = float(np.mean((y - yp) ** 2))
            mae = float(np.mean(np.abs(y - yp)))
            rmse = float(np.sqrt(mse))
            r2 = float(1.0 - mse / mse_he)
            neg_pct = float(100.0 * np.mean(yp <= 0.0))
            ql_full = float(np.mean(compute_qlike(y, yp))) if np.all(yp > 0) else np.nan
            ql_comm = float(np.mean(compute_qlike(y[common_mask_neural], yp[common_mask_neural]))) if np.all(yp[common_mask_neural] > 0) else np.nan
            metric_rows.append(
                {
                    "horizon": h,
                    "horizon_label": f"h={h}",
                    "model_key": ref_name,
                    "display_name": ref_name,
                    "training_regime": "Benchmark",
                    "n_total": len(y),
                    "mse": mse,
                    "mse_1e6": mse * 1e6,
                    "mae": mae,
                    "mae_1e3": mae * 1e3,
                    "rmse": rmse,
                    "rmse_1e3": rmse * 1e3,
                    "r2_oos": r2,
                    "neg_count": int(np.sum(yp <= 0.0)),
                    "neg_pct": neg_pct,
                    "all_positive": bool(np.all(yp > 0.0)),
                    "qlike_full335": ql_full,
                    "n_common_pos_neural": n_common_neural,
                    "qlike_common_pos": ql_comm,
                }
            )

        for m, fs in MODELS_4:
            disp = DISPLAY_NAMES[(m, fs)]
            for regime, suffix in [("Fixed", "fixed"), ("Refit20", "refit")]:
                col = f"{m}_{fs}_{suffix}"
                yp = p[col].to_numpy()
                mse = float(np.mean((y - yp) ** 2))
                mae = float(np.mean(np.abs(y - yp)))
                rmse = float(np.sqrt(mse))
                r2 = float(1.0 - mse / mse_he)
                neg_cnt = int(np.sum(yp <= 0.0))
                neg_pct = float(100.0 * neg_cnt / len(yp))
                all_pos = bool(np.all(yp > 0.0))
                ql_full = float(np.mean(compute_qlike(y, yp))) if all_pos else np.nan
                ql_comm = float(np.mean(compute_qlike(y[common_mask_neural], yp[common_mask_neural])))
                metric_rows.append(
                    {
                        "horizon": h,
                        "horizon_label": f"h={h}",
                        "model_key": f"{m}_{fs}",
                        "display_name": disp,
                        "training_regime": regime,
                        "n_total": len(y),
                        "mse": mse,
                        "mse_1e6": mse * 1e6,
                        "mae": mae,
                        "mae_1e3": mae * 1e3,
                        "rmse": rmse,
                        "rmse_1e3": rmse * 1e3,
                        "r2_oos": r2,
                        "neg_count": neg_cnt,
                        "neg_pct": neg_pct,
                        "all_positive": all_pos,
                        "qlike_full335": ql_full,
                        "n_common_pos_neural": n_common_neural,
                        "qlike_common_pos": ql_comm,
                    }
                )
                ql_str = f"{ql_full:.3f}" if not np.isnan(ql_full) else "  -- "
                print(
                    f"  {disp:23s} [{regime:7s}]: MSE={mse*1e6:.3f}, MAE={mae*1e3:.3f}, "
                    f"R2={r2:+.3f}, Neg={neg_pct:5.2f}% ({neg_cnt:2d}), "
                    f"QLIKE(335)={ql_str}, QLIKE(Com.{n_common_neural})={ql_comm:.3f}"
                )

    df_metrics = pd.DataFrame(metric_rows)
    OUT_METRICS.parent.mkdir(parents=True, exist_ok=True)
    df_metrics.to_csv(OUT_METRICS, index=False)

    # STEP 2b: HLN-corrected DM tests
    print("\n" + "=" * 80)
    print("STEP 2b: HLN-CORRECTED DM TESTS (exploratory family: neural refit robustness)")
    print("=" * 80)

    # Exact list of contrasts requested in STEP 2b:
    # (comparison_id, label, cand_spec, bench_spec, losses)
    # spec = (source_type, col_name)
    dm_specs = [
        ("TFT_NOSPILL_refit_vs_HAR", "TFT-NOSPILL (Refit) vs. HAR", "TFT_NOSPILL_refit", "HAR", ["SE"]),
        ("TFT_SPILL_refit_vs_HAR", "TFT-SPILL (Refit) vs. HAR", "TFT_SPILL_refit", "HAR", ["SE"]),
        (
            "TFT_SPILL_refit_vs_TFT_NOSPILL_refit",
            "TFT-SPILL (Refit) vs. TFT-NOSPILL (Refit)",
            "TFT_SPILL_refit",
            "TFT_NOSPILL_refit",
            ["SE", "QLIKE"],
        ),
        (
            "NBEATSx_SPILL_refit_vs_NBEATSx_NOSPILL_refit",
            "N-BEATSx-SPILL (Refit) vs. N-BEATSx-NOSPILL (Refit)",
            "NBEATSx_SPILL_refit",
            "NBEATSx_NOSPILL_refit",
            ["SE", "AE"],
        ),
        (
            "TFT_NOSPILL_refit_vs_fixed",
            "TFT-NOSPILL (Refit) vs. TFT-NOSPILL (Fixed)",
            "TFT_NOSPILL_refit",
            "TFT_NOSPILL_fixed",
            ["SE"],
        ),
        (
            "TFT_SPILL_refit_vs_fixed",
            "TFT-SPILL (Refit) vs. TFT-SPILL (Fixed)",
            "TFT_SPILL_refit",
            "TFT_SPILL_fixed",
            ["SE"],
        ),
    ]

    # Count m per loss across both horizons (h in {1, 10})
    m_per_loss: Dict[str, int] = {}
    for _, _, _, _, losses in dm_specs:
        for loss in losses:
            m_per_loss[loss] = m_per_loss.get(loss, 0) + 2  # h=1 and h=10

    print("Exploratory Bonferroni family 'neural refit robustness' test counts per loss:")
    for loss, m_val in m_per_loss.items():
        print(f"  {loss:5s}: m = {m_val}")

    dm_rows: List[Dict[str, object]] = []
    for h in [1, 10]:
        p = panels[h]
        mask_full = np.ones(len(p), dtype=bool)
        mask_no_feb = ~p["is_feb_outlier"].to_numpy()
        assert int(np.sum(mask_no_feb)) == 333

        for comp_id, comp_label, cand_col, bench_col, losses in dm_specs:
            for loss in losses:
                m_bonf = m_per_loss[loss]
                for sample_name, mask in [("Full_335", mask_full), ("Excl_Feb5_6_333", mask_no_feb)]:
                    y = p["y_true_har"].to_numpy()[mask]
                    yc = p[cand_col].to_numpy()[mask]
                    yb = p[bench_col].to_numpy()[mask]
                    if loss == "SE":
                        lc = (y - yc) ** 2
                        lb = (y - yb) ** 2
                    elif loss == "AE":
                        lc = np.abs(y - yc)
                        lb = np.abs(y - yb)
                    elif loss == "QLIKE":
                        assert np.all(yc > 0) and np.all(yb > 0), (
                            f"Non-positive prediction in QLIKE DM test {comp_id} h={h} {sample_name}"
                        )
                        lc = compute_qlike(y, yc)
                        lb = compute_qlike(y, yb)
                    else:
                        raise ValueError(f"Unknown loss: {loss}")

                    res = dm_hln_test(lc, lb, h=h)
                    p_bonf = min(1.0, m_bonf * res["p_raw"]) if sample_name == "Full_335" else np.nan
                    dm_rows.append(
                        {
                            "comparison_id": comp_id,
                            "comparison_label": comp_label,
                            "candidate": cand_col,
                            "benchmark": bench_col,
                            "horizon": h,
                            "horizon_label": f"h={h}",
                            "loss": loss,
                            "sample": sample_name,
                            "n": res["n"],
                            "m_bonf": m_bonf if sample_name == "Full_335" else np.nan,
                            "mean_loss_diff": res["mean_diff"],
                            "dm_stat_hln": res["stat_hln"],
                            "p_raw": res["p_raw"],
                            "p_bonf": p_bonf,
                        }
                    )

    df_dm = pd.DataFrame(dm_rows)
    df_dm.to_csv(OUT_DM, index=False)

    for _, r in df_dm.iterrows():
        pb_str = f"{r['p_bonf']:.4f} (m={int(r['m_bonf'])})" if not np.isnan(r["p_bonf"]) else "unadj."
        print(
            f"  [h={int(r['horizon']):2d} | {r['loss']:5s} | {r['sample']:15s}] "
            f"{r['comparison_label']:52s}: DM={r['dm_stat_hln']:+7.3f}, "
            f"p_raw={r['p_raw']:.4f}, p_bonf={pb_str}"
        )

    # STEP 3: Contradiction checks
    print("\n" + "=" * 80)
    print("STEP 3: CONTRADICTION CHECKS")
    print("=" * 80)
    full_dm = df_dm[df_dm["sample"] == "Full_335"]

    # C1: any refit model significantly beats HAR at p_bonf < 0.05 in SE (stat < 0 and p_bonf < 0.05)
    c1_rows = full_dm[
        (full_dm["benchmark"] == "HAR")
        & (full_dm["loss"] == "SE")
        & (full_dm["dm_stat_hln"] < 0)
        & (full_dm["p_bonf"] < 0.05)
    ]
    c1_triggered = len(c1_rows) > 0

    # C2: any SPILL vs NOSPILL refit contrast significant at p_bonf < 0.05 in SE or QLIKE with SPILL better (stat < 0)
    c2_rows = full_dm[
        (full_dm["comparison_id"].isin(["TFT_SPILL_refit_vs_TFT_NOSPILL_refit", "NBEATSx_SPILL_refit_vs_NBEATSx_NOSPILL_refit"]))
        & (full_dm["loss"].isin(["SE", "QLIKE"]))
        & (full_dm["dm_stat_hln"] < 0)
        & (full_dm["p_bonf"] < 0.05)
    ]
    c2_triggered = len(c2_rows) > 0

    # C3: refit N-BEATSx or TFT changes sign or ordering of full-sample h=1 or h=10 SE ranking among HAR, TFT-NOSPILL, TFT-SPILL
    def _order_3(df_sub: pd.DataFrame) -> List[str]:
        return df_sub.sort_values("mse")["model_key"].tolist()

    c3_details = []
    c3_triggered = False
    for h in [1, 10]:
        har_row = df_metrics[(df_metrics["horizon"] == h) & (df_metrics["model_key"] == "HAR (OLS)")].copy()
        har_row["model_key"] = "HAR"
        fix_3 = pd.concat(
            [
                har_row,
                df_metrics[
                    (df_metrics["horizon"] == h)
                    & (df_metrics["training_regime"] == "Fixed")
                    & (df_metrics["model_key"].isin(["TFT_NOSPILL", "TFT_SPILL"]))
                ],
            ],
            ignore_index=True,
        )
        ref_3 = pd.concat(
            [
                har_row,
                df_metrics[
                    (df_metrics["horizon"] == h)
                    & (df_metrics["training_regime"] == "Refit20")
                    & (df_metrics["model_key"].isin(["TFT_NOSPILL", "TFT_SPILL"]))
                ],
            ],
            ignore_index=True,
        )
        ord_fix = _order_3(fix_3)
        ord_ref = _order_3(ref_3)

        # Also check signs of R2_OOS across all 4 neural models (Refit20 vs Fixed)
        signs_match = True
        for mk in ["TFT_NOSPILL", "TFT_SPILL", "NBEATSx_NOSPILL", "NBEATSx_SPILL"]:
            r2_f = float(
                df_metrics[
                    (df_metrics["horizon"] == h)
                    & (df_metrics["training_regime"] == "Fixed")
                    & (df_metrics["model_key"] == mk)
                ]["r2_oos"].iloc[0]
            )
            r2_r = float(
                df_metrics[
                    (df_metrics["horizon"] == h)
                    & (df_metrics["training_regime"] == "Refit20")
                    & (df_metrics["model_key"] == mk)
                ]["r2_oos"].iloc[0]
            )
            if (r2_f > 0) != (r2_r > 0):
                signs_match = False
                c3_details.append(f"h={h} {mk} R2_OOS sign flipped: Fixed={r2_f:+.3f}, Refit={r2_r:+.3f}")

        if ord_fix != ord_ref:
            c3_triggered = True
            c3_details.append(f"h={h} SE ordering changed: Fixed={ord_fix} vs Refit={ord_ref}")
        if not signs_match:
            c3_triggered = True
        print(f"  [Ordering h={h}] Fixed={ord_fix} | Refit={ord_ref}")

    print(f"  Contradiction 1 (refit beats HAR at p_bonf < 0.05 in SE)       : {c1_triggered}")
    print(f"  Contradiction 2 (refit SPILL beats NOSPILL in SE/QLIKE p<0.05): {c2_triggered}")
    print(f"  Contradiction 3 (sign/ordering change among HAR/TFT-NO/TFT-SP) : {c3_triggered}")
    if c3_details:
        for d in c3_details:
            print(f"    Detail: {d}")

    print(f"\nSaved: {OUT_METRICS}")
    print(f"Saved: {OUT_DM}")

    if c1_triggered or c2_triggered or c3_triggered:
        print("\n>>> CONTRADICTION DETECTED: STOP BEFORE EDITING PAPER <<<")
        return 3

    print("\n>>> ALL STEP 1 AND STEP 3 CHECKS PASSED: SAFE TO PROCEED TO STEP 4 <<<")
    return 0


if __name__ == "__main__":
    sys.exit(main())

