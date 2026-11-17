#!/usr/bin/env python3
"""STEP A: 13-model MSE and MSE ranks (Full N=335 and Excl. Feb 5-6 N=333)
for Fixed (reproducing Table 13 / tab:outlier_ranks) vs. Refit20 designs.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
HAR_H1_PATH = ROOT / "inputs" / "v5" / "har_preds_T0_h1_v2.csv"
HAR_H10_PATH = ROOT / "inputs" / "v5" / "har_preds_T0_h10_v2.csv"
GARCH_V4_PATH = ROOT / "inputs" / "v5" / "GARCH_Predictions_V4.csv"
BASELINES_NEW_PATH = ROOT / "revision" / "outputs" / "step1h_new_baselines_predictions.csv"
FIXED_NN_PATH = ROOT / "revision" / "kaggle_outputs" / "outputs_3a3b_v2" / "predictions_ensemble.csv"
REFIT_NN_PATH = ROOT / "results" / "refit20_kaggle" / "predictions_refit20.csv"
OUT_CSV = ROOT / "results_new" / "neural_refit_ranks.csv"

FEB_OUTLIERS = {"2026-02-05", "2026-02-06"}

MODELS_13 = [
    "Hist-Exp",
    "Roll-20",
    "GARCH(1,1)-N",
    "GARCH(1,1)-t",
    "EGARCH(1,1)-N",
    "EGARCH(1,1)-t",
    "HAR (OLS)",
    "HAR-X-NOSPILL",
    "HAR-X-SPILL",
    "TFT-NOSPILL",
    "TFT-SPILL",
    "N-BEATSx-NOSPILL (raw)",
    "N-BEATSx-SPILL (raw)",
]

# Expected Table 13 (tab:outlier_ranks) fixed-design ranks from main.tex
EXPECTED_T13_RANKS = {
    (1, "Full"): {
        "HAR (OLS)": 1, "HAR-X-NOSPILL": 2, "EGARCH(1,1)-N": 3, "HAR-X-SPILL": 4,
        "TFT-SPILL": 5, "GARCH(1,1)-N": 6, "N-BEATSx-SPILL (raw)": 7,
        "TFT-NOSPILL": 8, "EGARCH(1,1)-t": 9, "GARCH(1,1)-t": 10,
        "Roll-20": 11, "N-BEATSx-NOSPILL (raw)": 12, "Hist-Exp": 13,
    },
    (1, "Excl"): {
        "TFT-NOSPILL": 1, "HAR-X-SPILL": 2, "HAR-X-NOSPILL": 3, "HAR (OLS)": 4,
        "GARCH(1,1)-t": 5, "TFT-SPILL": 6, "EGARCH(1,1)-t": 7, "Roll-20": 8,
        "EGARCH(1,1)-N": 9, "N-BEATSx-NOSPILL (raw)": 10, "GARCH(1,1)-N": 11,
        "N-BEATSx-SPILL (raw)": 12, "Hist-Exp": 13,
    },
    (10, "Full"): {
        "TFT-NOSPILL": 1, "HAR-X-SPILL": 2, "TFT-SPILL": 3, "HAR (OLS)": 4,
        "HAR-X-NOSPILL": 5, "GARCH(1,1)-t": 6, "GARCH(1,1)-N": 7, "EGARCH(1,1)-t": 8,
        "EGARCH(1,1)-N": 9, "Roll-20": 10, "N-BEATSx-NOSPILL (raw)": 11,
        "Hist-Exp": 12, "N-BEATSx-SPILL (raw)": 13,
    },
    (10, "Excl"): {
        "TFT-NOSPILL": 1, "HAR-X-SPILL": 2, "TFT-SPILL": 3, "HAR (OLS)": 4,
        "HAR-X-NOSPILL": 5, "GARCH(1,1)-t": 6, "N-BEATSx-NOSPILL (raw)": 7,
        "EGARCH(1,1)-t": 8, "GARCH(1,1)-N": 9, "Roll-20": 10, "EGARCH(1,1)-N": 11,
        "Hist-Exp": 12, "N-BEATSx-SPILL (raw)": 13,
    },
}


def build_panel(h: int, design: str, har_df: pd.DataFrame, garch_df: pd.DataFrame, base_new: pd.DataFrame, nn_df: pd.DataFrame) -> pd.DataFrame:
    """Build 335-row prediction matrix for the 13 Table-13 models."""
    p = har_df[["Date", "True", "Hist_Exp", "Roll_20", "HAR"]].copy()
    p = p.rename(columns={
        "Date": "real_date",
        "True": "y_true",
        "Hist_Exp": "Hist-Exp",
        "Roll_20": "Roll-20",
        "HAR": "HAR (OLS)",
    })
    p["real_date"] = p["real_date"].astype(str)

    bn_h = base_new[base_new["h"] == h][
        ["real_date", "GARCH11_t", "EGARCH_t", "HAR_X_NOSPILL", "HAR_X_SPILL"]
    ].rename(columns={
        "GARCH11_t": "GARCH(1,1)-t",
        "EGARCH_t": "EGARCH(1,1)-t",
        "HAR_X_NOSPILL": "HAR-X-NOSPILL",
        "HAR_X_SPILL": "HAR-X-SPILL",
    })
    p = p.merge(bn_h, on="real_date", how="inner")

    g_h = garch_df[garch_df["h"] == h][["real_date", "GARCH11", "EGARCH"]].rename(columns={
        "GARCH11": "GARCH(1,1)-N",
        "EGARCH": "EGARCH(1,1)-N",
    })
    p = p.merge(g_h, on="real_date", how="inner")

    nn_map = {
        ("TFT", "NOSPILL"): "TFT-NOSPILL",
        ("TFT", "SPILL"): "TFT-SPILL",
        ("NBEATSx", "NOSPILL"): "N-BEATSx-NOSPILL (raw)",
        ("NBEATSx", "SPILL"): "N-BEATSx-SPILL (raw)",
    }
    nn_h = nn_df[nn_df["horizon"] == h].copy()
    for (m, fs), label in nn_map.items():
        sub = nn_h[(nn_h["model"] == m) & (nn_h["feature_set"] == fs)][
            ["target_date", "y_true", "y_pred"]
        ].rename(columns={"target_date": "real_date", "y_pred": label, "y_true": "y_nn"})
        p = p.merge(sub, on="real_date", how="inner")
        assert np.max(np.abs(p["y_true"] - p["y_nn"])) < 1e-15, f"{design} h={h} {label} y_true mismatch"
        p = p.drop(columns=["y_nn"])

    assert len(p) == 335, f"h={h} {design} merged length {len(p)} != 335"
    return p


def score_13(panel: pd.DataFrame, h: int, design: str) -> pd.DataFrame:
    """Compute MSE (Full N=335 & Excl. Feb 5-6 N=333) and exact ranks 1..13."""
    y = panel["y_true"].to_numpy()
    no_feb = ~panel["real_date"].isin(FEB_OUTLIERS).to_numpy()
    assert int(np.sum(no_feb)) == 333

    rows: List[Dict[str, object]] = []
    for model in MODELS_13:
        yp = panel[model].to_numpy()
        se = (y - yp) ** 2
        mse_full = float(np.mean(se))
        mse_excl = float(np.mean(se[no_feb]))
        rows.append({
            "design": design,
            "horizon": h,
            "horizon_label": f"h={h}",
            "model": model,
            "mse_full_1e6": mse_full * 1e6,
            "mse_excl_1e6": mse_excl * 1e6,
            "feb5_6_err_share_pct": 100.0 * float(np.sum(se[~no_feb]) / np.sum(se)),
        })

    df = pd.DataFrame(rows)
    df["rank_full"] = df["mse_full_1e6"].rank(method="min", ascending=True).astype(int)
    df["rank_excl"] = df["mse_excl_1e6"].rank(method="min", ascending=True).astype(int)
    df["rank_shift"] = df["rank_full"] - df["rank_excl"]
    return df


def main() -> int:
    har_h1 = pd.read_csv(HAR_H1_PATH)
    har_h10 = pd.read_csv(HAR_H10_PATH)

    garch_raw = pd.read_csv(GARCH_V4_PATH)
    garch_raw = garch_raw.sort_values(["cutoff", "ds"]).reset_index(drop=True)
    garch_raw["h"] = garch_raw.groupby("cutoff").cumcount() + 1
    garch_raw["real_date"] = garch_raw["real_date"].astype(str)

    base_new = pd.read_csv(BASELINES_NEW_PATH)
    base_new["real_date"] = base_new["real_date"].astype(str)

    fixed_nn = pd.read_csv(FIXED_NN_PATH)
    fixed_nn = fixed_nn[fixed_nn["variant"] == "NoWarmup250"].copy()
    refit_nn = pd.read_csv(REFIT_NN_PATH)

    all_dfs: List[pd.DataFrame] = []
    for design, nn_df in [("Fixed", fixed_nn), ("Refit20", refit_nn)]:
        for h, har_df in [(1, har_h1), (10, har_h10)]:
            panel = build_panel(h, design, har_df, garch_raw, base_new, nn_df)
            scored = score_13(panel, h, design)
            all_dfs.append(scored)

    res = pd.concat(all_dfs, ignore_index=True)

    # Verify Fixed reproduces Table 13 exactly
    print("=== Verifying Fixed design reproduces Table 13 (tab:outlier_ranks) ===")
    for h in [1, 10]:
        sub = res[(res["design"] == "Fixed") & (res["horizon"] == h)]
        for _, r in sub.iterrows():
            m = r["model"]
            if r["rank_full"] != EXPECTED_T13_RANKS[(h, "Full")][m]:
                print(f"MISMATCH Fixed h={h} Full rank for {m}: got {r['rank_full']}, expected {EXPECTED_T13_RANKS[(h, 'Full')][m]}")
                return 2
            if r["rank_excl"] != EXPECTED_T13_RANKS[(h, "Excl")][m]:
                print(f"MISMATCH Fixed h={h} Excl rank for {m}: got {r['rank_excl']}, expected {EXPECTED_T13_RANKS[(h, 'Excl')][m]}")
                return 2
    print("[PASS] Fixed design reproduces all Table 13 ranks identically across h=1 and h=10 (Full & Excl).")

    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    res.to_csv(OUT_CSV, index=False)
    print(f"Saved: {OUT_CSV}\n")

    print("=== First-ranked (Rank 1) models across the 4 cases ===")
    for design in ["Fixed", "Refit20"]:
        for h in [1, 10]:
            sub = res[(res["design"] == design) & (res["horizon"] == h)]
            top_full = sub.sort_values("mse_full_1e6").iloc[0]
            top_excl = sub.sort_values("mse_excl_1e6").iloc[0]
            print(
                f"  {design:7s} | h={h:2d} | Full (N=335): {top_full['model']:22s} ({top_full['mse_full_1e6']:.3f}e-6) | "
                f"Excl Feb 5-6 (N=333): {top_excl['model']:22s} ({top_excl['mse_excl_1e6']:.3f}e-6)"
            )

    print("\n=== Full 13-model Refit20 vs Fixed Table ===")
    for h in [1, 10]:
        print(f"\n--- h = {h} (Refit20) ---")
        sub = res[(res["design"] == "Refit20") & (res["horizon"] == h)].sort_values("rank_excl")
        for _, r in sub.iterrows():
            print(
                f"  {r['model']:24s}: Full={r['mse_full_1e6']:.3f} (rank {r['rank_full']:2d}) | "
                f"Excl={r['mse_excl_1e6']:.3f} (rank {r['rank_excl']:2d}) [{r['rank_shift']:+d}] | "
                f"Share={r['feb5_6_err_share_pct']:.1f}%"
            )

    return 0


if __name__ == "__main__":
    sys.exit(main())
