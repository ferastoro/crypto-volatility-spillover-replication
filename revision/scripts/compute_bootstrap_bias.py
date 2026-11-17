#!/usr/bin/env python3
"""
Computes Bootstrap Bias (Boot_mean - Point_NET) across every asset and proxy
(both raw NET and standard DY12 NET) from revision/outputs/step1c_bootstrap_net_ci_dy12.csv
and saves to revision/outputs/step1c_bootstrap_bias.csv.
"""
from pathlib import Path
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent.parent.parent
OUT_DIR = BASE_DIR / "revision" / "outputs"


def main():
    df = pd.read_csv(OUT_DIR / "step1c_bootstrap_net_ci_dy12.csv")
    out = df[
        [
            "Proxy",
            "Asset",
            "N_boot",
            "Block_len",
            "Point_NET_pct",
            "Boot_mean",
            "Point_NET_pct_DY12",
            "Boot_mean_DY12",
            "CI_05_DY12",
            "Median_DY12",
            "CI_95_DY12",
            "Share_Pos_pct",
            "Excludes_Zero_90",
        ]
    ].copy()
    out["Bias_Raw_BootMinusPoint"] = out["Boot_mean"] - out["Point_NET_pct"]
    out["Bias_DY12_BootMinusPoint"] = out["Boot_mean_DY12"] - out["Point_NET_pct_DY12"]
    ci_width = out["CI_95_DY12"] - out["CI_05_DY12"]
    dist_lo = out["Point_NET_pct_DY12"] - out["CI_05_DY12"]
    dist_hi = out["CI_95_DY12"] - out["Point_NET_pct_DY12"]
    rel_pos = (dist_lo / ci_width).clip(-1.0, 2.0)
    out["Point_Frac_In_CI90"] = rel_pos
    out["Near_Or_Outside_Edge_15pct"] = (rel_pos <= 0.15) | (rel_pos >= 0.85)
    out = out[
        [
            "Proxy",
            "Asset",
            "N_boot",
            "Block_len",
            "Point_NET_pct",
            "Boot_mean",
            "Bias_Raw_BootMinusPoint",
            "Point_NET_pct_DY12",
            "Boot_mean_DY12",
            "Bias_DY12_BootMinusPoint",
            "CI_05_DY12",
            "Median_DY12",
            "CI_95_DY12",
            "Point_Frac_In_CI90",
            "Near_Or_Outside_Edge_15pct",
            "Share_Pos_pct",
            "Excludes_Zero_90",
        ]
    ]
    out_path = OUT_DIR / "step1c_bootstrap_bias.csv"
    out.to_csv(out_path, index=False)
    print(f"Wrote {out_path} ({len(out)} rows)")
    print(
        out[
            [
                "Proxy",
                "Asset",
                "Point_NET_pct_DY12",
                "Boot_mean_DY12",
                "Bias_DY12_BootMinusPoint",
                "CI_05_DY12",
                "CI_95_DY12",
                "Near_Or_Outside_Edge_15pct",
            ]
        ]
        .round(3)
        .to_string(index=False)
    )


if __name__ == "__main__":
    main()

