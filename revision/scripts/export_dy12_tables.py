#!/usr/bin/env python3
"""
Convert system-divided Stage-1 spillover tables into standard Diebold-Yilmaz (2012)
asset-level percentage-point CSVs (TO_i = sum_{j!=i} theta_{ji}*100, FROM_i = sum_{j!=i} theta_{ij}*100,
NET_i = TO_i - FROM_i, summing across assets to 0 and averaging to TSI).
All derived directly from revision/outputs/step1a_*, step1c_*, step1e_* CSVs.
"""

from pathlib import Path
import pandas as pd

OUT_DIR = Path(__file__).resolve().parent.parent / "outputs"


def main():
    # 1. Main & Robustness Net Spillovers (M=8)
    df_net = pd.read_csv(OUT_DIR / "step1a_net_spillovers.csv")
    df_dy12 = df_net.copy()
    for col in ["TO_pct", "FROM_pct", "NET_pct"]:
        df_dy12[col.replace("_pct", "_DY12_pct")] = df_dy12[col] * 8.0
    df_dy12.to_csv(OUT_DIR / "step1a_net_spillovers_dy12.csv", index=False)

    # 2. Bootstrap CIs in standard DY12 units (M=8)
    df_boot = pd.read_csv(OUT_DIR / "step1c_bootstrap_net_ci.csv")
    df_boot_dy12 = df_boot.copy()
    for col in ["Point_NET_pct", "Boot_mean", "Boot_sd", "CI_05", "Median", "CI_95", "CI_025", "CI_975"]:
        df_boot_dy12[f"{col}_DY12"] = df_boot_dy12[col] * 8.0
    df_boot_dy12.to_csv(OUT_DIR / "step1c_bootstrap_net_ci_dy12.csv", index=False)

    # 3. Sensitivity NET spillovers in standard DY12 units (N_assets varies: 7, 10, 8)
    df_sens_sum = pd.read_csv(OUT_DIR / "step1e_sensitivity_summary.csv")
    spec_M = df_sens_sum.set_index("Spec")["N_assets"].to_dict()
    df_sens_net = pd.read_csv(OUT_DIR / "step1e_sensitivity_net.csv")
    df_sens_net_dy12 = df_sens_net.copy()
    M_col = df_sens_net_dy12["Spec"].map(spec_M)
    for col in ["TO_pct", "FROM_pct", "NET_pct"]:
        df_sens_net_dy12[col.replace("_pct", "_DY12_pct")] = df_sens_net_dy12[col] * M_col
    df_sens_net_dy12.to_csv(OUT_DIR / "step1e_sensitivity_net_dy12.csv", index=False)

    df_sens_sum_dy12 = df_sens_sum.copy()
    df_sens_sum_dy12["BTCIDR_NET_DY12_pct"] = df_sens_sum_dy12["BTCIDR_NET_pct"] * df_sens_sum_dy12["N_assets"]
    df_sens_sum_dy12.to_csv(OUT_DIR / "step1e_sensitivity_summary_dy12.csv", index=False)

    # Print formatted Table 2 preview (Main 1677 post-warmup, DY12 units, 4 canonical proxies)
    canon = ["log_r2", "abs_r", "r2", "log_Parkinson_8asset_interval"]
    sub = df_boot_dy12[df_boot_dy12["Proxy"].isin(canon)]
    print("=== Table 2 DY12 NET & [5%, 95%] Bootstrap CIs (Post-Warmup, Main N=1,677) ===")
    for p in canon:
        print(f"\n[{p}]")
        sp = sub[sub["Proxy"] == p]
        for _, r in sp.iterrows():
            print(f"  {r['Asset']:8s}: {r['Point_NET_pct_DY12']:+6.2f}%  [{r['CI_05_DY12']:+6.2f}%, {r['CI_95_DY12']:+6.2f}%]  P(>0)={r['Share_Pos_pct']:5.1f}%")


if __name__ == "__main__":
    main()

