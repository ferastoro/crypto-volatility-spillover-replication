#!/usr/bin/env python3
"""
Step 2b of Round 4:
Compare pre-Round-4 Stage-2 HAR-X, VIF, DM, MCS, and RV_60m econometric outputs
(in revision/audit/pre_round4_sigma0_snapshot/) against the updated outputs
computed from revision/outputs/spill_features_main_us1677_causal.csv.
Saves revision/outputs/round4_old_vs_new_harx.csv.
"""

from pathlib import Path
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent.parent.parent
SNAP_DIR = BASE_DIR / "revision" / "audit" / "pre_round4_sigma0_snapshot"
OUT_DIR = BASE_DIR / "revision" / "outputs"


def main():
    old_met = pd.read_csv(SNAP_DIR / "step1g_main_metrics.csv")
    new_met = pd.read_csv(OUT_DIR / "step1g_main_metrics.csv")
    old_dm = pd.read_csv(SNAP_DIR / "step1h_dm_tests.csv")
    new_dm = pd.read_csv(OUT_DIR / "step1h_dm_tests.csv")
    old_vif = pd.read_csv(SNAP_DIR / "step1i_spill_vif.csv")
    new_vif = pd.read_csv(OUT_DIR / "step1i_spill_vif.csv")
    old_rv = pd.read_csv(SNAP_DIR / "step3d_rv60m_econometric.csv")
    new_rv = pd.read_csv(OUT_DIR / "step3d_rv60m_econometric.csv")
    old_mcs = pd.read_csv(SNAP_DIR / "step1h_mcs_results.csv")
    new_mcs = pd.read_csv(OUT_DIR / "step1h_mcs_results.csv")
    old_ql = pd.read_csv(SNAP_DIR / "step1g_qlike_table.csv")
    new_ql = pd.read_csv(OUT_DIR / "step1g_qlike_table.csv")

    rows = []

    def add(cat, item, ov, nv, src, fmt="{:.6e}", in_paper=False, paper_loc=""):
        ov_f = float(ov)
        nv_f = float(nv)
        rows.append({
            "category": cat,
            "item": item,
            "old": ov_f,
            "new": nv_f,
            "difference": nv_f - ov_f,
            "old_formatted": fmt.format(ov_f),
            "new_formatted": fmt.format(nv_f),
            "changed_at_fmt": fmt.format(ov_f) != fmt.format(nv_f),
            "in_main_tex": bool(in_paper),
            "location_in_main_tex": paper_loc,
            "source_csv": src,
        })

    # 1. HAR-X MSE / MAE / R2 / QLIKE / No_Feb5_6
    for hv in ["h=1", "h=10"]:
        for m in ["HAR-X-NOSPILL", "HAR-X-SPILL", "HAR-X-FULL"]:
            ro = old_met[(old_met["Horizon"] == hv) & (old_met["Model"] == m)].iloc[0]
            rn = new_met[(new_met["Horizon"] == hv) & (new_met["Model"] == m)].iloc[0]
            in_main = m in ("HAR-X-NOSPILL", "HAR-X-SPILL") or (m == "HAR-X-FULL")
            add("HARX_Metrics", f"{hv}.{m}.MSE", ro["MSE"], rn["MSE"], "step1g_main_metrics.csv", "{:.3e}", in_main, "Table 6, Results 4.2, Abstract")
            add("HARX_Metrics", f"{hv}.{m}.MAE", ro["MAE"], rn["MAE"], "step1g_main_metrics.csv", "{:.3e}", m != "HAR-X-FULL", "Table 6")
            add("HARX_Metrics", f"{hv}.{m}.RMSE", ro["RMSE"], rn["RMSE"], "step1g_main_metrics.csv", "{:.3e}", m != "HAR-X-FULL", "Table 6")
            add("HARX_Metrics", f"{hv}.{m}.R2_OOS_HistExp", ro["R2_OOS_HistExp"], rn["R2_OOS_HistExp"], "step1g_main_metrics.csv", "{:+.3f}", m != "HAR-X-FULL", "Table 6")
            add("HARX_Metrics", f"{hv}.{m}.MSE_No_Feb5_6", ro["MSE_No_Feb5_6"], rn["MSE_No_Feb5_6"], "step1g_main_metrics.csv", "{:.3e}", m != "HAR-X-FULL", "Table 9")
            add("HARX_Metrics", f"{hv}.{m}.Feb5_6_Error_Share_pct", ro["Feb5_6_Error_Share_pct"], rn["Feb5_6_Error_Share_pct"], "step1g_main_metrics.csv", "{:.1f}", hv == "h=1" and m != "HAR-X-FULL", "Table 9")

            qo = old_ql[(old_ql["Horizon"] == hv) & (old_ql["Model"] == m)].iloc[0]
            qn = new_ql[(new_ql["Horizon"] == hv) & (new_ql["Model"] == m)].iloc[0]
            add("HARX_QLIKE", f"{hv}.{m}.QLIKE_Full335", qo["QLIKE_Full335"], qn["QLIKE_Full335"], "step1g_qlike_table.csv", "{:.3f}", m != "HAR-X-FULL", "Table 7")
            add("HARX_QLIKE", f"{hv}.{m}.QLIKE_Common_Positive", qo["QLIKE_Common_Positive"], qn["QLIKE_Common_Positive"], "step1g_qlike_table.csv", "{:.3f}", m != "HAR-X-FULL", "Table 7")

            if m in ("HAR-X-NOSPILL", "HAR-X-SPILL"):
                mo = old_mcs[(old_mcs["Horizon"] == hv) & (old_mcs["Model"] == m)].iloc[0]
                mn = new_mcs[(new_mcs["Horizon"] == hv) & (new_mcs["Model"] == m)].iloc[0]
                add("HARX_MCS", f"{hv}.{m}.MCS_pval_MSE", mo["MCS_pval_MSE"], mn["MCS_pval_MSE"], "step1h_mcs_results.csv", "{:.3f}", True, "Table 7")

    # Also check global min MCS p-val in main.tex L349-350
    mcs_h1_min_o = old_mcs[(old_mcs["Horizon"] == "h=1") & (old_mcs["Model"] != "Hist-Exp")]["MCS_pval_MSE"].min()
    mcs_h1_min_n = new_mcs[(new_mcs["Horizon"] == "h=1") & (new_mcs["Model"] != "Hist-Exp")]["MCS_pval_MSE"].min()
    add("HARX_MCS", "h=1.Min_MCS_MSE_excl_HistExp", mcs_h1_min_o, mcs_h1_min_n, "step1h_mcs_results.csv", "{:.3f}", True, "Results 4.2")

    # 2. DM tests involving HAR-X
    dm_sub_new = new_dm[new_dm["Candidate"].str.contains("HAR-X") | new_dm["Benchmark"].str.contains("HAR-X")]
    for _, rn in dm_sub_new.iterrows():
        ro = old_dm[
            (old_dm["Family_ID"] == rn["Family_ID"])
            & (old_dm["Sample"] == rn["Sample"])
            & (old_dm["Horizon"] == rn["Horizon"])
            & (old_dm["Candidate"] == rn["Candidate"])
            & (old_dm["Benchmark"] == rn["Benchmark"])
        ].iloc[0]
        tag = f"{rn['Family_ID']}.{rn['Sample']}.{rn['Horizon']}.{rn['Candidate']}_vs_{rn['Benchmark']}"
        in_p = rn["Sample"] == "Full_335" and rn["Family_ID"] in ("F5_RQ3_HARX_FairTest_MSE", "F7_RQ3_SPILL_vs_NOSPILL_MAE")
        add("HARX_DM", f"{tag}.HLN_Stat", ro["HLN_Stat"], rn["HLN_Stat"], "step1h_dm_tests.csv", "{:+.3f}", in_p, "Table 8, Results 4.2")
        add("HARX_DM", f"{tag}.p_raw_t", ro["p_raw_t"], rn["p_raw_t"], "step1h_dm_tests.csv", "{:.4f}", in_p, "Table 8, Results 4.2")
        add("HARX_DM", f"{tag}.p_Bonf_t", ro["p_Bonf_t"], rn["p_Bonf_t"], "step1h_dm_tests.csv", "{:.3f}", in_p, "Table 8, Results 4.2, Abstract")

    # 3. VIF per feature (Full & PostWarmup)
    for _, rn in new_vif.iterrows():
        feat = rn["Feature"]
        ro = old_vif[old_vif["Feature"] == feat].iloc[0]
        in_p_full = feat in ("net_spill_crypto_btc_usd", "net_spill_crypto_eth_usd", "net_spill_saham_googl")
        add("SPILL_VIF", f"{feat}.VIF_Full", ro["VIF_Full_N1676"], rn["VIF_Full_N1676"], "step1i_spill_vif.csv", "{:.2f}", in_p_full, "Methodology Sec 3.1")
        add("SPILL_VIF", f"{feat}.VIF_PostWarmup", ro["VIF_PostWarmup_N1426"], rn["VIF_PostWarmup_N1426"], "step1i_spill_vif.csv", "{:.2f}", True, "Methodology Sec 3.1 (range)")

    add("SPILL_VIF", "VIF_PostWarmup.min", old_vif["VIF_PostWarmup_N1426"].min(), new_vif["VIF_PostWarmup_N1426"].min(), "step1i_spill_vif.csv", "{:.2f}", True, "Methodology Sec 3.1")
    add("SPILL_VIF", "VIF_PostWarmup.max", old_vif["VIF_PostWarmup_N1426"].max(), new_vif["VIF_PostWarmup_N1426"].max(), "step1i_spill_vif.csv", "{:.2f}", True, "Methodology Sec 3.1")

    # 4. RV_60m econometric results
    for _, rn in new_rv.iterrows():
        hz, md = rn["Horizon"], rn["Model"]
        ro = old_rv[(old_rv["Horizon"] == hz) & (old_rv["Model"] == md)].iloc[0]
        in_p_r2 = hz == "h=1" and md in ("HAR_RV", "HAR-X-SPILL_RV")
        in_p_dm = hz == "h=1" and md == "HAR-X-SPILL_RV"
        add("RV60m", f"{hz}.{md}.MSE", ro["MSE"], rn["MSE"], "step3d_rv60m_econometric.csv", "{:.4e}", False, "")
        add("RV60m", f"{hz}.{md}.R2_OOS_HistExp", ro["R2_OOS_HistExp"], rn["R2_OOS_HistExp"], "step3d_rv60m_econometric.csv", "{:.4f}", in_p_r2, "Results 4.2")
        add("RV60m", f"{hz}.{md}.DM_vs_HAR_p_raw", ro["DM_vs_HAR_p_raw"], rn["DM_vs_HAR_p_raw"], "step3d_rv60m_econometric.csv", "{:.4f}", False, "")
        add("RV60m", f"{hz}.{md}.DM_vs_HARX_NOSPILL_p_raw", ro["DM_vs_HARX_NOSPILL_p_raw"], rn["DM_vs_HARX_NOSPILL_p_raw"], "step3d_rv60m_econometric.csv", "{:.4f}", in_p_dm, "Results 4.2")

    df_out = pd.DataFrame(rows)
    out_path = OUT_DIR / "round4_old_vs_new_harx.csv"
    df_out.to_csv(out_path, index=False)
    print(f"Wrote {out_path} ({len(df_out)} comparisons)")

    chg = df_out[df_out["changed_at_fmt"] & df_out["in_main_tex"]]
    print(f"\n=== Paper-Referenced HAR-X / VIF / RV60m Items Changed ({len(chg)} rows) ===")
    if len(chg) > 0:
        print(chg[["item", "old_formatted", "new_formatted", "location_in_main_tex"]].to_string(index=False))

    print("\n=== Key RQ3 DM Summary (Old vs New) ===")
    dm_key = df_out[df_out["category"].isin(["HARX_DM", "RV60m"]) & df_out["item"].str.contains("p_raw|p_Bonf|R2_OOS") & df_out["item"].str.contains("SPILL|FULL")]
    print(dm_key[["item", "old_formatted", "new_formatted", "changed_at_fmt"]].to_string(index=False))


if __name__ == "__main__":
    main()

