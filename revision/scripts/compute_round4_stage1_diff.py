#!/usr/bin/env python3
"""
1) Recomputes revision/outputs/step1a_tsi_events.csv from the updated
   causal_sigma0=True Stage-1 Main_US_1677 log_r2 TVP-VAR outputs.
2) Compares pre-Round-4 snapshot (revision/audit/pre_round4_sigma0_snapshot/)
   vs new Stage-1 CSVs (revision/outputs/) across every Stage-1 quantity
   referenced in main.tex, writing:
     - revision/outputs/round4_sigma0_old_vs_new.csv
     - revision/outputs/round4_manuscript_impact.csv
"""

from pathlib import Path
import sys
import numpy as np
import pandas as pd

SCRIPTS_DIR = Path(__file__).resolve().parent
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from run_step1_stage1 import (  # noqa: E402
    ASSETS_8,
    OUT_DIR,
    WARMUP_OBS,
    build_aligned_panel,
    estimate_tvp_var_gfevd,
    make_proxy_matrices,
)

BASE_DIR = SCRIPTS_DIR.parent.parent
SNAP_DIR = BASE_DIR / "revision" / "audit" / "pre_round4_sigma0_snapshot"


def update_tsi_events():
    panel = build_aligned_panel(ASSETS_8, join_ihsg=False)
    labels_8 = [a[1] for a in ASSETS_8]
    proxies = make_proxy_matrices(panel, labels_8)
    y_lr2 = proxies["log_r2"]
    dates_ret = pd.to_datetime(panel["df_ret"]["Date"].values).strftime("%Y-%m-%d").to_numpy()
    res1 = estimate_tvp_var_gfevd(y_lr2, p=1, warmup=WARMUP_OBS, causal_sigma0=True)

    dates_eff = dates_ret[1:]
    tsi_all = res1["tsi_all"]
    ts_df = pd.read_csv(OUT_DIR / "step1a_tsi_timeseries_postwarmup.csv")
    ts_df["Date"] = ts_df["Date"].astype(str)

    warm_tsi = tsi_all[:WARMUP_OBS]
    warm_dates = dates_eff[:WARMUP_OBS]
    w_max_i = int(np.argmax(warm_tsi))

    def win_max(d0, d1):
        sub = ts_df[(ts_df["Date"] >= d0) & (ts_df["Date"] <= d1)]
        idx = int(sub["TSI_log_r2"].idxmax())
        return str(sub.loc[idx, "Date"]), float(sub.loc[idx, "TSI_log_r2"])

    post_min_i = int(ts_df["TSI_log_r2"].idxmin())
    post_max_i = int(ts_df["TSI_log_r2"].idxmax())

    tl_d, tl_v = win_max("2022-05-01", "2022-05-20")
    ftx_d, ftx_v = str(ts_df.loc[post_max_i, "Date"]), float(ts_df.loc[post_max_i, "TSI_log_r2"])
    cg_d, cg_v = win_max("2022-11-21", "2022-12-10")
    sp_d, sp_v = str(ts_df.loc[post_min_i, "Date"]), float(ts_df.loc[post_min_i, "TSI_log_r2"])
    ap_d, ap_v = win_max("2025-04-01", "2025-04-20")
    fb_d, fb_v = win_max("2026-02-01", "2026-02-15")

    rows = [
        {"Event": "Full_series_max_in_warmup", "Date": str(warm_dates[w_max_i]), "TSI_log_r2_pct": float(warm_tsi[w_max_i]), "Note": "Global max inside first-250 warmup window"},
        {"Event": "Warmup_first250_mean", "Date": f"{warm_dates[0]}..{warm_dates[-1]}", "TSI_log_r2_pct": float(np.mean(warm_tsi)), "Note": "Mean TSI over first 250 VAR steps"},
        {"Event": "Post_warmup_mean", "Date": f"{ts_df['Date'].iloc[0]}..{ts_df['Date'].iloc[-1]}", "TSI_log_r2_pct": float(ts_df["TSI_log_r2"].mean()), "Note": f"Main post-warmup mean (N={len(ts_df)})"},
        {"Event": "Terra_Luna_window_max", "Date": tl_d, "TSI_log_r2_pct": tl_v, "Note": "Terra-Luna collapse window peak"},
        {"Event": "FTX_window_max", "Date": ftx_d, "TSI_log_r2_pct": ftx_v, "Note": "FTX collapse post-warmup global peak"},
        {"Event": "ChatGPT_window_max", "Date": cg_d, "TSI_log_r2_pct": cg_v, "Note": "ChatGPT release window peak"},
        {"Event": "Spring2025_min", "Date": sp_d, "TSI_log_r2_pct": sp_v, "Note": "Post-warmup minimum"},
        {"Event": "April2025_tariff_peak", "Date": ap_d, "TSI_log_r2_pct": ap_v, "Note": "April 2025 tariff turmoil peak"},
        {"Event": "Feb2026_dislocation_max", "Date": fb_d, "TSI_log_r2_pct": fb_v, "Note": "Feb 5-6 2026 dislocation window peak"},
    ]
    for dt in ["2022-05-11", "2022-11-10", "2022-12-02", "2025-03-07", "2025-04-10", "2026-02-05", "2026-02-06"]:
        r = ts_df[ts_df["Date"] == dt].iloc[0]
        rows.append({
            "Event": f"Exact_{dt}",
            "Date": dt,
            "TSI_log_r2_pct": float(r["TSI_log_r2"]),
            "Note": f"abs_r={r['TSI_abs_r']:.2f}%, r2={r['TSI_r2']:.2f}%, park={r['TSI_log_Parkinson_8asset_interval']:.2f}%",
        })
    ev_path = OUT_DIR / "step1a_tsi_events.csv"
    pd.DataFrame(rows).to_csv(ev_path, index=False)
    print(f"Updated {ev_path}")


def main():
    update_tsi_events()

    old_sum = pd.read_csv(SNAP_DIR / "step1a_stage1_summary.csv")
    new_sum = pd.read_csv(OUT_DIR / "step1a_stage1_summary.csv")
    old_perm = pd.read_csv(SNAP_DIR / "step1a_permutation_test.csv").set_index("Proxy")
    new_perm = pd.read_csv(OUT_DIR / "step1a_permutation_test.csv").set_index("Proxy")
    old_ev = pd.read_csv(SNAP_DIR / "step1a_tsi_events.csv").set_index("Event")
    new_ev = pd.read_csv(OUT_DIR / "step1a_tsi_events.csv").set_index("Event")
    old_net = pd.read_csv(SNAP_DIR / "step1a_net_spillovers_dy12.csv")
    new_net = pd.read_csv(OUT_DIR / "step1a_net_spillovers_dy12.csv")
    old_bas = pd.read_csv(SNAP_DIR / "step1c_bootstrap_basic_ci.csv")
    new_bas = pd.read_csv(OUT_DIR / "step1c_bootstrap_basic_ci.csv")
    old_bias = pd.read_csv(SNAP_DIR / "step1c_bootstrap_bias.csv")
    new_bias = pd.read_csv(OUT_DIR / "step1c_bootstrap_bias.csv")
    old_btsi = pd.read_csv(SNAP_DIR / "step1c_bootstrap_tsi_ci.csv").set_index("Proxy")
    new_btsi = pd.read_csv(OUT_DIR / "step1c_bootstrap_tsi_ci.csv").set_index("Proxy")
    old_blk = pd.read_csv(SNAP_DIR / "step1d_block_decomposition.csv")
    new_blk = pd.read_csv(OUT_DIR / "step1d_block_decomposition.csv")
    old_pair = pd.read_csv(SNAP_DIR / "step1a_pairwise_fev.csv")
    new_pair = pd.read_csv(OUT_DIR / "step1a_pairwise_fev.csv")
    old_sens = pd.read_csv(SNAP_DIR / "step1e_sensitivity_summary_dy12.csv")
    new_sens = pd.read_csv(OUT_DIR / "step1e_sensitivity_summary_dy12.csv")

    diffs = []
    impacts = []

    def add_num(item, old_v, new_v, src_csv, fmt="{:.2f}", loc=""):
        ov = float(old_v)
        nv = float(new_v)
        diffs.append({
            "item": item,
            "old": ov,
            "new": nv,
            "difference": nv - ov,
            "source_csv": src_csv,
        })
        ofmt = fmt.format(ov)
        nfmt = fmt.format(nv)
        sign_flip = (ov > 0 and nv < 0) or (ov < 0 and nv > 0)
        if ofmt != nfmt or sign_flip:
            impacts.append({
                "item": item,
                "old_raw": ov,
                "new_raw": nv,
                "old_formatted": ofmt,
                "new_formatted": nfmt,
                "sign_or_flag_changed": bool(sign_flip),
                "location_in_main_tex": loc,
                "source_csv": src_csv,
            })

    def add_bool(item, old_b, new_b, src_csv, loc=""):
        ob = bool(old_b)
        nb = bool(new_b)
        diffs.append({
            "item": item,
            "old": int(ob),
            "new": int(nb),
            "difference": int(nb) - int(ob),
            "source_csv": src_csv,
        })
        if ob != nb:
            impacts.append({
                "item": item,
                "old_raw": str(ob),
                "new_raw": str(nb),
                "old_formatted": str(ob),
                "new_formatted": str(nb),
                "sign_or_flag_changed": True,
                "location_in_main_tex": loc,
                "source_csv": src_csv,
            })

    # 1. Stage-1 summary (Main_US_1677 x 4 canon proxies + Robust_JKSE_1556 log_r2)
    canon = ["log_r2", "abs_r", "r2", "log_Parkinson_8asset_interval"]
    sum_cols = [
        ("TSI_full_mean", "{:.2f}", "Results RQ1 (Sec 4.1)"),
        ("TSI_warmup_first250_mean", "{:.2f}", "Results RQ1 / Fig 2 legend"),
        ("TSI_post_mean", "{:.2f}", "Abstract, Table 4, Table 5, Results RQ1, Conclusion"),
        ("TSI_post_sd", "{:.2f}", "Results RQ1"),
        ("TSI_post_min", "{:.2f}", "Results RQ1"),
        ("TSI_post_max", "{:.2f}", "Results RQ1"),
        ("TSI_p0_post_mean", "{:.2f}", "Results RQ1"),
        ("TSI_p0_ratio_pct", "{:.2f}", "Abstract, Results RQ1, Lit Comparison, Conclusion"),
    ]
    for p in canon:
        ro = old_sum[(old_sum["Sample"] == "Main_US_1677") & (old_sum["Proxy"] == p)].iloc[0]
        rn = new_sum[(new_sum["Sample"] == "Main_US_1677") & (new_sum["Proxy"] == p)].iloc[0]
        for col, fmt, loc in sum_cols:
            add_num(f"Main_US_1677.{p}.{col}", ro[col], rn[col], "step1a_stage1_summary.csv", fmt, loc)

    ro_r = old_sum[(old_sum["Sample"] == "Robust_JKSE_1556") & (old_sum["Proxy"] == "log_r2")].iloc[0]
    rn_r = new_sum[(new_sum["Sample"] == "Robust_JKSE_1556") & (new_sum["Proxy"] == "log_r2")].iloc[0]
    add_num("Robust_JKSE_1556.log_r2.TSI_post_mean", ro_r["TSI_post_mean"], rn_r["TSI_post_mean"], "step1a_stage1_summary.csv", "{:.2f}", "Robustness Check (Sec 4.4)")

    # 2. TSI events
    for ev_k in ["Full_series_max_in_warmup", "Terra_Luna_window_max", "FTX_window_max", "ChatGPT_window_max", "April2025_tariff_peak"]:
        add_num(f"TSI_event.{ev_k}", old_ev.loc[ev_k, "TSI_log_r2_pct"], new_ev.loc[ev_k, "TSI_log_r2_pct"], "step1a_tsi_events.csv", "{:.2f}", "Results RQ1 (Sec 4.1)")

    # 3. Permutation test
    for p in canon:
        for col in ["Null_mean", "Null_p05", "Null_p95", "Excess_TSI", "Empirical_p"]:
            fmt = "{:.4f}" if col == "Empirical_p" else "{:.2f}"
            add_num(f"Perm.{p}.{col}", old_perm.loc[p, col], new_perm.loc[p, col], "step1a_permutation_test.csv", fmt, "Abstract, Table 4, Results RQ1, Conclusion")

    # 4. Robust_JKSE_1556 log_r2 NET DY12 (BTC-USD & BTC/IDR)
    for asset in ["BTC-USD", "BTC/IDR"]:
        vo = old_net[(old_net["Sample"] == "Robust_JKSE_1556") & (old_net["Proxy"] == "log_r2") & (old_net["Asset"] == asset)]["NET_DY12_pct"].iloc[0]
        vn = new_net[(new_net["Sample"] == "Robust_JKSE_1556") & (new_net["Proxy"] == "log_r2") & (new_net["Asset"] == asset)]["NET_DY12_pct"].iloc[0]
        add_num(f"Robust_JKSE_1556.log_r2.{asset}.NET_DY12_pct", vo, vn, "step1a_net_spillovers_dy12.csv", "{:+.2f}", "Robustness Check (Sec 4.4)")

    # 5. All 32 asset x proxy pairs: Point NET, Percentile 90%, Basic 90%, Flags, Bias, Boot_mean
    for idx in range(len(new_bas)):
        rn = new_bas.iloc[idx]
        a, p = rn["Asset"], rn["Proxy"]
        ro = old_bas[(old_bas["Asset"] == a) & (old_bas["Proxy"] == p)].iloc[0]
        bo = old_bias[(old_bias["Asset"] == a) & (old_bias["Proxy"] == p)].iloc[0]
        bn = new_bias[(new_bias["Asset"] == a) & (new_bias["Proxy"] == p)].iloc[0]
        for col in ["Point", "Pct_lo", "Pct_hi", "Basic_lo", "Basic_hi"]:
            add_num(f"Table4.{p}.{a}.{col}", ro[col], rn[col], "step1c_bootstrap_basic_ci.csv", "{:+.2f}", "Table 4, Results RQ1, Abstract")
        for bcol in ["Boot_mean_DY12", "Bias_DY12_BootMinusPoint", "Share_Pos_pct"]:
            fmt = "{:.1f}" if bcol == "Share_Pos_pct" else "{:+.2f}"
            add_num(f"BootBias.{p}.{a}.{bcol}", bo[bcol], bn[bcol], "step1c_bootstrap_bias.csv", fmt, "Methodology, Results RQ1")
        for fl in ["Excl0_pct", "Excl0_basic", "Agree"]:
            add_bool(f"Table4.{p}.{a}.{fl}", ro[fl], rn[fl], "step1c_bootstrap_basic_ci.csv", "Table 4 bolding, Results RQ1, Conclusion")

    add_num(
        "Max_Abs_BootBias_DY12",
        old_bias["Bias_DY12_BootMinusPoint"].abs().max(),
        new_bias["Bias_DY12_BootMinusPoint"].abs().max(),
        "step1c_bootstrap_bias.csv",
        "{:.2f}",
        "Methodology (Sec 3.1)",
    )

    # 6. Bootstrap TSI CIs
    for p in canon:
        ro_t = old_btsi.loc[p]
        rn_t = new_btsi.loc[p]
        for col in ["Point", "P05", "P95"]:
            add_num(f"Table4_TSI.{p}.{col}", ro_t[col], rn_t[col], "step1c_bootstrap_tsi_ci.csv", "{:.2f}", "Table 4")
        blo_o = 2.0 * ro_t["Point"] - ro_t["P95"]
        bhi_o = 2.0 * ro_t["Point"] - ro_t["P05"]
        blo_n = 2.0 * rn_t["Point"] - rn_t["P95"]
        bhi_n = 2.0 * rn_t["Point"] - rn_t["P05"]
        add_num(f"Table4_TSI.{p}.Basic_lo", blo_o, blo_n, "step1c_bootstrap_tsi_ci.csv", "{:.2f}", "Table 4")
        add_num(f"Table4_TSI.{p}.Basic_hi", bhi_o, bhi_n, "step1c_bootstrap_tsi_ci.csv", "{:.2f}", "Table 4")

    # 7. Block decomposition (Table 5)
    for p in canon:
        ro = old_blk[(old_blk["Sample"] == "Main_US_1677") & (old_blk["Proxy"] == p)].iloc[0]
        rn = new_blk[(new_blk["Sample"] == "Main_US_1677") & (new_blk["Proxy"] == p)].iloc[0]
        for col in ["TSI_post_mean", "Within_AI_pts", "Within_AI_share_of_TSI_pct",
                    "Within_Crypto_pts", "Within_Crypto_share_of_TSI_pct",
                    "AI_to_Crypto_pts", "AI_to_Crypto_share_of_TSI_pct",
                    "Crypto_to_AI_pts", "Crypto_to_AI_share_of_TSI_pct"]:
            fmt = "{:.1f}" if "share" in col else "{:.2f}"
            add_num(f"Table5.{p}.{col}", ro[col], rn[col], "step1d_block_decomposition.csv", fmt, "Table 5, Results RQ1")

    # 8. Pairwise FEV (Main_US_1677 log_r2)
    ro_p = old_pair[(old_pair["Sample"] == "Main_US_1677") & (old_pair["Proxy"] == "log_r2")].iloc[0]
    rn_p = new_pair[(new_pair["Sample"] == "Main_US_1677") & (new_pair["Proxy"] == "log_r2")].iloc[0]
    for col in ["BTCUSD_to_BTCIDR_pct", "BTCIDR_to_BTCUSD_pct", "AI_sum_to_BTCUSD_pct",
                "AI_sum_to_BTCIDR_pct", "BTCIDR_to_AI_mean_row_pct"]:
        add_num(f"Pairwise.log_r2.{col}", ro_p[col], rn_p[col], "step1a_pairwise_fev.csv", "{:.2f}", "Results RQ1 (Sec 4.1)")

    # 9. Sensitivity summary (DY12)
    for spec, cols in [
        ("No_BTCUSD_7asset", ["TSI_post_mean", "AI_sum_to_BTCIDR_pct", "Crypto_other_to_BTCIDR_pct"]),
        ("Plus_VIX_IXIC_10asset", ["TSI_post_mean", "AI_sum_to_BTCIDR_pct", "VIX_to_BTCIDR_pct", "BroadIdx_to_BTCIDR_pct", "BTCIDR_NET_DY12_pct"]),
        ("Plus_VIX_GSPC_10asset", ["TSI_post_mean", "AI_sum_to_BTCIDR_pct", "VIX_to_BTCIDR_pct", "BroadIdx_to_BTCIDR_pct", "BTCIDR_NET_DY12_pct"]),
        ("FX_Adjusted_BTCIDR_8asset", ["TSI_post_mean", "BTCIDR_NET_DY12_pct"]),
    ]:
        ro_s = old_sens[(old_sens["Spec"] == spec) & (old_sens["Proxy"] == "log_r2")].iloc[0]
        rn_s = new_sens[(new_sens["Spec"] == spec) & (new_sens["Proxy"] == "log_r2")].iloc[0]
        for col in cols:
            add_num(f"Sens.{spec}.log_r2.{col}", ro_s[col], rn_s[col], "step1e_sensitivity_summary_dy12.csv", "{:.2f}", "Robustness Check (Sec 4.4)")

    df_diffs = pd.DataFrame(diffs)
    df_impacts = pd.DataFrame(impacts)

    out_diffs = OUT_DIR / "round4_sigma0_old_vs_new.csv"
    out_imp = OUT_DIR / "round4_manuscript_impact.csv"
    df_diffs.to_csv(out_diffs, index=False)
    df_impacts.to_csv(out_imp, index=False)

    print(f"Wrote {out_diffs} ({len(df_diffs)} items checked)")
    print(f"Wrote {out_imp} ({len(df_impacts)} items changed at paper display precision or flag/sign)")
    if len(df_impacts) > 0:
        print("\n=== Changed Manuscript Items Preview ===")
        print(df_impacts[["item", "old_formatted", "new_formatted", "sign_or_flag_changed", "location_in_main_tex"]].to_string(index=False))


if __name__ == "__main__":
    main()

