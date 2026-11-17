#!/usr/bin/env python3
"""
STEP 1 audit computations:
a) Excl. Feb 5-6 (N=333) SPILL vs NOSPILL HLN p-values and Bonferroni (m=6 per loss)
b) RV60m m=28 (SE/AE) and m=14 (QLIKE) families from v2_3c3d_dm_tests.csv
c) Source of p_bonf >= 0.957 in C2_all_existing_dm_tests_with_holm.csv and v2_dm_tests.csv
d) Seed counts for RV60m neural runs (TFT and N-BEATSx)
"""
from pathlib import Path
import pandas as pd
import numpy as np
from scipy import stats

BASE = Path(__file__).resolve().parent.parent
RES_NEW = BASE / "results_new"
REV = BASE / "revision"
AUDIT = REV / "audit"

# --- STEP 1a ---
ci_df = pd.read_csv(RES_NEW / "loss_diff_ci.csv")
sub = ci_df[(ci_df["Sample"] == "Excl_Feb5_6") & (ci_df["Comparison"].str.contains("SPILL vs.")) & (ci_df["Comparison"].str.contains("NOSPILL"))].copy()
sub["p_raw_hln"] = 2.0 * stats.t.sf(np.abs(sub["dm_stat"]), df=sub["N"] - 1)
sub["m_bonf"] = 6
sub["p_bonf"] = np.minimum(1.0, sub["p_raw_hln"] * sub["m_bonf"])
sub["survives_05"] = sub["p_bonf"] < 0.05

cols_1a = [
    "Sample", "Comparison", "Loss", "Horizon", "N",
    "d_bar_pct", "ci_lo_pct", "ci_hi_pct", "mde_pct",
    "dm_stat", "p_raw_hln", "m_bonf", "p_bonf", "survives_05"
]
out_1a = sub[cols_1a]
out_1a.to_csv(RES_NEW / "excl_feb_spill_pvalues.csv", index=False)

print("=== STEP 1a: Excl. Feb 5-6 SPILL vs NOSPILL (m=6 per loss) ===")
for _, r in out_1a.iterrows():
    print(f"{r['Comparison']:50s} | {r['Loss']:5s} | {r['Horizon']:4s} | d%={r['d_bar_pct']:+7.2f} [{r['ci_lo_pct']:+7.2f}, {r['ci_hi_pct']:+7.2f}] | t={r['dm_stat']:+6.3f} | p_raw={r['p_raw_hln']:.6f} | p_bonf={r['p_bonf']:.6f} | <0.05={r['survives_05']}")
print(f"Total rows: {len(out_1a)}, rows with p_bonf < 0.05: {int(out_1a['survives_05'].sum())}")

# --- STEP 1b ---
dm3d = pd.read_csv(AUDIT / "v2_3c3d_dm_tests.csv")
print("\n=== STEP 1b: Families in v2_3c3d_dm_tests.csv ===")
print(dm3d.groupby(["Section", "Family", "m_Bonf", "Loss"]).size())

f_3d = dm3d[(dm3d["Section"] == "3d") & (dm3d["Family"] == "Expl_3d_RV60m")].copy()
print("\n=== All Expl_3d_RV60m (m=28 SE/AE, m=14 QLIKE) rows ===")
for _, r in f_3d.iterrows():
    direction = "Cand improves" if r["md"] < 0 else "Cand worsens"
    print(f"m={r['m_Bonf']:2d} | {r['Loss']:5s} | {r['Horizon']:4s} | {r['Cand']:22s} vs {r['Bench']:18s} | md={r['md']:+.4e} | stat={r['stat_hln']:+7.3f} | p_raw={r['p_raw_t']:.6g} | p_bonf={r['p_Bonf_t']:.6g} | {direction}")

# SPILL vs NOSPILL rows in 3d
print("\n=== STEP 1b (i): SPILL vs NOSPILL improvements in 3d ===")
spill_3d = f_3d[f_3d["Cand"].str.contains("SPILL") & f_3d["Bench"].str.contains("NOSPILL")]
for loss in ["SE", "AE", "QLIKE"]:
    sub_l = spill_3d[(spill_3d["Loss"] == loss) & (spill_3d["md"] < 0)]
    if len(sub_l) == 0:
        print(f"Loss={loss}: NO rows where SPILL improves over NOSPILL (all worsen or none tested)")
    else:
        min_p = sub_l["p_Bonf_t"].min()
        print(f"Loss={loss}: min p_bonf among SPILL improvements = {min_p:.6f}")
        for _, r in sub_l.iterrows():
            print(f"   {r['Horizon']} {r['Cand']} vs {r['Bench']}: md={r['md']:+.4e}, p_raw={r['p_raw_t']:.6f}, p_bonf={r['p_Bonf_t']:.6f}")

print("\n=== STEP 1b (i-alt): ALL comparisons in m=28 / m=14 where Cand improves ===")
for loss in ["SE", "AE", "QLIKE"]:
    sub_l = f_3d[(f_3d["Loss"] == loss) & (f_3d["md"] < 0)]
    print(f"Loss={loss} (any Cand improves): count={len(sub_l)}, min p_bonf={sub_l['p_Bonf_t'].min() if len(sub_l) else 'N/A'}")

print("\n=== STEP 1b (ii): 3d tests with p_bonf < 0.05 ===")
sig_3d = f_3d[f_3d["p_Bonf_t"] < 0.05]
for _, r in sig_3d.iterrows():
    direction = "Cand improves" if r["md"] < 0 else "Cand worsens"
    print(f"m={r['m_Bonf']:2d} | {r['Loss']:5s} | {r['Horizon']:4s} | {r['Cand']} vs {r['Bench']} | md={r['md']:+.4e} | p_raw={r['p_raw_t']:.6g} | p_bonf={r['p_Bonf_t']:.6g} | {direction}")

print("\n=== STEP 1b (ii-alt): 3d tests with p_bonf < 0.10 ===")
sig10_3d = f_3d[f_3d["p_Bonf_t"] < 0.10]
for _, r in sig10_3d.iterrows():
    direction = "Cand improves" if r["md"] < 0 else "Cand worsens"
    print(f"m={r['m_Bonf']:2d} | {r['Loss']:5s} | {r['Horizon']:4s} | {r['Cand']} vs {r['Bench']} | md={r['md']:+.4e} | p_raw={r['p_raw_t']:.6g} | p_bonf={r['p_Bonf_t']:.6g} | {direction}")

# --- STEP 1c ---
c2 = pd.read_csv(REV / "C2_all_existing_dm_tests_with_holm.csv")
print("\n=== STEP 1c: Search for 0.957 in C2_all_existing_dm_tests_with_holm.csv ===")
m957 = c2[(c2["p_Bonf_t"].round(3) == 0.957) | (c2["p_Holm_t"].round(3) == 0.957)]
print(m957.to_string())

v2_dm = pd.read_csv(AUDIT / "v2_dm_tests.csv")
print("\n=== STEP 1c: Search for ~0.957 in v2_dm_tests.csv ===")
m957_v2 = v2_dm[(v2_dm["p_Bonf_t"].round(3) == 0.957) | (v2_dm["p_raw_t"].round(3) == 0.957)]
print(m957_v2.to_string())

print("\n=== STEP 1c: All h=10 QLIKE tests in v2_dm_tests.csv ===")
q10 = v2_dm[(v2_dm["Horizon"] == "h=10") & (v2_dm["Loss"] == "QLIKE")]
print(q10[["Family", "m_Bonf", "Cand", "Bench", "md", "stat_hln", "p_raw_t", "p_Bonf_t"]].to_string())

# --- STEP 1d ---
print("\n=== STEP 1d: RV60m seed counts ===")
for fn in ["v2_3c3d_collapse_diagnostics_per_seed.csv", "v2_3c3d_metrics_recomputation.csv"]:
    df_s = pd.read_csv(AUDIT / fn)
    print(f"--- {fn} columns: {list(df_s.columns)} ---")
    if "variant" in df_s.columns:
        sub_rv = df_s[df_s["variant"].astype(str).str.contains("RV60", na=False)]
        if "seed" in sub_rv.columns:
            print(sub_rv.groupby(["variant", "model", "feature_set"])["seed"].nunique() if "feature_set" in sub_rv.columns else sub_rv.groupby(["variant", "model"])["seed"].nunique())
        else:
            print(sub_rv.head(10))
