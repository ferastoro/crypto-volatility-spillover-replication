#!/usr/bin/env python3
"""
Step 1a & 1b: Compute and verify RV60m SPILL vs NOSPILL HLN-DM statistics under AE and QLIKE,
and list every test in the RV60m QLIKE m=14 family from v2_3c3d_dm_tests.csv and predictions_ensemble.csv.
"""
from pathlib import Path
import numpy as np
import pandas as pd
from scipy import stats

BASE_DIR = Path(__file__).resolve().parent.parent
AUDIT_DIR = BASE_DIR / "revision" / "audit"
KAGGLE_3C3D = BASE_DIR / "revision" / "kaggle_outputs" / "outputs_3c3d_v2"
PAPER_DIR = Path(__file__).resolve().parent.parent
OUT_DIR = PAPER_DIR / "results_new"
OUT_DIR.mkdir(parents=True, exist_ok=True)


def hln_dm_two_sided(d: np.ndarray, h: int):
    d = np.asarray(d, dtype=np.float64)
    n = len(d)
    dbar = float(np.mean(d))
    dc = d - dbar
    gamma0 = float(np.dot(dc, dc) / n)
    L = h - 1
    var_sum = gamma0
    for lag in range(1, L + 1):
        w = 1.0 - lag / (L + 1.0)
        cov = float(np.dot(dc[lag:], dc[:-lag]) / n)
        var_sum += 2.0 * w * cov
    dm_stat = dbar / np.sqrt(var_sum / n)
    # Harvey-Leybourne-Newbold (1997) adjustment factor sqrt((n + 1 - 2h + h(h-1)/n)/n)
    adj = np.sqrt((n + 1.0 - 2.0 * h + (h * (h - 1.0)) / n) / n)
    hln_stat = float(dm_stat * adj)
    p_raw = float(2.0 * stats.t.sf(abs(hln_stat), df=n - 1))
    return float(dbar), hln_stat, p_raw


def main():
    dm = pd.read_csv(AUDIT_DIR / "v2_3c3d_dm_tests.csv")
    pe = pd.read_csv(KAGGLE_3C3D / "predictions_ensemble.csv")
    rv_main = pd.read_csv(OUT_DIR / "rv60m_main.csv")

    dm3d = dm[(dm["Section"] == "3d") & (dm["Sample"] == "Full_335")].copy()
    pe_rv = pe[pe["variant"] == "RV60m"].copy()

    rows = []
    configs = [
        ("TFT", "AE", 4),
        ("NBEATSx", "AE", 4),
        ("TFT", "QLIKE", 2),
    ]
    for model, loss, m_sub in configs:
        for h in [1, 10]:
            h_str = f"h={h}"
            sub_sp = pe_rv[(pe_rv["model"] == model) & (pe_rv["feature_set"] == "SPILL") & (pe_rv["horizon"] == h)].sort_values("origin_idx")
            sub_no = pe_rv[(pe_rv["model"] == model) & (pe_rv["feature_set"] == "NOSPILL") & (pe_rv["horizon"] == h)].sort_values("origin_idx")
            assert np.allclose(sub_sp["y_true"].values, sub_no["y_true"].values, rtol=0, atol=1e-15)
            y = sub_sp["y_true"].values
            p_sp = sub_sp["y_pred"].values
            p_no = sub_no["y_pred"].values

            if loss == "AE":
                l_sp = np.abs(y - p_sp)
                l_no = np.abs(y - p_no)
            elif loss == "QLIKE":
                l_sp = (y / p_sp) - np.log(y / p_sp) - 1.0
                l_no = (y / p_no) - np.log(y / p_no) - 1.0
            else:
                raise ValueError(loss)

            mean_sp = float(np.mean(l_sp))
            mean_no = float(np.mean(l_no))
            dbar, hln_stat_strict, p_raw_strict = hln_dm_two_sided(l_sp - l_no, h)

            dm_cand = f"RV60m:{model}-SPILL"
            dm_bench = f"RV60m:{model}-NOSPILL"
            dm_row = dm3d[(dm3d["Horizon"] == h_str) & (dm3d["Loss"] == loss) & (dm3d["Cand"] == dm_cand) & (dm3d["Bench"] == dm_bench)].iloc[0]

            # Saved audit CSV values (which match run_step1_stage2.py: factor 1.0 at h=1, full HLN factor at h=10)
            hln_stat_csv = float(dm_row["stat_hln"])
            p_raw_csv = float(dm_row["p_raw_t"])
            if h > 1:
                assert abs(hln_stat_strict - hln_stat_csv) < 1e-9, (hln_stat_strict, hln_stat_csv)
                assert abs(p_raw_strict - p_raw_csv) < 1e-9, (p_raw_strict, p_raw_csv)
            assert abs(dbar - float(dm_row["md"])) < 1e-12, (dbar, dm_row["md"])

            m_fam3d = int(dm_row["m_Bonf"])
            p_bonf_sub_csv = min(1.0, m_sub * p_raw_csv)
            p_bonf_sub_strict = min(1.0, m_sub * p_raw_strict)
            p_bonf_3d = float(dm_row["p_Bonf_t"])
            direction = "SPILL improves" if dbar < 0 else "SPILL worsens"

            rows.append({
                "Model": model,
                "Horizon": h,
                "Loss": loss,
                "Mean_SPILL": mean_sp,
                "Mean_NOSPILL": mean_no,
                "Mean_Diff_SPILL_minus_NO": dbar,
                "Direction": direction,
                "HLN_Stat_AuditCSV": hln_stat_csv,
                "p_raw_AuditCSV": p_raw_csv,
                "m_sub": m_sub,
                "p_bonf_sub_AuditCSV": p_bonf_sub_csv,
                "m_3d_family": m_fam3d,
                "p_bonf_3d_family": p_bonf_3d,
                "HLN_Stat_Strict": hln_stat_strict,
                "p_raw_Strict": p_raw_strict,
                "p_bonf_sub_Strict": p_bonf_sub_strict,
            })

    out_df = pd.DataFrame(rows)
    out_df.to_csv(OUT_DIR / "rv60m_spill_vs_no_ae_qlike.csv", index=False)
    print("=== STEP 1a: RV60m SPILL vs NOSPILL (AE m=4, TFT QLIKE m=2, and full RV60m family m) ===")
    print(out_df.to_string(index=False))

    print("\n=== STEP 1b: RV60m QLIKE m=14 family ===")
    q14 = dm3d[(dm3d["Loss"] == "QLIKE") & (dm3d["m_Bonf"] == 14)].copy()
    q14["Direction"] = np.where(q14["md"] < 0, "Cand better than Bench", "Cand worse than Bench")
    print(q14[["Horizon", "Cand", "Bench", "md", "stat_hln", "p_raw_t", "p_Bonf_t", "Direction"]].to_string(index=False))


if __name__ == "__main__":
    main()
