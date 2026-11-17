#!/usr/bin/env python3
"""
Analysis 3: Assemble main-text RV60m table from saved outputs.
"""
from pathlib import Path
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent.parent
REV_OUT = BASE_DIR / "revision" / "outputs"
KAGGLE_3C3D = BASE_DIR / "revision" / "kaggle_outputs" / "outputs_3c3d_v2"
AUDIT_DIR = BASE_DIR / "revision" / "audit"
OUT_DIR = Path(__file__).resolve().parent.parent / "results_new"
OUT_DIR.mkdir(parents=True, exist_ok=True)


def main():
    econ = pd.read_csv(REV_OUT / "step3d_rv60m_econometric.csv")
    nn = pd.read_csv(KAGGLE_3C3D / "metrics_by_model.csv")
    dm = pd.read_csv(AUDIT_DIR / "v2_3c3d_dm_tests.csv")

    nn_rv = nn[(nn["variant"] == "RV60m") & (nn["horizon"].isin([1, 10]))].copy()
    dm_rv = dm[(dm["Section"] == "3d") & (dm["Sample"] == "Full_335") & (dm["Horizon"].isin(["h=1", "h=10"]))].copy()

    rows = []
    for hv in [1, 10]:
        h_str = f"h={hv}"
        # Econometric models
        for mname, label in [
            ("HAR_RV", "HAR-RV"),
            ("HAR-X-NOSPILL_RV", "HAR-X-NOSPILL-RV"),
            ("HAR-X-SPILL_RV", "HAR-X-SPILL-RV"),
        ]:
            r = econ[(econ["Horizon"] == h_str) & (econ["Model"] == mname)].iloc[0]
            tft_no_ql = float(nn_rv[(nn_rv["horizon"] == hv) & (nn_rv["model"] == "TFT") & (nn_rv["feature_set"] == "NOSPILL")]["QLIKE"].iloc[0])
            dm_ql_row = dm_rv[(dm_rv["Horizon"] == h_str) & (dm_rv["Loss"] == "QLIKE") & (dm_rv["Cand"] == "RV60m:TFT-NOSPILL") & (dm_rv["Bench"] == mname)].iloc[0]
            m_qlike = tft_no_ql - float(dm_ql_row["md"])
            rows.append({
                "Horizon": h_str,
                "Model": label,
                "MSE": float(r["MSE"]),
                "MAE": float(r["MAE"]),
                "QLIKE": m_qlike,
                "R2_OOS": float(r["R2_OOS_HistExp"]),
                "DM_SE_vs_HAR_p_raw": float(r["DM_vs_HAR_p_raw"]),
                "DM_SE_vs_HAR_p_bonf": min(1.0, 28.0 * float(r["DM_vs_HAR_p_raw"])) if mname != "HAR_RV" else float("nan"),
                "DM_SE_vs_NOSPILL_p_raw": float(r["DM_vs_HARX_NOSPILL_p_raw"]) if mname == "HAR-X-SPILL_RV" else float("nan"),
            })

        # Neural models
        for m, fs, label, dm_cand, dm_nospill in [
            ("TFT", "NOSPILL", "TFT-NOSPILL", "RV60m:TFT-NOSPILL", None),
            ("TFT", "SPILL", "TFT-SPILL", "RV60m:TFT-SPILL", "RV60m:TFT-NOSPILL"),
            ("NBEATSx", "NOSPILL", "N-BEATSx-NOSPILL", "RV60m:NBEATSx-NOSPILL", None),
            ("NBEATSx", "SPILL", "N-BEATSx-SPILL", "RV60m:NBEATSx-SPILL", "RV60m:NBEATSx-NOSPILL"),
        ]:
            r = nn_rv[(nn_rv["horizon"] == hv) & (nn_rv["model"] == m) & (nn_rv["feature_set"] == fs)].iloc[0]
            se_har = dm_rv[(dm_rv["Horizon"] == h_str) & (dm_rv["Loss"] == "SE") & (dm_rv["Cand"] == dm_cand) & (dm_rv["Bench"] == "HAR_RV")].iloc[0]
            ae_har = dm_rv[(dm_rv["Horizon"] == h_str) & (dm_rv["Loss"] == "AE") & (dm_rv["Cand"] == dm_cand) & (dm_rv["Bench"] == "HAR_RV")].iloc[0]
            ql_har_rows = dm_rv[(dm_rv["Horizon"] == h_str) & (dm_rv["Loss"] == "QLIKE") & (dm_rv["Cand"] == dm_cand) & (dm_rv["Bench"] == "HAR_RV")]
            ql_p_raw = float(ql_har_rows["p_raw_t"].iloc[0]) if len(ql_har_rows) else float("nan")
            ql_p_bonf = float(ql_har_rows["p_Bonf_t"].iloc[0]) if len(ql_har_rows) else float("nan")

            sp_no_se_raw = float("nan")
            sp_no_se_bonf = float("nan")
            sp_no_ql_raw = float("nan")
            sp_no_ql_bonf = float("nan")
            if dm_nospill:
                sp_se = dm_rv[(dm_rv["Horizon"] == h_str) & (dm_rv["Loss"] == "SE") & (dm_rv["Cand"] == dm_cand) & (dm_rv["Bench"] == dm_nospill)].iloc[0]
                sp_no_se_raw = float(sp_se["p_raw_t"])
                sp_no_se_bonf = float(sp_se["p_Bonf_t"])
                sp_ql = dm_rv[(dm_rv["Horizon"] == h_str) & (dm_rv["Loss"] == "QLIKE") & (dm_rv["Cand"] == dm_cand) & (dm_rv["Bench"] == dm_nospill)]
                if len(sp_ql):
                    sp_no_ql_raw = float(sp_ql["p_raw_t"].iloc[0])
                    sp_no_ql_bonf = float(sp_ql["p_Bonf_t"].iloc[0])

            rows.append({
                "Horizon": h_str,
                "Model": label,
                "MSE": float(r["MSE"]),
                "MAE": float(r["MAE"]),
                "QLIKE": float(r["QLIKE"]) if pd.notna(r["QLIKE"]) else float("nan"),
                "R2_OOS": float(r["R2_OOS"]),
                "DM_SE_vs_HAR_p_raw": float(se_har["p_raw_t"]),
                "DM_SE_vs_HAR_p_bonf": float(se_har["p_Bonf_t"]),
                "DM_AE_vs_HAR_p_raw": float(ae_har["p_raw_t"]),
                "DM_AE_vs_HAR_p_bonf": float(ae_har["p_Bonf_t"]),
                "DM_QL_vs_HAR_p_raw": ql_p_raw,
                "DM_QL_vs_HAR_p_bonf": ql_p_bonf,
                "DM_SE_vs_NOSPILL_p_raw": sp_no_se_raw,
                "DM_SE_vs_NOSPILL_p_bonf": sp_no_se_bonf,
                "DM_QL_vs_NOSPILL_p_raw": sp_no_ql_raw,
                "DM_QL_vs_NOSPILL_p_bonf": sp_no_ql_bonf,
            })

    out_df = pd.DataFrame(rows)
    out_df.to_csv(OUT_DIR / "rv60m_main.csv", index=False)
    print(out_df.to_string(index=False))


if __name__ == "__main__":
    main()

