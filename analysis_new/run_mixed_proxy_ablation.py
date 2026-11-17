#!/usr/bin/env python3
"""
Controlled ablation of the -4.10% BTC/IDR mixed-proxy sign reversal.
Copies/imports TVP-VAR functions from:
  - revision/scripts/run_step1_stage1.py (current estimator)
  - audit_scripts/run_checkpoint3a.py (legacy estimator; extracted cleanly to avoid top-level execution)
Writes outputs strictly to results_new/.
"""

import sys
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.linalg import pinv

REPO_ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = REPO_ROOT / "results_new"
OUT_DIR.mkdir(parents=True, exist_ok=True)

sys.path.insert(0, str(REPO_ROOT / "revision" / "scripts"))
import run_step1_stage1 as s1  # safe: main() is behind __name__ == "__main__"


# -----------------------------------------------------------------------------
# Legacy TVP-VAR & GFEVD verbatim from audit_scripts/run_checkpoint3a.py (L235-370)
# -----------------------------------------------------------------------------
class LegacyTVPVAR:
    def __init__(self, y, p=1, lambda_ff=0.99, kappa_ff=0.96):
        self.y = y
        self.T, self.M = y.shape
        self.p = p
        self.K = self.M * self.p + 1
        self.lambda_ff = lambda_ff
        self.kappa_ff = kappa_ff
        self.Z = np.zeros((self.T - p, self.M, self.K * self.M))
        self.Y = self.y[p:]
        for t in range(self.T - p):
            z_t = np.concatenate([[1.0], self.y[t : t + p].flatten()[::-1]])
            self.Z[t] = np.kron(np.eye(self.M), z_t)

    def estimate(self, causal_sigma0=False, warmup=250):
        T_est = self.T - self.p
        beta_t = np.zeros((self.K * self.M, 1))
        P_t = np.eye(self.K * self.M) * 10.0
        if causal_sigma0:
            Sigma_t = np.cov(self.y[: min(warmup, self.T)].T)
        else:
            Sigma_t = np.cov(self.y.T)

        self.beta_history = np.zeros((T_est, self.K * self.M))
        self.Sigma_history = np.zeros((T_est, self.M, self.M))

        for t in range(T_est):
            Z_t = self.Z[t]
            y_t = self.Y[t].reshape(self.M, 1)
            P_t_pred = P_t / self.lambda_ff
            v_t = y_t - Z_t @ beta_t
            Sigma_t = self.kappa_ff * Sigma_t + (1 - self.kappa_ff) * (v_t @ v_t.T)
            F_t = Z_t @ P_t_pred @ Z_t.T + Sigma_t
            K_t = P_t_pred @ Z_t.T @ pinv(F_t)
            beta_t = beta_t + K_t @ v_t
            P_t = P_t_pred - K_t @ Z_t @ P_t_pred
            self.beta_history[t] = beta_t.flatten()
            self.Sigma_history[t] = Sigma_t
        return self.beta_history, self.Sigma_history


def legacy_calculate_gfevd(beta_history, sigma_history, M, p, H=10):
    T_est = beta_history.shape[0]
    gfevd_history = np.zeros((T_est, M, M))
    for t in range(T_est):
        beta_t = beta_history[t]
        Sigma_t = sigma_history[t]
        sigma_diag = np.diag(Sigma_t)
        A = np.zeros((p, M, M))
        idx = 0
        for i in range(M):
            idx += 1
            for lag in range(p):
                for j in range(M):
                    A[lag, i, j] = beta_t[idx]
                    idx += 1
        Phi = np.zeros((H, M, M))
        Phi[0] = np.eye(M)
        for h in range(1, H):
            for j in range(1, min(h, p) + 1):
                Phi[h] += Phi[h - j] @ A[j - 1]
        theta = np.zeros((M, M))
        for i in range(M):
            for j in range(M):
                e_j = np.zeros((M, 1))
                e_j[j] = 1.0
                num = 0.0
                den = 0.0
                for h in range(H):
                    term = (Phi[h][i, :] @ Sigma_t @ e_j)[0]
                    num += term ** 2
                    den += Phi[h][i, :] @ Sigma_t @ Phi[h][i, :].T
                theta[i, j] = (1.0 / sigma_diag[j]) * (num / den)
        row_sums = theta.sum(axis=1)
        gfevd_history[t] = theta / row_sums[:, np.newaxis]
    return gfevd_history


def summarize_gfevd_dy12(gfevd_slice):
    """Exact DY12 metrics on a slice of GFEVD matrices (T_sub, M, M)."""
    M = gfevd_slice.shape[1]
    mg = gfevd_slice.mean(axis=0) * 100.0
    diag = np.diag(mg)
    from_dy12 = mg.sum(axis=1) - diag
    to_dy12 = mg.sum(axis=0) - diag
    net_dy12 = to_dy12 - from_dy12
    tsi = float((mg.sum() - np.trace(mg)) / M)
    return {
        "TSI": tsi,
        "BTC_USD_NET": float(net_dy12[5]),
        "BTC_IDR_NET": float(net_dy12[7]),
        "net_dy12": net_dy12,
    }


def main():
    # =========================================================================
    # STEP 1: Ablation A (current pipeline, Main_US_1677)
    # =========================================================================
    main_panel = s1.build_aligned_panel(s1.ASSETS_8, join_ihsg=False)
    labels_8 = [lbl for _, lbl, _, _ in s1.ASSETS_8]
    proxies_main = s1.make_proxy_matrices(main_panel, labels_8)

    ab_a_specs = [
        ("Mixed legacy (BTC/IDR log Park., peers log r2)", "log_Parkinson_mixed_legacy"),
        ("Symmetric log Parkinson (all 8 assets)", "log_Parkinson_8asset_interval"),
        ("Symmetric log(r^2) (all 8 assets)", "log_r2"),
    ]
    rows_a = []
    for plabel, pname in ab_a_specs:
        y = proxies_main[pname]
        res = s1.estimate_tvp_var_gfevd(
            y, p=1, lam=s1.LAMBDA_FF, kap=s1.KAPPA_FF, H=s1.GFEVD_H, warmup=s1.WARMUP_OBS, causal_sigma0=True
        )
        for tname, gsl in [
            ("Un-trimmed (all 1,676 steps)", res["gfevd_all"]),
            ("Post-warm-up (N_eff=1,426)", res["gfevd_post"]),
        ]:
            m = summarize_gfevd_dy12(gsl)
            rows_a.append({
                "Proxy_Spec": plabel,
                "Proxy_Key": pname,
                "BurnIn_Treatment": tname,
                "N_steps": int(len(gsl)),
                "BTC_IDR_NET": round(m["BTC_IDR_NET"], 4),
                "BTC_USD_NET": round(m["BTC_USD_NET"], 4),
                "TSI": round(m["TSI"], 4),
                "BTC_IDR_NET_exact": m["BTC_IDR_NET"],
                "BTC_USD_NET_exact": m["BTC_USD_NET"],
                "TSI_exact": m["TSI"],
            })

    df_a = pd.DataFrame(rows_a)
    df_a.to_csv(OUT_DIR / "mixed_proxy_ablation_a.csv", index=False)

    # Assert exact reproduction of Table 4 / Table B1
    val_pk_post = df_a.loc[(df_a["Proxy_Key"] == "log_Parkinson_8asset_interval") & (df_a["BurnIn_Treatment"].str.startswith("Post")), "BTC_IDR_NET_exact"].iloc[0]
    val_lr2_post = df_a.loc[(df_a["Proxy_Key"] == "log_r2") & (df_a["BurnIn_Treatment"].str.startswith("Post")), "BTC_IDR_NET_exact"].iloc[0]
    val_mix_post = df_a.loc[(df_a["Proxy_Key"] == "log_Parkinson_mixed_legacy") & (df_a["BurnIn_Treatment"].str.startswith("Post")), "BTC_IDR_NET_exact"].iloc[0]
    assert abs(val_pk_post - 1.2316) < 0.005, f"Symmetric Park post mismatch: {val_pk_post}"
    assert abs(val_lr2_post - 4.4205) < 0.005, f"Symmetric log_r2 post mismatch: {val_lr2_post}"
    assert abs(val_mix_post - 3.3435) < 0.005, f"Mixed legacy post mismatch: {val_mix_post}"

    print("=== ABLATION A (Current Pipeline, Main_US_1677) ===")
    print(df_a[["Proxy_Spec", "BurnIn_Treatment", "N_steps", "BTC_IDR_NET", "BTC_USD_NET", "TSI"]].to_string(index=False))

    # =========================================================================
    # STEP 2: Ablation B (Legacy Pipeline, 5 binary switches)
    # Switches:
    #   sw1_burnin:    False -> un-trimmed, True -> discard first 250 steps
    #   sw2_cal1677:   False -> 1,556 joint US-JKSE sample, True -> 1,677 US calendar
    #   sw3_nozscore:  False -> full-sample z-scored, True -> unstandardized raw log(.)
    #   sw4_currest:   False -> full-sample Sigma_0 + LegacyTVPVAR (non-Joseph/pinv),
    #                  True  -> causal Sigma_0 (first 250) + current pipeline estimator (Joseph/solve)
    #   sw5_intpark:   False -> target_alternatives.csv['parkinson'] (log(p + 1e-10)),
    #                  True  -> current pipeline interval-based log(max(p_int, 0) + 1e-10)
    # =========================================================================
    core_vars = [
        "ret_saham_nvda", "ret_saham_msft", "ret_saham_tsla", "ret_saham_googl",
        "ret_saham_meta", "ret_crypto_btc_usd", "ret_crypto_eth_usd", "ret_crypto_btc_idr_indodax",
    ]
    df_1556 = pd.read_csv(REPO_ROOT / "data" / "clean" / "dataset_garch_baseline.csv")
    df_1556["Date"] = pd.to_datetime(df_1556["Date"])
    alt_df = pd.read_csv(REPO_ROOT / "data" / "clean" / "target_alternatives.csv")
    alt_df["Date"] = pd.to_datetime(alt_df["Date"])

    rob1556_panel = s1.build_aligned_panel(s1.ASSETS_8, join_ihsg=True, join_legacy_1556=True)

    # Verify calendar parity between df_1556 and rob1556_panel["df_ret"]
    assert len(df_1556) == 1556
    assert (df_1556["Date"].values == rob1556_panel["df_ret"]["Date"].values).all(), "1556 dates mismatch!"
    max_ret_diff = float(np.max(np.abs(df_1556[core_vars].values - rob1556_panel["df_ret"][labels_8].values)))
    print(f"\nParity check (1,556 returns legacy vs rob1556_panel): max_abs_diff = {max_ret_diff:.3e}")

    # Verify US-1677 date parity with target_alternatives
    assert (main_panel["df_ret"]["Date"].values == alt_df["Date"].values).all(), "1677 alt_df dates mismatch!"

    def build_legacy_or_switched_y(sw2_cal1677: bool, sw3_nozscore: bool, sw5_intpark: bool) -> np.ndarray:
        eps = 1e-10
        if not sw2_cal1677:
            # 1,556 joint sample
            r_mat = df_1556[core_vars].values
            Y_raw = np.log(r_mat ** 2 + eps)
            if not sw5_intpark:
                merged = pd.merge(df_1556[["Date"]], alt_df[["Date", "parkinson"]], on="Date", how="left")
                Y_raw[:, 7] = np.log(merged["parkinson"].values + eps)
            else:
                Y_raw[:, 7] = np.log(np.maximum(rob1556_panel["df_park_int"]["BTC/IDR"].values, 0.0) + eps)
        else:
            # 1,677 US calendar sample
            r_mat = main_panel["df_ret"][labels_8].values
            Y_raw = np.log(r_mat ** 2 + eps)
            if not sw5_intpark:
                merged = pd.merge(main_panel["df_ret"][["Date"]], alt_df[["Date", "parkinson"]], on="Date", how="left")
                Y_raw[:, 7] = np.log(merged["parkinson"].values + eps)
            else:
                Y_raw[:, 7] = np.log(np.maximum(main_panel["df_park_int"]["BTC/IDR"].values, 0.0) + eps)

        if not sw3_nozscore:
            mu = np.mean(Y_raw, axis=0)
            sd = np.std(Y_raw, axis=0)
            return (Y_raw - mu) / sd
        return Y_raw

    def run_ablation_b_cell(label: str, sw1_burnin: bool, sw2_cal1677: bool,
                            sw3_nozscore: bool, sw4_currest: bool, sw5_intpark: bool):
        y = build_legacy_or_switched_y(sw2_cal1677=sw2_cal1677, sw3_nozscore=sw3_nozscore, sw5_intpark=sw5_intpark)
        if not sw4_currest:
            mod = LegacyTVPVAR(y, p=1, lambda_ff=0.99, kappa_ff=0.96)
            beta_h, sigma_h = mod.estimate(causal_sigma0=False, warmup=250)
            gfevd_all = legacy_calculate_gfevd(beta_h, sigma_h, M=8, p=1, H=10)
            gsl = gfevd_all[250:] if sw1_burnin else gfevd_all
        else:
            res = s1.estimate_tvp_var_gfevd(
                y, p=1, lam=s1.LAMBDA_FF, kap=s1.KAPPA_FF, H=s1.GFEVD_H, warmup=s1.WARMUP_OBS, causal_sigma0=True
            )
            gsl = res["gfevd_post"] if sw1_burnin else res["gfevd_all"]
        m = summarize_gfevd_dy12(gsl)
        return {
            "Step_Label": label,
            "sw1_discard_250": sw1_burnin,
            "sw2_sample_1677": sw2_cal1677,
            "sw3_no_zscore": sw3_nozscore,
            "sw4_causal_joseph": sw4_currest,
            "sw5_interval_park": sw5_intpark,
            "N_steps": int(len(gsl)),
            "BTC_IDR_NET": round(m["BTC_IDR_NET"], 4),
            "BTC_USD_NET": round(m["BTC_USD_NET"], 4),
            "TSI": round(m["TSI"], 4),
            "BTC_IDR_NET_exact": m["BTC_IDR_NET"],
        }

    # (a) Legacy reproduction + (b) One-at-a-time
    rows_b_oat = [
        run_ablation_b_cell("(a) Legacy baseline (checkpoint3a exact)", False, False, False, False, False),
        run_ablation_b_cell("(1) + Discard first 250 steps (N_eff=1,305)", True, False, False, False, False),
        run_ablation_b_cell("(2) + Sample = 1,677 US calendar", False, True, False, False, False),
        run_ablation_b_cell("(3) + Remove full-sample z-scoring", False, False, True, False, False),
        run_ablation_b_cell("(4) + Causal Sigma_0 & Joseph/solve estimator", False, False, False, True, False),
        run_ablation_b_cell("(5) + Interval Parkinson from raw Indodax", False, False, False, False, True),
    ]
    df_b_oat = pd.DataFrame(rows_b_oat)
    df_b_oat.to_csv(OUT_DIR / "mixed_proxy_ablation_b_oat.csv", index=False)

    legacy_val = df_b_oat.loc[0, "BTC_IDR_NET_exact"]
    assert abs(legacy_val - (-4.0968)) < 0.01, f"Legacy reproduction failed: {legacy_val}"

    # (c) Cumulative: (a) -> +(4) -> +(3) -> +(5) -> +(2) -> +(1)
    rows_b_cum = [
        run_ablation_b_cell("0. Legacy baseline (a)", False, False, False, False, False),
        run_ablation_b_cell("1. + (4) Causal Sigma_0 & Joseph/solve", False, False, False, True, False),
        run_ablation_b_cell("2. + (3) Remove z-scoring", False, False, True, True, False),
        run_ablation_b_cell("3. + (5) Interval Parkinson", False, False, True, True, True),
        run_ablation_b_cell("4. + (2) Sample = 1,677 US calendar", False, True, True, True, True),
        run_ablation_b_cell("5. + (1) Discard first 250 steps [== Abl. A (i)-(b)]", True, True, True, True, True),
    ]
    df_b_cum = pd.DataFrame(rows_b_cum)
    df_b_cum.to_csv(OUT_DIR / "mixed_proxy_ablation_b_cum.csv", index=False)

    assert abs(df_b_cum.loc[5, "BTC_IDR_NET_exact"] - val_mix_post) < 1e-10, "Final cumulative != Ablation A (i)-(b)"

    print("\n=== ABLATION B(a)+(b): One-at-a-Time from Legacy Pipeline ===")
    print(df_b_oat[["Step_Label", "N_steps", "BTC_IDR_NET", "BTC_USD_NET", "TSI"]].to_string(index=False))

    print("\n=== ABLATION B(c): Cumulative (4) -> (3) -> (5) -> (2) -> (1) ===")
    print(df_b_cum[["Step_Label", "N_steps", "BTC_IDR_NET", "BTC_USD_NET", "TSI"]].to_string(index=False))


if __name__ == "__main__":
    main()
