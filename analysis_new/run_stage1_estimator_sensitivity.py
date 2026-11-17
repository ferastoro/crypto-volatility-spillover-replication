#!/usr/bin/env python3
"""
Task B: Stage-1 TVP-VAR estimator sensitivity (grid over lambda/kappa, r^2*1e4 scale check,
and causal Sigma_0 window N_init=100 vs N_init=250) on Main_US_1677 (N_eff=1,426).
Imports data loading & proxy construction directly from revision/scripts/run_step1_stage1.py
without modifying it.
"""

import sys
from pathlib import Path
import numpy as np
import pandas as pd

ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR / "revision" / "scripts"))

import run_step1_stage1 as s1  # noqa: E402


def estimate_tvp_var_gfevd_sens(
    y: np.ndarray,
    p: int = 1,
    lam: float = 0.99,
    kap: float = 0.96,
    H: int = 10,
    warmup: int = 250,
    sigma0_n: int = 250,
):
    """Exact copy of s1.estimate_tvp_var_gfevd(causal_sigma0=True) allowing separate sigma0_n."""
    T, M = y.shape
    K = M * p + 1
    n_params = K * M

    if p == 0:
        Y = y
        X = np.ones((T, 1), dtype=np.float64)
        obs_offset = 0
    else:
        Y = y[p:]
        X = np.empty((T - p, K), dtype=np.float64)
        X[:, 0] = 1.0
        for lag in range(1, p + 1):
            X[:, 1 + (lag - 1) * M : 1 + lag * M] = y[p - lag : T - lag]
        obs_offset = p

    T_eff = len(Y)
    beta = np.zeros(n_params, dtype=np.float64)
    P = np.eye(n_params, dtype=np.float64) * 10.0
    init_n = min(sigma0_n, T)
    Sigma = np.cov(y[:init_n].T)
    if Sigma.ndim == 0:
        Sigma = np.array([[float(Sigma)]])

    lam_inv = 1.0 / lam
    kap1 = 1.0 - kap
    eye_m = np.eye(M, dtype=np.float64)
    eye_n = np.eye(n_params, dtype=np.float64)

    beta_hist = np.empty((T_eff, M, K), dtype=np.float64)
    sigma_hist = np.empty((T_eff, M, M), dtype=np.float64)

    for t in range(T_eff):
        xt = X[t]
        Z = np.kron(eye_m, xt)
        err = Y[t] - Z @ beta
        Sigma = kap * Sigma + kap1 * np.outer(err, err)
        P_pred = P * lam_inv
        PZ_T = P_pred @ Z.T
        S_t = Z @ PZ_T + Sigma
        K_t = np.linalg.solve(S_t, PZ_T.T).T
        beta = beta + K_t @ err
        IKZ = eye_n - K_t @ Z
        P = IKZ @ P_pred @ IKZ.T + K_t @ Sigma @ K_t.T
        P = 0.5 * (P + P.T)
        beta_hist[t] = beta.reshape(M, K)
        sigma_hist[t] = Sigma

    if p == 0:
        diag_s = np.maximum(np.diagonal(sigma_hist, axis1=1, axis2=2), 1e-12)
        num = (sigma_hist ** 2) / diag_s[:, None, :]
        denom = np.maximum(np.sum(num, axis=2, keepdims=True), 1e-12)
        gfevd_all = num / denom
    else:
        A = [beta_hist[:, :, 1 + j * M : 1 + (j + 1) * M] for j in range(p)]
        Phi = np.zeros((T_eff, H, M, M), dtype=np.float64)
        Phi[:, 0] = eye_m
        for h in range(1, H):
            acc = np.zeros((T_eff, M, M), dtype=np.float64)
            for j in range(1, min(p, h) + 1):
                acc += np.einsum("tij,tjk->tik", A[j - 1], Phi[:, h - j])
            Phi[:, h] = acc
        diag_s = np.maximum(np.diagonal(sigma_hist, axis1=1, axis2=2), 1e-12)
        Phi_S = np.einsum("thij,tjk->thik", Phi, sigma_hist)
        num = np.sum(Phi_S ** 2, axis=1) / diag_s[:, None, :]
        denom = np.maximum(np.sum(num, axis=2, keepdims=True), 1e-12)
        gfevd_all = num / denom

    start_idx = warmup + max(0, 1 - obs_offset)
    tsi_all = (gfevd_all.sum(axis=(1, 2)) - np.trace(gfevd_all, axis1=1, axis2=2)) / M * 100.0
    gfevd_post = gfevd_all[start_idx:]
    tsi_post = tsi_all[start_idx:]
    mean_gfevd = gfevd_post.mean(axis=0)
    from_i = (mean_gfevd.sum(axis=1) - np.diag(mean_gfevd)) / M * 100.0
    to_i = (mean_gfevd.sum(axis=0) - np.diag(mean_gfevd)) / M * 100.0
    net_i = to_i - from_i

    return {
        "tsi_post_mean": float(np.mean(tsi_post)),
        "net_i": net_i,
        "n_post": int(len(tsi_post)),
        "mean_gfevd": mean_gfevd,
    }


def eval_cell(y: np.ndarray, labels: list[str], lam: float, kap: float, sigma0_n: int):
    res1 = estimate_tvp_var_gfevd_sens(y, p=1, lam=lam, kap=kap, H=10, warmup=250, sigma0_n=sigma0_n)
    res0 = estimate_tvp_var_gfevd_sens(y, p=0, lam=lam, kap=kap, H=10, warmup=250, sigma0_n=sigma0_n)
    idx_btcusd = labels.index("BTC-USD")
    idx_ethusd = labels.index("ETH-USD")
    idx_btcidr = labels.index("BTC/IDR")
    ai_idx = [labels.index(a) for a in ["NVDA", "MSFT", "TSLA", "GOOGL", "META"]]
    lag0_share = (res0["tsi_post_mean"] / res1["tsi_post_mean"]) * 100.0
    # Convert from system-share (divided by M) to DY12 % (row-sum normalization, x M = 8)
    m_assets = float(len(labels))
    net_dy12 = res1["net_i"] * m_assets
    return {
        "N_eff": res1["n_post"],
        "TSI": res1["tsi_post_mean"],
        "BTC_USD_NET": float(net_dy12[idx_btcusd]),
        "ETH_USD_NET": float(net_dy12[idx_ethusd]),
        "BTC_IDR_NET": float(net_dy12[idx_btcidr]),
        "AI_NET_SUM": float(np.sum(net_dy12[ai_idx])),
        "Lag0_Share_pct": float(lag0_share),
    }


def main():
    main_panel = s1.build_aligned_panel(s1.ASSETS_8, join_ihsg=False)
    labels = [lbl for _, lbl, _, _ in s1.ASSETS_8]
    proxies = s1.make_proxy_matrices(main_panel, labels)

    # Sanity assertion against original s1 function
    for pname in ["log_r2", "abs_r", "r2", "log_Parkinson_8asset_interval"]:
        ref = s1.estimate_tvp_var_gfevd(proxies[pname], p=1, lam=0.99, kap=0.96, H=10, warmup=250, causal_sigma0=True)
        chk = estimate_tvp_var_gfevd_sens(proxies[pname], p=1, lam=0.99, kap=0.96, H=10, warmup=250, sigma0_n=250)
        assert np.max(np.abs(ref["net_i"] - chk["net_i"])) < 1e-12, f"Mismatch on {pname}"

    lambdas = [0.96, 0.98, 0.99, 0.995]
    kappas = [0.94, 0.96, 0.98]
    proxy_specs = [
        ("log_r2", "log(r^2)", proxies["log_r2"]),
        ("abs_r", "|r|", proxies["abs_r"]),
        ("r2", "r^2", proxies["r2"]),
        ("log_Parkinson_8asset_interval", "log(Parkinson)", proxies["log_Parkinson_8asset_interval"]),
    ]

    grid_rows = []
    for pkey, plabel, ymat in proxy_specs:
        # 1) 4 x 3 (lambda, kappa) grid with sigma0_n = 250
        for lam in lambdas:
            for kap in kappas:
                m = eval_cell(ymat, labels, lam=lam, kap=kap, sigma0_n=250)
                grid_rows.append({
                    "Check_Type": "grid_lam_kap",
                    "Proxy_Key": pkey,
                    "Proxy_Label": plabel,
                    "lambda": lam,
                    "kappa": kap,
                    "sigma0_n": 250,
                    "scale_factor": 1.0,
                    **m,
                })
        # 2) Prior check: sigma0_n = 100 at baseline lambda=0.99, kappa=0.96 (and across all 12 cells? Also baseline!)
        m100 = eval_cell(ymat, labels, lam=0.99, kap=0.96, sigma0_n=100)
        grid_rows.append({
            "Check_Type": "prior_sigma0_100",
            "Proxy_Key": pkey,
            "Proxy_Label": plabel,
            "lambda": 0.99,
            "kappa": 0.96,
            "sigma0_n": 100,
            "scale_factor": 1.0,
            **m100,
        })

    # 3) Scale check: r^2 * 1e4 at baseline (0.99, 0.96) and full grid
    r2_scaled = proxies["r2"] * 1e4
    for lam in lambdas:
        for kap in kappas:
            ms = eval_cell(r2_scaled, labels, lam=lam, kap=kap, sigma0_n=250)
            grid_rows.append({
                "Check_Type": "scale_r2_1e4",
                "Proxy_Key": "r2_scaled_1e4",
                "Proxy_Label": "r^2 * 1e4",
                "lambda": lam,
                "kappa": kap,
                "sigma0_n": 250,
                "scale_factor": 1e4,
                **ms,
            })

    df_grid = pd.DataFrame(grid_rows)
    # Assert exact parity with manuscript log(r^2) DY12 baseline
    b0 = df_grid[(df_grid["Check_Type"] == "grid_lam_kap") & (df_grid["Proxy_Key"] == "log_r2") & (df_grid["lambda"] == 0.99) & (df_grid["kappa"] == 0.96)].iloc[0]
    assert abs(b0["TSI"] - 35.127547) < 1e-4
    assert abs(b0["BTC_USD_NET"] - 8.166267) < 1e-4
    assert abs(b0["ETH_USD_NET"] + 3.417351) < 1e-4
    assert abs(b0["BTC_IDR_NET"] - 4.420494) < 1e-4
    assert abs(b0["AI_NET_SUM"] + 9.169410) < 1e-4
    assert abs(b0["Lag0_Share_pct"] - 96.033714) < 1e-4

    out_dir = ROOT_DIR / "results_new"
    out_dir.mkdir(parents=True, exist_ok=True)
    df_grid.to_csv(out_dir / "stage1_estimator_sensitivity_grid.csv", index=False)

    # Summary Table B6 across the 12-cell grid per proxy + prior & scale diagnostics
    sum_rows = []
    for pkey, plabel, _ in proxy_specs:
        sub = df_grid[(df_grid["Check_Type"] == "grid_lam_kap") & (df_grid["Proxy_Key"] == pkey)]
        sub_p100 = df_grid[(df_grid["Check_Type"] == "prior_sigma0_100") & (df_grid["Proxy_Key"] == pkey)].iloc[0]
        base = sub[(sub["lambda"] == 0.99) & (sub["kappa"] == 0.96)].iloc[0]
        row = {
            "Proxy_Key": pkey,
            "Proxy_Label": plabel,
            "n_cells": int(len(sub)),
            "frac_BTCIDR_pos": float((sub["BTC_IDR_NET"] > 0).mean()),
            "count_BTCIDR_pos": int((sub["BTC_IDR_NET"] > 0).sum()),
            "frac_BTCUSD_pos": float((sub["BTC_USD_NET"] > 0).mean()),
        }
        for col in ["TSI", "BTC_USD_NET", "ETH_USD_NET", "BTC_IDR_NET", "AI_NET_SUM", "Lag0_Share_pct"]:
            row[f"{col}_base"] = float(base[col])
            row[f"{col}_min"] = float(sub[col].min())
            row[f"{col}_median"] = float(sub[col].median())
            row[f"{col}_max"] = float(sub[col].max())
            row[f"{col}_sigma100"] = float(sub_p100[col])
            row[f"{col}_sigma100_diff"] = float(sub_p100[col] - base[col])
        sum_rows.append(row)

    # Add r2_scaled_1e4 summary comparison vs raw r2
    sub_r2_raw = df_grid[(df_grid["Check_Type"] == "grid_lam_kap") & (df_grid["Proxy_Key"] == "r2")].reset_index(drop=True)
    sub_r2_scl = df_grid[(df_grid["Check_Type"] == "scale_r2_1e4")].reset_index(drop=True)
    max_diff_r2_scale = {
        col: float(np.max(np.abs(sub_r2_raw[col] - sub_r2_scl[col])))
        for col in ["TSI", "BTC_USD_NET", "ETH_USD_NET", "BTC_IDR_NET", "AI_NET_SUM", "Lag0_Share_pct"]
    }
    base_scl = sub_r2_scl[(sub_r2_scl["lambda"] == 0.99) & (sub_r2_scl["kappa"] == 0.96)].iloc[0]
    row_s = {
        "Proxy_Key": "r2_scaled_1e4",
        "Proxy_Label": "r^2 * 1e4",
        "n_cells": int(len(sub_r2_scl)),
        "frac_BTCIDR_pos": float((sub_r2_scl["BTC_IDR_NET"] > 0).mean()),
        "count_BTCIDR_pos": int((sub_r2_scl["BTC_IDR_NET"] > 0).sum()),
        "frac_BTCUSD_pos": float((sub_r2_scl["BTC_USD_NET"] > 0).mean()),
    }
    for col in ["TSI", "BTC_USD_NET", "ETH_USD_NET", "BTC_IDR_NET", "AI_NET_SUM", "Lag0_Share_pct"]:
        row_s[f"{col}_base"] = float(base_scl[col])
        row_s[f"{col}_min"] = float(sub_r2_scl[col].min())
        row_s[f"{col}_median"] = float(sub_r2_scl[col].median())
        row_s[f"{col}_max"] = float(sub_r2_scl[col].max())
        row_s[f"{col}_sigma100"] = np.nan
        row_s[f"{col}_sigma100_diff"] = np.nan
    sum_rows.append(row_s)

    df_sum = pd.DataFrame(sum_rows)
    df_sum.to_csv(out_dir / "stage1_estimator_sensitivity_summary.csv", index=False)

    print("=== SUMMARY TABLE ===")
    for _, r in df_sum.iterrows():
        print(
            f"{r['Proxy_Label']:16s} | n={int(r['n_cells'])} | "
            f"TSI [{r['TSI_min']:.2f}, {r['TSI_median']:.2f}, {r['TSI_max']:.2f}] | "
            f"BTC-USD [{r['BTC_USD_NET_min']:+.2f}, {r['BTC_USD_NET_median']:+.2f}, {r['BTC_USD_NET_max']:+.2f}] | "
            f"ETH-USD [{r['ETH_USD_NET_min']:+.2f}, {r['ETH_USD_NET_median']:+.2f}, {r['ETH_USD_NET_max']:+.2f}] | "
            f"BTC/IDR [{r['BTC_IDR_NET_min']:+.2f}, {r['BTC_IDR_NET_median']:+.2f}, {r['BTC_IDR_NET_max']:+.2f}] (pos={int(r['count_BTCIDR_pos'])}/{int(r['n_cells'])}) | "
            f"AI_sum [{r['AI_NET_SUM_min']:+.2f}, {r['AI_NET_SUM_median']:+.2f}, {r['AI_NET_SUM_max']:+.2f}] | "
            f"Lag0 [{r['Lag0_Share_pct_min']:.2f}, {r['Lag0_Share_pct_median']:.2f}, {r['Lag0_Share_pct_max']:.2f}]"
        )
    print("\n=== PRIOR CHECK Sigma0(100) vs Sigma0(250) at (0.99, 0.96) ===")
    for _, r in df_sum.iloc[:4].iterrows():
        print(
            f"{r['Proxy_Label']:16s} | TSI: {r['TSI_base']:.2f} vs {r['TSI_sigma100']:.2f} | "
            f"BTC/IDR NET: {r['BTC_IDR_NET_base']:+.2f} vs {r['BTC_IDR_NET_sigma100']:+.2f} | "
            f"BTC-USD NET: {r['BTC_USD_NET_base']:+.2f} vs {r['BTC_USD_NET_sigma100']:+.2f} | "
            f"Lag0: {r['Lag0_Share_pct_base']:.2f} vs {r['Lag0_Share_pct_sigma100']:.2f}"
        )
    print("\n=== SCALE CHECK r^2 vs r^2*1e4 at (0.99, 0.96) and max abs diff across all 12 cells ===")
    r2_base = sub_r2_raw[(sub_r2_raw["lambda"] == 0.99) & (sub_r2_raw["kappa"] == 0.96)].iloc[0]
    print(f"r2 (0.99,0.96)     : TSI={r2_base['TSI']:.4f}, BTC_USD={r2_base['BTC_USD_NET']:+.4f}, ETH_USD={r2_base['ETH_USD_NET']:+.4f}, BTC_IDR={r2_base['BTC_IDR_NET']:+.4f}, AI={r2_base['AI_NET_SUM']:+.4f}, Lag0={r2_base['Lag0_Share_pct']:.4f}")
    print(f"r2*1e4 (0.99,0.96) : TSI={base_scl['TSI']:.4f}, BTC_USD={base_scl['BTC_USD_NET']:+.4f}, ETH_USD={base_scl['ETH_USD_NET']:+.4f}, BTC_IDR={base_scl['BTC_IDR_NET']:+.4f}, AI={base_scl['AI_NET_SUM']:+.4f}, Lag0={base_scl['Lag0_Share_pct']:.4f}")
    print("Max abs diff:", max_diff_r2_scale)


if __name__ == "__main__":
    main()
