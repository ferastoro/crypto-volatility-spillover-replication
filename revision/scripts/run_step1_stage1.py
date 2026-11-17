#!/usr/bin/env python3
"""
Step 1 Stage-1 Revisions (Items 1a, 1b, 1c, 1d, 1e, 1f)
Runs all local CPU re-analyses for Stage-1 connectedness on raw data without modifying original files.
Outputs saved to revision/outputs/*.csv.
"""

import os
import time
import warnings
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.tsa.stattools import adfuller

warnings.filterwarnings("ignore")

BASE_DIR = Path(__file__).resolve().parent.parent.parent
RAW_DIR = BASE_DIR / "data" / "raw"
CLEAN_DIR = BASE_DIR / "data" / "clean"
EXT_DIR = BASE_DIR / "revision" / "data_ext"
INPUTS_V5 = BASE_DIR / "inputs" / "v5"
OUT_DIR = BASE_DIR / "revision" / "outputs"
OUT_DIR.mkdir(parents=True, exist_ok=True)

ASSETS_8 = [
    ("saham_nvda.csv", "NVDA", "AI", "equity"),
    ("saham_msft.csv", "MSFT", "AI", "equity"),
    ("saham_tsla.csv", "TSLA", "AI", "equity"),
    ("saham_googl.csv", "GOOGL", "AI", "equity"),
    ("saham_meta.csv", "META", "AI", "equity"),
    ("crypto_btc_usd.csv", "BTC-USD", "Crypto", "crypto"),
    ("crypto_eth_usd.csv", "ETH-USD", "Crypto", "crypto"),
    ("crypto_btc_idr_indodax.csv", "BTC/IDR", "Crypto", "crypto"),
]

WARMUP_OBS = 250
LAMBDA_FF = 0.99
KAPPA_FF = 0.96
GFEVD_H = 10
N_BOOT = 300
BOOT_BLOCK = 15
N_SHUFFLE = 300
SHUFFLE_BLOCK = 10
RNG_SEED = 20260904


# =============================================================================
# 1. FAST VECTORIZED TVP-VAR & GFEVD
# =============================================================================

def estimate_tvp_var_gfevd(y: np.ndarray, p: int = 1, lam: float = LAMBDA_FF,
                           kap: float = KAPPA_FF, H: int = GFEVD_H,
                           warmup: int = WARMUP_OBS, causal_sigma0: bool = True):
    """
    Estimate Forgetting-Factor TVP-VAR(p) and H-step GFEVD.
    Returns dict with full and post-warmup GFEVD, TSI, TO, FROM, NET, and aligned indices.
    For calendar parity between p=0 and p=1, `discard_rows` removes the first `warmup + p_ref_offset`
    so when comparing p=0 against p=1 on the same return matrix `y` (T rows), both post-warmup windows
    begin at observation index `1 + warmup` (i.e. date index `251` of returns `0..T-1`).
    """
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
    if causal_sigma0:
        init_n = min(warmup, T)
        Sigma = np.cov(y[:init_n].T)
    else:
        Sigma = np.cov(y.T)
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
        # Solve S_t @ K_t^T = PZ_T^T
        K_t = np.linalg.solve(S_t, PZ_T.T).T
        beta = beta + K_t @ err
        IKZ = eye_n - K_t @ Z
        P = IKZ @ P_pred @ IKZ.T + K_t @ Sigma @ K_t.T
        P = 0.5 * (P + P.T)
        beta_hist[t] = beta.reshape(M, K)
        sigma_hist[t] = Sigma

    # Vectorized GFEVD across all T_eff steps
    if p == 0:
        # Static shock with zero lag: MA(0) = I_M, higher Phi_h = 0
        # For p=0, contemporaneous GFEVD only uses h=0 (Sigma_ii^-1 Sigma_ij^2 / sum_j ...)
        diag_s = np.maximum(np.diagonal(sigma_hist, axis1=1, axis2=2), 1e-12)
        num = (sigma_hist ** 2) / diag_s[:, None, :]
        denom = np.maximum(np.sum(num, axis=2, keepdims=True), 1e-12)
        gfevd_all = num / denom
    else:
        # Build companion MA matrices Phi_0 = I, Phi_h = sum_{j=1}^{min(p,h)} A_j Phi_{h-j}
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

    # Index alignment: for p=1, t=0 corresponds to return index 1; discarding first `warmup` VAR steps
    # keeps t in [warmup .. T_eff-1], i.e. return index >= 1 + warmup = 251.
    # For p=0, t=0 corresponds to return index 0; to match exact same calendar return indices (>= 1 + warmup),
    # start at `start_idx = warmup + (1 - obs_offset)`.
    start_idx = warmup + max(0, 1 - obs_offset)
    tsi_all = (gfevd_all.sum(axis=(1, 2)) - np.trace(gfevd_all, axis1=1, axis2=2)) / M * 100.0
    gfevd_post = gfevd_all[start_idx:]
    tsi_post = tsi_all[start_idx:]

    mean_gfevd = gfevd_post.mean(axis=0)
    # Row i = recipient, Col j = transmitter (standardized so row sums to 1)
    from_i = (mean_gfevd.sum(axis=1) - np.diag(mean_gfevd)) / M * 100.0
    to_i = (mean_gfevd.sum(axis=0) - np.diag(mean_gfevd)) / M * 100.0
    net_i = to_i - from_i

    return {
        "gfevd_all": gfevd_all,
        "tsi_all": tsi_all,
        "gfevd_post": gfevd_post,
        "tsi_post": tsi_post,
        "mean_gfevd": mean_gfevd,
        "to_i": to_i,
        "from_i": from_i,
        "net_i": net_i,
        "obs_offset": obs_offset,
        "start_idx": start_idx,
        "y_indices_post": np.arange(obs_offset + start_idx, T),
    }


def compute_net_and_tsi_only(y: np.ndarray, p: int = 1, warmup: int = WARMUP_OBS):
    """Lightweight return of (tsi_mean, net_i) for bootstrap/permutation workers."""
    res = estimate_tvp_var_gfevd(y, p=p, warmup=warmup, causal_sigma0=True)
    return float(res["tsi_post"].mean()), res["net_i"]


def _boot_worker(args):
    seed_i, y, block_len, warmup = args
    rng = np.random.default_rng(seed_i)
    T, M = y.shape
    # Moving block bootstrap (overlapping circular blocks of length `block_len`)
    n_blocks = int(np.ceil(T / block_len))
    starts = rng.integers(0, T, size=n_blocks)
    idx = np.concatenate([np.arange(s, s + block_len) % T for s in starts])[:T]
    y_boot = y[idx]
    return compute_net_and_tsi_only(y_boot, p=1, warmup=warmup)


def _shuffle_worker(args):
    seed_i, y, block_len, warmup = args
    rng = np.random.default_rng(seed_i)
    T, M = y.shape
    y_shuf = np.empty_like(y)
    # Break cross-series contemporaneous/lag dependence while preserving within-series block dynamics
    for m in range(M):
        n_blocks = int(np.ceil(T / block_len))
        # Partition into non-overlapping blocks of length ~block_len + circular random shift
        shift = int(rng.integers(0, T))
        col_shifted = np.roll(y[:, m], shift)
        blocks = [col_shifted[k : k + block_len] for k in range(0, T, block_len)]
        perm = rng.permutation(len(blocks))
        y_shuf[:, m] = np.concatenate([blocks[k] for k in perm])[:T]
    tsi_m, _ = compute_net_and_tsi_only(y_shuf, p=1, warmup=warmup)
    return tsi_m


# =============================================================================
# 2. DATA CONSTRUCTION FROM RAW FILES
# =============================================================================

def load_raw_series(filepath: Path) -> pd.DataFrame:
    df = pd.read_csv(filepath)
    df["Date"] = pd.to_datetime(df["Date"])
    df = df.sort_values("Date").drop_duplicates("Date").reset_index(drop=True)
    return df


def build_aligned_panel(asset_specs, join_ihsg: bool = False, join_legacy_1556: bool = False,
                        extra_specs=None, fx_adjust_btcidr: bool = False):
    """
    Build clean price/return/Parkinson panel from raw CSVs on the equity trading calendar
    (or joint US-JKSE calendar when `join_ihsg=True`, plus VIX/USDIDR when `join_legacy_1556=True`).
    Continuous crypto High/Low over multi-day equity gaps are aggregated (max High, min Low)
    on `(prev_trading_date, curr_trading_date]` matching close-to-close return interval.
    """
    specs = list(asset_specs) + (list(extra_specs) if extra_specs else [])
    raw_dfs = {}
    for fname, label, _, _ in specs:
        fpath = EXT_DIR / fname if (EXT_DIR / fname).exists() and not (RAW_DIR / fname).exists() else (RAW_DIR / fname)
        if not fpath.exists():
            raise SystemExit(f"Raw vendor file missing: {fpath}. Run src/collect_data.py first.")
        raw_dfs[label] = load_raw_series(fpath)

    # Calendar defined by intersection of all equity/index assets (plus ihsg if requested)
    cal_labels = [lbl for _, lbl, _, kind in specs if kind != "crypto"]
    common_dates = set(raw_dfs[cal_labels[0]]["Date"])
    for lbl in cal_labels[1:]:
        common_dates &= set(raw_dfs[lbl]["Date"])
    if join_ihsg:
        ihsg = load_raw_series(RAW_DIR / "konteks_ihsg.csv")
        common_dates &= set(ihsg["Date"])
    if join_legacy_1556:
        common_dates &= set(load_raw_series(RAW_DIR / "konteks_vix.csv")["Date"])
        common_dates &= set(load_raw_series(RAW_DIR / "konteks_usdidr.csv")["Date"])

    # Also ensure every crypto asset exists on those dates
    for _, lbl, _, kind in specs:
        if kind == "crypto":
            common_dates &= set(raw_dfs[lbl]["Date"])

    cal_dates = pd.Series(sorted(common_dates), name="Date")
    cal_df = pd.DataFrame({"Date": cal_dates})

    # Optional FX adjustment for BTC/IDR
    if fx_adjust_btcidr:
        usdidr = load_raw_series(RAW_DIR / "konteks_usdidr.csv").set_index("Date")
        btc_idr_raw = raw_dfs["BTC/IDR"].copy()
        # Reindex USD/IDR forward-fill onto all daily BTC/IDR dates so weekend/gap days carry Friday rate
        all_days = btc_idr_raw[["Date"]].merge(usdidr[["Close"]].rename(columns={"Close": "FX"}), on="Date", how="left")
        all_days["FX"] = all_days["FX"].ffill().bfill()
        btc_idr_raw["Close"] = btc_idr_raw["Close"] / all_days["FX"]
        btc_idr_raw["High"] = btc_idr_raw["High"] / all_days["FX"]
        btc_idr_raw["Low"] = btc_idr_raw["Low"] / all_days["FX"]
        raw_dfs["BTC/IDR"] = btc_idr_raw

    # Map each raw date for interval High/Low aggregation onto next calendar trading date
    cal_idx_map = {d: i for i, d in enumerate(cal_dates)}
    returns_dict = {"Date": cal_dates.iloc[1:].values}
    park_interval_dict = {"Date": cal_dates.iloc[1:].values}
    park_single_dict = {"Date": cal_dates.iloc[1:].values}
    close_dict = {"Date": cal_dates.values}

    eps = 1e-10
    coef_p = 4.0 * np.log(2.0)

    for _, lbl, _, kind in specs:
        df_a = raw_dfs[lbl].sort_values("Date").reset_index(drop=True)
        # Close on calendar dates
        sub_cal = cal_df.merge(df_a[["Date", "Close", "High", "Low"]], on="Date", how="left")
        close_dict[lbl] = sub_cal["Close"].values
        lr = np.log(sub_cal["Close"].values[1:] / sub_cal["Close"].values[:-1])
        returns_dict[lbl] = lr

        # Single trading-day Parkinson on calendar dates[1:]
        hl_s = np.maximum(sub_cal["High"].values[1:] / np.maximum(sub_cal["Low"].values[1:], 1e-12), 1.0)
        park_single_dict[lbl] = (np.log(hl_s) ** 2) / coef_p

        if kind == "crypto":
            # Interval (prev_cal_date, curr_cal_date] cumulative High/Low
            # Assign each raw row in (cal_dates[0], cal_dates[-1]] to the index k in 1..len-1
            d_vals = cal_dates.values
            raw_dates = df_a["Date"].values
            pos = np.searchsorted(d_vals, raw_dates, side="left")
            # A raw date d falls into interval (d_{k-1}, d_k] iff d_{k-1} < d <= d_k -> pos == k (when d == d_k) or pos == k (when d_{k-1} < d < d_k)
            valid = (raw_dates > d_vals[0]) & (raw_dates <= d_vals[-1])
            df_int = pd.DataFrame({
                "k": pos[valid],
                "High": df_a["High"].values[valid],
                "Low": df_a["Low"].values[valid],
            })
            agg = df_int.groupby("k").agg(High=("High", "max"), Low=("Low", "min")).reindex(range(1, len(cal_dates)))
            hl_int = np.maximum(agg["High"].values / np.maximum(agg["Low"].values, 1e-12), 1.0)
            park_interval_dict[lbl] = (np.log(hl_int) ** 2) / coef_p
        else:
            park_interval_dict[lbl] = park_single_dict[lbl]

    df_ret = pd.DataFrame(returns_dict)
    df_park_int = pd.DataFrame(park_interval_dict)
    df_park_sng = pd.DataFrame(park_single_dict)
    df_close = pd.DataFrame(close_dict)

    # Holiday indicator on return dates
    ihsg_dates = set(load_raw_series(RAW_DIR / "konteks_ihsg.csv")["Date"])
    df_ret["is_jkse_holiday"] = (~df_ret["Date"].isin(ihsg_dates)).astype(int)

    return {
        "df_ret": df_ret,
        "df_park_int": df_park_int,
        "df_park_sng": df_park_sng,
        "df_close": df_close,
        "specs": specs,
    }


def make_proxy_matrices(panel, labels):
    r_mat = panel["df_ret"][labels].values
    p_int = panel["df_park_int"][labels].values
    p_sng = panel["df_park_sng"][labels].values
    eps = 1e-10
    proxies = {
        "log_r2": np.log(r_mat ** 2 + eps),
        "abs_r": np.abs(r_mat),
        "r2": r_mat ** 2,
        "log_Parkinson_8asset_interval": np.log(np.maximum(p_int, 0.0) + eps),
        "log_Parkinson_8asset_sameday": np.log(np.maximum(p_sng, 0.0) + eps),
    }
    # Also construct mixed proxy matching legacy audit script for exact provenance comparison (only BTC/IDR uses Parkinson)
    if "BTC/IDR" in labels:
        mixed = proxies["log_r2"].copy()
        j_btc = labels.index("BTC/IDR")
        mixed[:, j_btc] = proxies["log_Parkinson_8asset_interval"][:, j_btc]
        proxies["log_Parkinson_mixed_legacy"] = mixed
    return proxies


def main():
    t_start = time.time()
    print("=== Step 1 Stage-1 Re-runs ===")

    # -------------------------------------------------------------------------
    # Build Main (1,677) and Robustness (1,556) panels
    # -------------------------------------------------------------------------
    main_panel = build_aligned_panel(ASSETS_8, join_ihsg=False)
    rob1556_panel = build_aligned_panel(ASSETS_8, join_ihsg=True, join_legacy_1556=True)
    labels_8 = [lbl for _, lbl, _, _ in ASSETS_8]

    print(f"Main US-calendar prices={len(main_panel['df_close'])}, returns={len(main_panel['df_ret'])} "
          f"[{main_panel['df_ret']['Date'].min().date()}..{main_panel['df_ret']['Date'].max().date()}], "
          f"JKSE holidays inside return dates={main_panel['df_ret']['is_jkse_holiday'].sum()}")
    print(f"Robustness JKSE-joined prices={len(rob1556_panel['df_close'])}, returns={len(rob1556_panel['df_ret'])} "
          f"[{rob1556_panel['df_ret']['Date'].min().date()}..{rob1556_panel['df_ret']['Date'].max().date()}]")

    # -------------------------------------------------------------------------
    # 1f. Reconciliation (2437 vs 2440, SD/mean(r^2) samples, Feb 5-6, ADF on proxies)
    # -------------------------------------------------------------------------
    raw_btcidr = load_raw_series(RAW_DIR / "crypto_btc_idr_indodax.csv")
    raw_btcusd = load_raw_series(RAW_DIR / "crypto_btc_usd.csv")
    raw_usdidr = load_raw_series(RAW_DIR / "konteks_usdidr.csv")
    raw_nvda = load_raw_series(RAW_DIR / "saham_nvda.csv")
    dl_df = pd.read_csv(CLEAN_DIR / "dataset_deep_learning.csv")
    dl_df["Date"] = pd.to_datetime(dl_df["Date"])
    garch_df = pd.read_csv(CLEAN_DIR / "dataset_garch_baseline.csv")
    garch_df["Date"] = pd.to_datetime(garch_df["Date"])
    ens_df = pd.read_csv(BASE_DIR / "results" / "v6" / "v6_ensemble_predictions.csv")
    ens_df["real_date"] = pd.to_datetime(ens_df["real_date"])

    # Calendar gap math
    ret_start = main_panel["df_ret"]["Date"].min()
    ret_end = main_panel["df_ret"]["Date"].max()
    cal_span_ret = (ret_end - ret_start).days + 1
    cal_span_price = (main_panel["df_close"]["Date"].max() - main_panel["df_close"]["Date"].min()).days + 1
    non_us_ret_days = cal_span_ret - len(main_panel["df_ret"])
    non_us_raw_days = len(raw_btcidr) - len(main_panel["df_close"])

    recon_rows = [
        {"Metric": "Raw Indodax/BTC-USD/ETH-USD 24/7 price rows", "Start": str(raw_btcidr["Date"].min().date()), "End": str(raw_btcidr["Date"].max().date()), "Count": len(raw_btcidr), "Notes": "2440 daily UTC bars -> 2439 24h returns"},
        {"Metric": "US equity price dates (all 5 AI equities)", "Start": str(raw_nvda["Date"].min().date()), "End": str(raw_nvda["Date"].max().date()), "Count": len(main_panel["df_close"]), "Notes": "1678 US trading dates"},
        {"Metric": "Raw non-US-trading calendar days in full raw span", "Start": str(raw_btcidr["Date"].min().date()), "End": str(raw_btcidr["Date"].max().date()), "Count": non_us_raw_days, "Notes": f"2440 - 1678 = {non_us_raw_days} ({non_us_raw_days/len(raw_btcidr)*100:.2f}%)"},
        {"Metric": "Return-aligned US equity trading dates (Stage 1 main)", "Start": str(ret_start.date()), "End": str(ret_end.date()), "Count": len(main_panel["df_ret"]), "Notes": "1677 close-to-close return dates (Stage 1 main & Stage 2 full)"},
        {"Metric": "Calendar days in return span 2020-01-03..2026-09-04", "Start": str(ret_start.date()), "End": str(ret_end.date()), "Count": cal_span_ret, "Notes": f"1677 US trading + {non_us_ret_days} weekends/US holidays ({non_us_ret_days/cal_span_ret*100:.2f}%)"},
        {"Metric": "Calendar days in US price span 2020-01-02..2026-09-04", "Start": str(main_panel["df_close"]["Date"].min().date()), "End": str(main_panel["df_close"]["Date"].max().date()), "Count": cal_span_price, "Notes": "2438 calendar days; 2440 raw includes 2020-01-01 and 2026-09-05"},
        {"Metric": "JKSE-intersected Stage-1 robustness return dates", "Start": str(rob1556_panel["df_ret"]["Date"].min().date()), "End": str(rob1556_panel["df_ret"]["Date"].max().date()), "Count": len(rob1556_panel["df_ret"]), "Notes": f"Drops {len(main_panel['df_ret']) - len(rob1556_panel['df_ret'])} US-open/JKSE-holiday days"},
        {"Metric": "Post-warmup Stage-1 main return dates (discard 250)", "Start": str(main_panel["df_ret"]["Date"].iloc[251].date()), "End": str(ret_end.date()), "Count": len(main_panel["df_ret"]) - 1 - WARMUP_OBS, "Notes": "p=1 VAR has 1676 steps; discarding first 250 leaves 1426 effective dates"},
        {"Metric": "Stage-2 causal SPILL feature dates (raw un-ffilled)", "Start": "2020-01-06", "End": str(ret_end.date()), "Count": len(pd.read_csv(INPUTS_V5 / "tvp_var_dynamic_features_causal.csv")), "Notes": "JKSE-joined p=1 output forward-filled across 121 JKSE holidays onto Stage-2 US calendar"},
        {"Metric": "Stage-2 OOS unique cutoffs / h=1 evaluation dates", "Start": str(ens_df["real_date"].min().date()), "End": "2026-08-21", "Count": 335, "Notes": "335 rolling origins; 344 unique target dates; 3350 origin-horizon pairs"},
    ]
    pd.DataFrame(recon_rows).to_csv(OUT_DIR / "step1f_sample_reconciliation.csv", index=False)

    # Descriptive stats across all samples
    desc_rows = []
    # (A) All 8 assets on Main Stage-1 N=1,677 (close-to-close)
    for lbl in labels_8:
        r = main_panel["df_ret"][lbl].values
        r2 = r ** 2
        p_int = main_panel["df_park_int"][lbl].values
        desc_rows.append({
            "Sample": "Main Stage-1 (US cal, close-to-close)",
            "N": len(r),
            "Asset": lbl,
            "Mean_r": float(np.mean(r)),
            "SD_r": float(np.std(r, ddof=1)),
            "Skew_r": float(stats.skew(r)),
            "Kurt_excess_r": float(stats.kurtosis(r)),
            "Mean_r2_1e4": float(np.mean(r2) * 1e4),
            "SD_r2_1e4": float(np.std(r2, ddof=1) * 1e4),
            "Skew_r2": float(stats.skew(r2)),
            "Max_r2_1e4": float(np.max(r2) * 1e4),
            "Mean_Parkinson_1e4": float(np.mean(p_int) * 1e4),
        })
    # (B) dataset_deep_learning daily-only crypto return (N=1,677) vs full weekend-accumulated
    for col, lbl in [("ret_crypto_btc_idr_indodax", "BTC/IDR (DL daily-only col)"),
                     ("ret_crypto_btc_usd", "BTC-USD (DL daily-only col)"),
                     ("ret_crypto_eth_usd", "ETH-USD (DL daily-only col)")]:
        r = dl_df[col].values
        r2 = r ** 2
        desc_rows.append({
            "Sample": "Stage-2 DL Target (US cal, daily-only no weekend gap)",
            "N": len(r),
            "Asset": lbl,
            "Mean_r": float(np.mean(r)),
            "SD_r": float(np.std(r, ddof=1)),
            "Skew_r": float(stats.skew(r)),
            "Kurt_excess_r": float(stats.kurtosis(r)),
            "Mean_r2_1e4": float(np.mean(r2) * 1e4),
            "SD_r2_1e4": float(np.std(r2, ddof=1) * 1e4),
            "Skew_r2": float(stats.skew(r2)),
            "Max_r2_1e4": float(np.max(r2) * 1e4),
            "Mean_Parkinson_1e4": np.nan,
        })
    # (C) JKSE-joined N=1,556 Stage-1 robustness / legacy GARCH baseline sample
    for lbl in labels_8:
        r = rob1556_panel["df_ret"][lbl].values
        r2 = r ** 2
        desc_rows.append({
            "Sample": "Robustness JKSE-joined (US+JKSE cal)",
            "N": len(r),
            "Asset": lbl,
            "Mean_r": float(np.mean(r)),
            "SD_r": float(np.std(r, ddof=1)),
            "Skew_r": float(stats.skew(r)),
            "Kurt_excess_r": float(stats.kurtosis(r)),
            "Mean_r2_1e4": float(np.mean(r2) * 1e4),
            "SD_r2_1e4": float(np.std(r2, ddof=1) * 1e4),
            "Skew_r2": float(stats.skew(r2)),
            "Max_r2_1e4": float(np.max(r2) * 1e4),
            "Mean_Parkinson_1e4": float(np.mean(rob1556_panel["df_park_int"][lbl].values) * 1e4),
        })
    # (D) Stage-2 OOS samples: h=1 (335), h=10 (335), and stacked 3,350
    sub_dl_oos = dl_df[dl_df["Date"] >= "2025-04-24"].copy()
    sub_dl_oos_h1 = ens_df[ens_df["Config"] == "TFT_SPILL_T0"].copy()
    sub_dl_oos_h1["h"] = sub_dl_oos_h1.groupby("cutoff").cumcount() + 1
    for h_lbl, mask_h in [("OOS T0 target h=1 (N=335)", sub_dl_oos_h1["h"] == 1),
                          ("OOS T0 target h=10 (N=335)", sub_dl_oos_h1["h"] == 10),
                          ("OOS T0 stacked h=1..10 (N=3350)", np.ones(len(sub_dl_oos_h1), dtype=bool))]:
        y_vals = sub_dl_oos_h1.loc[mask_h, "y"].values
        desc_rows.append({
            "Sample": h_lbl,
            "N": len(y_vals),
            "Asset": "BTC/IDR (T0 target r2)",
            "Mean_r": np.nan,
            "SD_r": np.nan,
            "Skew_r": np.nan,
            "Kurt_excess_r": np.nan,
            "Mean_r2_1e4": float(np.mean(y_vals) * 1e4),
            "SD_r2_1e4": float(np.std(y_vals, ddof=1) * 1e4),
            "Skew_r2": float(stats.skew(y_vals)),
            "Max_r2_1e4": float(np.max(y_vals) * 1e4),
            "Mean_Parkinson_1e4": np.nan,
        })
    # Also train/val splits on dl_df['ret_crypto_btc_idr_indodax']**2 matching exact OOS boundaries
    first_oos_date = sub_dl_oos_h1.loc[sub_dl_oos_h1["h"] == 1, "real_date"].min()
    dl_train_val = dl_df[dl_df["Date"] < first_oos_date].copy()
    dl_val = dl_train_val.tail(167)
    dl_tr = dl_train_val.iloc[:-167]
    for slbl, df_s in [("Stage-2 DL Train (1174 dates)", dl_tr), ("Stage-2 DL Val (167 dates)", dl_val),
                       ("Stage-2 DL Unique OOS (336 dates from 2025-04-24..2026-09-04)", dl_df[dl_df["Date"] >= first_oos_date])]:
        r = df_s["ret_crypto_btc_idr_indodax"].values
        r2 = r ** 2
        desc_rows.append({
            "Sample": slbl,
            "N": len(r),
            "Asset": "BTC/IDR (DL daily-only col)",
            "Mean_r": float(np.mean(r)),
            "SD_r": float(np.std(r, ddof=1)),
            "Skew_r": float(stats.skew(r)),
            "Kurt_excess_r": float(stats.kurtosis(r)),
            "Mean_r2_1e4": float(np.mean(r2) * 1e4),
            "SD_r2_1e4": float(np.std(r2, ddof=1) * 1e4),
            "Skew_r2": float(stats.skew(r2)),
            "Max_r2_1e4": float(np.max(r2) * 1e4),
            "Mean_Parkinson_1e4": np.nan,
        })

    pd.DataFrame(desc_rows).to_csv(OUT_DIR / "step1f_descriptive_stats.csv", index=False)

    # Feb 5-6 raw audit
    feb_rows = []
    for fname, lbl, _, _ in ASSETS_8 + [("konteks_usdidr.csv", "USD/IDR", "FX", "fx")]:
        df_a = load_raw_series(RAW_DIR / fname)
        df_a["log_ret"] = np.log(df_a["Close"] / df_a["Close"].shift(1))
        df_a["r2"] = df_a["log_ret"] ** 2
        sub = df_a[(df_a["Date"] >= "2026-02-04") & (df_a["Date"] <= "2026-02-07")]
        for _, row in sub.iterrows():
            feb_rows.append({
                "Asset": lbl,
                "Date": str(row["Date"].date()),
                "Open": row["Open"],
                "High": row["High"],
                "Low": row["Low"],
                "Close": row["Close"],
                "Volume": row["Volume"],
                "Log_Return": row["log_ret"],
                "Log_Return_Pct": row["log_ret"] * 100.0,
                "r2": row["r2"],
                "r2_1e4": row["r2"] * 1e4,
            })
    pd.DataFrame(feb_rows).to_csv(OUT_DIR / "step1f_feb5_6_raw_check.csv", index=False)

    # ADF stationarity tests on returns AND all 4 proxies (N=1677 full and post-warmup N=1426)
    main_proxies = make_proxy_matrices(main_panel, labels_8)
    adf_rows = []
    series_map = {"log_return": main_panel["df_ret"][labels_8].values}
    for pname in ["log_r2", "abs_r", "r2", "log_Parkinson_8asset_interval", "log_Parkinson_8asset_sameday"]:
        series_map[pname] = main_proxies[pname]

    for sname, mat in series_map.items():
        for wname, sl in [("Full_N1677", slice(0, None)), ("PostWarmup_N1426", slice(251, None))]:
            for j, lbl in enumerate(labels_8):
                arr = mat[sl, j]
                stat, pval, usedlag, nobs, crit, _ = adfuller(arr, regression="c", autolag="AIC")
                adf_rows.append({
                    "Sample": wname,
                    "Series": sname,
                    "Asset": lbl,
                    "N_obs": int(nobs),
                    "Lags": int(usedlag),
                    "ADF_Stat": float(stat),
                    "p_value": float(pval),
                    "Crit_1pct": float(crit["1%"]),
                    "Crit_5pct": float(crit["5%"]),
                    "Stationary_1pct": bool(pval < 0.01),
                })
    pd.DataFrame(adf_rows).to_csv(OUT_DIR / "step1f_adf_tests.csv", index=False)
    print("Saved Step 1f reconciliation, descriptive stats, Feb 5-6 check, ADF CSVs.")

    # -------------------------------------------------------------------------
    # 1a, 1b, 1d: Core Stage-1 Re-runs after 250-obs warm-up discard across ALL proxies
    # -------------------------------------------------------------------------
    proxy_list = ["log_r2", "abs_r", "r2", "log_Parkinson_8asset_interval",
                  "log_Parkinson_8asset_sameday", "log_Parkinson_mixed_legacy"]
    sample_configs = [
        ("Main_US_1677", main_panel, labels_8, True),
        ("Robust_JKSE_1556", rob1556_panel, labels_8, False),
    ]

    summary_rows = []
    net_rows = []
    pairwise_rows = []
    block_rows = []
    gfevd_rows = []
    tsi_ts_df = pd.DataFrame({"Date": main_panel["df_ret"]["Date"].iloc[251:].values})

    ai_idx = [0, 1, 2, 3, 4]
    cr_idx = [5, 6, 7]
    btcusd_i = 5
    btcidr_i = 7

    for samp_name, panel, lbls, is_main in sample_configs:
        proxies = make_proxy_matrices(panel, lbls)
        dates_ret = panel["df_ret"]["Date"].values
        for pname in proxy_list:
            y = proxies[pname]
            res1 = estimate_tvp_var_gfevd(y, p=1, warmup=WARMUP_OBS, causal_sigma0=True)
            res0 = estimate_tvp_var_gfevd(y, p=0, warmup=WARMUP_OBS, causal_sigma0=True)
            # Diagnostic check vs legacy full-sample Sigma_0 on Main log_r2
            if is_main and pname == "log_r2":
                res1_fullcov = estimate_tvp_var_gfevd(y, p=1, warmup=WARMUP_OBS, causal_sigma0=False)
                max_gdiff = np.max(np.abs(res1["gfevd_post"] - res1_fullcov["gfevd_post"]))
                print(f"Check causal Sigma_0 vs legacy full-sample Sigma_0 post-warmup max GFEVD diff = {max_gdiff:.2e}")

            post_dates = dates_ret[res1["y_indices_post"]]
            tsi_p = res1["tsi_post"]
            tsi_pre = res1["tsi_all"][:WARMUP_OBS]
            tsi_full = res1["tsi_all"]
            tsi_p0 = res0["tsi_post"]

            min_k = int(np.argmin(tsi_p))
            max_k = int(np.argmax(tsi_p))

            if is_main and pname in ["log_r2", "abs_r", "r2", "log_Parkinson_8asset_interval"]:
                tsi_ts_df[f"TSI_{pname}"] = tsi_p
                tsi_ts_df[f"NET_BTCI_{pname}"] = (res1["gfevd_post"][:, :, btcidr_i].sum(axis=1) - res1["gfevd_post"][:, btcidr_i, :].sum(axis=1)) / len(lbls) * 100.0

            summary_rows.append({
                "Sample": samp_name,
                "Proxy": pname,
                "N_returns": len(y),
                "N_post_warmup": len(tsi_p),
                "Post_Start": str(pd.Timestamp(post_dates[0]).date()),
                "Post_End": str(pd.Timestamp(post_dates[-1]).date()),
                "TSI_full_mean": float(np.mean(tsi_full)),
                "TSI_warmup_first250_mean": float(np.mean(tsi_pre)),
                "TSI_post_mean": float(np.mean(tsi_p)),
                "TSI_post_sd": float(np.std(tsi_p, ddof=1)),
                "TSI_post_min": float(np.min(tsi_p)),
                "TSI_post_min_date": str(pd.Timestamp(post_dates[min_k]).date()),
                "TSI_post_max": float(np.max(tsi_p)),
                "TSI_post_max_date": str(pd.Timestamp(post_dates[max_k]).date()),
                "TSI_p0_post_mean": float(np.mean(tsi_p0)),
                "TSI_p0_ratio_pct": float(np.mean(tsi_p0) / np.mean(tsi_p) * 100.0),
                "TSI_p1_minus_p0": float(np.mean(tsi_p) - np.mean(tsi_p0)),
            })

            mg = res1["mean_gfevd"]
            for j, lbl in enumerate(lbls):
                net_rows.append({
                    "Sample": samp_name,
                    "Proxy": pname,
                    "Asset": lbl,
                    "TO_pct": float(res1["to_i"][j]),
                    "FROM_pct": float(res1["from_i"][j]),
                    "NET_pct": float(res1["net_i"][j]),
                    "Own_diag_pct": float(mg[j, j] * 100.0),
                })

            # Pairwise FEV shares (mg[i, j] = share of asset i's H-step FEV due to shock in j, in [0,1])
            btcusd_to_btcidr = float(mg[btcidr_i, btcusd_i] * 100.0)
            btcidr_to_btcusd = float(mg[btcusd_i, btcidr_i] * 100.0)
            ai_to_btcusd = float(mg[btcusd_i, ai_idx].sum() * 100.0)
            ai_to_btcidr = float(mg[btcidr_i, ai_idx].sum() * 100.0)
            btcidr_to_ai_mean = float(mg[ai_idx, btcidr_i].mean() * 100.0)
            btcidr_to_ai_sum_fev = float(mg[ai_idx, btcidr_i].sum()) # sum across 5 AI rows (fractional)
            btcusd_to_ai_mean = float(mg[ai_idx, btcusd_i].mean() * 100.0)
            pairwise_rows.append({
                "Sample": samp_name,
                "Proxy": pname,
                "BTCUSD_to_BTCIDR_pct": btcusd_to_btcidr,
                "BTCIDR_to_BTCUSD_pct": btcidr_to_btcusd,
                "AI_sum_to_BTCUSD_pct": ai_to_btcusd,
                "AI_sum_to_BTCIDR_pct": ai_to_btcidr,
                "BTCIDR_to_AI_mean_row_pct": btcidr_to_ai_mean,
                "BTCUSD_to_AI_mean_row_pct": btcusd_to_ai_mean,
            })

            # 1d: Block decomposition of total off-diagonal connectedness
            # Total off-diagonal sum = sum_{i!=j} mg[i,j]; TSI = total_off / M * 100
            mg_off = mg.copy()
            np.fill_diagonal(mg_off, 0.0)
            total_off = float(mg_off.sum())
            M_cur = len(lbls)
            within_ai = float(mg_off[np.ix_(ai_idx, ai_idx)].sum())
            within_cr = float(mg_off[np.ix_(cr_idx, cr_idx)].sum())
            ai_to_cr = float(mg_off[np.ix_(cr_idx, ai_idx)].sum())
            cr_to_ai = float(mg_off[np.ix_(ai_idx, cr_idx)].sum())
            block_rows.append({
                "Sample": samp_name,
                "Proxy": pname,
                "TSI_post_mean": float(total_off / M_cur * 100.0),
                "Within_AI_pts": float(within_ai / M_cur * 100.0),
                "Within_AI_share_of_TSI_pct": float(within_ai / total_off * 100.0),
                "Within_Crypto_pts": float(within_cr / M_cur * 100.0),
                "Within_Crypto_share_of_TSI_pct": float(within_cr / total_off * 100.0),
                "AI_to_Crypto_pts": float(ai_to_cr / M_cur * 100.0),
                "AI_to_Crypto_share_of_TSI_pct": float(ai_to_cr / total_off * 100.0),
                "Crypto_to_AI_pts": float(cr_to_ai / M_cur * 100.0),
                "Crypto_to_AI_share_of_TSI_pct": float(cr_to_ai / total_off * 100.0),
                "Net_Cross_AI_to_Crypto_pts": float((ai_to_cr - cr_to_ai) / M_cur * 100.0),
            })

            if is_main and pname in ["log_r2", "abs_r", "r2", "log_Parkinson_8asset_interval"]:
                for ri, rl in enumerate(lbls):
                    row_d = {"Proxy": pname, "Recipient": rl}
                    for ci, cl in enumerate(lbls):
                        row_d[cl] = float(mg[ri, ci] * 100.0)
                    gfevd_rows.append(row_d)

    pd.DataFrame(summary_rows).to_csv(OUT_DIR / "step1a_stage1_summary.csv", index=False)
    pd.DataFrame(net_rows).to_csv(OUT_DIR / "step1a_net_spillovers.csv", index=False)
    pd.DataFrame(pairwise_rows).to_csv(OUT_DIR / "step1a_pairwise_fev.csv", index=False)
    pd.DataFrame(block_rows).to_csv(OUT_DIR / "step1d_block_decomposition.csv", index=False)
    pd.DataFrame(gfevd_rows).to_csv(OUT_DIR / "step1a_full_gfevd_matrices.csv", index=False)
    tsi_ts_df.to_csv(OUT_DIR / "step1a_tsi_timeseries_postwarmup.csv", index=False)

    print("\n--- Post-Warmup Summary (Main 1677 vs Robust 1556) ---")
    print(pd.DataFrame(summary_rows)[["Sample", "Proxy", "TSI_full_mean", "TSI_warmup_first250_mean", "TSI_post_mean", "TSI_p0_post_mean", "TSI_p0_ratio_pct"]].to_string(index=False))
    print("\n--- Post-Warmup NET Spillovers (Main 1677) ---")
    df_net_main = pd.DataFrame(net_rows)
    piv = df_net_main[df_net_main["Sample"] == "Main_US_1677"].pivot(index="Asset", columns="Proxy", values="NET_pct")
    print(piv[["log_r2", "abs_r", "r2", "log_Parkinson_8asset_interval", "log_Parkinson_8asset_sameday", "log_Parkinson_mixed_legacy"]].round(2).to_string())

    # -------------------------------------------------------------------------
    # 1a Shuffle Test + 1c Moving-Block Bootstrap across 4 proxies (Main 1,677)
    # Use `log_Parkinson_8asset_interval` as canonical Parkinson + also run mixed check if needed
    # -------------------------------------------------------------------------
    canon_proxies = ["log_r2", "abs_r", "r2", "log_Parkinson_8asset_interval"]
    main_proxies_dict = make_proxy_matrices(main_panel, labels_8)

    shuf_rows = []
    boot_rows = []
    boot_dist_rows = []
    n_workers = min(8, os.cpu_count() or 4)

    with ProcessPoolExecutor(max_workers=n_workers) as ex:
        # 1a Circular Block Shuffle
        print(f"\nRunning Circular Block-Shuffle Permutation Test ({N_SHUFFLE} reps/proxy, block={SHUFFLE_BLOCK}) on {n_workers} cores...")
        t_s0 = time.time()
        for pi, pname in enumerate(canon_proxies):
            y = main_proxies_dict[pname]
            obs_tsi = float(pd.DataFrame(summary_rows).query("Sample == 'Main_US_1677' and Proxy == @pname")["TSI_post_mean"].iloc[0])
            tasks = [(RNG_SEED + 10000 * pi + b, y, SHUFFLE_BLOCK, WARMUP_OBS) for b in range(N_SHUFFLE)]
            null_vals = np.array(list(ex.map(_shuffle_worker, tasks)))
            p_emp = float((1 + np.sum(null_vals >= obs_tsi)) / (N_SHUFFLE + 1))
            shuf_rows.append({
                "Proxy": pname,
                "N_reps": N_SHUFFLE,
                "Block_len": SHUFFLE_BLOCK,
                "Observed_TSI_post": obs_tsi,
                "Null_mean": float(np.mean(null_vals)),
                "Null_sd": float(np.std(null_vals, ddof=1)),
                "Null_p05": float(np.percentile(null_vals, 5)),
                "Null_p95": float(np.percentile(null_vals, 95)),
                "Null_p99": float(np.percentile(null_vals, 99)),
                "Null_max": float(np.max(null_vals)),
                "Excess_TSI": float(obs_tsi - np.mean(null_vals)),
                "Empirical_p": p_emp,
            })
        print(f"Permutation test done in {time.time() - t_s0:.1f}s")

        # 1c Moving-Block Bootstrap
        print(f"Running Moving-Block Bootstrap ({N_BOOT} reps/proxy, block={BOOT_BLOCK}) on {n_workers} cores...")
        t_b0 = time.time()
        for pi, pname in enumerate(canon_proxies):
            y = main_proxies_dict[pname]
            point_nets = df_net_main.query("Sample == 'Main_US_1677' and Proxy == @pname").set_index("Asset")["NET_pct"]
            point_tsi = float(pd.DataFrame(summary_rows).query("Sample == 'Main_US_1677' and Proxy == @pname")["TSI_post_mean"].iloc[0])
            tasks = [(RNG_SEED + 50000 + 10000 * pi + b, y, BOOT_BLOCK, WARMUP_OBS) for b in range(N_BOOT)]
            res_list = list(ex.map(_boot_worker, tasks))
            tsis = np.array([r[0] for r in res_list])
            nets = np.array([r[1] for r in res_list]) # (N_BOOT, 8)

            boot_dist_rows.append({
                "Proxy": pname,
                "Metric": "TSI_post_mean",
                "Point": point_tsi,
                "Boot_mean": float(np.mean(tsis)),
                "Boot_sd": float(np.std(tsis, ddof=1)),
                "P025": float(np.percentile(tsis, 2.5)),
                "P05": float(np.percentile(tsis, 5)),
                "P50": float(np.percentile(tsis, 50)),
                "P95": float(np.percentile(tsis, 95)),
                "P975": float(np.percentile(tsis, 97.5)),
            })
            for j, lbl in enumerate(labels_8):
                arr = nets[:, j]
                p05, p95 = np.percentile(arr, [5, 95])
                p025, p975 = np.percentile(arr, [2.5, 97.5])
                boot_rows.append({
                    "Proxy": pname,
                    "Asset": lbl,
                    "N_boot": N_BOOT,
                    "Block_len": BOOT_BLOCK,
                    "Point_NET_pct": float(point_nets[lbl]),
                    "Boot_mean": float(np.mean(arr)),
                    "Boot_sd": float(np.std(arr, ddof=1)),
                    "CI_05": float(p05),
                    "Median": float(np.median(arr)),
                    "CI_95": float(p95),
                    "CI_025": float(p025),
                    "CI_975": float(p975),
                    "Share_Pos_pct": float(np.mean(arr > 0) * 100.0),
                    "Excludes_Zero_90": bool((p05 > 0) or (p95 < 0)),
                    "Excludes_Zero_95": bool((p025 > 0) or (p975 < 0)),
                })
        print(f"Moving-Block Bootstrap done in {time.time() - t_b0:.1f}s")

    pd.DataFrame(shuf_rows).to_csv(OUT_DIR / "step1a_permutation_test.csv", index=False)
    pd.DataFrame(boot_rows).to_csv(OUT_DIR / "step1c_bootstrap_net_ci.csv", index=False)
    pd.DataFrame(boot_dist_rows).to_csv(OUT_DIR / "step1c_bootstrap_tsi_ci.csv", index=False)

    # -------------------------------------------------------------------------
    # 1e: Sensitivity Analyses (all post-warmup on Main US trading calendar)
    # -------------------------------------------------------------------------
    sens_sum_rows = []
    sens_net_rows = []

    # (i) System without BTC-USD (7 assets)
    assets_no_btcusd = [a for a in ASSETS_8 if a[1] != "BTC-USD"]
    panel_no_btc = build_aligned_panel(assets_no_btcusd, join_ihsg=False)
    lbls_no_btc = [a[1] for a in assets_no_btcusd]
    px_no_btc = make_proxy_matrices(panel_no_btc, lbls_no_btc)
    for pname in ["log_r2", "abs_r", "log_Parkinson_8asset_interval"]:
        res = estimate_tvp_var_gfevd(px_no_btc[pname], p=1, warmup=WARMUP_OBS, causal_sigma0=True)
        mg = res["mean_gfevd"]
        j_btcidr = lbls_no_btc.index("BTC/IDR")
        ai_to_btcidr_s = float(mg[j_btcidr, [lbls_no_btc.index(a[1]) for a in assets_no_btcusd if a[2] == "AI"]].sum() * 100.0)
        eth_to_btcidr_s = float(mg[j_btcidr, lbls_no_btc.index("ETH-USD")] * 100.0)
        sens_sum_rows.append({
            "Spec": "No_BTCUSD_7asset",
            "Proxy": pname,
            "N_assets": len(lbls_no_btc),
            "N_post": len(res["tsi_post"]),
            "TSI_post_mean": float(res["tsi_post"].mean()),
            "AI_sum_to_BTCIDR_pct": ai_to_btcidr_s,
            "Crypto_other_to_BTCIDR_pct": eth_to_btcidr_s,
            "BTCIDR_NET_pct": float(res["net_i"][j_btcidr]),
        })
        for j, lbl in enumerate(lbls_no_btc):
            sens_net_rows.append({
                "Spec": "No_BTCUSD_7asset",
                "Proxy": pname,
                "Asset": lbl,
                "TO_pct": float(res["to_i"][j]),
                "FROM_pct": float(res["from_i"][j]),
                "NET_pct": float(res["net_i"][j]),
            })

    # (ii) System + ^VIX + broad index (^IXIC and also ^GSPC) as controls (10 assets)
    for idx_file, idx_lbl in [("idx_ixic.csv", "IXIC"), ("idx_gspc.csv", "GSPC")]:
        extra = [
            ("konteks_vix.csv", "VIX", "Control", "equity"),
            (idx_file, idx_lbl, "Control", "equity"),
        ]
        panel_ctrl = build_aligned_panel(ASSETS_8, join_ihsg=False, extra_specs=extra)
        lbls_ctrl = labels_8 + ["VIX", idx_lbl]
        px_ctrl = make_proxy_matrices(panel_ctrl, lbls_ctrl)
        for pname in ["log_r2", "abs_r"]:
            res = estimate_tvp_var_gfevd(px_ctrl[pname], p=1, warmup=WARMUP_OBS, causal_sigma0=True)
            mg = res["mean_gfevd"]
            j_btcidr = lbls_ctrl.index("BTC/IDR")
            j_btcusd = lbls_ctrl.index("BTC-USD")
            j_vix = lbls_ctrl.index("VIX")
            j_idx = lbls_ctrl.index(idx_lbl)
            sens_sum_rows.append({
                "Spec": f"Plus_VIX_{idx_lbl}_10asset",
                "Proxy": pname,
                "N_assets": len(lbls_ctrl),
                "N_post": len(res["tsi_post"]),
                "TSI_post_mean": float(res["tsi_post"].mean()),
                "AI_sum_to_BTCIDR_pct": float(mg[j_btcidr, ai_idx].sum() * 100.0),
                "Crypto_other_to_BTCIDR_pct": float(mg[j_btcidr, [btcusd_i, 6]].sum() * 100.0),
                "BTCIDR_NET_pct": float(res["net_i"][j_btcidr]),
                "VIX_to_BTCIDR_pct": float(mg[j_btcidr, j_vix] * 100.0),
                "BroadIdx_to_BTCIDR_pct": float(mg[j_btcidr, j_idx] * 100.0),
                "BTCUSD_to_BTCIDR_pct": float(mg[j_btcidr, j_btcusd] * 100.0),
            })
            for j, lbl in enumerate(lbls_ctrl):
                sens_net_rows.append({
                    "Spec": f"Plus_VIX_{idx_lbl}_10asset",
                    "Proxy": pname,
                    "Asset": lbl,
                    "TO_pct": float(res["to_i"][j]),
                    "FROM_pct": float(res["from_i"][j]),
                    "NET_pct": float(res["net_i"][j]),
                })

    # (iii) FX-adjusted BTC/IDR (P_BTCIDR / S_USDIDR)
    panel_fx = build_aligned_panel(ASSETS_8, join_ihsg=False, fx_adjust_btcidr=True)
    px_fx = make_proxy_matrices(panel_fx, labels_8)
    for pname in ["log_r2", "abs_r", "log_Parkinson_8asset_interval"]:
        res = estimate_tvp_var_gfevd(px_fx[pname], p=1, warmup=WARMUP_OBS, causal_sigma0=True)
        mg = res["mean_gfevd"]
        sens_sum_rows.append({
            "Spec": "FX_Adjusted_BTCIDR_8asset",
            "Proxy": pname,
            "N_assets": len(labels_8),
            "N_post": len(res["tsi_post"]),
            "TSI_post_mean": float(res["tsi_post"].mean()),
            "AI_sum_to_BTCIDR_pct": float(mg[btcidr_i, ai_idx].sum() * 100.0),
            "Crypto_other_to_BTCIDR_pct": float(mg[btcidr_i, [btcusd_i, 6]].sum() * 100.0),
            "BTCIDR_NET_pct": float(res["net_i"][btcidr_i]),
            "BTCUSD_to_BTCIDR_pct": float(mg[btcidr_i, btcusd_i] * 100.0),
        })
        for j, lbl in enumerate(labels_8):
            sens_net_rows.append({
                "Spec": "FX_Adjusted_BTCIDR_8asset",
                "Proxy": pname,
                "Asset": lbl,
                "TO_pct": float(res["to_i"][j]),
                "FROM_pct": float(res["from_i"][j]),
                "NET_pct": float(res["net_i"][j]),
            })

    # Compare return correlation between BTC/IDR nominal vs FX-adjusted vs BTC-USD
    corr_nom = np.corrcoef(main_panel["df_ret"]["BTC-USD"], main_panel["df_ret"]["BTC/IDR"])[0, 1]
    corr_fx = np.corrcoef(panel_fx["df_ret"]["BTC-USD"], panel_fx["df_ret"]["BTC/IDR"])[0, 1]
    print(f"Return corr(BTC-USD, BTC/IDR): nominal={corr_nom:.4f}, FX-adjusted={corr_fx:.4f}")

    pd.DataFrame(sens_sum_rows).to_csv(OUT_DIR / "step1e_sensitivity_summary.csv", index=False)
    pd.DataFrame(sens_net_rows).to_csv(OUT_DIR / "step1e_sensitivity_net.csv", index=False)

    print(f"\nAll Step 1 Stage-1 outputs written to {OUT_DIR} in {time.time() - t_start:.1f}s.")


if __name__ == "__main__":
    main()
