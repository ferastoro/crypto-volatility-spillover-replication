#!/usr/bin/env python3
"""Task 4 (Optional): 1-day-ahead BTC/IDR Student-t VaR backtest across 335 OOS origins."""

from __future__ import annotations

import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from arch import arch_model
from scipy import stats

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parent.parent
OUT_PATH = Path(__file__).resolve().parent.parent / "results_new" / "var_backtest_h1.csv"
ROLL_WINDOW = 1174


def kupiec_christoffersen(exc: np.ndarray, alpha: float) -> dict[str, float]:
    """Compute Kupiec (1995) LR_uc and Christoffersen (1998) LR_cc tests."""
    T = len(exc)
    x = int(np.sum(exc))
    p_hat = x / T

    def _xlogp(cnt: int, prob: float) -> float:
        return 0.0 if cnt == 0 else cnt * float(np.log(prob))

    ll_null = _xlogp(x, alpha) + _xlogp(T - x, 1.0 - alpha)
    ll_alt = _xlogp(x, p_hat) + _xlogp(T - x, 1.0 - p_hat)
    lr_uc = max(0.0, -2.0 * (ll_null - ll_alt))
    p_uc = float(1.0 - stats.chi2.cdf(lr_uc, df=1))

    # Markov transition counts (t-1 -> t)
    n00 = int(np.sum((exc[:-1] == 0) & (exc[1:] == 0)))
    n01 = int(np.sum((exc[:-1] == 0) & (exc[1:] == 1)))
    n10 = int(np.sum((exc[:-1] == 1) & (exc[1:] == 0)))
    n11 = int(np.sum((exc[:-1] == 1) & (exc[1:] == 1)))
    n0 = n00 + n01
    n1 = n10 + n11
    pi = (n01 + n11) / (n0 + n1)
    pi0 = n01 / n0 if n0 > 0 else 0.0
    pi1 = n11 / n1 if n1 > 0 else 0.0

    ll_ind_null = _xlogp(n01 + n11, pi) + _xlogp(n00 + n10, 1.0 - pi)
    ll_ind_alt = _xlogp(n01, pi0) + _xlogp(n00, 1.0 - pi0) + _xlogp(n11, pi1) + _xlogp(n10, 1.0 - pi1)
    lr_ind = max(0.0, -2.0 * (ll_ind_null - ll_ind_alt))
    lr_cc = lr_uc + lr_ind
    p_cc = float(1.0 - stats.chi2.cdf(lr_cc, df=2))

    return {
        "n_obs": T,
        "exceptions": x,
        "expected": alpha * T,
        "rate_pct": 100.0 * p_hat,
        "lr_uc": lr_uc,
        "p_uc": p_uc,
        "lr_ind": lr_ind,
        "lr_cc": lr_cc,
        "p_cc": p_cc,
    }


def main() -> None:
    dl = pd.read_csv(ROOT / "data" / "clean" / "dataset_deep_learning.csv")
    har1 = pd.read_csv(ROOT / "inputs" / "v5" / "har_preds_T0_h1_v2.csv").sort_values("Date").reset_index(drop=True)
    g4 = pd.read_csv(ROOT / "inputs" / "v5" / "GARCH_Predictions_V4.csv")
    g4["h"] = g4.groupby("cutoff").cumcount() + 1
    g1 = g4[g4["h"] == 1].sort_values("real_date").reset_index(drop=True)
    hx = pd.read_csv(ROOT / "revision" / "outputs" / "step1h_new_baselines_predictions.csv")
    hx1 = hx[hx["h"] == 1].sort_values("real_date").reset_index(drop=True)
    kr = pd.read_csv(ROOT / "revision" / "kaggle_outputs" / "outputs_3a3b_v2" / "predictions_ensemble.csv")
    t_no = kr[(kr["model"] == "TFT") & (kr["feature_set"] == "NOSPILL") & (kr["variant"] == "NoWarmup250") & (kr["horizon"] == 1)].sort_values("target_date").reset_index(drop=True)
    t_sp = kr[(kr["model"] == "TFT") & (kr["feature_set"] == "SPILL") & (kr["variant"] == "NoWarmup250") & (kr["horizon"] == 1)].sort_values("target_date").reset_index(drop=True)

    target_dates = har1["Date"].values
    assert (g1["real_date"].values == target_dates).all()
    assert (hx1["real_date"].values == target_dates).all()
    assert (t_no["target_date"].values == target_dates).all()
    assert (t_sp["target_date"].values == target_dates).all()

    # Exact causal Student-t nu_t from rolling GARCH(1,1)-t fits (matching run_step1_stage2.py)
    date_to_idx = {d: i for i, d in enumerate(dl["Date"].values)}
    r_all = dl["ret_crypto_btc_idr_indodax"].values
    nu_series = np.empty(len(target_dates), dtype=float)
    garch_t_check = np.empty(len(target_dates), dtype=float)
    r_oos = np.empty(len(target_dates), dtype=float)

    for k, d_str in enumerate(target_dates):
        t_idx = date_to_idx[d_str]
        o_idx = t_idx - 1
        r_oos[k] = r_all[t_idx]
        win = r_all[o_idx - ROLL_WINDOW + 1 : o_idx + 1] * 100.0
        gm = arch_model(win, mean="Constant", vol="GARCH", p=1, q=1, dist="StudentsT").fit(disp="off", show_warning=False)
        if gm.convergence_flag != 0:
            gm = arch_model(win, mean="Constant", vol="GARCH", p=1, q=1, dist="StudentsT").fit(
                disp="off", show_warning=False, options={"maxiter": 1000}
            )
        nu_series[k] = float(gm.params["nu"])
        garch_t_check[k] = float(gm.forecast(horizon=1, reindex=False).variance.values[-1, 0] / 10000.0)

    assert np.max(np.abs(garch_t_check - hx1["GARCH11_t"].values)) < 1e-12
    assert np.max(np.abs(r_oos**2 - har1["True"].values)) < 1e-12

    models = [
        ("HAR", har1["HAR"].values),
        ("HAR-X-NOSPILL", hx1["HAR_X_NOSPILL"].values),
        ("HAR-X-SPILL", hx1["HAR_X_SPILL"].values),
        ("GARCH(1,1)-N", g1["GARCH11"].values),
        ("GARCH(1,1)-t", hx1["GARCH11_t"].values),
        ("EGARCH(1,1)-N", g1["EGARCH"].values),
        ("TFT-NOSPILL", t_no["y_pred"].values),
        ("TFT-SPILL", t_sp["y_pred"].values),
    ]

    rows = []
    for level_pct, alpha in [(95, 0.05), (99, 0.01)]:
        # Standardized unit-variance zero-mean Student-t quantile
        q_std = stats.t.ppf(alpha, df=nu_series) * np.sqrt((nu_series - 2.0) / nu_series)
        for mname, var_hat in models:
            assert np.all(var_hat > 0)
            var_threshold = np.sqrt(var_hat) * q_std
            exc = (r_oos < var_threshold).astype(int)
            res = kupiec_christoffersen(exc, alpha)
            res["level"] = f"{level_pct}%"
            res["alpha"] = alpha
            res["model"] = mname
            res["nu_median"] = float(np.median(nu_series))
            rows.append(res)

    out_df = pd.DataFrame(rows)[
        ["level", "model", "n_obs", "exceptions", "expected", "rate_pct", "lr_uc", "p_uc", "lr_cc", "p_cc", "nu_median"]
    ]
    out_df.to_csv(OUT_PATH, index=False)
    print(f"Rolling GARCH-t nu: min={np.min(nu_series):.3f}, median={np.median(nu_series):.3f}, max={np.max(nu_series):.3f}")
    print(out_df.to_string(index=False))


if __name__ == "__main__":
    main()
