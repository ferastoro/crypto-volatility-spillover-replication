#!/usr/bin/env python3
"""
Computes 90% (and 95% where available) basic (reverse-percentile) bootstrap
confidence intervals from revision/outputs/step1c_bootstrap_net_ci_dy12.csv:
    [2 * point - q_hi, 2 * point - q_lo]
Saves revision/outputs/step1c_bootstrap_basic_ci.csv.
"""
from pathlib import Path
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent.parent.parent
OUT_DIR = BASE_DIR / "revision" / "outputs"


def excl_zero(lo: float, hi: float) -> bool:
    return bool((lo > 0.0 and hi > 0.0) or (lo < 0.0 and hi < 0.0))


def main():
    df = pd.read_csv(OUT_DIR / "step1c_bootstrap_net_ci_dy12.csv")
    has_95 = {"CI_025_DY12", "CI_975_DY12"}.issubset(df.columns)

    rows = []
    for _, r in df.iterrows():
        pt = float(r["Point_NET_pct_DY12"])
        p05 = float(r["CI_05_DY12"])
        p95 = float(r["CI_95_DY12"])
        b_lo = 2.0 * pt - p95
        b_hi = 2.0 * pt - p05
        ex_p = excl_zero(p05, p95)
        ex_b = excl_zero(b_lo, b_hi)
        agree = bool(ex_p == ex_b and (not ex_p or ((p05 > 0) == (b_lo > 0))))
        rec = {
            "Asset": r["Asset"],
            "Proxy": r["Proxy"],
            "Point": pt,
            "Pct_lo": p05,
            "Pct_hi": p95,
            "Basic_lo": b_lo,
            "Basic_hi": b_hi,
            "Excl0_pct": ex_p,
            "Excl0_basic": ex_b,
            "Agree": agree,
        }
        if has_95:
            p025 = float(r["CI_025_DY12"])
            p975 = float(r["CI_975_DY12"])
            b95_lo = 2.0 * pt - p975
            b95_hi = 2.0 * pt - p025
            ex_p95 = excl_zero(p025, p975)
            ex_b95 = excl_zero(b95_lo, b95_hi)
            rec.update({
                "Pct95_lo": p025,
                "Pct95_hi": p975,
                "Basic95_lo": b95_lo,
                "Basic95_hi": b95_hi,
                "Excl0_pct95": ex_p95,
                "Excl0_basic95": ex_b95,
                "Agree_95": bool(ex_p95 == ex_b95 and (not ex_p95 or ((p025 > 0) == (b95_lo > 0)))),
            })
        rows.append(rec)

    out = pd.DataFrame(rows)
    out_path = OUT_DIR / "step1c_bootstrap_basic_ci.csv"
    out.to_csv(out_path, index=False)
    print(f"Wrote {out_path} ({len(out)} rows)")
    print(out[["Asset", "Proxy", "Point", "Pct_lo", "Pct_hi", "Basic_lo", "Basic_hi", "Excl0_pct", "Excl0_basic", "Agree"]].round(2).to_string(index=False))
    if has_95:
        print("\n=== 95% Basic vs Percentile ===")
        print(out[["Asset", "Proxy", "Point", "Pct95_lo", "Pct95_hi", "Basic95_lo", "Basic95_hi", "Excl0_pct95", "Excl0_basic95", "Agree_95"]].round(2).to_string(index=False))


if __name__ == "__main__":
    main()

