#!/usr/bin/env python3
"""Verify Feb 3-7, 2026 daily closes, log returns, and Indodax BTC volumes from re-fetched raw CSVs."""

from __future__ import annotations

import math
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent.parent
BTC_USD_PATH = ROOT / "data" / "raw" / "crypto_btc_usd.csv"
BTC_IDR_PATH = ROOT / "data" / "raw" / "crypto_btc_idr_indodax.csv"

DATES_PRINT = ["2026-02-03", "2026-02-04", "2026-02-05", "2026-02-06", "2026-02-07"]


def load_series(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    df = df.sort_values("Date").reset_index(drop=True)
    df["log_ret_pct"] = 100.0 * (df["Close"].apply(math.log) - df["Close"].shift(1).apply(lambda x: math.log(x) if pd.notna(x) else float("nan")))
    return df.set_index("Date")


def main() -> int:
    if not BTC_USD_PATH.exists() or not BTC_IDR_PATH.exists():
        print("SKIP: Raw files missing in data/raw/. Run src/collect_data.py first.")
        return 0
    usd = load_series(BTC_USD_PATH)
    idr = load_series(BTC_IDR_PATH)

    print("Date (UTC) | BTC-USD Close | BTC-USD r (%) | BTC/IDR Close    | BTC/IDR r (%) | Indodax Vol (BTC)")
    print("-" * 102)
    for d in DATES_PRINT:
        u = usd.loc[d]
        i = idr.loc[d]
        print(
            f"{d} | {u['Close']:13,.2f} | {u['log_ret_pct']:+12.2f}% | "
            f"{i['Close']:16,.0f} | {i['log_ret_pct']:+12.2f}% | {i['Volume']:17.2f}"
        )

    checks = [
        ("BTC-USD r(2026-02-05)", float(usd.loc["2026-02-05", "log_ret_pct"]), -15.23, 0.01),
        ("BTC-USD r(2026-02-06)", float(usd.loc["2026-02-06", "log_ret_pct"]), +11.80, 0.01),
        ("BTC/IDR r(2026-02-05)", float(idr.loc["2026-02-05", "log_ret_pct"]), -14.85, 0.01),
        ("BTC/IDR r(2026-02-06)", float(idr.loc["2026-02-06", "log_ret_pct"]), +11.75, 0.01),
        ("Indodax Vol(2026-02-04)", float(idr.loc["2026-02-04", "Volume"]), 45.78, 0.01),
        ("Indodax Vol(2026-02-05)", float(idr.loc["2026-02-05", "Volume"]), 176.51, 0.01),
        ("Indodax Vol(2026-02-06)", float(idr.loc["2026-02-06", "Volume"]), 162.21, 0.01),
        ("BTC-USD Close(2026-02-04)", float(usd.loc["2026-02-04", "Close"]), 73019.70, 0.01),
        ("BTC-USD Close(2026-02-05)", float(usd.loc["2026-02-05", "Close"]), 62702.10, 0.01),
        ("BTC-USD Close(2026-02-06)", float(usd.loc["2026-02-06", "Close"]), 70555.39, 0.01),
    ]

    print("\nASSERTIONS (tol = 0.01):")
    failures = []
    for name, actual, expected, tol in checks:
        diff = abs(actual - expected)
        ok = diff <= tol
        status = "PASS" if ok else "FAIL"
        print(f"  [{status}] {name:28s}: actual={actual:.6f}, expected={expected:.2f}, |diff|={diff:.6f}")
        if not ok:
            failures.append(name)

    if failures:
        print(f"\nFAILURES: {failures}")
        return 1
    print("\nALL 10 FEB-2026 CHECKS PASSED")
    return 0


if __name__ == "__main__":
    sys.exit(main())
