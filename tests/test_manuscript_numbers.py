#!/usr/bin/env python3
"""Automated consistency test suite for numeric claims and tables in main.tex."""

from __future__ import annotations

import math
import re
import sys
from dataclasses import dataclass
from pathlib import Path

import os
TEX_PATH = Path(os.environ.get("MANUSCRIPT_TEX", str(Path(__file__).resolve().parent.parent / "main.tex")))


@dataclass
class CheckResult:
    group: str
    item: str
    passed: bool
    detail: str


def strip_tex_num(token: str) -> str:
    """Strip TeX wrappers such as \\textbf{}, math $, \\% and spaces."""
    s = token.strip()
    s = re.sub(r"\\textbf\{([^}]*)\}", r"\1", s)
    s = s.replace("$", "").replace("\\%", "").replace(",", "").strip()
    return s


def parse_float(token: str) -> float:
    return float(strip_tex_num(token))


def extract_table_block(tex: str, label: str) -> str:
    """Return the single LaTeX table environment containing \\label{label}."""
    target = f"\\label{{{label}}}"
    for m in re.finditer(r"\\begin\{table\}.*?\\end\{table\}", tex, re.DOTALL):
        if target in m.group(0):
            return m.group(0)
    raise ValueError(f"Could not find table with label {label}")


def extract_tabular_blocks(table_block: str) -> list[str]:
    return re.findall(r"\\begin\{tabular\}.*?\\end\{tabular\}", table_block, re.DOTALL)


def split_latex_rows(tabular_block: str) -> list[list[str]]:
    """Split a tabular block into non-rule data rows of trimmed cells."""
    body_match = re.search(r"\\midrule(.*?)\\bottomrule", tabular_block, re.DOTALL)
    if not body_match:
        return []
    body = body_match.group(1)
    rows: list[list[str]] = []
    for raw_line in body.split("\\\\"):
        line = re.sub(r"\\midrule|\\cmidrule(?:\([^)]*\))?\{[^}]*\}|\\multicolumn\{[^}]*\}\{[^}]*\}\{[^}]*\}", "", raw_line)
        line = line.strip()
        if not line:
            continue
        cells = [c.strip() for c in line.split("&")]
        rows.append(cells)
    return rows


def check_table_a1(tex: str, results: list[CheckResult]) -> None:
    """Check 1: Table A1 NET = TO - FROM, mean TO = mean FROM = TSI, group sums."""
    block_a1 = extract_table_block(tex, "tab:appendix_to_from")
    rows = split_tabular_blocks_single(block_a1)
    # Table 5 holds TSI per proxy for comparison with mean(TO) and mean(FROM)
    block_t5 = extract_table_block(tex, "tab:block_decomp")
    t5_rows = split_tabular_blocks_single(block_t5)
    t5_tsi = [parse_float(r[1]) for r in t5_rows]

    proxies = ["log(r^2)", "|r|", "r^2", "log(Parkinson)"]
    ai_assets = {"NVDA", "MSFT", "TSLA", "GOOGL", "META"}
    crypto_assets = {"BTC-USD", "ETH-USD", "BTC/IDR"}

    if len(rows) != 8:
        results.append(CheckResult("Check 1 (Table A1)", "Row count == 8", False, f"Found {len(rows)} rows"))
        return

    for p_idx, pname in enumerate(proxies):
        col_start = 1 + p_idx * 3
        tos, froms, nets = [], [], []
        ai_net = 0.0
        crypto_net = 0.0
        for row in rows:
            asset = strip_tex_num(row[0])
            to_v = parse_float(row[col_start])
            fr_v = parse_float(row[col_start + 1])
            net_v = parse_float(row[col_start + 2])
            tos.append(to_v)
            froms.append(fr_v)
            nets.append(net_v)
            diff = abs((to_v - fr_v) - net_v)
            ok = diff <= 0.015  # 2-decimal independent rounding tolerance (max 0.01)
            results.append(
                CheckResult(
                    "Check 1 (Table A1)",
                    f"{pname} {asset}: NET == TO - FROM",
                    ok,
                    f"{to_v:.2f} - {fr_v:.2f} = {to_v - fr_v:.2f} vs {net_v:.2f} (diff={diff:.3f})",
                )
            )
            if asset in ai_assets:
                ai_net += net_v
            elif asset in crypto_assets:
                crypto_net += net_v

        mean_to = sum(tos) / len(tos)
        mean_fr = sum(froms) / len(froms)
        tsi_t5 = t5_tsi[p_idx]
        ok_to = abs(mean_to - tsi_t5) <= 0.015
        ok_fr = abs(mean_fr - tsi_t5) <= 0.015
        results.append(
            CheckResult(
                "Check 1 (Table A1)",
                f"{pname}: mean(TO) == TSI",
                ok_to,
                f"mean(TO)={mean_to:.3f} vs Table 5 TSI={tsi_t5:.2f}",
            )
        )
        results.append(
            CheckResult(
                "Check 1 (Table A1)",
                f"{pname}: mean(FROM) == TSI",
                ok_fr,
                f"mean(FROM)={mean_fr:.3f} vs Table 5 TSI={tsi_t5:.2f}",
            )
        )
        total_net = ai_net + crypto_net
        ok_sum = abs(total_net) <= 0.035  # up to 8 rounded terms
        results.append(
            CheckResult(
                "Check 1 (Table A1)",
                f"{pname}: AI_sum + Crypto_sum == 0",
                ok_sum,
                f"AI={ai_net:+.2f}, Crypto={crypto_net:+.2f}, total={total_net:+.2f}",
            )
        )


def split_tabular_blocks_single(table_block: str, idx: int = 0) -> list[list[str]]:
    tabs = extract_tabular_blocks(table_block)
    return split_latex_rows(tabs[idx])


def check_table_5(tex: str, results: list[CheckResult]) -> None:
    """Check 2: Table 5 block sums equal TSI and shares sum to 100%."""
    block = extract_table_block(tex, "tab:block_decomp")
    rows = split_tabular_blocks_single(block)
    val_share_re = re.compile(r"([0-9.]+)\s*\(([0-9.]+)\\%\)")

    for row in rows:
        proxy = row[0]
        tsi = parse_float(row[1])
        vals, shares = [], []
        for cell in row[2:6]:
            m = val_share_re.search(cell)
            if not m:
                results.append(CheckResult("Check 2 (Table 5)", f"{proxy} parse", False, f"Unparsed cell: {cell}"))
                return
            vals.append(float(m.group(1)))
            shares.append(float(m.group(2)))

        val_sum = sum(vals)
        share_sum = sum(shares)
        ok_v = abs(val_sum - tsi) <= 0.025
        ok_s = abs(share_sum - 100.0) <= 0.25
        results.append(
            CheckResult(
                "Check 2 (Table 5)",
                f"{proxy}: block sum == TSI",
                ok_v,
                f"sum={val_sum:.2f} vs TSI={tsi:.2f}",
            )
        )
        results.append(
            CheckResult(
                "Check 2 (Table 5)",
                f"{proxy}: share sum == 100%",
                ok_s,
                f"sum={share_sum:.1f}%",
            )
        )
        for idx, (v, s) in enumerate(zip(vals, shares), start=1):
            calc_s = 100.0 * v / tsi
            ok_pair = abs(calc_s - s) <= 0.15
            results.append(
                CheckResult(
                    "Check 2 (Table 5)",
                    f"{proxy} block {idx} share consistency",
                    ok_pair,
                    f"{v:.2f}/{tsi:.2f}*100={calc_s:.2f}% vs {s:.1f}%",
                )
            )


def strip_multirow_prefix(cell: str) -> tuple[str | None, str]:
    m = re.match(r"\\multirow\{[^}]*\}\{[^}]*\}\{([^}]*)\}", cell.strip())
    if m:
        return strip_tex_num(m.group(1)), ""
    return None, cell


def check_r2_oos(tex: str, results: list[CheckResult]) -> None:
    """Check 3: R2_OOS = 1 - MSE / MSE_HistExp in Tables 6, 9, B4, and implied consistency in Table 12."""
    # Table 6
    b6 = extract_table_block(tex, "tab:rq2_forecasting")
    r6 = split_tabular_blocks_single(b6)
    hist_exp_mse: dict[str, float] = {}
    cur_h = ""
    t6_parsed: list[tuple[str, str, float, float, float]] = []
    for row in r6:
        h_lbl, _ = strip_multirow_prefix(row[0])
        if h_lbl:
            cur_h = h_lbl
        model = strip_tex_num(row[1])
        mse = parse_float(row[2])
        rmse = parse_float(row[4])
        r2 = parse_float(row[5])
        if model == "Hist-Exp":
            hist_exp_mse[cur_h] = mse
        t6_parsed.append((cur_h, model, mse, rmse, r2))

    for h, model, mse, rmse, r2 in t6_parsed:
        calc_r2 = 1.0 - mse / hist_exp_mse[h]
        ok_r2 = abs(calc_r2 - r2) <= 0.002
        ok_rmse = abs(math.sqrt(mse) - rmse) <= 0.002
        results.append(
            CheckResult(
                "Check 3 (Table 6)",
                f"{h} {model}: R2_OOS & RMSE",
                ok_r2 and ok_rmse,
                f"R2={calc_r2:+.4f} vs {r2:+.3f}; sqrt(MSE)={math.sqrt(mse):.4f} vs {rmse:.3f}",
            )
        )

    # Table 9 Panel A
    b9 = extract_table_block(tex, "tab:matched_harx")
    r9_a = split_tabular_blocks_single(b9, idx=0)
    cur_h = ""
    for row in r9_a:
        if any("Comparison" in c for c in row):
            break
        if len(row) < 6 or "Model" in row[1]:
            continue
        h_lbl, _ = strip_multirow_prefix(row[0])
        if h_lbl:
            cur_h = h_lbl
        model = strip_tex_num(row[1])
        mse = parse_float(row[2])
        r2 = parse_float(row[5])
        calc_r2 = 1.0 - mse / hist_exp_mse[cur_h]
        ok = abs(calc_r2 - r2) <= 0.002
        results.append(
            CheckResult(
                "Check 3 (Table 9)",
                f"{cur_h} {model}: R2_OOS",
                ok,
                f"R2={calc_r2:+.4f} vs {r2:+.3f}",
            )
        )

    # Table B4 Panel A
    b_ref = extract_table_block(tex, "tab:appendix_refit20")
    r_ref = split_tabular_blocks_single(b_ref, idx=0)
    cur_h = ""
    cur_m = ""
    for row in r_ref:
        if len(row) < 8 or "Horizon" in row[0]:
            continue
        h_lbl, _ = strip_multirow_prefix(row[0])
        if h_lbl:
            cur_h = h_lbl
        m_lbl, _ = strip_multirow_prefix(row[1])
        if m_lbl:
            cur_m = m_lbl
        regime = strip_tex_num(row[2])
        mse = parse_float(row[3])
        r2 = parse_float(row[5])
        calc_r2 = 1.0 - mse / hist_exp_mse[cur_h]
        ok = abs(calc_r2 - r2) <= 0.002
        results.append(
            CheckResult(
                "Check 3 (Table B4)",
                f"{cur_h} {cur_m} {regime}: R2_OOS",
                ok,
                f"R2={calc_r2:+.4f} vs {r2:+.3f}",
            )
        )

    # Table 12 (RV60m): check internal Hist-Exp MSE consistency across models per horizon
    b12 = extract_table_block(tex, "tab:rv60m_main")
    r12 = split_tabular_blocks_single(b12)
    cur_h = ""
    by_h: dict[str, list[tuple[str, float, float]]] = {}
    for row in r12:
        h_lbl, _ = strip_multirow_prefix(row[0])
        if h_lbl:
            cur_h = h_lbl
        model = strip_tex_num(row[1])
        mse = parse_float(row[2])
        r2 = parse_float(row[5])
        by_h.setdefault(cur_h, []).append((model, mse, r2))

    for h, items in by_h.items():
        implied = [mse / (1.0 - r2) for _, mse, r2 in items]
        mean_imp = sum(implied) / len(implied)
        for model, mse, r2 in items:
            calc_r2 = 1.0 - mse / mean_imp
            ok = abs(calc_r2 - r2) <= 0.0025
            results.append(
                CheckResult(
                    "Check 3 (Table 12 RV60m)",
                    f"{h} {model}: R2_OOS consistency",
                    ok,
                    f"1-MSE/{mean_imp:.4f}={calc_r2:+.4f} vs {r2:+.3f}",
                )
            )


def check_bonferroni_pair(
    raw_tok: str, bonf_tok: str, m: int, label: str, results: list[CheckResult]
) -> None:
    raw_s = strip_tex_num(raw_tok)
    bonf_s = strip_tex_num(bonf_tok)
    if raw_s.startswith("<"):
        # True p_raw is in (0, bound); any p_bonf < m*bound is consistent
        bound = float(raw_s[1:])
        max_b = min(1.0, m * bound)
        if bonf_s.startswith("<"):
            b_bound = float(bonf_s[1:])
            ok = b_bound <= max_b + 0.002 and b_bound > 0.0
        else:
            b_val = float(bonf_s)
            ok = 0.0 <= b_val <= max_b + 0.001
        results.append(CheckResult("Check 4 (Bonferroni)", label, ok, f"m={m}, p_raw={raw_s}, p_bonf={bonf_s}"))
        return

    p_raw = float(raw_s)
    dec_raw = len(raw_s.split(".")[1]) if "." in raw_s else 4
    eps_raw = 0.5 * (10.0 ** (-dec_raw))

    if bonf_s.startswith("<"):
        b_bound = float(bonf_s[1:])
        ok = min(1.0, m * (p_raw - eps_raw)) < b_bound
        results.append(CheckResult("Check 4 (Bonferroni)", label, ok, f"m={m}, p_raw={raw_s}, p_bonf={bonf_s}"))
        return

    p_bonf = float(bonf_s)
    dec_bonf = len(bonf_s.split(".")[1]) if "." in bonf_s else 3
    eps_bonf = 0.5 * (10.0 ** (-dec_bonf)) + 1e-6
    lo = min(1.0, m * max(0.0, p_raw - eps_raw)) - eps_bonf
    hi = min(1.0, m * (p_raw + eps_raw)) + eps_bonf
    ok = lo <= p_bonf <= hi
    results.append(
        CheckResult(
            "Check 4 (Bonferroni)",
            label,
            ok,
            f"m={m}, m*p_raw={min(1.0, m * p_raw):.4f} vs {p_bonf:.4f} (allowed [{lo:.4f}, {hi:.4f}])",
        )
    )


def check_bonferroni_all(tex: str, results: list[CheckResult]) -> None:
    """Check 4: p_bonf = min(1, m*p_raw) in Tables 8, 9, B3, B4 and family sizes vs Table 3."""
    pair_re = re.compile(r"([<0-9.]+)\s*\(([<0-9.]+)\)")

    # Table 8
    b8 = extract_table_block(tex, "tab:dm_tests")
    r8 = split_tabular_blocks_single(b8)
    for row in r8:
        comp = row[0]
        m_match = re.search(r"m=(\d+)", row[1])
        if not m_match:
            results.append(CheckResult("Check 4 (Table 8)", comp, False, f"Missing m in {row[1]}"))
            continue
        m = int(m_match.group(1))
        for h_idx, h_name in [(3, "h=1"), (5, "h=10")]:
            pm = pair_re.search(strip_tex_num(row[h_idx]))
            if not pm:
                results.append(CheckResult("Check 4 (Table 8)", f"{comp} {h_name}", False, f"Parse fail: {row[h_idx]}"))
                continue
            check_bonferroni_pair(pm.group(1), pm.group(2), m, f"T8 {comp} ({row[1]}) {h_name}", results)

    # Table 9 Panel B (m=6)
    b9 = extract_table_block(tex, "tab:matched_harx")
    r9_b = split_tabular_blocks_single(b9, idx=0)
    in_b = False
    for row in r9_b:
        if any("Comparison" in c for c in row):
            in_b = True
            continue
        if not in_b or len(row) < 6:
            continue
        comp = f"{row[0]} ({row[1]})"
        for h_idx, h_name in [(3, "h=1"), (5, "h=10")]:
            pm = pair_re.search(strip_tex_num(row[h_idx]))
            if pm:
                check_bonferroni_pair(pm.group(1), pm.group(2), 6, f"T9 {comp} {h_name}", results)

    # Table B3
    bb3 = extract_table_block(tex, "tab:appendix_tail_egarch")
    rb3 = split_tabular_blocks_single(bb3)
    for row in rb3:
        comp = strip_tex_num(row[0])
        p_raw = row[4]
        check_bonferroni_pair(p_raw, row[5], 26, f"TB3 {comp} m=26", results)
        if "--" not in row[6]:
            check_bonferroni_pair(p_raw, row[6], 16, f"TB3 {comp} m=16", results)

    # Table B4 Panel B
    bb4 = extract_table_block(tex, "tab:appendix_refit20")
    rb4 = split_tabular_blocks_single(bb4, idx=1)
    cur_h = ""
    for row in rb4:
        if len(row) < 8 or "Horizon" in row[0]:
            continue
        h_lbl, _ = strip_multirow_prefix(row[0])
        if h_lbl:
            cur_h = h_lbl
        comp = f"{cur_h} {row[1]} ({row[2]})"
        m = int(strip_tex_num(row[3]))
        check_bonferroni_pair(row[5], row[6], m, f"TB4 {comp}", results)

    # Table 12 (RV60m): SE p_raw (p_bonf) [m=28] and SPILL vs. NO [m=28 where numeric]
    b12 = extract_table_block(tex, "tab:rv60m_main")
    r12 = split_tabular_blocks_single(b12)
    cur_h = ""
    for row in r12:
        h_lbl, _ = strip_multirow_prefix(row[0])
        if h_lbl:
            cur_h = h_lbl
        model = strip_tex_num(row[1])
        pm_se = pair_re.search(strip_tex_num(row[6]))
        if pm_se:
            check_bonferroni_pair(pm_se.group(1), pm_se.group(2), 28, f"T12 SE {cur_h} {model}", results)
        pm_sp = pair_re.search(strip_tex_num(row[8]))
        if pm_sp and pm_sp.group(2).lower() != "n/a":
            check_bonferroni_pair(pm_sp.group(1), pm_sp.group(2), 28, f"T12 SPILL-NO {cur_h} {model}", results)

    # Table 3 confirmatory family sizes vs Sec. 3.3.3 prose
    b3 = extract_table_block(tex, "tab:bonferroni_families")
    r3 = split_tabular_blocks_single(b3)
    ok_rows = len(r3) == 9
    results.append(CheckResult("Check 4 (Table 3)", "9 confirmatory family rows", ok_rows, f"found {len(r3)} rows"))
    has_seven_prose = "Seven loss-specific exploratory and auxiliary families" in tex
    # Sec. 3.3.3 prose and Table 3 note both enumerate Softplus (10), RV60m (28, 28, 14), refit (12, 2, 2)
    sec333 = tex[tex.find("\\label{subsubsec:eval_metrics}"):tex.find("\\label{tab:bonferroni_families}")]
    t3_note = b3[b3.find("\\emph{Note:}"):]
    required_nums = ["10", "28", "14", "12", "2"]
    ok_note = has_seven_prose and all(n in t3_note and n in sec333 for n in required_nums)
    results.append(
        CheckResult(
            "Check 4 (Table 3)",
            "Prose & Table 3 note agree on 7 auxiliary families (10; 28/28/14; 12/2/2)",
            ok_note,
            f"prose_has_seven={has_seven_prose}",
        )
    )


def check_tables_10_11(tex: str, results: list[CheckResult]) -> None:
    """Check 5: Table 10 CI midpoint == d_bar, MDE ~= 1.43 * half_width, Table 11 match, Abstract/Conclusion bounds."""
    ci_re = re.compile(r"\[([+-]?[0-9.]+),\s*([+-]?[0-9.]+)\]")

    b10 = extract_table_block(tex, "tab:loss_diff_ci")
    r10 = split_tabular_blocks_single(b10)
    t10_cis: dict[tuple[str, str], tuple[tuple[float, float], tuple[float, float]]] = {}

    # Two-sided alpha=0.05, 80% power Student-t ratio with df=334 is ~1.4284 (~1.43)
    for row in r10:
        key = (strip_tex_num(row[0]), strip_tex_num(row[1]))
        parsed_samples = []
        for s_name, d_col, ci_col, mde_col in [("N=335", 2, 3, 4), ("N=333", 5, 6, 7)]:
            d_bar = parse_float(row[d_col])
            m = ci_re.search(row[ci_col])
            if not m:
                results.append(CheckResult("Check 5 (Table 10)", f"{key} {s_name}", False, f"Bad CI: {row[ci_col]}"))
                continue
            lo, hi = float(m.group(1)), float(m.group(2))
            mde = parse_float(row[mde_col])
            mid = 0.5 * (lo + hi)
            hw = 0.5 * (hi - lo)
            ok_mid = abs(mid - d_bar) <= 0.015
            # MDE uses 2.80 * SE while HLN CI uses t_{0.975, df} * SE (~1.9671 * SE), ratio ~1.4234 (~1.43 approx)
            mde_143 = 1.43 * hw
            ok_mde = abs(mde - mde_143) <= max(0.05, 0.006 * mde)
            results.append(
                CheckResult(
                    "Check 5 (Table 10)",
                    f"{key[0]} h={key[1]} {s_name}: midpoint & MDE",
                    ok_mid and ok_mde,
                    f"mid={mid:+.2f} vs d_bar={d_bar:+.2f}; MDE={mde:.2f} vs 1.43*hw={mde_143:.2f}",
                )
            )
            parsed_samples.append((lo, hi))
        if len(parsed_samples) == 2:
            t10_cis[key] = (parsed_samples[0], parsed_samples[1])

    # Table 11 matches Table 10 HLN CIs
    b11 = extract_table_block(tex, "tab:boot_loss_diff_ci")
    r11 = split_tabular_blocks_single(b11)
    for row in r11:
        key = (strip_tex_num(row[0]), strip_tex_num(row[1]))
        if key not in t10_cis:
            results.append(CheckResult("Check 5 (Table 11)", f"{key} in Table 10", False, "Key missing"))
            continue
        m335 = ci_re.search(row[2])
        m333 = ci_re.search(row[4])
        ci335 = (float(m335.group(1)), float(m335.group(2)))
        ci333 = (float(m333.group(1)), float(m333.group(2)))
        ok_335 = ci335 == t10_cis[key][0]
        ok_333 = ci333 == t10_cis[key][1]
        results.append(
            CheckResult(
                "Check 5 (Table 11)",
                f"{key[0]} h={key[1]}: HLN CI matches Table 10",
                ok_335 and ok_333,
                f"N=335 {ci335} vs {t10_cis[key][0]}; N=333 {ci333} vs {t10_cis[key][1]}",
            )
        )

    # Conclusion bounds match union of HLN + Bootstrap CIs in Table 11 (N=335, h=1)
    harx_h1 = [r for r in r11 if "HAR-X SP vs. NO match (SE)" in r[0] and strip_tex_num(r[1]) == "1"][0]
    harx_hln = ci_re.search(harx_h1[2])
    harx_boot = ci_re.search(harx_h1[3])
    harx_union = (min(float(harx_hln.group(1)), float(harx_boot.group(1))), max(float(harx_hln.group(2)), float(harx_boot.group(2))))

    tft_h1 = [r for r in r11 if "TFT SP vs. NO (SE)" in r[0] and strip_tex_num(r[1]) == "1"][0]
    tft_hln = ci_re.search(tft_h1[2])
    tft_boot = ci_re.search(tft_h1[3])
    tft_union = (min(float(tft_hln.group(1)), float(tft_boot.group(1))), max(float(tft_hln.group(2)), float(tft_boot.group(2))))

    ok_conc_bounds = ("[-1.87\\%, +3.63\\%]" in tex and harx_union == (-1.87, 3.63)) and (
        "[-11.89\\%, +5.62\\%]" in tex and tft_union == (-11.89, 5.62)
    )
    results.append(
        CheckResult(
            "Check 5 (Conclusion CIs)",
            "Union of HLN & Boot CIs matches text bounds",
            ok_conc_bounds,
            f"matched HAR-X union={harx_union}, TFT union={tft_union}",
        )
    )


def check_counts_and_dates(tex: str, results: list[CheckResult]) -> None:
    """Check 6: Date/row identities and Chow N_pre + N_post == 1,677 in Table B2."""
    checks = [
        ("2,437 = 1,677 + 760", 1677 + 760 == 2437 and "2,437" in tex and "1,677" in tex and "760" in tex),
        ("915 + 167 = 1,082", 915 + 167 == 1082 and "915" in tex and "167" in tex and "1,082" in tex),
        (
            "915 + 167 + 335 + 9 = 1,426 (344 target dates - 335 origins = 9)",
            915 + 167 + 335 + 9 == 1426 and "335" in tex and "344" in tex and "1,426" in tex,
        ),
        ("250 + 1,426 = 1,676", 250 + 1426 == 1676 and "250" in tex and "1,676" in tex),
    ]
    for name, ok in checks:
        results.append(CheckResult("Check 6 (Row counts)", name, ok, "present & exact" if ok else "mismatch"))

    bb2 = extract_table_block(tex, "tab:appendix_chow")
    rb2 = split_tabular_blocks_single(bb2)
    for row in rb2:
        dt = row[0]
        parts = [int(p.strip().replace(",", "")) for p in row[2].split("/")]
        ok = len(parts) == 2 and sum(parts) == 1677
        results.append(
            CheckResult(
                "Check 6 (Table B2 Chow)",
                f"{dt}: N_pre + N_post == 1,677",
                ok,
                f"{parts[0]} + {parts[1]} = {sum(parts)}",
            )
        )


def main() -> int:
    if len(sys.argv) > 1:
        tex_path = Path(sys.argv[1])
    else:
        tex_path = TEX_PATH
    if not tex_path.exists():
        print(f"SKIP: Manuscript TeX not bundled at {tex_path}. Pass path or set MANUSCRIPT_TEX to verify LaTeX tables.")
        return 0
    tex = tex_path.read_text(encoding="utf-8")
    results: list[CheckResult] = []

    check_table_a1(tex, results)
    check_table_5(tex, results)
    check_r2_oos(tex, results)
    check_bonferroni_all(tex, results)
    check_tables_10_11(tex, results)
    check_counts_and_dates(tex, results)

    by_group: dict[str, list[CheckResult]] = {}
    for r in results:
        by_group.setdefault(r.group, []).append(r)

    print(f"{'Group':32s} | {'Pass':>4s} | {'Fail':>4s} | Status")
    print("-" * 60)
    total_pass = 0
    total_fail = 0
    for g, items in by_group.items():
        p = sum(1 for x in items if x.passed)
        f = len(items) - p
        total_pass += p
        total_fail += f
        print(f"{g:32s} | {p:4d} | {f:4d} | {'PASS' if f == 0 else 'FAIL'}")
    print("-" * 60)
    print(f"{'TOTAL':32s} | {total_pass:4d} | {total_fail:4d} | {'ALL PASS' if total_fail == 0 else 'FAILURES PRESENT'}")

    fails = [r for r in results if not r.passed]
    if fails:
        print("\nFAIL DETAILS:")
        for r in fails:
            print(f"  [{r.group}] {r.item}: {r.detail}")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
