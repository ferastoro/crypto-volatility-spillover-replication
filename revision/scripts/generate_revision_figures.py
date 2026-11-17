#!/usr/bin/env python3
"""
Generate all publication-grade vector PDF figures for the manuscript.
Reads 100% dynamically from revision/outputs/*.csv and data/raw/*.csv.
- Vector PDF with empty metadata (double-blind anonymity compliant).
- All text sizes chosen so printed sizes remain >= 6.0 pt at 0.90*textwidth (A4).
- US English spelling; 'N-BEATSx', 'Hist-Exp', 'JKSE', 'warm-up' terminology.
"""

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.patches import FancyBboxPatch
import textwrap
from matplotlib.patches import Rectangle, FancyArrowPatch

import numpy as np
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent.parent.parent
OUT_CSV = BASE_DIR / "revision" / "outputs"
RAW_DIR = BASE_DIR / "data" / "raw"
FIG_DIR = BASE_DIR / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)

sys.path.append(str(BASE_DIR / "revision" / "scripts"))
import run_step1_stage1 as s1

EMPTY_META = {
    "Title": "",
    "Author": "",
    "Subject": "",
    "Keywords": "",
    "Creator": "",
    "Producer": "",
}

plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
    "axes.edgecolor": "#333333",
    "axes.linewidth": 0.8,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
})


def save_both(fig, filename: str):
    fig.savefig(FIG_DIR / filename, format="pdf", bbox_inches="tight", dpi=300, metadata=EMPTY_META)
    plt.close(fig)
    print(f"Saved {filename} -> {FIG_DIR}")



# Hapus blok ini jika save_both sudah didefinisikan di notebook/proyekmu.
if "save_both" not in globals():
    def save_both(fig, name):
        plt.show()


def make_fig1_methodology_flow():
    """Fig 1: Two-Stage Research Framework (placed at width=0.98\\textwidth -> scale ~0.78)."""
    plt.rcParams["font.family"] = "sans-serif"
    plt.rcParams["font.sans-serif"] = ["DejaVu Sans", "Arial", "Helvetica"]
    fig, ax = plt.subplots(figsize=(8.6, 5.6))
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 100)
    ax.axis("off")
    fig.subplots_adjust(left=0.01, right=0.99, top=0.99, bottom=0.01)

    HEAD, STEP, FS = 6.8, 2.85, 7.9

    def box(x, y, w, h, num, title, items, fc, ec, wrap=58):
        """items: str -> bullet; ("sub", str) -> baris turunan tanpa bullet."""
        # bayangan, badan, header
        ax.add_patch(Rectangle((x + 0.45, y - 0.6), w, h, facecolor="black",
                               edgecolor="none", alpha=0.09, zorder=1))
        ax.add_patch(Rectangle((x, y), w, h, facecolor=fc, edgecolor=ec,
                               linewidth=1.3, zorder=2))
        ax.add_patch(Rectangle((x, y + h - HEAD), w, HEAD, facecolor=ec,
                               edgecolor=ec, linewidth=1.3, zorder=3))
        cy = y + h - HEAD / 2
        ax.add_patch(Rectangle((x + 0.9, cy - 2.0), 3.2, 4.0, facecolor="white",
                               edgecolor="none", zorder=4))
        ax.text(x + 2.5, cy - 0.05, num, ha="center", va="center", fontsize=9.5,
                fontweight="bold", color=ec, zorder=5)
        ax.text(x + 5.4, cy, title, ha="left", va="center", fontsize=8.8,
                fontweight="bold", color="white", zorder=5)

        ty = y + h - HEAD - 2.8
        for it in items:
            sub = isinstance(it, tuple)
            text = it[1] if sub else it
            for i, ln in enumerate(textwrap.wrap(text, wrap - (3 if sub else 0))):
                if not sub and i == 0:
                    ax.add_patch(Rectangle((x + 1.5, ty - 0.4), 0.8, 0.8,
                                           facecolor=ec, edgecolor="none", zorder=5))
                ax.text(x + (4.6 if sub else 3.2), ty, ln, ha="left", va="center",
                        fontsize=FS, color="#111111", zorder=5)
                ty -= STEP
            ty -= 0.35

    def arrow(x0, y0, x1, y1):
        ax.add_patch(FancyArrowPatch((x0, y0), (x1, y1), arrowstyle="-|>",
                                     mutation_scale=13, lw=1.5, color="#333333",
                                     shrinkA=0, shrinkB=0, zorder=6))

    LX, RX, CW = 1.5, 51.5, 47.0
    H = 27.0
    Y1, Y2, Y3, H3 = 71.0, 39.0, 6.0, 24.0

    # Row 1 left: Data Collection & Alignment
    box(LX, Y1, CW, H, "1", "Data Collection & Alignment (2020–2026)", [
        "5 AI Equities (NVDA, MSFT, TSLA, GOOGL, META)",
        "Global & Domestic Crypto (BTC-USD, ETH-USD, BTC/IDR)",
        "Main US Calendar: 1,677 return dates (JKSE holiday flag)",
        "Robustness: 1,556 joint US–JKSE dates | Indodax 60m RV",
    ], fc="#EEF3FA", ec="#1D4E89")

    # Row 1 right: Stage 1 TVP-VAR
    box(RX, Y1, CW, H, "2", "Stage 1: Connectedness & Sensitivity", [
        "TVP-VAR(1), forgetting factors λ=0.99, κ=0.96, H=10",
        "250-obs warm-up discard (1,426 effective obs); p=0 vs p=1",
        "4 Volatility Proxies (log r², |r|, r², log Parkinson)",
        "300-rep block-shuffle null & 300-rep moving-block CI",
    ], fc="#FDF3E7", ec="#B45309")

    arrow(LX + CW + 0.3, Y1 + H / 2, RX - 0.3, Y1 + H / 2)

    # Row 2 right: Feature Engineering
    box(RX, Y2, CW, H, "3", "Stage-2 Feature Engineering (Causal Lag t)", [
        "NOSPILL (11): 7 US returns + weekend gap + 3 calendar",
        "SPILL (20): NOSPILL + 9 causal TVP-VAR spillovers",
        ("sub", "(TSI, AI→BTC/IDR, 5 AI, BTC-USD & ETH-USD NET)"),
        "Audit: Target NET = −∑7; VIF drops after warm-up",
    ], fc="#F3F7EC", ec="#3F6212")

    arrow(RX + CW / 2, Y1 - 0.3, RX + CW / 2, Y2 + H + 0.3)

    # Row 2 left: Stage 2 Forecasting
    box(LX, Y2, CW, H, "4", "Stage 2: Multi-Horizon Forecasting (h = 1..10)", [
        "Train/Val/Test: 915 / 167 / 335 cutoffs (post-warm-up)",
        "Econometric: Hist-Exp, Roll-20, GARCH(1,1)-N/t, EGARCH(1,1)-N/t, HAR, parsimonious HAR-X",
        "Deep Learning: TFT (MSE) & N-BEATSx (MAE), 5 seeds (+RV_60m)",
    ], fc="#EEF3FA", ec="#1D4E89")

    arrow(RX - 0.3, Y2 + H / 2, LX + CW + 0.3, Y2 + H / 2)

    # Row 3: Statistical Evaluation (full width)
    box(LX, Y3, 97.0, H3, "5", "Statistical Evaluation, Robustness & Tail Sensitivity (RQ1–RQ3)", [
        "Accuracy & Tail Calibration: MSE, MAE, RMSE, OOS R², QLIKE (positive-only & common-positive sets)",
        "Formal Testing: HLN Diebold-Mariano (MSE, MAE, QLIKE), explicit Bonferroni families, 90% MCS",
        "Outlier Concentration Audit: Full N=335 vs 333 dates omitting 5–6 Feb 2026 (ranks & Indodax volume)",
        "Stage-1 Sensitivity: Excl. BTC-USD (7 assets), +VIX/+IXIC/+GSPC (10 assets), FX-adjusted BTC/IDR",
    ], fc="#F6F2FA", ec="#5B21B6", wrap=118)

    arrow(LX + CW / 2, Y2 - 0.3, LX + CW / 2, Y3 + H3 + 0.3)
    save_both(fig, "fig1_methodology_flow.pdf")

def make_fig2_tsi():
    """Fig 2 (file fig1_tsi.pdf): Dynamic TSI 2020-2026 with warm-up shading, permutation null, & events."""
    main_panel = s1.build_aligned_panel(s1.ASSETS_8, join_ihsg=False)
    lbls = [a[1] for a in s1.ASSETS_8]
    proxies = s1.make_proxy_matrices(main_panel, lbls)
    res1 = s1.estimate_tvp_var_gfevd(proxies["log_r2"], p=1, warmup=250, causal_sigma0=True)
    dates_all = pd.to_datetime(main_panel["df_ret"]["Date"].values[1:])
    tsi_all = res1["tsi_all"]

    df_sum = pd.read_csv(OUT_CSV / "step1a_stage1_summary.csv").query("Sample == 'Main_US_1677' and Proxy == 'log_r2'").iloc[0]
    df_perm = pd.read_csv(OUT_CSV / "step1a_permutation_test.csv").query("Proxy == 'log_r2'").iloc[0]
    df_ev = pd.read_csv(OUT_CSV / "step1a_tsi_events.csv").set_index("Event")

    fig, ax = plt.subplots(figsize=(9.8, 4.0))
    w_end = dates_all[250]
    ax.axvspan(dates_all[0], w_end, color="#E5E7EB", alpha=0.85, zorder=0,
               label=f"First 250 warm-up steps (mean {df_sum['TSI_warmup_first250_mean']:.2f}%)")
    ax.axvline(w_end, color="#6B7280", linestyle=":", linewidth=1.1, zorder=1)

    ax.plot(dates_all[:251], tsi_all[:251], color="#9CA3AF", linewidth=1.0, zorder=2)
    ax.plot(dates_all[250:], tsi_all[250:], color="#1D4ED8", linewidth=1.35, zorder=3,
            label=r"Post-warm-up TSI ($\log r^2$, $p=1$)")

    post_mean = float(df_sum["TSI_post_mean"])
    null_mean = float(df_perm["Null_mean"])
    ax.axhline(post_mean, color="#DC2626", linestyle="--", linewidth=1.25, zorder=4,
               label=f"Post-warm-up mean ({post_mean:.2f}%)")
    ax.axhline(null_mean, color="#059669", linestyle="-.", linewidth=1.25, zorder=4,
               label=f"Block-shuffle null ({null_mean:.2f}%)")

    # Annotations >= 10.5 pt (gives >= 6.4 pt in print at 0.90 textwidth)
    anns = [
        ("2022-05-09", df_ev.loc["Terra_Luna_window_max", "TSI_log_r2_pct"],
         f"Terra-Luna\n({df_ev.loc['Terra_Luna_window_max', 'TSI_log_r2_pct']:.2f}%)", (-42, 14)),
        ("2022-11-10", df_ev.loc["FTX_window_max", "TSI_log_r2_pct"],
         f"FTX peak\n({df_ev.loc['FTX_window_max', 'TSI_log_r2_pct']:.2f}%)", (15, 11)),
        ("2025-03-07", df_ev.loc["Spring2025_min", "TSI_log_r2_pct"],
         f"Post-warm-up min\n({df_ev.loc['Spring2025_min', 'TSI_log_r2_pct']:.2f}%)", (-118, 2)),
        ("2025-04-10", df_ev.loc["April2025_tariff_peak", "TSI_log_r2_pct"],
         f"Apr 2025 spike\n({df_ev.loc['April2025_tariff_peak', 'TSI_log_r2_pct']:.2f}%)", (10, 12)),
    ]
    for dstr, yval, txt, xyoff in anns:
        dx = pd.Timestamp(dstr)
        ax.scatter([dx], [yval], color="#1E3A8A", s=26, zorder=5)
        ax.annotate(txt, xy=(dx, yval), xytext=xyoff, textcoords="offset points",
                    fontsize=10.2, color="#111827",
                    arrowprops=dict(arrowstyle="->", color="#4B5563", lw=0.9), zorder=6)

    ax.set_ylabel("Total Spillover Index (%)", fontsize=11.5)
    ax.set_xlabel("Date (US equity trading calendar, $N=1{,}677$ returns)", fontsize=11.5)
    ax.set_ylim(5, 95)
    ax.xaxis.set_major_locator(mdates.YearLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    ax.tick_params(axis="both", labelsize=10.5)
    ax.grid(axis="y", linestyle=":", alpha=0.45)
    ax.legend(loc="upper right", fontsize=9.8, frameon=True, facecolor="white", framealpha=0.92, ncol=2)

    save_both(fig, "fig1_tsi.pdf")


def make_fig3_net_spillover():
    """Fig 3 (file fig2_net_spillover.pdf): Directional NET spillovers with 5-95% bootstrap CIs across 4 proxies."""
    df_b = pd.read_csv(OUT_CSV / "step1c_bootstrap_net_ci_dy12.csv")
    assets = ["NVDA", "MSFT", "TSLA", "GOOGL", "META", "BTC-USD", "ETH-USD", "BTC/IDR"]
    proxies = [
        ("log_r2", r"$\log(r^2)$", "#1D4ED8"),
        ("abs_r", r"$|r|$", "#D97706"),
        ("r2", r"$r^2$", "#059669"),
        ("log_Parkinson_8asset_interval", r"$\log(\mathrm{Parkinson})$", "#7C3AED"),
    ]

    fig, ax = plt.subplots(figsize=(9.8, 4.1))
    x = np.arange(len(assets))
    w = 0.19
    offsets = [-1.5 * w, -0.5 * w, 0.5 * w, 1.5 * w]

    for (pkey, plabel, col), off in zip(proxies, offsets):
        sub = df_b[df_b["Proxy"] == pkey].set_index("Asset").loc[assets]
        pts = sub["Point_NET_pct_DY12"].values
        err_low = np.maximum(0.0, pts - sub["CI_05_DY12"].values)
        err_high = np.maximum(0.0, sub["CI_95_DY12"].values - pts)
        ax.bar(x + off, pts, width=w, color=col, alpha=0.88, edgecolor="#111111", linewidth=0.5, label=plabel, zorder=3)
        ax.errorbar(x + off, pts, yerr=[err_low, err_high], fmt="none", ecolor="#111827",
                    elinewidth=0.95, capsize=2.0, capthick=0.85, zorder=4)

    ax.axhline(0.0, color="#111827", linewidth=1.0, zorder=2)
    ax.axvline(4.5, color="#6B7280", linestyle="--", linewidth=1.0, zorder=1)
    ax.set_xticks(x)
    ax.set_xticklabels(assets, fontsize=10.8, fontweight="bold")
    ax.set_ylabel("Directional Net Spillover (%)", fontsize=11.5)
    ax.set_xlabel("Asset (Left: AI Technology Equities | Right: Cryptocurrencies)", fontsize=11.2)
    ax.tick_params(axis="y", labelsize=10.5)
    ax.grid(axis="y", linestyle=":", alpha=0.45)
    ax.legend(loc="upper left", fontsize=10.2, frameon=True, facecolor="white", ncol=4)
    ax.set_ylim(-15.5, 13.0)

    save_both(fig, "fig2_net_spillover.pdf")


def make_fig4_error_concentration():
    """Fig 4 (file fig3_error_concentration.pdf): h=1 error share (left) + Feb 5-6 Indodax returns & volume (right)."""
    df_r = pd.read_csv(BASE_DIR / "revision" / "audit" / "v2_outlier_ranks_13main.csv")
    sub = df_r[df_r["Horizon"] == "h=1"].copy()
    # Clean display order matching main models
    name_map = {
        "Hist-Exp": "Hist-Exp",
        "Roll-20": "Roll-20",
        "GARCH(1,1)-N": "GARCH(1,1)-N",
        "EGARCH(1,1)-N": "EGARCH(1,1)-N",
        "GARCH(1,1)-t": "GARCH(1,1)-t",
        "EGARCH(1,1)-t": "EGARCH(1,1)-t",
        "HAR": "HAR",
        "HAR-X-NOSPILL": "HAR-X-NOSPILL",
        "HAR-X-SPILL": "HAR-X-SPILL",
        "HAR-X-FULL": "HAR-X-FULL",
        "TFT-NOSPILL": "TFT-NOSPILL",
        "TFT-SPILL": "TFT-SPILL",
        "NBEATSx-NOSPILL": "N-BEATSx-NOSPILL",
        "NBEATSx-SPILL": "N-BEATSx-SPILL",
        "N-BEATSx-NOSPILL_raw": "N-BEATSx-NOSPILL",
        "N-BEATSx-SPILL_raw": "N-BEATSx-SPILL",
    }
    sub = sub[sub["Model"] != "HAR-X-FULL"].copy()
    sub["Label"] = sub["Model"].map(name_map)
    share_col = "Feb5_6_Err_Share_pct" if "Feb5_6_Err_Share_pct" in sub.columns else "Feb5_6_Error_Share_pct"
    sub = sub.sort_values(share_col, ascending=True).reset_index(drop=True)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10.6, 4.7), gridspec_kw={"width_ratios": [1.15, 1.0]})

    # Panel (a): Error concentration bar chart across 13 evaluated models
    y_pos = np.arange(len(sub))
    feb_sh = sub[share_col].values
    rem_sh = 100.0 - feb_sh
    ax1.barh(y_pos, feb_sh, color="#DC2626", alpha=0.88, label="5–6 Feb 2026 (2 dates)")
    ax1.barh(y_pos, rem_sh, left=feb_sh, color="#93C5FD", alpha=0.85, label="Remaining 333 dates")
    for i, val in enumerate(feb_sh):
        ax1.text(val - 3.0, i, f"{val:.1f}%", va="center", ha="right", color="white", fontsize=9.8, fontweight="bold")
    ax1.set_yticks(y_pos)
    ax1.set_yticklabels(sub["Label"].values, fontsize=10.0)
    ax1.set_xlim(0, 100)
    ax1.set_xlabel("Share of $h=1$ Total Squared Error (%)", fontsize=10.8)
    ax1.set_title("(a) $h=1$ Error Concentration (13 Main Models)", fontsize=11.2, fontweight="bold", pad=22)
    ax1.tick_params(axis="x", labelsize=10.0)
    ax1.legend(loc="lower center", bbox_to_anchor=(0.5, 1.005), fontsize=9.2, frameon=True, facecolor="white", ncol=2)
    ax1.grid(axis="x", linestyle=":", alpha=0.45)

    # Panel (b): Indodax BTC/IDR daily log return & volume around 5-6 Feb 2026
    raw_ind_path = RAW_DIR / "crypto_btc_idr_indodax.csv"
    if not raw_ind_path.exists():
        plt.close(fig)
        raise SystemExit(f"Raw vendor file missing for Figure 3b: {raw_ind_path}. Run src/collect_data.py first.")
    raw_ind = pd.read_csv(raw_ind_path)
    raw_ind["Date"] = pd.to_datetime(raw_ind["Date"])
    raw_ind = raw_ind.sort_values("Date").reset_index(drop=True)
    raw_ind["ret_pct"] = np.log(raw_ind["Close"] / raw_ind["Close"].shift(1)) * 100.0
    win = raw_ind[(raw_ind["Date"] >= "2026-01-22") & (raw_ind["Date"] <= "2026-02-20")].copy()

    ax2.bar(win["Date"], win["Volume"], width=0.72, color=[
        "#DC2626" if d.strftime("%Y-%m-%d") in {"2026-02-05", "2026-02-06"} else "#CBD5E1"
        for d in win["Date"]
    ], alpha=0.9, label="Indodax Volume (BTC)")
    ax2.set_ylabel("Daily Volume (BTC)", fontsize=10.8, color="#374151")
    ax2.set_ylim(0, 225)
    ax2.tick_params(axis="y", labelsize=10.0)

    ax2_r = ax2.twinx()
    ax2_r.plot(win["Date"], win["ret_pct"], color="#1D4ED8", marker="o", markersize=3.5, linewidth=1.35, label="Log Return (%)")
    ax2_r.axhline(0.0, color="#6B7280", linestyle=":", linewidth=0.8)
    ax2_r.set_ylabel("BTC/IDR Log Return (%)", fontsize=10.8, color="#1D4ED8")
    ax2_r.set_ylim(-20, 18)
    ax2_r.tick_params(axis="y", labelsize=10.0, colors="#1D4ED8")

    # Annotate exact Feb 5 and Feb 6 values from raw_ind
    r5 = win[win["Date"] == "2026-02-05"].iloc[0]
    r6 = win[win["Date"] == "2026-02-06"].iloc[0]
    ax2.annotate(f"5 Feb\n{r5['Volume']:.1f} BTC\n({r5['ret_pct']:.2f}%)",
                 xy=(r5["Date"], r5["Volume"]), xytext=(-55, 10), textcoords="offset points",
                 fontsize=9.3, color="#991B1B", fontweight="bold",
                 arrowprops=dict(arrowstyle="->", color="#991B1B", lw=0.9))
    ax2.annotate(f"6 Feb\n{r6['Volume']:.1f} BTC\n(+{r6['ret_pct']:.2f}%)",
                 xy=(r6["Date"], r6["Volume"]), xytext=(18, -14), textcoords="offset points",
                 fontsize=9.3, color="#1E40AF", fontweight="bold",
                 arrowprops=dict(arrowstyle="->", color="#1E40AF", lw=0.9))

    ax2.xaxis.set_major_locator(mdates.DayLocator(interval=7))
    ax2.xaxis.set_major_formatter(mdates.DateFormatter("%b %d"))
    ax2.tick_params(axis="x", labelsize=9.8)
    ax2.set_xlabel("UTC Date (Jan–Feb 2026 Window)", fontsize=10.8)
    ax2.set_title("(b) Indodax BTC/IDR Volume & Log Return", fontsize=11.2, fontweight="bold", pad=22)

    plt.tight_layout(w_pad=2.0)
    save_both(fig, "fig3_error_concentration.pdf")


def main():
    make_fig1_methodology_flow()
    make_fig2_tsi()
    make_fig3_net_spillover()
    make_fig4_error_concentration()


if __name__ == "__main__":
    main()
