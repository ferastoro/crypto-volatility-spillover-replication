# Replication Package: Global AI Stock Volatility Connectedness and Bitcoin Volatility Forecasting in Indonesia

This anonymized replication package contains the code, processed/derived datasets, figure scripts, and verification tests accompanying the manuscript *"Global AI Stock Volatility Connectedness and Bitcoin Volatility Forecasting in Indonesia: Empirical and Deep Learning Evidence"*.

---

## 1. Repository Layout

- `src/`: Public market data downloader (`collect_data.py` using Yahoo Finance via `yfinance` and Indodax public `history_v2`, requiring no API credentials), preprocessing pipeline (`preprocess_data.py`), split verification (`verify_stock_splits.py`), and shared Stage-2 helpers (`v5_helpers.py`).
- `revision/scripts/`: Stage-1 TVP-VAR / GFEVD (`run_step1_stage1.py`), DY12 export (`export_dy12_tables.py`), basic bootstrap CIs (`compute_bootstrap_basic_ci.py`), causal sigma_0 comparison (`compute_round4_stage1_diff.py`), Stage-2 econometric evaluation (`run_step1_stage2.py`), intraday RV60m local script (`run_step3d_indodax_rv_local.py`), and publication figure generator (`generate_revision_figures.py`).
- `analysis_new/`: Self-contained evaluation scripts for matched-sample HAR-X (`run_analysis1_matched_harx.py`), loss-differential CIs (`run_analysis2_loss_diff_ci.py`), stationary bootstrap CIs (`run_bootstrap_loss_diff_ci.py`), intraday RV60m table (`run_analysis3_rv60m_table.py`, `run_rv60m_spill_vs_no_ae_qlike.py`), mixed-proxy ablation (`run_mixed_proxy_ablation.py`), rolling neural refit evaluation (`run_neural_refit_eval.py`, `run_neural_refit_ranks.py`), TVP-VAR estimator sensitivity (`run_stage1_estimator_sensitivity.py`), VaR backtests (`run_var_backtest_appendix.py`), and HLN-DM audit (`step1_audit.py`).
- `revision/kaggle/`: GPU training/refit Python scripts and stripped Jupyter notebooks (`kaggle_3a3b_v2.ipynb`, `kaggle_3c3d_v2.ipynb`, `refit20_kaggle.py`, `run_3a_warmup_excluded.py`, `run_3b_nonneg_nbeatsx.py`, `run_3c_tft_rolling_refit.py`, `run_3d_intraday_rv_dl.py`). All cell outputs, execution counts, and identifying metadata have been removed.
- `data/clean/`: Derived daily log-returns, target variance proxies, calendar indicators, and Stage-1 TVP-VAR feature series (no raw vendor OHLCV or volume columns).
- `inputs/v5/`: Derived causal Stage-1 covariates and fixed-window econometric baseline predictions (`HAR`, `GARCH11`, `EGARCH`).
- `revision/outputs/` & `revision/kaggle_outputs/`: Precomputed derived outputs, econometric baseline predictions, neural ensemble predictions, and diagnostic CSVs.
- `results/` & `results_new/`: Saved statistical summary tables and rolling-refit logs backing manuscript Tables 1–13, A1, and B1–B6. Raw OHLCV/volume columns (`Open`, `High`, `Low`, `Close`, `Volume`, `vol_60m_sum`) have been stripped.
- `figures/`: Manuscript vector PDF figures (`fig1_methodology_flow.pdf`, `fig1_tsi.pdf`, `fig2_net_spillover.pdf`, `fig3_error_concentration.pdf`).
- `tests/`: Consistency verification suite (`test_manuscript_numbers.py` and `check_feb2026.py`).
- `requirements.txt`, `LICENSE`, `CITATION.cff`: Pinned environment specification, MIT/CC BY 4.0 licenses, and anonymous citation metadata.

---

## 2. Environment Setup

Create a fresh virtual environment and install the pinned packages from `requirements.txt` (recorded from the local Python 3.13 verification environment):

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

**Environment note on PyTorch versions:** The local CPU verification environment recorded in `requirements.txt` uses `torch==2.13.0` (`neuralforecast==3.3.0`, `pytorch_lightning==2.6.6`, `arch==8.0.0`, `statsmodels==0.14.6`). The GPU deep learning runs reported in the manuscript were executed on Kaggle Tesla T4 GPUs using `PyTorch 2.11.0+cu130` (`neuralforecast==3.3.0`).

**Optional Kaggle path configuration:** GPU scripts in `revision/kaggle/` look for datasets relative to the repository (`./data/clean`, `./inputs/v5`, `./revision/outputs`) or under an environment-configured root `KAGGLE_INPUT_ROOT`:

```bash
export KAGGLE_INPUT_ROOT="/kaggle/input/anonymous-dataset"
```

---

## 3. Data Provenance & Raw-Data Retrieval Notice

Raw vendor market price feeds (`data/raw/*.csv`, `revision/data_ext/idx_gspc.csv`, `revision/data_ext/idx_ixic.csv`, and `revision/data_ext/indodax_btcidr_60m.csv`) are **not redistributed** in this repository per provider terms of use. Instead, all derived return, volatility proxy, spillover, and prediction files are provided alongside public download scripts (`src/collect_data.py`).

To re-fetch public raw feeds (requires internet access, ~2–5 minutes):

```bash
python src/collect_data.py
python src/verify_stock_splits.py
python src/preprocess_data.py
```

**Which outputs require re-fetching raw feeds vs. run directly from bundled derived files:**
- **Requires raw feeds first (`python src/collect_data.py`):**
  - Full Stage-1 TVP-VAR estimation from OHLCV bars (`revision/scripts/run_step1_stage1.py`): Table 1, Table 4 & Table A1's 8-asset $\log(\text{Parkinson})$ column, Table 5, and Table B1 macro/NY-close/mixed-proxy checks (`analysis_new/run_mixed_proxy_ablation.py`, `analysis_new/run_stage1_estimator_sensitivity.py`).
  - Re-aggregating 60-minute Indodax OHLCV bars (`revision/scripts/run_step3d_indodax_rv_local.py`).
  - Figure 3(b) Indodax daily volume bar overlay (`revision/scripts/generate_revision_figures.py`).
  - February 3–7, 2026 raw close/volume spot check (`tests/check_feb2026.py`, which automatically prints `SKIP` when `data/raw/` is absent).
- **Runs directly offline from bundled processed/derived data:**
  - Stage-1 DY12 table assembly & 90% percentile/basic bootstrap CIs (`revision/scripts/export_dy12_tables.py`, `revision/scripts/compute_bootstrap_basic_ci.py` -> Table 4 & Table A1).
  - Causal $\Sigma_0$ comparison (`revision/scripts/compute_round4_stage1_diff.py` -> Table B1 diffs).
  - Full Stage-2 econometric baselines, HLN-DM tests, Bonferroni families, MCS, outlier ranks (`revision/scripts/run_step1_stage2.py` -> Tables 3, 6, 7, 8, 13, B3).
  - Matched-sample HAR-X ablation (`analysis_new/run_analysis1_matched_harx.py` -> Table 9).
  - Parametric & stationary bootstrap loss-differential CIs (`analysis_new/run_analysis2_loss_diff_ci.py`, `analysis_new/run_bootstrap_loss_diff_ci.py`, `analysis_new/step1_audit.py` -> Tables 10–11).
  - Intraday `RV_60m` summary table & `AE`/`QLIKE` significance checks from derived `step3d_rv60m_econometric.csv` and `metrics_by_model.csv` (`analysis_new/run_analysis3_rv60m_table.py`, `analysis_new/run_rv60m_spill_vs_no_ae_qlike.py` -> Table 12).
  - 20-origin rolling-refit neural evaluation (`analysis_new/run_neural_refit_eval.py`, `analysis_new/run_neural_refit_ranks.py` -> Table B4).
  - Student-$t$ VaR & Kupiec backtest appendix (`analysis_new/run_var_backtest_appendix.py` -> Table B6).

Note on static reference tables: `revision/outputs/step1g_dl_specs.csv` (Table 2 specification summary, matching `revision/kaggle_outputs/outputs_3a3b_v2/config_used.json`) and `revision/outputs/step1f_chow_breaks.csv` (Table B2 Chow break output) are bundled as derived CSVs.

---

## 4. Commands to Regenerate Tables, Figures, and Verification Checks

| Manuscript Target | Command(s) | Data Mode | Expected Runtime |
|---|---|---|---|
| **Tables 1, 4, 5, A1, B1, B5** (from raw feeds) | `python src/collect_data.py && python revision/scripts/run_step1_stage1.py && python analysis_new/run_stage1_estimator_sensitivity.py && python analysis_new/run_mixed_proxy_ablation.py` | Raw fetch + CPU | ~5–12 min |
| **Table 4 & Table A1** (DY12 & CIs from Stage-1 CSVs) | `python revision/scripts/export_dy12_tables.py && python revision/scripts/compute_bootstrap_basic_ci.py` | Derived (offline) | < 2 s |
| **Tables 3, 6, 7, 8, 13, B3** (Stage-2 econometrics, DM, MCS) | `python revision/scripts/run_step1_stage2.py` | Derived (offline) | ~30–60 s |
| **Table 9** (Matched HAR-X) | `python analysis_new/run_analysis1_matched_harx.py` | Derived (offline) | < 3 s |
| **Tables 10 & 11** (Loss-diff & bootstrap CIs) | `python analysis_new/run_analysis2_loss_diff_ci.py && python analysis_new/run_bootstrap_loss_diff_ci.py` | Derived (offline) | ~10 s |
| **Table 12** (Intraday `RV_60m`) | `python analysis_new/run_analysis3_rv60m_table.py && python analysis_new/run_rv60m_spill_vs_no_ae_qlike.py` | Derived (offline) | < 3 s |
| **Table B4** (Rolling-refit neural eval) | `python analysis_new/run_neural_refit_eval.py && python analysis_new/run_neural_refit_ranks.py` | Derived (offline) | < 5 s |
| **Table B6** (VaR & Kupiec backtest) | `python analysis_new/run_var_backtest_appendix.py` | Derived (offline) | < 2 s |
| **Stage-2 Neural Models** (GPU refits / fixed runs) | Run `revision/kaggle/run_3a_warmup_excluded.py`, `run_3b_nonneg_nbeatsx.py`, `run_3c_tft_rolling_refit.py`, `run_3d_intraday_rv_dl.py`, `refit20_kaggle.py` | Derived + GPU | ~20–90 min (T4 GPU) |
| **Figures 1–3** | `python revision/scripts/generate_revision_figures.py` | Needs `data/raw/` for Fig. 3b | ~5 s |
| **Consistency Tests** | `python tests/check_feb2026.py` and `python tests/test_manuscript_numbers.py /path/to/main.tex` | Skips cleanly if raw / TeX absent | < 2 s |

---

## 5. Random Seeds & Licenses

- **Stage-1 permutation (300 reps) & moving-block bootstrap (300 reps):** `RNG_SEED = 20260904`.
- **Stage-2 stationary bootstrap CIs (10,000 reps):** `SEED = 20260101`.
- **Stage-2 neural training & rolling refits (`TFT`, `N-BEATSx`):** Seeds `[42, 7, 13, 21, 99]`.
- **Code License:** MIT License (`Anonymous`).
- **Derived Data & Results License:** Creative Commons Attribution 4.0 International (`CC BY 4.0`).
