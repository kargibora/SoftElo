# Soft-Elo

**From Uncertain Judgments to Calibrated Rankings: Conformal Elo Estimation for LLM Evaluation**

Bora Kargi and David Salinas · **NeurIPS 2026 · Poster**

[Paper](https://arxiv.org/abs/2606.13221) · [Project page](https://kargibora.github.io/conformal-elo/) · [Data](data/) · [Citation](#citation)

Soft-Elo converts an AI judge's score differences into calibrated preference probabilities, then fits Elo ratings to those probabilities. Conformal intervals capture the remaining disagreement with human ratings.

This repository runs the core evaluation pipeline from saved judgments. It includes calibration, leave-one-model-out evaluation, bootstrap summaries, and conformal diagnostics. No judge API calls or GPU are needed.

## Quick start

Requires Python 3.10+.

```bash
git clone https://github.com/kargibora/SoftElo.git
cd SoftElo
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt "Jinja2>=3.1"
python run_demo.py --alpha 0.20
python make_tables.py
python plot_results.py
```

Jinja2 is required for LaTeX table export. The 1,000-row sample covers 14 models; `--alpha 0.20` requests 80% coverage because each split has only seven calibration models. The CLI default is 90%.

Expected sample results with seed 42:

| Method | Elo MAE ↓ | Spearman correlation ↑ |
|---|---:|---:|
| Hard-Elo | 57.63 | 0.960 |
| Soft-Elo | 19.54 | 0.969 |

These are demo results, not the full-paper results. Outputs include CSV summaries, `tables.md`, LaTeX tables, and `diagnostics.png` under `outputs/`.

## Implementation

All numerical calculations are in [`softelo_minimal/core.py`](softelo_minimal/core.py).

**Calibrate score differences.** `score_gap` averages the score differences across rubric dimensions shared by both answers. `fit_beta_mle` fits β on human non-tie preferences by minimizing binary cross-entropy:

$$p_{AB}=\sigma(\beta\Delta s),\qquad \sigma(z)=\frac{1}{1+e^{-z}}.$$

**Fit ratings.** `fit_bt` fits Bradley–Terry strengths, with predicted win probability $q_{AB}=\sigma(\theta_A-\theta_B)$. It minimizes cross-entropy against the comparison targets, plus L2 regularization (`λ = 0.01`). Hard-Elo uses win/tie/loss targets `1 / 0.5 / 0`; Soft-Elo uses $p_{AB}$. Centered strengths become Elo ratings through

$$R_m=1500+\frac{400}{\ln 10}\theta_m.$$

**Evaluate held-out models.** `loo_evaluation` excludes one model, fits β and opponent ratings using comparisons between the remaining models, then estimates the excluded model against those fixed opponents. Its human votes are used only to obtain the evaluation reference, not to calibrate β or estimate its judge-derived rating. `fit_bt` uses L-BFGS-B; the single-model fit uses `solve_target_theta`.

**Compute intervals.** `bootstrap_target_se` resamples a model's comparisons while keeping β and opponent ratings fixed. `conformal_summary` splits the held-out results into calibration/test halves, calibrates the normalized error $|R^{human}-R^{judge}|/\mathrm{SE}$, and evaluates intervals $R^{judge}\pm\hat q\,\mathrm{SE}$. Coverage relies on exchangeability; the reported coverage is empirical across repeated splits.

`run_demo.py` runs this pipeline. `make_tables.py` and `plot_results.py` format its outputs without refitting ratings. `beta_full_fit.csv` is a separate whole-input diagnostic, not the β used in held-out evaluation.

## Data and full-size runs

Required CSV columns:

```text
model_a,model_b,human_pref,judge_pref,scores_a,scores_b
```

For both preference columns, **`0` means A wins, `1` means B wins, and `0.5` means tie**. The implementation converts these labels to A-win probabilities internally. Scores are dictionaries such as `{"helpfulness": 8, "accuracy": 7}`.

[`data/manifest.csv`](data/manifest.csv) lists the released LMArena 100K, LMArena 140K, and ComparIA judge files. They include preferences, scores, metadata, and hashed identifiers, but no raw prompts or completions. Run each judge file separately:

```bash
python run_demo.py \
  --data data/lmarena100k/annotations_qwen3_5_27b.csv \
  --out outputs_qwen \
  --alpha 0.10
python make_tables.py --out outputs_qwen
python plot_results.py --out outputs_qwen
```

Use `python run_demo.py --help` for bootstrap, split, and seed settings. This package covers the core pipeline, not every paper ablation. `export_release_data.py` is a maintainer utility requiring the original research files; it is not needed for the released CSVs.

## Current limitations

- The conformal quantile clips to the largest calibration error when there are too few models. A finite 90% threshold requires at least nine calibration models; the sample has seven.
- The single-model Newton solver can return extreme values without detecting non-convergence. It needs a safeguarded update before use on arbitrary new inputs.
- Very small bootstrap counts produce unstable intervals. Inputs must follow the documented schema; malformed labels are not always rejected.

Source data and terms: [LMArena 100K](https://huggingface.co/datasets/lmarena-ai/arena-human-preference-100k), [LMArena 140K](https://huggingface.co/datasets/lmarena-ai/arena-human-preference-140k), [ComparIA](https://huggingface.co/datasets/ministere-culture/comparia-votes).

## Citation

```bibtex
@inproceedings{kargi2026softelo,
  title     = {From Uncertain Judgments to Calibrated Rankings:
               Conformal Elo Estimation for LLM Evaluation},
  author    = {Kargi, Bora and Salinas, David},
  booktitle = {Advances in Neural Information Processing Systems},
  year      = {2026},
  url       = {https://arxiv.org/abs/2606.13221}
}
```
