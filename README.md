# Soft-Elo Reproducibility Package

This directory is an anonymizable, standalone package for reproducing the core
Hard-Elo vs. Soft-Elo pipeline without the full internal experiment tree. It is
not a toy reimplementation: it includes leave-one-model-out Elo estimation,
held-out temperature fitting, bootstrap summaries, score calibration diagnostics,
beta-stability checks, and model-level split conformal intervals.

## Contents

- `softelo_minimal/core.py`: self-contained Hard-Elo, Soft-Elo, LOO evaluation,
  beta fitting, bootstrap summaries, score diagnostics, and model-level split
  conformal intervals.
- `run_demo.py`: command-line entry point for all CSV outputs.
- `make_tables.py`: converts the summary CSVs into compact Markdown and LaTeX
  tables.
- `plot_results.py`: creates a compact diagnostic figure from the outputs.
- `data/sample_annotations.csv`: 1,000-row test annotation file.
- `data/lmarena100k/`, `data/lmarena140k/`, `data/comparia/`: sanitized
  paper annotation CSVs for the judges reported in the paper.
- `data/manifest.csv`: row counts, source JSON paths, and export policy for
  each sanitized CSV.
- `export_release_data.py`: regenerates the sanitized CSVs from the configured
  raw annotation roots in the full repository.
- `outputs/`: generated result CSVs.

The exported data contains only model IDs, preference labels, score
dictionaries, lengths, lightweight metadata, and deterministic hashed IDs. It
excludes prompts and completions.

## Input Format

The expected CSV columns are:

```text
model_a,model_b,human_pref,judge_pref,scores_a,scores_b
```

The full release CSVs include additional metadata columns such as `battle_id`,
`instruction_id_hash`, `original_question_id_hash`, `language`, and `source`;
these are optional for the reference implementation.

Preference convention:

- `human_pref = 0` means model A wins.
- `human_pref = 1` means model B wins.
- `human_pref = 0.5` means tie.

The same convention is used for `judge_pref`. `scores_a` and `scores_b` are
Python/JSON-like dictionaries containing rubric scores for each completion.

## Run

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python run_demo.py
python make_tables.py
python plot_results.py
```

The default run settings match the paper experiment config where applicable:
`--bootstrap-resamples 20`, `--split-repeats 5`, `--alpha 0.10`, and
`--seed 42`.

The script writes:

- `outputs/loo_summary.csv`
- `outputs/heldout_models.csv`
- `outputs/conformal_summary.csv`
- `outputs/bootstrap_summary.csv`
- `outputs/beta_full_fit.csv`
- `outputs/beta_stability_draws.csv`
- `outputs/beta_stability_summary.csv`
- `outputs/calibration_bins.csv`
- `outputs/score_signal_summary.csv`
- `outputs/run_config.json`
- `outputs/tables.md`
- `outputs/tables_tex/elo_summary.tex`
- `outputs/tables_tex/conformal_summary.tex`
- `outputs/diagnostics.png`

`run_demo.py` is the source of the numerical outputs. `make_tables.py` only
formats `loo_summary.csv` and `conformal_summary.csv`; `plot_results.py` only
plots `heldout_models.csv`, `calibration_bins.csv`, and
`beta_stability_summary.csv`.

## Method Summary

Hard-Elo fits Bradley--Terry to the discrete judge label after converting it to
`P(A beats B)`.

Soft-Elo first fits a scalar temperature `beta` on human non-tie comparisons:

```text
P(A beats B | x) = sigmoid(beta * (score_a - score_b))
```

In leave-one-model-out evaluation, beta is fit on anchor-only battles, excluding
the held-out model's human-labeled battles. The resulting soft targets replace
the hard judge labels in the same Bradley--Terry objective.

The conformal example uses normalized residuals:

```text
abs(human_elo - judge_elo) / bootstrap_se(judge_elo)
```

and returns model-level intervals centered at the LLM-derived Elo estimate.

## Typical Custom Run

```bash
python run_demo.py \
  --data data/sample_annotations.csv \
  --out outputs \
  --bootstrap-resamples 50 \
  --summary-bootstrap-repeats 1000 \
  --beta-stability-repeats 50 \
  --split-repeats 5 \
  --alpha 0.10 \
  --seed 42
python make_tables.py --out outputs
python plot_results.py --out outputs
```

To run a paper-sized sanitized CSV, pass any file listed in `data/manifest.csv`:

```bash
python run_demo.py \
  --data data/lmarena100k/annotations_qwen3_5_27b.csv \
  --out outputs_qwen3_5_27b \
  --bootstrap-resamples 20 \
  --summary-bootstrap-repeats 1000 \
  --beta-stability-repeats 20 \
  --split-repeats 5 \
  --seed 42
python make_tables.py --out outputs_qwen3_5_27b
python plot_results.py --out outputs_qwen3_5_27b
```

Avoid using very small `--bootstrap-resamples` values for reported conformal
widths. The conformal score is normalized by a bootstrap standard error, so
smoke-test settings such as `--bootstrap-resamples 2` can produce near-zero SE
estimates and artificially huge interval widths. The default value, `20`,
matches the paper experiment config.

## Notes for Anonymized Submission

Before submission, you can zip this directory as a standalone artifact. The
included sanitized CSVs are generated from the configured paper annotation
roots and intentionally omit raw prompts and completions. Check the source
dataset and model-provider terms before redistributing cached annotations.
