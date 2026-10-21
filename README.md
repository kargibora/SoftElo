# Soft-Elo

[![Paper](https://img.shields.io/badge/Paper-OpenReview-8c1b13)](https://openreview.net/forum?id=QRZ8c3QtOj)
[![NeurIPS 2026](https://img.shields.io/badge/NeurIPS_2026-Poster-6847a3)](https://neurips.cc/virtual/2026/loc/paris/poster/152776)
[![Dataset](https://img.shields.io/badge/Hugging_Face-Dataset-ffd21e)](https://huggingface.co/datasets/kbora/SoftElo)

**From Uncertain Judgments to Calibrated Rankings: Conformal Elo Estimation for LLM Evaluation**

Bora Kargi and David Salinas · NeurIPS 2026

Soft-Elo maps a judge's score differences to calibrated win probabilities and
fits Bradley–Terry ratings using those probabilities. Conformal calibration
then estimates intervals for the difference between judge-derived and human-derived
ratings. The method targets the Elo scale; it does not remove every judge bias.

On LMArena100K, the reproduced mean Elo MAE across eight judges is **45.85 for
Hard-Elo and 17.87 for Soft-Elo**. See the
[verification report](reproduction/VERIFICATION.md) for the comparisons and scope.

This repository provides annotation-loading examples, paper-reproduction commands,
and a compact CSV demo. Reproduction compares newly fitted results with archived
measurements and exits with an error when they disagree.

## Reproduce one judge

The paper-reproduction environment uses Python 3.12 and a committed `uv.lock`.
The computations run on CPU using existing annotations. They make no judge API
calls and need no GPU.

```bash
git clone https://github.com/kargibora/SoftElo.git
cd SoftElo
uv sync --project reproduction --locked --python 3.12
```

The [Hugging Face dataset](https://huggingface.co/datasets/kbora/SoftElo) is currently
private for author review. An account with access must authenticate first:

```bash
uv run --project reproduction hf auth login
uv run --project reproduction python reproduce.py \
  --suite main --judge GPT-OSS-120B --out outputs/gpt120
```

The command downloads a pinned dataset revision, fits Hard-Elo and Soft-Elo for
each held-out model, and compares all predictions and fitted temperatures with
the saved reference. Outputs include `summary.csv`, `heldout_models.csv`,
`run_config.json`, and `validation.json`.

For an existing local dataset checkout, add `--data /path/to/dataset`. Install
without uv using `pip install ./reproduction`; this does not reproduce the lock file.

## Reproduction experiments

Omit `--judge` to evaluate every judge in a suite. `--jobs` controls parallel CPU
workers. Two workers are the default.

```bash
# Main Table 2: eight judges, fold-specific temperature fitting.
uv run --project reproduction python reproduce.py --suite main --jobs 2

# Appendix E.1: constant smoothing and the pooled-temperature baseline.
uv run --project reproduction python reproduce.py --suite smoothing --jobs 2

# Main Table 3: the original bootstrap and conformal calculation.
uv run --project reproduction python reproduce.py --suite conformal-historical --jobs 2

# LMArena 140K: seven judges, fold-specific temperature fitting.
uv run --project reproduction python reproduce.py --suite lmarena140k --jobs 2

# All 31 judge/protocol combinations.
uv run --project reproduction python reproduce.py --suite all --jobs 2
```

| Experiment | Reference comparison |
| --- | --- |
| LMArena 100K main | Every held-out Hard/Soft rating, human reference rating and fitted temperature |
| Constant smoothing | Per-judge MAE for all five constants and both baselines |
| Historical conformal | Model ratings, bootstrap standard errors and all five split outcomes |
| LMArena 140K main | Archived per-judge summary statistics; model-level references were not saved |

The comparison uses absolute tolerance `1e-5` and requires matching model/method
identities. The aggregate `validation.json` lists each comparison and any failures.
See [reproduction/VERIFICATION.md](reproduction/VERIFICATION.md) for completed checks.

The main and smoothing analyses use different temperature-fitting protocols.
Constants are compared retrospectively using human Elo. The historical conformal
command preserves the original quantile interpolation and seeds to reproduce the
saved results; it is not an implementation of the direct conformal order statistic.
The original Hard-Elo bootstrap resamples anchor battles, while Soft-Elo keeps
its anchors fixed. The historical reproduction preserves this asymmetry.

To rebuild tables from saved measurements without running fits:

```bash
uv run --project reproduction softelo-reproduce \
  --cached-results /path/to/dataset/results --out outputs/archived-tables
```

## Load the annotations

```bash
uv run --project reproduction python examples/load_annotations.py \
  --corpus lmarena100k --judge GPT-OSS-120B --csv annotations.csv
```

The dataset contains LMArena 100K and LMArena 140K annotations. Each row is one
judge's annotation of one battle. Use each judge's valid rows for the paper
reproduction; intersecting battles across all judges changes the experiment.
The dataset card documents score orientation, missing per-order scores, upstream
source revisions and licenses.

## Compact CSV demo

`run_demo.py` retains the original compact implementation for exploring the method
with CSV input. It treats weak preferences and tied judge scores differently
from the paper protocols. Use `reproduce.py` above when checking paper numbers.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python run_demo.py --data data/sample_annotations.csv --out outputs/demo
python make_tables.py --out outputs/demo
python plot_results.py --out outputs/demo
```

The input columns are `model_a`, `model_b`, `human_pref`, `judge_pref`, `scores_a`
and `scores_b`. Preferences use 0 for A wins, 1 for B wins and 0.5 for a tie;
score dictionaries are JSON strings. The loading example above exports compatible
CSV files. The original CSV snapshots remain available under `data/`.

## Tests and provenance

```bash
uv run --project reproduction python -m unittest discover -s reproduction/tests
```

The reference implementation preserves the original numerical operations.
[core_provenance.json](reproduction/softelo_release/core_provenance.json) records
source paths and hashes. LMArena140K summary references and their hashes are in
[reproduction/references](reproduction/references). The extracted implementation
uses the Apache-2.0 license; see [LICENSE](reproduction/LICENSE) and
[NOTICE](reproduction/NOTICE). Dataset terms are documented separately on Hugging Face.

## Citation

```bibtex
@inproceedings{kargi2026softelo,
  title = {From Uncertain Judgments to Calibrated Rankings: Conformal Elo Estimation for LLM Evaluation},
  author = {Kargi, Bora and Salinas, David},
  booktitle = {Advances in Neural Information Processing Systems},
  year = {2026},
  url = {https://openreview.net/forum?id=QRZ8c3QtOj}
}
```

Please also cite the upstream LMArena datasets listed in the Hugging Face dataset card.
