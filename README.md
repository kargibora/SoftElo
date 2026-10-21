# Soft-Elo

[![Paper](https://img.shields.io/badge/Paper-OpenReview-8c1b13)](https://openreview.net/forum?id=QRZ8c3QtOj)
[![NeurIPS 2026](https://img.shields.io/badge/NeurIPS_2026-Poster-6847a3)](https://neurips.cc/virtual/2026/loc/paris/poster/152776)
[![Dataset](https://img.shields.io/badge/Hugging_Face-Dataset-ffd21e)](https://huggingface.co/datasets/kbora/SoftElo)

**From Uncertain Judgments to Calibrated Rankings: Conformal Elo Estimation for LLM Evaluation**

Bora Kargi and David Salinas · NeurIPS 2026

Soft-Elo converts judge score differences into calibrated win probabilities,
then uses them to fit Bradley–Terry ratings and construct conformal intervals
for human-derived Elo.

## Installation

```bash
git clone https://github.com/kargibora/SoftElo.git
cd SoftElo
uv sync --project reproduction --locked --python 3.12
```

The experiments run on CPU using the released annotations. No GPU or judge API
calls are required.

## Reproduce the results

The [annotation dataset](https://huggingface.co/datasets/kbora/SoftElo) is public
and downloads automatically. No authentication is required.

Run one judge:

```bash
uv run --project reproduction python reproduce.py \
  --suite main --judge GPT-OSS-120B --out outputs/gpt120
```

Run an experiment across all available judges:

```bash
# LMArena 100K: Hard-Elo and Soft-Elo
uv run --project reproduction python reproduce.py --suite main

# Constant label smoothing
uv run --project reproduction python reproduce.py --suite smoothing

# Conformal intervals
uv run --project reproduction python reproduce.py --suite conformal-historical

# LMArena 140K
uv run --project reproduction python reproduce.py --suite lmarena140k

# All experiments above
uv run --project reproduction python reproduce.py --suite all
```

Results are saved under `outputs/`. Use `--out` to choose another directory,
`--jobs` to set the number of CPU workers, or `--data /path/to/dataset` to use
a local dataset copy. Each experiment fits the annotations and checks the
results against the saved paper measurements.

## Load the annotations

```bash
uv run --project reproduction python examples/load_annotations.py \
  --corpus lmarena100k --judge GPT-OSS-120B --csv annotations.csv
```

The dataset includes LMArena 100K and LMArena 140K. Each row contains one judge's
annotation of one battle. The [dataset card](https://huggingface.co/datasets/kbora/SoftElo)
describes the fields, source datasets, and licenses.

## CSV demo

For a small example using the bundled CSV:

```bash
pip install -r requirements.txt
python run_demo.py --data data/sample_annotations.csv --out outputs/demo
python make_tables.py --out outputs/demo
python plot_results.py --out outputs/demo
```

CSV inputs contain `model_a`, `model_b`, `human_pref`, `judge_pref`, `scores_a`
and `scores_b`. Preferences use 0 for an A win, 1 for a B win and 0.5 for a tie;
score dictionaries are JSON strings. The annotation-loading command above exports
this format. Use `reproduce.py` for the paper experiments; the compact demo uses
different handling of weak preferences and tied judge scores.

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
