# Reproduction verification

Verified on 21 October 2026 by fitting the released annotations again on CPU.
All **31 judge/protocol comparisons pass**, with absolute tolerance `1e-5`
and zero relative tolerance. The largest absolute difference across checked
numeric outputs is **1.137e-12**.

| Suite | Judges | Compared with the archive |
| --- | ---: | --- |
| LMArena 100K main | 8 | All held-out ratings, human reference ratings and fitted temperatures |
| Constant smoothing | 8 | MAE for five constants, Hard-Elo and pooled-temperature Soft-Elo |
| Historical conformal | 8 | Ratings, judge/human bootstrap standard errors and all five split outcomes |
| LMArena 140K main | 7 | Per-judge summary statistics |

The machine-readable [validation report](verification/validation.json) records
individual checks, input and reference hashes, and run identifiers.
The [fresh summaries](verification/summary.csv) contain the newly computed metrics.
Smoothing and LMArena140K comparisons are limited to the archived statistics;
original model-level references were not saved for those analyses.

## Correction found by the audit

The first full run passed the main, smoothing and LMArena140K comparisons but
failed the eight historical conformal comparisons. The release code had held
Hard-Elo anchors fixed during bootstrapping. The original implementation
resampled those anchor battles for Hard-Elo and held them fixed for Soft-Elo.
Restoring this setting and rerunning all eight judges reproduced the archive.
No archived result was changed to obtain a match.

The historical conformal suite retains the original pooled temperature,
20 bootstrap resamples, five splits, seed 42 and higher-interpolation quantile.
The split-width ranges are the 2.5th and 97.5th percentiles. This command
reproduces the original calculation; its quantile differs from the direct
conformal order-statistic rule.

## Environment and additional checks

The fresh full runs used Python 3.13.11 with the committed dependency lock:
NumPy 2.2.6, pandas 2.3.3, SciPy 1.17.1 and scikit-learn 1.8.0.
The documented Python 3.12 environment was separately installed and checked.
Fifteen unit tests pass. The loading example was checked against both a local
release and Hugging Face, and its CSV output was loaded by the original demo.
An installed wheel includes all seven LMArena140K reference files.
The numerical function bodies in the public package were checked against the
code used in the recorded runs.

Earlier release validation checked all 368,777 exported rows, their checksums,
source IDs, model pairs, human labels and subset membership. The reproduction
uses each judge's valid annotations, rather than the cross-judge intersection.
Input annotations match dataset revision
`6223cb4116d76a1bae376c0730df4d968d7172be`.

Run records:

- `c1406b29-1181-4627-9c6e-61e1d6790765`: all 23 non-conformal comparisons passed;
  the eight initial conformal mismatches caused the run to exit with an error.
- `f3dcd6c1-a910-4b5c-b25e-fd20a8616848`: all eight corrected conformal comparisons passed.

This verifies the four suites above. It does not rerun judge inference,
auxiliary prompt experiments, the human re-annotation study or every appendix figure.
