# Validation

Last checked: 2026-09-15.

## Numerical comparison

Features were compared with the reference implementation and the saved
Heartbeat feature arrays used for the paper.

| Comparison | Training / query sequences | Channels | Maximum context difference | Maximum query difference |
| --- | ---: | ---: | ---: | ---: |
| Variable lengths, missing values, a constant channel, and a singleton class | 12 / 2 | 4 | 0 | 0 |
| Effective PCA rank cap | 8 / 1 | 2 | 0 | 0 |
| Heartbeat subset | 12 / 3 | 61 | 0 | 0 |
| Complete Heartbeat split vs saved features | 204 / 205 | 61 | 4.753e-12 | 5.582e-12 |

Selected feature indices were identical in all three direct code comparisons.
The complete Heartbeat comparison passed `atol=1e-7, rtol=1e-6`.

## Executed checks

- Ten unit checks passed: context standardization/selection, excluded singleton
  class banks, query-state immutability and batch independence, input validation,
  two logsignature identities, UEA split/label handling, probability column order,
  native class-limit routing, and feature coverage.
- `run.py --help` and shell syntax checks passed.
- The classification runner was tested with real feature extraction and a
  predictor stub. Saved labels and probability column order were checked.
- Python syntax, import, formatting, and static checks passed.

Numerical checks used Python 3.12.14, NumPy 1.26.4, SciPy 1.15.3,
scikit-learn 1.7.2, and one CPU thread on macOS Intel.

## Scope

These checks cover feature construction and the prediction interface. They do
not include neural inference or installation of the full CUDA environment.
Classification accuracy requires the TabPFN checkpoint and the evaluation
command in the README.
