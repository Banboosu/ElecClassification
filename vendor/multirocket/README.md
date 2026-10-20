# Vendored MultiRocket

Source: https://github.com/ChangWeiTan/MultiRocket (main branch, downloaded 2026-10-20).
Files: `multirocket.py`, `logistic_regression.py`, `__init__.py`; license: GPL-3.0 (see `LICENSE`).

Only the `fit`/`transform` functions are used in E14; the vendored classifier is not used.
This directory is not a pip package; `scripts/run_e14_multirocket.py` adds `vendor/` to `sys.path`.
