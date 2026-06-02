# CLAUDE.md — Commerce ML Lab

Coding standards and conventions for this repository.

## Language and tooling

- Python 3.11+, managed with `uv`
- `ruff` for linting and formatting (line length 100)
- `mypy` for type checking (strict-ish — see pyproject.toml)
- `pytest` for tests
- Run `make ci` (lint + typecheck + test) before every commit

## Code style

- **Type hints on every public function signature** — no bare `def foo(x):`.
- **NumPy-style docstrings** on every public function and class.
- **No magic numbers** — use named constants or config dicts.
- **No hardcoded paths** — use `Path(__file__).parent` relative paths or the `DATA_DIR` constant from `commerce_ml.data.loaders`.
- **Fixed random seeds** — pass `random_state=42` or `seed=42` everywhere.

## What to build first: tests, then implementation

1. Write the function signature + docstring + type hints.
2. Write the unit tests against the spec.
3. Implement to make tests pass.
4. Run `make ci`.

## Data

- All data lives in `data/` (gitignored).
- Download via `make data` or individual `make data-*` targets.
- Never commit data files or model artifacts.

## Notebooks

- `projects/XX/notebooks/` — exploration notebooks, messy is fine, for learning.
  Clear outputs before committing.
- `projects/XX/report.ipynb` — polished narrative, clean outputs committed.

## Project structure convention

Each project in `projects/` follows:
```
projects/XX_name/
├── README.md          # framing, data, results, limitations
├── report.ipynb       # polished end-to-end narrative
├── notebooks/         # exploration (run by hand, outputs cleared)
├── src/<module>/      # production Python — importable, testable
├── api/               # FastAPI service (projects 01 and 02 only)
│   ├── main.py
│   └── schemas.py
├── results/           # saved charts and metrics tables (committed)
└── tests/             # pytest — at least 5 tests per project
```

## Senior signals to preserve in every PR

- Baselines before models — always establish a simple baseline first.
- Honest limitations — every README has a "limitations" section.
- Cost-aware decisions — thresholds, policies, and recommendations should connect to a stated business loss or cost.
- "What I'd do with more time" — every README has a future work section.
