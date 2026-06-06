.PHONY: install install-pip data data-m5 data-criteo data-synthetic \
        lint format typecheck test ci \
        train-forecast train-returns train-intent train-shipping \
        serve-forecast serve-returns serve-shipping clean

# ── Python — works with uv (preferred) or plain conda/pip ────────────────────
# Override: PYTHON=/path/to/python make data-m5
PYTHON ?= python

# ── Setup ────────────────────────────────────────────────────────────────────

install:
	uv sync --all-extras

# For conda/non-uv environments: pip install -e . then use make with PYTHON=python
install-pip:
	$(PYTHON) -m pip install -e ".[dev]" --quiet

# ── Data ─────────────────────────────────────────────────────────────────────
# PYTHONPATH=src makes commerce_ml importable without installing the package.
# Requires: kaggle CLI available and ~/.kaggle/kaggle.json in place for M5.
#   Install kaggle: pip install kaggle  (or: conda install -c conda-forge kaggle)

data: data-m5 data-criteo

data-m5:
	@echo "Downloading M5 Forecasting dataset (requires Kaggle account + M5 terms accepted)..."
	@echo "  If kaggle is not installed: pip install kaggle"
	PYTHONPATH=src $(PYTHON) -m commerce_ml.data.loaders download-m5

data-criteo:
	@echo "Downloading Criteo Uplift dataset..."
	PYTHONPATH=src $(PYTHON) -m commerce_ml.data.loaders download-criteo

data-synthetic:
	@echo "Generating synthetic returns data..."
	PYTHONPATH=src $(PYTHON) -m commerce_ml.data.synthetic generate

# ── Code quality ─────────────────────────────────────────────────────────────

lint:
	uv run ruff check src/ projects/ tests/

format:
	uv run ruff format src/ projects/ tests/

typecheck:
	uv run mypy src/ --ignore-missing-imports

test:
	uv run pytest

test-cov:
	uv run pytest --cov=src/commerce_ml --cov-report=html

ci: lint typecheck test

# ── Training ─────────────────────────────────────────────────────────────────

train-forecast:
	cd projects/01_demand_forecasting && PYTHONPATH=src uv run python -m forecasting.train

train-returns:
	cd projects/02_returns_intelligence && PYTHONPATH=src uv run python -m returns.train

train-intent:
	cd projects/03_checkout_intent && PYTHONPATH=src uv run python -m intent.train

train-shipping:
	cd projects/04_shipping_optimization && PYTHONPATH=src uv run python -m shipping.train

# ── Serving ──────────────────────────────────────────────────────────────────

serve-forecast:
	cd projects/01_demand_forecasting && uv run uvicorn api.main:app --reload --port 8001

serve-returns:
	cd projects/02_returns_intelligence && uv run uvicorn api.main:app --reload --port 8002

serve-shipping:
	cd projects/04_shipping_optimization && uv run uvicorn api.main:app --reload --port 8003

# ── Cleanup ───────────────────────────────────────────────────────────────────

clean:
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name .mypy_cache -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name .ruff_cache -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name .pytest_cache -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name "*.egg-info" -exec rm -rf {} + 2>/dev/null || true
	@echo "Cleaned."
