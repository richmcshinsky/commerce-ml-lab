.PHONY: install data data-m5 data-criteo lint format typecheck test ci \
        train-forecast train-returns train-intent \
        serve-forecast serve-returns clean

# ── Setup ────────────────────────────────────────────────────────────────────

install:
	uv sync --all-extras

# ── Data ─────────────────────────────────────────────────────────────────────

data: data-m5 data-criteo

data-m5:
	@echo "Downloading M5 Forecasting dataset..."
	python -m commerce_ml.data.loaders download-m5

data-criteo:
	@echo "Downloading Criteo Uplift dataset..."
	python -m commerce_ml.data.loaders download-criteo

data-synthetic:
	@echo "Generating synthetic returns data..."
	python -m commerce_ml.data.synthetic generate

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
	cd projects/01_demand_forecasting && uv run python -m forecasting.train

train-returns:
	cd projects/02_returns_intelligence && uv run python -m returns.train

train-intent:
	cd projects/03_checkout_intent && uv run python -m intent.train

# ── Serving ──────────────────────────────────────────────────────────────────

serve-forecast:
	cd projects/01_demand_forecasting && uv run uvicorn api.main:app --reload --port 8001

serve-returns:
	cd projects/02_returns_intelligence && uv run uvicorn api.main:app --reload --port 8002

# ── Cleanup ───────────────────────────────────────────────────────────────────

clean:
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name .mypy_cache -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name .ruff_cache -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name .pytest_cache -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name "*.egg-info" -exec rm -rf {} + 2>/dev/null || true
	@echo "Cleaned."
