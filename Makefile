.PHONY: help install install-dev format lint typecheck test clean \
        download-data process-data build-features train serve \
        docker-build docker-up docker-down monitor retrain

# Default target: show help
help:
	@echo "Available commands:"
	@echo ""
	@echo "  Setup:"
	@echo "    make install        Install production dependencies"
	@echo "    make install-dev    Install all dependencies (including dev)"
	@echo ""
	@echo "  Code Quality:"
	@echo "    make format         Format code with black"
	@echo "    make lint           Run ruff linter"
	@echo "    make typecheck      Run mypy type checker"
	@echo "    make test           Run pytest"
	@echo "    make check          Run all checks (format, lint, typecheck, test)"
	@echo ""
	@echo "  Data Pipeline (Phase 2):"
	@echo "    make download-data  Download MovieLens dataset"
	@echo "    make process-data   Preprocess and split data"
	@echo "    make build-features Build feature matrices"
	@echo ""
	@echo "  Training (Phase 3):"
	@echo "    make train          Train all models"
	@echo ""
	@echo "  Serving (Phase 4):"
	@echo "    make serve          Start FastAPI server locally"
	@echo ""
	@echo "  Docker (Phase 4-5):"
	@echo "    make docker-build   Build Docker image"
	@echo "    make docker-up      Start all containers"
	@echo "    make docker-down    Stop all containers"
	@echo ""
	@echo "  Monitoring (Phase 5):"
	@echo "    make monitor        Launch Streamlit dashboard"
	@echo "    make retrain        Trigger retraining pipeline"
	@echo ""
	@echo "  Utilities:"
	@echo "    make clean          Remove generated files"

# =============================================================================
# Setup
# =============================================================================

install:
	pip install -r requirements.txt

install-dev:
	pip install -r requirements-dev.txt
	pre-commit install

# =============================================================================
# Code Quality
# =============================================================================

format:
	black src/ tests/

lint:
	ruff check src/ tests/

typecheck:
	mypy src/

test:
	python -m pytest tests/ -v

check: format lint typecheck test
	@echo "All checks passed!"

# =============================================================================
# Data Pipeline (Phase 2)
# =============================================================================

download-data:
	python -m src.data.download

process-data:
	python -m src.data.preprocess
	python -m src.data.split

build-features:
	python -m src.features.build

# =============================================================================
# Training (Phase 3)
# =============================================================================

train:
	python -m src.training.train

# =============================================================================
# Serving (Phase 4)
# =============================================================================

serve:
	uvicorn src.api.main:app --host 0.0.0.0 --port 8000 --reload

# =============================================================================
# Docker (Phase 4-5)
# =============================================================================

docker-build:
	docker build -t movie-recommender .

docker-up:
	docker-compose up -d

docker-down:
	docker-compose down

# =============================================================================
# Monitoring (Phase 5)
# =============================================================================

monitor:
	streamlit run src/monitoring/dashboard.py

retrain:
	python -m src.pipeline.retrain

# =============================================================================
# Utilities
# =============================================================================

clean:
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete 2>/dev/null || true
	find . -type d -name ".pytest_cache" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name ".mypy_cache" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name ".ruff_cache" -exec rm -rf {} + 2>/dev/null || true
	rm -rf htmlcov/ .coverage 2>/dev/null || true
	@echo "Cleaned generated files"
