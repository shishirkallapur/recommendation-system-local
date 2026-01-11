# Implementation Plan: Movie Recommendation System

## Overview

This document breaks down each phase into granular steps. Use this to track progress and resume work in new conversation threads.

**Status Legend:**
- ⬜ Not started
- 🟡 In progress
- ✅ Completed

---

## Phase 1: Project Skeleton

**Goal:** Create foundation with proper structure, dependencies, tooling, and containerization.

| Step | Task | Files Created | Status |
|------|------|---------------|--------|
| 1.1 | Create `pyproject.toml` with tool configs | `pyproject.toml` | ✅ |
| 1.2 | Create requirements files | `requirements.txt`, `requirements-dev.txt` | ✅ |
| 1.3 | Create directory structure and `__init__.py` files | `src/*/`, `data/*/`, `models/*/`, `tests/`, `configs/`, `scripts/`, `.github/workflows/` | ✅ |
| 1.4 | Create `.gitignore` | `.gitignore` | ✅ |
| 1.5 | Create config YAML files | `configs/data.yaml`, `configs/training.yaml`, `configs/serving.yaml`, `configs/monitoring.yaml`, `configs/retrain.yaml` | ✅ |
| 1.6 | Create Makefile | `Makefile` | ✅ |
| 1.7 | Create pre-commit config | `.pre-commit-config.yaml` | ✅ |
| 1.8 | Create Dockerfile | `Dockerfile` | ✅ |
| 1.9 | Create docker-compose | `docker-compose.yml` | ✅ |
| 1.10 | Verification and README | `README.md`, verify all commands work | ✅ |

---

## Phase 2: Data Pipeline

**Goal:** Download MovieLens data, preprocess into implicit feedback, split by time, generate features.

| Step | Task | Files Created | Status |
|------|------|---------------|--------|
| 2.1 | Create config loader utility | `src/config.py` | ✅ |
| 2.2 | Implement data download | `src/data/download.py` | ✅ |
| 2.3 | Implement preprocessing (implicit conversion, filtering) | `src/data/preprocess.py` | ✅ |
| 2.4 | Implement time-based splitting | `src/data/split.py` | ✅ |
| 2.5 | Implement feature building (genre encoding, popularity) | `src/features/build.py` | ✅ |
| 2.6 | Create ID mapping utilities | `src/data/mappings.py` | ✅ |
| 2.7 | Write unit tests for data pipeline | `tests/test_data.py` | ✅ |
| 2.8 | Verification: end-to-end data pipeline | Run `make download-data process-data build-features` | ✅ |

**Outputs:**
- `data/raw/` — Original MovieLens CSVs
- `data/processed/` — `interactions.csv`, `train.csv`, `val.csv`, `test.csv`
- `data/features/` — `item_features.csv`, `popularity.csv`
- `data/processed/` — `user_mapping.json`, `item_mapping.json`

---

## Phase 3: Model Training

**Goal:** Implement recommender models, evaluation metrics, MLflow tracking, model registry.

| Step | Task | Files Created | Status |
|------|------|---------------|--------|
| 3.1 | Create abstract base recommender class | `src/models/base.py` | ✅ |
| 3.2 | Implement item-item similarity model | `src/models/item_item.py` | ✅ |
| 3.3 | Implement ALS matrix factorization model | `src/models/als.py` | ✅ |
| 3.4 | Implement FAISS index builder | `src/models/index.py` | ✅ |
| 3.5 | Implement ranking metrics (precision, recall, NDCG, MRR) | `src/training/evaluate.py` | ✅ |
| 3.6 | Implement MLflow utilities (logging, registry) | `src/training/mlflow_utils.py` | ✅ |
| 3.7 | Implement training orchestrator | `src/training/train.py` | ✅ |
| 3.8 | Implement model export (save production artifacts) | `src/training/export.py` | ✅ |
| 3.9 | Write unit tests for models and evaluation | `tests/test_models.py`, `tests/test_evaluation.py` | ✅ |
| 3.10 | Verification: train and register model | Run `make train`, verify MLflow UI | ✅ |

**Outputs:**
- MLflow experiment with logged runs
- `models/production/` — `model_metadata.json`, `user_embeddings.npy`, `item_embeddings.npy`, `item_index.faiss`, `user_mapping.json`, `item_mapping.json`, `item_features.json`

---

## Phase 4: API Serving

**Goal:** Create FastAPI service with recommendation endpoints, request logging.

| Step | Task | Files Created | Status |
|------|------|---------------|--------|
| 4.1 | Define Pydantic schemas (request/response models) | `src/api/schemas.py` | ✅ |
| 4.2 | Implement model loader (load artifacts at startup) | `src/api/model_loader.py` | ✅ |
| 4.3 | Implement recommendation engine (scoring, filtering) | `src/api/recommender.py` | ✅ |
| 4.4 | Implement fallback handler (cold-start logic) | `src/api/fallback.py` | ✅ |
| 4.5 | Implement request logger (SQLite async writes) | `src/api/logger.py` | ✅ |
| 4.6 | Create FastAPI app with all endpoints | `src/api/main.py` | ✅ |
| 4.7 | Write API tests | `tests/test_api.py` | ✅ |
| 4.8 | Verification: test all endpoints | Run `make serve`, test with curl/httpx | ✅ |

**Endpoints:**
- `GET /health` — Health check
- `POST /recommend` — Personalized recommendations
- `POST /similar` — Item similarity
- `GET /popular` — Popular items fallback

---

## Phase 5: User-Facing Frontend

**Goal:** Create a Streamlit-based web interface for users to interact with the recommendation system.

| Step | Task | Files Created | Status |
|------|------|---------------|--------|
| 5.1 | Create frontend directory structure | `src/frontend/__init__.py` | ⬜ |
| 5.2 | Implement API client utilities | `src/frontend/api_client.py` | ⬜ |
| 5.3 | Implement personalized recommendations page | `src/frontend/pages/personalized.py` | ⬜ |
| 5.4 | Implement similar movies page | `src/frontend/pages/similar.py` | ⬜ |
| 5.5 | Implement popular movies page | `src/frontend/pages/popular.py` | ⬜ |
| 5.6 | Implement about/status page | `src/frontend/pages/about.py` | ⬜ |
| 5.7 | Create main Streamlit app with navigation | `src/frontend/app.py` | ⬜ |
| 5.8 | Add styling and UI polish | CSS customizations, loading states | ⬜ |
| 5.9 | Write frontend tests | `tests/test_frontend.py` | ⬜ |
| 5.10 | Verification: end-to-end user flow | Test all pages manually | ⬜ |

**Pages:**
- **Personalized** — Enter user ID, get recommendations
- **Find Similar** — Select a movie, find similar ones
- **Popular** — Browse popular movies with genre filter
- **About** — System status, how it works

---

## Phase 6: Monitoring & Retraining

**Goal:** Implement KPI computation, replay evaluation, dashboard, retraining pipeline.

| Step | Task | Files Created | Status |
|------|------|---------------|--------|
| 6.1 | Implement KPI computation (latency, traffic, coverage) | `src/monitoring/kpis.py` | ✅ |
| 6.2 | Implement replay evaluation (offline metrics on logs) | `src/monitoring/replay_eval.py` | ✅ |
| 6.3 | Implement Streamlit monitoring dashboard | `src/monitoring/dashboard.py` | ⬜ |
| 6.4 | Implement data merge for retraining | `src/pipeline/data_merge.py` | ⬜ |
| 6.5 | Implement promotion logic | `src/pipeline/promote.py` | ⬜ |
| 6.6 | Implement retraining orchestrator | `src/pipeline/retrain.py` | ⬜ |
| 6.7 | Write tests for monitoring and pipeline | `tests/test_monitoring.py`, `tests/test_pipeline.py` | ⬜ |
| 6.8 | Verification: full retraining cycle | Run `make retrain`, verify promotion | ⬜ |

**Outputs:**
- `data/logs/reports/` — KPI reports
- Streamlit dashboard on port 8501
- Automated model promotion workflow

---

## Phase 7: CI/CD

**Goal:** Implement pre-commit hooks, GitHub Actions, finalize tests.

| Step | Task | Files Created | Status |
|------|------|---------------|--------|
| 7.1 | Configure pre-commit hooks | `.pre-commit-config.yaml` (if not done in 1.7) | ⬜ |
| 7.2 | Create GitHub Actions CI workflow | `.github/workflows/ci.yml` | ⬜ |
| 7.3 | Ensure all tests pass | Run `make check` | ⬜ |
| 7.4 | Create GitHub Actions Docker build workflow | `.github/workflows/docker.yml` | ⬜ |
| 7.5 | Final documentation | Update `README.md` with usage instructions | ⬜ |
| 7.6 | Verification: push to GitHub, verify Actions pass | Push code, check Actions tab | ⬜ |

**Outputs:**
- Pre-commit hooks running on every commit
- GitHub Actions running on every push/PR
- Complete, documented project

---

## Quick Reference: File to Phase Mapping

| Directory | Files | Phase |
|-----------|-------|-------|
| `src/data/` | `download.py`, `preprocess.py`, `split.py`, `mappings.py` | 2 |
| `src/features/` | `build.py` | 2 |
| `src/models/` | `base.py`, `item_item.py`, `als.py`, `index.py` | 3 |
| `src/training/` | `train.py`, `evaluate.py`, `mlflow_utils.py`, `export.py` | 3 |
| `src/api/` | `main.py`, `schemas.py`, `model_loader.py`, `recommender.py`, `fallback.py`, `logger.py` | 4 |
| `src/frontend/` | `app.py`, `api_client.py`, `pages/*.py` | 5 |
| `src/monitoring/` | `kpis.py`, `replay_eval.py`, `dashboard.py` | 6 |
| `src/pipeline/` | `retrain.py`, `promote.py`, `data_merge.py` | 6 |
| `.github/workflows/` | `ci.yml`, `docker.yml` | 7 |

---

## Running the Complete System

After all phases are complete, run the system with:

```bash
# Terminal 1: Start the API
python -m src.api.main
# API available at http://localhost:8000

# Terminal 2: Start the User Frontend
streamlit run src/frontend/app.py --server.port 8502
# Frontend available at http://localhost:8502

# Terminal 3: Start the Monitoring Dashboard
streamlit run src/monitoring/dashboard.py --server.port 8501
# Dashboard available at http://localhost:8501
```

Or use Docker Compose:

```bash
docker-compose up
```
