# Implementation Plan: Movie Recommendation System (v1)

## Overview

This document tracks the implementation progress for v1 of the Movie Recommendation System.

**v1 Scope:** End-to-end recommendation system with training, serving, frontend, and monitoring.

**Status Legend:**
- ⬜ Not started
- 🟡 In progress
- ✅ Completed
- ⏸️ Paused (planned for v2)

---

## Phase 1: Project Skeleton ✅

**Goal:** Create foundation with proper structure, dependencies, tooling, and containerization.

| Step | Task | Status |
|------|------|--------|
| 1.1 | Create `pyproject.toml` with tool configs | ✅ |
| 1.2 | Create requirements files | ✅ |
| 1.3 | Create directory structure and `__init__.py` files | ✅ |
| 1.4 | Create `.gitignore` | ✅ |
| 1.5 | Create config YAML files | ✅ |
| 1.6 | Create Makefile | ✅ |
| 1.7 | Create pre-commit config | ✅ |
| 1.8 | Create Dockerfile | ✅ |
| 1.9 | Create docker-compose | ✅ |
| 1.10 | Verification and README | ✅ |

---

## Phase 2: Data Pipeline ✅

**Goal:** Download MovieLens data, preprocess into implicit feedback, split by time, generate features.

| Step | Task | Status |
|------|------|--------|
| 2.1 | Create config loader utility | ✅ |
| 2.2 | Implement data download | ✅ |
| 2.3 | Implement preprocessing (implicit conversion, filtering) | ✅ |
| 2.4 | Implement time-based splitting | ✅ |
| 2.5 | Implement feature building (genre encoding, popularity) | ✅ |
| 2.6 | Create ID mapping utilities | ✅ |
| 2.7 | Write unit tests for data pipeline | ✅ |
| 2.8 | Verification: end-to-end data pipeline | ✅ |

**Outputs:**
- `data/raw/` — Original MovieLens CSVs
- `data/processed/` — `interactions.csv`, `train.csv`, `val.csv`, `test.csv`
- `data/features/` — `item_features.csv`, `popularity.csv`
- `data/processed/` — `user_mapping.json`, `item_mapping.json`

---

## Phase 3: Model Training ✅

**Goal:** Implement recommender models, evaluation metrics, MLflow tracking, model registry.

| Step | Task | Status |
|------|------|--------|
| 3.1 | Create abstract base recommender class | ✅ |
| 3.2 | Implement item-item similarity model | ✅ |
| 3.3 | Implement ALS matrix factorization model | ✅ |
| 3.4 | Implement FAISS index builder | ✅ |
| 3.5 | Implement ranking metrics (precision, recall, NDCG, MRR) | ✅ |
| 3.6 | Implement MLflow utilities (logging, registry) | ✅ |
| 3.7 | Implement training orchestrator | ✅ |
| 3.8 | Implement model export (save production artifacts) | ✅ |
| 3.9 | Write unit tests for models and evaluation | ✅ |
| 3.10 | Verification: train and register model | ✅ |

**Outputs:**
- MLflow experiment with logged runs
- `models/production/` — All serving artifacts

---

## Phase 4: API Serving ✅

**Goal:** Create FastAPI service with recommendation endpoints, request logging.

| Step | Task | Status |
|------|------|--------|
| 4.1 | Define Pydantic schemas (request/response models) | ✅ |
| 4.2 | Implement model loader (load artifacts at startup) | ✅ |
| 4.3 | Implement recommendation engine (scoring, filtering) | ✅ |
| 4.4 | Implement fallback handler (cold-start logic) | ✅ |
| 4.5 | Implement request logger (SQLite async writes) | ✅ |
| 4.6 | Create FastAPI app with all endpoints | ✅ |
| 4.7 | Write API tests | ✅ |
| 4.8 | Verification: test all endpoints | ✅ |

**Endpoints:**
- `GET /health` — Health check
- `POST /recommend` — Personalized recommendations
- `POST /similar` — Item similarity
- `GET /popular` — Popular items fallback

---

## Phase 5: User-Facing Frontend ✅

**Goal:** Create a Streamlit-based web interface for users to interact with the recommendation system.

| Step | Task | Status |
|------|------|--------|
| 5.1 | Create frontend directory structure | ✅ |
| 5.2 | Implement API client utilities | ✅ |
| 5.3 | Implement personalized recommendations page | ✅ |
| 5.4 | Implement similar movies page | ✅ |
| 5.5 | Implement popular movies page | ✅ |
| 5.6 | Implement about/status page | ✅ |
| 5.7 | Create main Streamlit app with navigation | ✅ |
| 5.8 | Add styling and UI polish | ✅ |
| 5.9 | Write frontend tests | ✅ |
| 5.10 | Verification: end-to-end user flow | ✅ |

**Pages:**
- **Personalized** — Enter user ID, get recommendations
- **Find Similar** — Select a movie, find similar ones
- **Popular** — Browse popular movies with genre filter
- **About** — System status, how it works

---

## Phase 6: Monitoring ✅ (Partial)

**Goal:** Implement KPI computation and monitoring dashboard.

| Step | Task | Status |
|------|------|--------|
| 6.1 | Implement KPI computation (latency, traffic, coverage) | ✅ |
| 6.2 | Implement replay evaluation (offline metrics on logs) | ✅ |
| 6.3 | Implement Streamlit monitoring dashboard | ✅ |
| 6.4 | Implement data merge for retraining | ✅ |
| 6.5 | Implement promotion logic | ⏸️ v2 |
| 6.6 | Implement retraining orchestrator | ⏸️ v2 |
| 6.7 | Write tests for monitoring and pipeline | ⏸️ v2 |
| 6.8 | Verification: full retraining cycle | ⏸️ v2 |

**v1 Outputs:**
- KPI computation module
- Monitoring dashboard (port 8501)
- Data merge utilities (foundation for v2)

**Paused for v2:** Automated retraining requires real user feedback. See `FUTURE_IMPLEMENTATIONS.md`.

---

## Phase 7: CI/CD ⏸️

**Goal:** Implement automated testing and deployment pipelines.

| Step | Task | Status |
|------|------|--------|
| 7.1 | Configure pre-commit hooks | ✅ |
| 7.2 | Create GitHub Actions CI workflow | ⏸️ v2 |
| 7.3 | Ensure all tests pass | ✅ |
| 7.4 | Create GitHub Actions Docker build workflow | ⏸️ v2 |
| 7.5 | Final documentation | ✅ |
| 7.6 | Verification: push to GitHub, verify Actions pass | ⏸️ v2 |

**v1 Outputs:**
- Pre-commit hooks (ruff, black, mypy)
- Manual test execution via `make test`

---

## v1 Summary

### What's Included

| Component | Description |
|-----------|-------------|
| Data Pipeline | MovieLens 100K ingestion, preprocessing, splitting |
| Models | Item-Item similarity, ALS matrix factorization |
| Evaluation | NDCG, Precision, Recall, Hit Rate, MRR |
| Experiment Tracking | MLflow logging and model registry |
| API | FastAPI with /recommend, /similar, /popular, /health |
| Frontend | Streamlit app with 4 pages |
| Monitoring | KPI dashboard with traffic, latency, quality metrics |
| Code Quality | Pre-commit hooks, unit tests |

### What's Deferred to v2

| Component | Reason |
|-----------|--------|
| User Authentication | Requires user database, sessions |
| Real-time Ratings | Requires auth + new API endpoints |
| Automated Retraining | Needs real user feedback to be meaningful |
| GitHub Actions | Keeping v1 simple |

See `FUTURE_IMPLEMENTATIONS.md` for detailed v2 plans.

---

## Running the System (v1)

```bash
# Terminal 1: Start the API
PYTHONPATH=. python -m src.api.main
# API at http://localhost:8000

# Terminal 2: Start the Frontend
PYTHONPATH=. streamlit run src/frontend/app.py --server.port 8502
# Frontend at http://localhost:8502

# Terminal 3: Start the Monitoring Dashboard
PYTHONPATH=. streamlit run src/monitoring/dashboard.py --server.port 8501
# Dashboard at http://localhost:8501
```

Or use Docker Compose:

```bash
docker-compose up
```
