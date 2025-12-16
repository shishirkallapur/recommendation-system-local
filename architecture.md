# Movie Recommendation System: Architecture

## 1. High-Level Overview

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                              OFFLINE LAYER                                  │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐    ┌───────────┐  │
│  │Data Ingestion│───▶│ Feature Eng. │───▶│   Training   │───▶│  MLflow   │  │
│  │  MovieLens   │    │  Pipeline    │    │   Pipeline   │    │ Registry  │  │
│  └──────────────┘    └──────────────┘    └──────────────┘    └─────┬─────┘  │
│                                                                     │       │
│         ┌───────────────────────────────────────────────────────────┘       │
│         ▼                                                                   │
│  ┌─────────────────┐                                                        │
│  │ Production      │  (model weights, embeddings, FAISS index, mappings)    │
│  │ Artifacts Store │                                                        │
│  └────────┬────────┘                                                        │
└───────────┼─────────────────────────────────────────────────────────────────┘
            │
            │ Load on startup / hot-reload on promotion
            ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                              ONLINE LAYER                                   │
│  ┌─────────────────────────────────────────────────────────────────────┐    │
│  │                         FastAPI Service                             │    │
│  │  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐  ┌────────────┐  │    │
│  │  │ /recommend  │  │  /similar   │  │  /popular   │  │  /health   │  │    │
│  │  └──────┬──────┘  └──────┬──────┘  └──────┬──────┘  └────────────┘  │    │
│  │         │                │                │                         │    │
│  │         ▼                ▼                ▼                         │    │
│  │  ┌─────────────────────────────────────────────────────────────┐    │    │
│  │  │              Recommendation Engine                          │    │    │
│  │  │  • User embedding lookup    • Item embedding lookup         │    │    │
│  │  │  • Real-time scoring        • FAISS similarity search       │    │    │
│  │  │  • Filtering & exclusion    • Cold-start fallback           │    │    │
│  │  └─────────────────────────────────────────────────────────────┘    │    │
│  └─────────────────────────────────────────────────────────────────────┘    │
│                                     │                                       │
│                                     ▼                                       │
│                           ┌─────────────────┐                               │
│                           │  Request Logger │                               │
│                           │    (SQLite)     │                               │
│                           └─────────────────┘                               │
└─────────────────────────────────────────────────────────────────────────────┘
            │
            │ Async write
            ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                           MONITORING LAYER                                  │
│  ┌─────────────────┐    ┌─────────────────┐    ┌─────────────────────────┐  │
│  │   KPI Scripts   │    │ Replay Evaluator│    │   Streamlit Dashboard   │  │
│  │ (latency, traffic)│   │ (offline CTR)  │    │   (health & metrics)    │  │
│  └─────────────────┘    └─────────────────┘    └─────────────────────────┘  │
│                                     │                                       │
│                                     ▼                                       │
│                           ┌─────────────────┐                               │
│                           │ Retrain Trigger │                               │
│                           │   (scheduled)   │                               │
│                           └────────┬────────┘                               │
│                                    │                                        │
└────────────────────────────────────┼────────────────────────────────────────┘
                                     │
                                     └──────────▶ Back to OFFLINE LAYER

┌─────────────────────────────────────────────────────────────────────────────┐
│                              CI/CD LAYER                                    │
│  ┌─────────────────┐    ┌─────────────────┐    ┌─────────────────────────┐  │
│  │  Pre-commit     │    │ GitHub Actions  │    │     Makefile            │  │
│  │  Hooks          │    │ (lint, test,    │    │  (local commands)       │  │
│  │ (lint, format)  │    │  docker build)  │    │                         │  │
│  └─────────────────┘    └─────────────────┘    └─────────────────────────┘  │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## 2. Component Details

### 2.1 Data Layer

| Component | Responsibility | Storage |
|-----------|---------------|---------|
| Raw Data Store | Original MovieLens CSVs | `data/raw/` |
| Processed Data | Cleaned interactions, splits | `data/processed/` |
| Feature Store | Item features, precomputed stats | `data/features/` |
| Request Logs | Online inference logs | `data/logs/requests.db` |

### 2.2 Training Layer

| Component | Responsibility | Output |
|-----------|---------------|--------|
| Preprocessor | Clean, filter, split data | Train/val/test CSVs, mappings |
| Feature Builder | Genre encoding, popularity stats | Feature matrices |
| Item-Item Trainer | Cosine similarity model | Similarity matrix, item index |
| ALS Trainer | Matrix factorization | User/item embeddings |
| Evaluator | Ranking metrics computation | Metrics dict |
| MLflow Logger | Experiment tracking | Runs, artifacts, registry |

### 2.3 Serving Layer

| Component | Responsibility | Notes |
|-----------|---------------|-------|
| Model Loader | Load production artifacts at startup | Supports hot-reload via signal |
| Embedding Store | In-memory user/item embeddings | NumPy arrays |
| FAISS Index | Fast approximate nearest neighbors | For item similarity |
| Scorer | Compute user×item scores | Real-time dot product |
| Filter | Exclude seen, apply genre/year filters | Post-scoring |
| Fallback Handler | Cold-start logic | Popular items, seed-based |
| Request Logger | Async write to SQLite | Non-blocking |

### 2.4 Monitoring Layer

| Component | Responsibility | Output |
|-----------|---------------|--------|
| KPI Computer | Aggregate latency, traffic, coverage | JSON/CSV metrics |
| Replay Evaluator | Offline evaluation on logged data | Precision, hit rate by model |
| Dashboard | Visual health monitoring | Streamlit app |

### 2.5 CI/CD Layer

| Component | Responsibility | Trigger |
|-----------|---------------|---------|
| Pre-commit Hooks | Lint (ruff), format (black), type-check (mypy) | On git commit |
| GitHub Actions | Run tests, lint, build Docker image | On push/PR to main |
| Makefile | Unified local commands for all workflows | Manual |

---

## 3. Data Flows

### 3.1 Training Flow

```
MovieLens CSV
    │
    ▼
[Preprocess] ─── filter users/items, convert to implicit ───▶ interactions.csv
    │
    ▼
[Split] ─── global time split ───▶ train.csv, val.csv, test.csv
    │
    ▼
[Feature Build] ─── genre encoding, popularity ───▶ item_features.csv
    │
    ▼
[Train ALS] ─── implicit ALS ───▶ user_embeddings.npy, item_embeddings.npy
    │
    ▼
[Build Index] ─── FAISS indexing ───▶ item_index.faiss
    │
    ▼
[Evaluate] ─── ranking metrics ───▶ metrics.json
    │
    ▼
[MLflow Log] ─── register if better ───▶ Model Registry (Production)
    │
    ▼
[Export] ─── copy to serving path ───▶ models/production/
```

### 3.2 Serving Flow

```
POST /recommend {user_id, k, exclude_seen}
    │
    ▼
[Lookup user_id in mapping]
    │
    ├── Found ──▶ [Get user embedding] ──▶ [Score all items] ──▶ [Filter & rank]
    │                                                                  │
    │                                                                  ▼
    │                                                          [Return top-K]
    │
    └── Not found ──▶ [Fallback: /popular] ──▶ [Return popular items]
    │
    ▼
[Log request to SQLite]
```

### 3.3 Retraining Flow

```
[Scheduled trigger or manual invoke]
    │
    ▼
[Load new interactions from logs]
    │
    ▼
[Merge with existing training data]
    │
    ▼
[Retrain models] ───▶ MLflow (new run)
    │
    ▼
[Evaluate on validation set]
    │
    ▼
[Compare NDCG@10 vs current Production]
    │
    ├── Improved ≥ 1% ──▶ [Promote to Production] ──▶ [Signal API to reload]
    │
    └── Not improved ──▶ [Keep current Production] ──▶ [Log decision]
```

### 3.4 CI/CD Flow

```
[Developer writes code]
    │
    ▼
[Git commit] ───▶ [Pre-commit hooks: ruff, black, mypy]
    │
    ├── Fails ──▶ [Fix issues locally]
    │
    └── Passes ──▶ [Push to GitHub]
                        │
                        ▼
                  [GitHub Actions triggered]
                        │
                        ├── [Run pytest]
                        ├── [Run ruff lint]
                        └── [Build Docker image]
                              │
                              ├── Any fails ──▶ [Block merge, notify]
                              │
                              └── All pass ──▶ [Ready to merge]
```

---

## 4. API Specification

### 4.1 Endpoints

#### GET /health

Returns service status and model version info.

**Response:** `status`, `model_version`, `model_loaded_at`, `uptime_seconds`

#### POST /recommend

Personalized recommendations for a user.

**Request body:**
- `user_id` (required): User identifier
- `k` (optional): Number of recommendations, default 10
- `exclude_seen` (optional): Exclude previously interacted items, default true
- `filters` (optional): Object with `genres`, `year_min`, `year_max`

**Response:** `user_id`, `recommendations` (list of movie_id, title, score), `model_version`, `is_fallback`, `fallback_reason` (if applicable)

#### POST /similar

Item-item similarity for "because you watched X" scenarios.

**Request body:**
- `movie_id` (required): Seed movie identifier
- `k` (optional): Number of similar items, default 10

**Response:** `movie_id`, `title`, `similar_items` (list of movie_id, title, score), `model_version`

**Error (404):** `error`, `message` for unknown items

#### GET /popular

Non-personalized popularity-based recommendations.

**Query params:**
- `k` (optional): Number of items, default 10
- `genre` (optional): Filter by genre

**Response:** `recommendations` (list of movie_id, title, popularity_score), `source`

---

## 5. Directory Structure

```
movie-recommender/
├── .github/
│   └── workflows/
│       └── ci.yml
│
├── configs/
│   ├── data.yaml
│   ├── training.yaml
│   ├── serving.yaml
│   ├── monitoring.yaml
│   └── retrain.yaml
│
├── data/
│   ├── raw/
│   │   ├── ratings.csv
│   │   └── movies.csv
│   ├── processed/
│   │   ├── interactions.csv
│   │   ├── train.csv
│   │   ├── val.csv
│   │   └── test.csv
│   ├── features/
│   │   ├── item_features.csv
│   │   └── popularity.csv
│   └── logs/
│       └── requests.db
│
├── models/
│   └── production/
│       ├── model_metadata.json
│       ├── user_embeddings.npy
│       ├── item_embeddings.npy
│       ├── item_index.faiss
│       ├── user_mapping.json
│       ├── item_mapping.json
│       └── item_features.json
│
├── src/
│   ├── __init__.py
│   ├── data/
│   │   ├── __init__.py
│   │   ├── download.py
│   │   ├── preprocess.py
│   │   └── split.py
│   │
│   ├── features/
│   │   ├── __init__.py
│   │   └── build.py
│   │
│   ├── models/
│   │   ├── __init__.py
│   │   ├── base.py
│   │   ├── item_item.py
│   │   ├── als.py
│   │   └── index.py
│   │
│   ├── training/
│   │   ├── __init__.py
│   │   ├── train.py
│   │   ├── evaluate.py
│   │   └── mlflow_utils.py
│   │
│   ├── api/
│   │   ├── __init__.py
│   │   ├── main.py
│   │   ├── schemas.py
│   │   ├── recommender.py
│   │   ├── fallback.py
│   │   └── logger.py
│   │
│   ├── monitoring/
│   │   ├── __init__.py
│   │   ├── kpis.py
│   │   ├── replay_eval.py
│   │   └── dashboard.py
│   │
│   └── pipeline/
│       ├── __init__.py
│       ├── retrain.py
│       └── promote.py
│
├── tests/
│   ├── __init__.py
│   ├── test_data.py
│   ├── test_models.py
│   ├── test_api.py
│   └── test_evaluation.py
│
├── scripts/
│   ├── setup_data.sh
│   ├── train_all.sh
│   └── run_retrain.sh
│
├── .pre-commit-config.yaml
├── Dockerfile
├── docker-compose.yml
├── requirements.txt
├── requirements-dev.txt
├── pyproject.toml
├── Makefile
└── README.md
```

---

## 6. Production Artifacts Structure

Contents of `models/production/` after training:

| File | Description |
|------|-------------|
| `model_metadata.json` | Model version, training date, metrics, hyperparameters |
| `user_embeddings.npy` | NumPy array of user latent factors (n_users × embedding_dim) |
| `item_embeddings.npy` | NumPy array of item latent factors (n_items × embedding_dim) |
| `item_index.faiss` | FAISS index for fast item similarity search |
| `user_mapping.json` | Dict mapping user_id → matrix index |
| `item_mapping.json` | Dict mapping movie_id → matrix index |
| `item_features.json` | Item metadata (title, genres, year) for response enrichment |

---

## 7. Request Logging Schema

SQLite table `request_logs`:

| Column | Type | Description |
|--------|------|-------------|
| request_id | TEXT | UUID for the request |
| timestamp | DATETIME | Request time |
| user_id | INTEGER | Requested user |
| endpoint | TEXT | /recommend, /similar, /popular |
| model_version | TEXT | Model version used |
| recommendations | TEXT | JSON array of movie_ids |
| scores | TEXT | JSON array of scores |
| latency_ms | REAL | Response time in milliseconds |
| is_fallback | INTEGER | 1 if cold-start path used, 0 otherwise |
| fallback_reason | TEXT | Reason for fallback (if applicable) |

---

## 8. Implementation Plan

| Phase | Components | Files to Create | Output |
|-------|------------|-----------------|--------|
| **1. Skeleton** | Project structure, configs, dependencies, Docker, Makefile, pre-commit | `pyproject.toml`, `requirements.txt`, `requirements-dev.txt`, `Makefile`, `Dockerfile`, `docker-compose.yml`, `.pre-commit-config.yaml`, all `__init__.py`, config YAMLs | Runnable empty project with tooling |
| **2. Data** | Download, preprocess, split, feature build | `src/data/*.py`, `src/features/build.py` | Train/val/test splits + features ready |
| **3. Training** | Models (baseline + ALS), evaluation, MLflow tracking | `src/models/*.py`, `src/training/*.py` | Trained model registered in MLflow |
| **4. Serving** | FastAPI endpoints, request logging | `src/api/*.py` | Working API returning recommendations |
| **5. Ops** | Monitoring KPIs, replay evaluation, retraining pipeline, dashboard | `src/monitoring/*.py`, `src/pipeline/*.py` | Complete local system with observability |
| **6. CI/CD** | Pre-commit hooks, GitHub Actions, tests | `.github/workflows/ci.yml`, `tests/*.py` | Automated quality gates on commit/push |
