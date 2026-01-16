# Movie Recommendation System: Architecture (v1)

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
            │ Load on startup
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
            │
            ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                            FRONTEND LAYER                                   │
│  ┌─────────────────────────────────────────────────────────────────────┐    │
│  │                      Streamlit Frontend                             │    │
│  │  ┌───────────────┐  ┌───────────────┐  ┌───────────────────────┐    │    │
│  │  │ Personalized  │  │ Find Similar  │  │   Popular Movies      │    │    │
│  │  │Recommendations│  │    Movies     │  │     Browser           │    │    │
│  │  └───────┬───────┘  └───────┬───────┘  └───────────┬───────────┘    │    │
│  │          │                  │                      │                │    │
│  │          └──────────────────┼──────────────────────┘                │    │
│  │                             ▼                                       │    │
│  │                    ┌─────────────────┐                              │    │
│  │                    │   API Client    │                              │    │
│  │                    │ (HTTP Requests) │                              │    │
│  │                    └─────────────────┘                              │    │
│  └─────────────────────────────────────────────────────────────────────┘    │
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
└─────────────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────────────┐
│                            CODE QUALITY LAYER                               │
│  ┌─────────────────┐    ┌─────────────────┐    ┌─────────────────────────┐  │
│  │  Pre-commit     │    │   Unit Tests    │    │     Makefile            │  │
│  │  Hooks          │    │   (pytest)      │    │  (local commands)       │  │
│  │ (ruff, black,   │    │                 │    │                         │  │
│  │  mypy)          │    │                 │    │                         │  │
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
| Model Loader | Load production artifacts at startup | In-memory storage |
| Embedding Store | In-memory user/item embeddings | NumPy arrays |
| FAISS Index | Fast approximate nearest neighbors | For item similarity |
| Scorer | Compute user×item scores | Real-time dot product |
| Filter | Exclude seen, apply genre/year filters | Post-scoring |
| Fallback Handler | Cold-start logic | Popular items, seed-based |
| Request Logger | Async write to SQLite | Non-blocking |

### 2.4 Frontend Layer

| Component | Responsibility | Notes |
|-----------|---------------|-------|
| Streamlit App | User-facing web interface | Port 8502 |
| API Client | HTTP requests to FastAPI | Handles errors gracefully |
| Personalized Page | Recommendations for known users | User ID input |
| Similar Movies Page | Seed-based recommendations | Movie selection |
| Popular Page | Browse trending movies | Genre filtering |
| About Page | System status and info | API health check |

### 2.5 Monitoring Layer

| Component | Responsibility | Output |
|-----------|---------------|--------|
| KPI Calculator | Aggregate latency, traffic, coverage | KPI reports |
| Replay Evaluator | Offline evaluation on logged data | Precision, hit rate |
| Dashboard | Visual health monitoring | Streamlit app (port 8501) |

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
[MLflow Log] ─── register model ───▶ Model Registry
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

### 3.3 Frontend Flow

```
User opens Streamlit app (port 8502)
    │
    ▼
[Check API health] ──▶ Show status indicator
    │
    ├── Personalized Page
    │   └── User enters ID ──▶ POST /recommend ──▶ Display movies
    │
    ├── Similar Movies Page
    │   └── User selects movie ──▶ POST /similar ──▶ Display similar
    │
    └── Popular Page
        └── User selects genre ──▶ GET /popular ──▶ Display popular
```

---

## 4. API Specification

### 4.1 Endpoints

#### GET /health

Returns service status and model version info.

**Response:** `status`, `model_version`, `model_loaded_at`, `uptime_seconds`, `n_users`, `n_items`

#### POST /recommend

Personalized recommendations for a user.

**Request body:**
- `user_id` (required): User identifier
- `k` (optional): Number of recommendations, default 10
- `exclude_seen` (optional): Exclude previously interacted items, default true
- `filters` (optional): Object with `genres`, `year_min`, `year_max`

**Response:** `user_id`, `recommendations` (list of movie_id, title, score, year, genres), `model_version`, `is_fallback`, `fallback_reason`

#### POST /similar

Item-item similarity for "because you watched X" scenarios.

**Request body:**
- `movie_id` (required): Seed movie identifier
- `k` (optional): Number of similar items, default 10

**Response:** `movie_id`, `title`, `similar_items` (list), `model_version`

#### GET /popular

Non-personalized popularity-based recommendations.

**Query params:**
- `k` (optional): Number of items, default 10
- `genre` (optional): Filter by genre

**Response:** `recommendations` (list), `source`

---

## 5. Directory Structure

```
movie-recommender/
├── configs/
│   ├── data.yaml
│   ├── training.yaml
│   ├── serving.yaml
│   └── monitoring.yaml
│
├── data/
│   ├── raw/                    # Original MovieLens CSVs
│   ├── processed/              # Train/val/test splits, mappings
│   ├── features/               # Item features, popularity
│   └── logs/                   # Request logs (SQLite)
│
├── models/
│   └── production/             # Serving artifacts
│
├── src/
│   ├── data/                   # Data pipeline
│   │   ├── download.py
│   │   ├── preprocess.py
│   │   ├── split.py
│   │   └── mappings.py
│   │
│   ├── features/               # Feature engineering
│   │   └── build.py
│   │
│   ├── models/                 # Recommender models
│   │   ├── base.py
│   │   ├── item_item.py
│   │   ├── als.py
│   │   └── index.py
│   │
│   ├── training/               # Training pipeline
│   │   ├── train.py
│   │   ├── evaluate.py
│   │   ├── export.py
│   │   └── mlflow_utils.py
│   │
│   ├── api/                    # FastAPI serving
│   │   ├── main.py
│   │   ├── schemas.py
│   │   ├── model_loader.py
│   │   ├── recommender.py
│   │   ├── fallback.py
│   │   └── logger.py
│   │
│   ├── frontend/               # Streamlit frontend
│   │   ├── app.py
│   │   ├── api_client.py
│   │   └── pages/
│   │       ├── personalized.py
│   │       ├── similar.py
│   │       ├── popular.py
│   │       └── about.py
│   │
│   ├── monitoring/             # Monitoring & KPIs
│   │   ├── kpis.py
│   │   ├── replay_eval.py
│   │   └── dashboard.py
│   │
│   └── pipeline/               # Retraining utilities (v2)
│       └── data_merge.py
│
├── tests/                      # Unit tests
│
├── .pre-commit-config.yaml
├── Dockerfile
├── docker-compose.yml
├── requirements.txt
├── Makefile
├── FUTURE_IMPLEMENTATIONS.md
└── README.md
```

---

## 6. Production Artifacts

Contents of `models/production/` after training:

| File | Description |
|------|-------------|
| `model_metadata.json` | Model version, training date, metrics, hyperparameters |
| `user_embeddings.npy` | User latent factors (n_users × embedding_dim) |
| `item_embeddings.npy` | Item latent factors (n_items × embedding_dim) |
| `item_index.faiss` | FAISS index for fast item similarity search |
| `user_mapping.json` | user_id → matrix index |
| `item_mapping.json` | movie_id → matrix index |
| `item_features.json` | Item metadata (title, genres, year) |

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
| is_fallback | INTEGER | 1 if cold-start path used |
| fallback_reason | TEXT | Reason for fallback |

---

## 8. v1 Limitations

| Limitation | Impact | Planned for v2 |
|------------|--------|----------------|
| No user authentication | Users identified by dataset IDs only | Yes |
| No new user registration | Cannot add users not in training data | Yes |
| No real-time ratings | No feedback loop for learning | Yes |
| Static model | Must manually retrain | Yes |
| MovieLens 100K only | Limited scale | Yes (1M/10M) |

See `FUTURE_IMPLEMENTATIONS.md` for detailed v2 roadmap.

---

## 9. Running the System

### Local Development

```bash
# Terminal 1: API (port 8000)
PYTHONPATH=. python -m src.api.main

# Terminal 2: Frontend (port 8502)
PYTHONPATH=. streamlit run src/frontend/app.py --server.port 8502

# Terminal 3: Monitoring (port 8501)
PYTHONPATH=. streamlit run src/monitoring/dashboard.py --server.port 8501
```

### Docker

```bash
docker-compose up
```

### Useful Commands

```bash
make download-data    # Download MovieLens dataset
make process-data     # Run preprocessing
make build-features   # Build feature matrices
make train            # Train models
make serve            # Start API server
make test             # Run tests
make lint             # Run linters
make check            # Run all checks
```
