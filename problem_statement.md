# Movie Recommendation System: Problem Statement

## 1. Objective

Design and implement a **production-style, end-to-end movie recommendation system** that suggests the top-N movies a user is likely to enjoy. The system must handle the full ML lifecycle: data ingestion, training, serving, **user-facing frontend**, monitoring, and retraining—all runnable on a local machine.

---

## 2. Core Requirements

### 2.1 Data & Preprocessing

- Ingest MovieLens 100K or 1M dataset (user–movie ratings with timestamps)
- Treat ratings as **implicit feedback**: rating ≥ 4 → positive interaction, else ignored
- Apply minimum interaction thresholds:
  - Users with < 5 ratings filtered out
  - Movies with < 10 ratings filtered out
- Perform **global time-based split** (all users share same cutoffs):
  - Train: oldest 70% of interactions by timestamp
  - Validation: next 15%
  - Test: most recent 15%
- Generate artifacts:
  - Interaction matrix (sparse)
  - Item features (genres one-hot, popularity stats)
  - ID mappings (user_id ↔ index, movie_id ↔ index)

### 2.2 Model Training & Evaluation

**Models to implement:**

1. **Baseline**: Item-item collaborative filtering using cosine similarity on co-occurrence vectors
2. **Primary**: Matrix factorization via Alternating Least Squares (ALS) for implicit feedback

**Evaluation approach:**

- Ranking metrics only (implicit feedback paradigm):
  - Precision@K, Recall@K, NDCG@K, Hit Rate@K (K = 5, 10, 20)
  - Mean Reciprocal Rank (MRR)
- Track all experiments in MLflow: parameters, metrics, model artifacts, embeddings

### 2.3 Model Management

**Registry workflow:**

- Register models in MLflow Model Registry: `None` → `Staging` → `Production`

**Promotion criteria:**

- Primary metric: **NDCG@10**
- Minimum relative improvement: **≥ 1%** over current Production model
- No regression > 5% on any secondary metric

**Production artifacts to export:**

- Model weights / factor matrices
- Precomputed user & item embeddings
- Approximate nearest neighbor index (FAISS) for item similarity
- ID mapping tables

### 2.4 Serving (REST API)

**Endpoints:**

| Endpoint | Method | Purpose |
|----------|--------|---------|
| `/health` | GET | Liveness check + model version info |
| `/recommend` | POST | Personalized recommendations for a user |
| `/similar` | POST | Item-item similarity ("because you watched X") |
| `/popular` | GET | Non-personalized fallback (trending/popular) |

**Serving modes:**

- **Known users**: Real-time dot product of user embedding × item embeddings, return top-K
- **Unknown users (cold-start)**: Return popular items, optionally filtered by genre
- **Unknown items**: Return 404 with explanation

**Features:**

- Exclude already-seen items by default
- Support optional filters: genre, release year range

### 2.5 User-Facing Frontend

**Requirements:**

A web-based interface that allows users to interact with the recommendation system:

| Feature | Description |
|---------|-------------|
| **Personalized Recommendations** | Known users enter their ID to get tailored recommendations |
| **Seed-Based Discovery** | New users select movies they like to get similar recommendations |
| **Popular Movies Browser** | Browse trending movies with genre filtering |
| **Movie Details** | Display movie metadata (title, year, genres, scores) |

**User Experience:**

- Clean, intuitive interface
- Real-time API status indicator
- Graceful handling of API errors
- Responsive feedback (loading states, success/error messages)

**Cold-Start Handling in UI:**

| Scenario | UI Behavior |
|----------|-------------|
| Known user | Show personalized recommendations |
| Unknown user | Show notice + popular movies fallback |
| User provides seed movie | Show similar movies to seed |

### 2.6 Cold-Start Strategy

| Scenario | Strategy |
|----------|----------|
| New user (no history) | Return popularity-based recommendations; optionally accept seed genres |
| New user (has seed item) | Return similar items to the seed via item-item similarity |
| New item (not in training) | Cannot recommend; log for next retraining batch |

### 2.7 Logging & Monitoring

**Request logging (to SQLite):**

- `request_id`, `timestamp`, `user_id`, `endpoint`, `model_version`
- `recommendations` (list of movie_ids), `scores`, `latency_ms`
- `is_fallback` (true if cold-start path was used)

**Monitoring metrics:**

- Traffic: request count, requests/minute, unique users
- Latency: mean, p50, p95, p99
- Coverage: % of catalog recommended at least once
- Fallback rate: % of requests using cold-start path

**Output:** Daily HTML/Markdown report or Streamlit dashboard

### 2.8 Offline Replay Evaluation

- Evaluate logged recommendations against subsequent interactions
- Join logged `recommendations` with future `interactions` within a time window
- Compute realized Precision@K, Hit Rate, CTR proxy
- Compare across model versions to validate online performance

### 2.9 Retraining Pipeline

**Trigger:** Manual or scheduled (simulated via cron/script)

**Steps:**

1. Ingest new interaction logs (simulated by sampling held-out data)
2. Merge with existing training data
3. Retrain both models, log to MLflow
4. Evaluate on validation set
5. If promotion criteria met → promote to Production, restart API container
6. If not → retain current Production model, log decision

---

## 3. Constraints

- **Fully local**: No cloud services required
- **Open data**: MovieLens dataset (free, well-documented)
- **Open-source stack**: Python, FastAPI, Streamlit, Docker, MLflow, SQLite
- **Single-machine scale**: Optimize for datasets up to ~1M interactions

---

## 4. Success Metrics

### Offline Metrics (Training)

- NDCG@10 ≥ 0.15 on test set
- Recall@20 ≥ 0.25
- Coverage ≥ 50% of catalog in top-100 recommendations

### Online Metrics (Serving)

- p95 latency < 100ms for `/recommend`
- p95 latency < 50ms for `/similar`
- Fallback rate < 10%
- Zero 5xx errors under normal load

### Frontend Metrics

- All pages load within 2 seconds
- API status clearly visible
- Graceful error handling (no crashes)

### Operational Metrics

- Retraining completes in < 10 minutes (100K dataset)
- Model promotion decision logged with full reasoning
- Request logs queryable for past 7 days

---

## 5. Tech Stack

| Layer | Technology | Purpose |
|-------|------------|---------|
| Language | Python 3.9+ | Core implementation |
| Data Processing | Pandas, NumPy | Data manipulation |
| ML Models | implicit (ALS), scikit-learn | Recommender algorithms |
| Similarity Search | FAISS | Fast nearest neighbors |
| Experiment Tracking | MLflow | Logging, registry, artifacts |
| API Framework | FastAPI | REST endpoints |
| Frontend | Streamlit | User-facing web interface |
| Validation | Pydantic | Request/response schemas |
| Database | SQLite | Request logs, MLflow backend |
| Containerization | Docker, docker-compose | Local deployment |
| Monitoring UI | Streamlit | Dashboard |
| Testing | pytest | Unit & integration tests |
