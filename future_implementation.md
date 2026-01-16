# Future Implementations

This document outlines planned features and improvements for future versions of the Movie Recommendation System.

## Current Scope (v1)

Version 1 is a **learning-focused, end-to-end recommendation system** that demonstrates:

- ✅ Data pipeline (ingestion, preprocessing, feature engineering)
- ✅ Model training (Item-Item similarity, ALS matrix factorization)
- ✅ Model evaluation (ranking metrics: NDCG, Precision, Recall, MRR)
- ✅ Experiment tracking (MLflow)
- ✅ REST API serving (FastAPI)
- ✅ User-facing frontend (Streamlit)
- ✅ Monitoring dashboard (KPIs, latency, quality metrics)
- ✅ Request logging (SQLite)

### Current Limitations

1. **No user authentication** — Users are identified by pre-existing IDs from the dataset
2. **No new user registration** — Cannot add users not in the training data
3. **No real-time feedback** — Users cannot rate movies; no feedback loop
4. **Static model** — Retraining pipeline exists but lacks real interaction data
5. **Item-Item model outperforms ALS** — On 100K dataset; may change with larger data

---

## Planned Features (v2+)

### 1. User Authentication & Management

**Goal:** Allow real users to create accounts and log in.

**Components:**
- User database (SQLite or PostgreSQL)
- Registration flow (username, email, password)
- Password hashing (bcrypt)
- Session management (JWT tokens or session cookies)
- Login/logout UI in frontend

**API Endpoints:**
```
POST /auth/register    — Create new account
POST /auth/login       — Authenticate and get token
POST /auth/logout      — Invalidate session
GET  /auth/me          — Get current user info
```

**Frontend Changes:**
- Login/Register pages
- Protected routes (recommendations require login)
- User profile page

---

### 2. Real-Time Rating System

**Goal:** Allow users to rate movies and capture genuine feedback.

**Components:**
- Ratings database table
- Rating submission API
- Star rating UI component
- Rating history per user

**API Endpoints:**
```
POST /ratings          — Submit a rating {movie_id, rating}
GET  /ratings/user/{id} — Get user's rating history
GET  /ratings/movie/{id} — Get movie's ratings
DELETE /ratings/{id}   — Remove a rating
```

**Database Schema:**
```sql
CREATE TABLE ratings (
    id INTEGER PRIMARY KEY,
    user_id INTEGER NOT NULL,
    movie_id INTEGER NOT NULL,
    rating INTEGER NOT NULL CHECK (rating >= 1 AND rating <= 5),
    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(user_id, movie_id)
);
```

**Frontend Changes:**
- Star rating component on movie cards
- "Rate this movie" modal
- User's rating history page
- Update recommendations after rating

---

### 3. New User Onboarding (Cold-Start)

**Goal:** Provide good recommendations for users with no history.

**Approaches:**
1. **Preference survey** — Ask users to rate 5-10 seed movies on signup
2. **Genre preferences** — Let users select favorite genres
3. **Popular-based start** — Show popular movies until preferences learned

**Frontend Flow:**
```
Register → Onboarding Survey → Rate 5 Movies → Get Personalized Recs
```

---

### 4. Automated Retraining Pipeline

**Goal:** Continuously improve the model with new interaction data.

**Components (partially built in v1):**
- Data merge module ✅ (built)
- Promotion logic ✅ (built)
- Scheduled retraining trigger
- A/B testing framework
- Model versioning and rollback

**Workflow:**
```
New ratings collected
    ↓
Scheduled job (daily/weekly)
    ↓
Merge new data with training set
    ↓
Retrain model
    ↓
Evaluate on validation set
    ↓
Compare vs production (NDCG@10 ≥ 1% improvement)
    ↓
Promote or reject
    ↓
Hot-reload API with new model
```

---

### 5. Scale to MovieLens 1M/10M

**Goal:** Test system with larger datasets.

**Expected Changes:**
- ALS may outperform Item-Item at scale
- Need batch processing for training
- May need approximate nearest neighbors (FAISS already supported)
- Database optimization for larger request logs

**Tasks:**
- Update data download to support ML-1M/10M
- Benchmark training time and memory
- Tune hyperparameters for larger data
- Evaluate model quality differences

---

### 6. Enhanced Monitoring

**Goal:** Production-grade observability.

**Components:**
- Prometheus metrics export
- Grafana dashboards
- Alerting (latency spikes, error rates)
- A/B test analysis
- User engagement metrics (CTR proxy)

---

### 7. Deployment Improvements

**Goal:** Production-ready deployment.

**Components:**
- Docker Compose for full stack
- Kubernetes manifests (optional)
- Health checks and readiness probes
- Horizontal scaling for API
- CDN for static assets

---

## Technical Debt

Items to address before v2:

1. **Test coverage** — Add more integration tests
2. **Error handling** — More granular error types in API
3. **Logging** — Structured logging (JSON format)
4. **Configuration** — Environment-based config (dev/staging/prod)
5. **Documentation** — API documentation (Swagger/OpenAPI)

---

## Discussion Notes

### Why Pause Retraining? (v1 Decision)

The retraining pipeline (Steps 6.5-6.8) was paused because:

1. **No real feedback loop** — Without user ratings, we're simulating interactions
2. **Item-Item outperforms ALS** — On 100K data, making ALS promotion criteria less meaningful
3. **Need user system first** — Real retraining requires real user interactions

The data merge (`src/pipeline/data_merge.py`) and promotion logic (`src/pipeline/promote.py`) foundations are built and can be activated once user ratings are implemented.

### Priority Order for v2

1. **User authentication** — Foundation for everything else
2. **Rating system** — Enables real feedback
3. **Cold-start onboarding** — Better new user experience
4. **Retraining pipeline** — Now meaningful with real data
5. **Scale to 1M** — Validate at larger scale

---

## References

- [MovieLens Dataset](https://grouplens.org/datasets/movielens/)
- [Implicit Library (ALS)](https://implicit.readthedocs.io/)
- [FAISS (Similarity Search)](https://faiss.ai/)
- [FastAPI Documentation](https://fastapi.tiangolo.com/)
- [Streamlit Documentation](https://docs.streamlit.io/)
