# Movie Recommendation System (v1)

An end-to-end, production-style movie recommendation system built for learning MLOps practices. Runs fully on a local machine using open-source tools.

## ✨ Features

- **Data Pipeline**: Download, preprocess, and split MovieLens 100K data
- **Training**: Item-item similarity and ALS matrix factorization models
- **Experiment Tracking**: MLflow for logging, versioning, and model registry
- **Serving**: FastAPI REST API for real-time recommendations
- **Frontend**: Streamlit web interface for users to explore recommendations
- **Monitoring**: Request logging, KPI computation, and monitoring dashboard

## 🖥️ Screenshots

| Frontend | Monitoring Dashboard |
|----------|---------------------|
| Personalized recommendations, similar movies, popular browser | Traffic, latency, and quality metrics |

## 🛠️ Tech Stack

| Component | Technology |
|-----------|------------|
| Language | Python 3.9+ |
| Data | Pandas, NumPy, SciPy |
| ML Models | scikit-learn, implicit |
| Similarity Search | FAISS |
| Experiment Tracking | MLflow |
| API | FastAPI, Uvicorn |
| Frontend | Streamlit |
| Database | SQLite (request logs) |
| Containerization | Docker, docker-compose |
| Code Quality | ruff, black, mypy, pre-commit |

## 📁 Project Structure

```
├── configs/             # YAML configuration files
├── data/                # Data storage (gitignored)
│   ├── raw/             # Original MovieLens files
│   ├── processed/       # Cleaned and split data
│   ├── features/        # Generated features
│   └── logs/            # Request logs (SQLite)
├── models/              # Model artifacts (gitignored)
│   └── production/      # Currently deployed model
├── src/
│   ├── data/            # Data download, preprocessing, splitting
│   ├── features/        # Feature engineering
│   ├── models/          # Recommender implementations
│   ├── training/        # Training and evaluation
│   ├── api/             # FastAPI service
│   ├── frontend/        # Streamlit user interface
│   ├── monitoring/      # KPIs and dashboard
│   └── pipeline/        # Retraining utilities
├── tests/               # Unit tests
├── Dockerfile
├── docker-compose.yml
├── Makefile
├── FUTURE_IMPLEMENTATIONS.md
└── README.md
```

## 🚀 Quick Start

### Prerequisites

- Python 3.9+
- ~2GB disk space for data and models
- (Optional) Docker and docker-compose

### Installation

```bash
# Clone the repository
git clone <repository-url>
cd movie-recommender

# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
pip install -r requirements-dev.txt

# Set up pre-commit hooks
pre-commit install
```

### Run the Full Pipeline

```bash
# 1. Download MovieLens data
make download-data

# 2. Preprocess and split data
make process-data

# 3. Build features
make build-features

# 4. Train models
make train

# 5. Start the system (see next section)
```

### Running the System

Open three terminals:

```bash
# Terminal 1: Start API (port 8000)
PYTHONPATH=. python -m src.api.main

# Terminal 2: Start Frontend (port 8502)
PYTHONPATH=. streamlit run src/frontend/app.py --server.port 8502

# Terminal 3: Start Monitoring Dashboard (port 8501)
PYTHONPATH=. streamlit run src/monitoring/dashboard.py --server.port 8501
```

### Access Services

| Service | URL | Description |
|---------|-----|-------------|
| Frontend | http://localhost:8502 | User-facing recommendation interface |
| API | http://localhost:8000 | REST API endpoints |
| API Docs | http://localhost:8000/docs | Interactive API documentation |
| Monitoring | http://localhost:8501 | KPI dashboard |
| MLflow UI | http://localhost:5000 | Experiment tracking (run `mlflow ui` separately) |

### Using Docker

```bash
# Build and start all services
docker-compose up --build

# Run in background
docker-compose up -d --build

# Stop services
docker-compose down
```

## 🎬 Frontend Pages

| Page | Description |
|------|-------------|
| **Personalized** | Enter a user ID to get tailored movie recommendations |
| **Find Similar** | Select a movie to discover similar films |
| **Popular** | Browse trending movies with optional genre filtering |
| **About** | System status, how it works, and usage tips |

## 📡 API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/health` | GET | Health check with model info |
| `/recommend` | POST | Personalized recommendations for a user |
| `/similar` | POST | Find movies similar to a given movie |
| `/popular` | GET | Get popular movies (optional genre filter) |

### Example Requests

```bash
# Health check
curl http://localhost:8000/health

# Get recommendations for user 196
curl -X POST "http://localhost:8000/recommend" \
  -H "Content-Type: application/json" \
  -d '{"user_id": 196, "k": 10}'

# Find movies similar to movie 1 (Toy Story)
curl -X POST "http://localhost:8000/similar" \
  -H "Content-Type: application/json" \
  -d '{"movie_id": 1, "k": 10}'

# Get popular Action movies
curl "http://localhost:8000/popular?k=10&genre=Action"
```

## 📊 Monitoring Dashboard

The monitoring dashboard displays:

- **API Health**: Status, model version, uptime
- **Traffic Metrics**: Request count, requests/minute, unique users
- **Latency Metrics**: Mean, p50, p95, p99 response times
- **Quality Metrics**: Fallback rate, catalog coverage

## 🧪 Development

### Code Quality

```bash
# Format code
make format

# Run linter
make lint

# Run type checker
make typecheck

# Run tests
make test

# Run all checks
make check
```

### Available Make Commands

```bash
make help           # Show all available commands
make download-data  # Download MovieLens dataset
make process-data   # Run preprocessing and splitting
make build-features # Build feature matrices
make train          # Train models with MLflow tracking
make serve          # Start API server
make test           # Run unit tests
make lint           # Run ruff linter
make format         # Format code with black
make check          # Run all quality checks
```

## ⚙️ Configuration

Configuration files in `configs/`:

| File | Purpose |
|------|---------|
| `data.yaml` | Data paths, preprocessing settings |
| `training.yaml` | Model hyperparameters, MLflow settings |
| `serving.yaml` | API settings, fallback behavior |
| `monitoring.yaml` | KPI thresholds, dashboard settings |

## 🚧 v1 Limitations

| Limitation | Description |
|------------|-------------|
| **No user authentication** | Users are identified by pre-existing IDs from the MovieLens dataset |
| **No new user registration** | Cannot add users that aren't in the training data |
| **No real-time ratings** | Users cannot rate movies; no feedback loop |
| **Static model** | Model must be manually retrained |
| **MovieLens 100K only** | Limited to ~100K ratings from ~1K users |

## 🔮 Future Plans (v2)

See [FUTURE_IMPLEMENTATION.md](future_implementation.md) for detailed v2 roadmap:

- User authentication and registration
- Real-time rating system
- New user onboarding (cold-start handling)
- Automated retraining pipeline
- Scale to MovieLens 1M/10M datasets
- GitHub Actions CI/CD

## 📚 Documentation

| Document | Description |
|----------|-------------|
| [architecture.md](architecture.md) | System design, data flows, API spec |
| [implementation_plan.md](implementation_plan.md) | Phase-by-phase task breakdown |
| [FUTURE_IMPLEMENTATIONS.md](FUTURE_IMPLEMENTATIONS.md) | v2 roadmap and planned features |

## 🙏 Acknowledgments

- [MovieLens](https://grouplens.org/datasets/movielens/) dataset by GroupLens Research
- [Implicit](https://implicit.readthedocs.io/) library for ALS implementation
- [FAISS](https://faiss.ai/) for similarity search

## 📄 License

MIT
