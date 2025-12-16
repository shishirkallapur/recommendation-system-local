# Movie Recommendation System

An end-to-end, production-style movie recommendation system built for learning MLOps practices. Runs fully on a local machine using open-source tools.

## Features

- **Data Pipeline**: Download, preprocess, and split MovieLens data
- **Training**: Item-item similarity and ALS matrix factorization models
- **Experiment Tracking**: MLflow for logging, versioning, and model registry
- **Serving**: FastAPI REST API for real-time recommendations
- **Monitoring**: Request logging, KPI computation, Streamlit dashboard
- **Retraining**: Automated pipeline with promotion criteria

## Tech Stack

| Component | Technology |
|-----------|------------|
| Language | Python 3.9 |
| Data | Pandas, NumPy, SciPy |
| ML Models | scikit-learn, implicit |
| Similarity Search | FAISS |
| Experiment Tracking | MLflow |
| API | FastAPI, Uvicorn |
| Monitoring | Streamlit |
| Containerization | Docker, docker-compose |

## Project Structure

```
├── configs/             # YAML configuration files
├── data/                # Data storage (gitignored)
│   ├── raw/             # Original MovieLens files
│   ├── processed/       # Cleaned and split data
│   ├── features/        # Generated features
│   └── logs/            # Request logs
├── models/              # Model artifacts (gitignored)
│   └── production/      # Currently deployed model
├── src/                 # Source code
│   ├── data/            # Data download, preprocessing, splitting
│   ├── features/        # Feature engineering
│   ├── models/          # Recommender implementations
│   ├── training/        # Training and evaluation
│   ├── api/             # FastAPI service
│   ├── monitoring/      # KPIs and dashboard
│   └── pipeline/        # Retraining pipeline
├── tests/               # Unit tests
├── Dockerfile           # Container definition
├── docker-compose.yml   # Multi-container orchestration
├── Makefile             # Task automation
└── README.md
```

## Quick Start

### Prerequisites

- Python 3.9+
- Docker and docker-compose (for containerized deployment)
- ~2GB disk space for data and models

### Installation

```bash
# Clone the repository
git clone <repository-url>
cd recommendation-system-local

# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install dependencies
make install-dev

# Set up pre-commit hooks
pre-commit install
```

### Run the Pipeline

```bash
# 1. Download MovieLens data
make download-data

# 2. Preprocess and split data
make process-data

# 3. Build features
make build-features

# 4. Train models
make train

# 5. Start API server
make serve
```

### Using Docker

```bash
# Build and start all services
docker-compose up --build

# Or run in background
docker-compose up -d --build

# View logs
docker-compose logs -f

# Stop services
docker-compose down
```

### Access Services

| Service | URL |
|---------|-----|
| API | http://localhost:8000 |
| API Docs | http://localhost:8000/docs |
| MLflow UI | http://localhost:5000 |
| Dashboard | http://localhost:8501 (with `--profile monitoring`) |

## API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/health` | GET | Health check |
| `/recommend` | POST | Get personalized recommendations |
| `/similar` | POST | Get similar items |
| `/popular` | GET | Get popular items |

### Example Request

```bash
curl -X POST "http://localhost:8000/recommend" \
  -H "Content-Type: application/json" \
  -d '{"user_id": 123, "k": 10}'
```

## Development

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
make help  # Show all available commands
```

## Configuration

Configuration files are in `configs/`:

| File | Purpose |
|------|---------|
| `data.yaml` | Data paths, preprocessing settings |
| `training.yaml` | Model hyperparameters, MLflow settings |
| `serving.yaml` | API settings, fallback behavior |
| `monitoring.yaml` | KPI thresholds, dashboard settings |
| `retrain.yaml` | Promotion criteria, schedule |

## Documentation

- [Problem Statement](PROBLEM_STATEMENT.md) - Project objectives and requirements
- [Architecture](ARCHITECTURE.md) - System design and data flows
- [Implementation Plan](IMPLEMENTATION_PLAN.md) - Detailed task breakdown

## License

MIT
