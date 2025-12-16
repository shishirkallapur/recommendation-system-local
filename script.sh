#!/bin/bash

# Run this from the movie-recommender/ root directory

echo "Creating directory structure..."

# Source code directories
mkdir -p src/data
mkdir -p src/features
mkdir -p src/models
mkdir -p src/training
mkdir -p src/api
mkdir -p src/monitoring
mkdir -p src/pipeline

# Create __init__.py files for Python packages
touch src/__init__.py
touch src/data/__init__.py
touch src/features/__init__.py
touch src/models/__init__.py
touch src/training/__init__.py
touch src/api/__init__.py
touch src/monitoring/__init__.py
touch src/pipeline/__init__.py

# Data directories (will be gitignored, but structure preserved)
mkdir -p data/raw
mkdir -p data/processed
mkdir -p data/features
mkdir -p data/logs

# Model artifacts directory
mkdir -p models/production

# Add .gitkeep to preserve empty directories
touch data/raw/.gitkeep
touch data/processed/.gitkeep
touch data/features/.gitkeep
touch data/logs/.gitkeep
touch models/production/.gitkeep

# Config directory
mkdir -p configs

# Tests directory
mkdir -p tests
touch tests/__init__.py

# Scripts directory
mkdir -p scripts

# GitHub Actions directory (for Phase 6)
mkdir -p .github/workflows

echo "Directory structure created!"
echo ""
echo "Verifying structure..."
echo ""

# Display the tree (if tree command available, otherwise use find)
if command -v tree &> /dev/null; then
    tree -a -I '.venv|.git' --dirsfirst
else
    find . -type f -name "*.py" -o -name ".gitkeep" | grep -v ".venv" | sort
fi
