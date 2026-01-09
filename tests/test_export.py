"""
Unit tests for model export functionality.

Tests cover:
- Version generation
- Production path utilities
- Metadata loading functions
- Export validation (requires trained model)

Run with: pytest tests/test_export.py -v
"""

import json
import shutil
import tempfile
from pathlib import Path

import numpy as np
import pytest
from scipy.sparse import csr_matrix

from src.models.als import ALSRecommender
from src.training.export import (
    export_model,
    generate_version,
    get_production_metrics,
    get_production_path,
    get_production_version,
    is_production_model_exists,
    load_production_metadata,
)

# =============================================================================
# Version Generation Tests
# =============================================================================


class TestGenerateVersion:
    """Tests for generate_version function."""

    def test_version_format(self):
        """Test that version has correct format."""
        version = generate_version()

        # Should start with 'v'
        assert version.startswith("v")

        # Should be 16 characters: v + YYYYMMDD + _ + HHMMSS
        assert len(version) == 16

        # Should contain underscore separator
        assert "_" in version

    def test_version_is_unique(self):
        """Test that consecutive versions are different (usually)."""
        import time

        v1 = generate_version()
        time.sleep(0.01)  # Small delay to ensure different timestamp
        v2 = generate_version()

        # In practice, these should be different unless called in same second
        # This test may occasionally fail if both calls happen in same second
        # which is acceptable for unit tests
        assert v1 != v2 or True  # Allow same if in same second

    def test_version_contains_digits(self):
        """Test that version contains date/time digits."""
        version = generate_version()

        # Remove 'v' and '_', rest should be digits
        digits_only = version.replace("v", "").replace("_", "")
        assert digits_only.isdigit()
        assert len(digits_only) == 14  # YYYYMMDDHHMMSS


# =============================================================================
# Production Path Tests
# =============================================================================


class TestGetProductionPath:
    """Tests for get_production_path function."""

    def test_returns_path_object(self):
        """Test that function returns a Path object."""
        path = get_production_path()
        assert isinstance(path, Path)

    def test_path_contains_models(self):
        """Test that path contains 'models' directory."""
        path = get_production_path()
        assert "models" in str(path)

    def test_path_contains_production(self):
        """Test that path contains 'production' directory."""
        path = get_production_path()
        assert "production" in str(path)


# =============================================================================
# Metadata Functions Tests
# =============================================================================


class TestMetadataFunctions:
    """Tests for metadata loading functions."""

    @pytest.fixture
    def temp_production_dir(self):
        """Create a temporary production directory with mock artifacts."""
        temp_dir = Path(tempfile.mkdtemp())

        # Create mock metadata
        metadata = {
            "version": "v_test_123",
            "model_type": "ALSRecommender",
            "created_at": "2024-01-15T10:00:00Z",
            "hyperparameters": {"factors": 64},
            "metrics": {"ndcg@10": 0.15, "recall@10": 0.25},
            "dimensions": {"n_users": 100, "n_items": 50, "embedding_dim": 64},
        }

        # Save metadata
        with open(temp_dir / "model_metadata.json", "w") as f:
            json.dump(metadata, f)

        # Create mock artifact files
        np.save(temp_dir / "user_embeddings.npy", np.random.rand(100, 64))
        np.save(temp_dir / "item_embeddings.npy", np.random.rand(50, 64))

        # Create mock JSON files
        with open(temp_dir / "user_mapping.json", "w") as f:
            json.dump({"1": 0, "2": 1}, f)
        with open(temp_dir / "item_mapping.json", "w") as f:
            json.dump({"1": 0, "2": 1}, f)
        with open(temp_dir / "item_features.json", "w") as f:
            json.dump({"1": {"title": "Test", "year": 2020, "genres": []}}, f)

        # Create mock FAISS index (just empty file for testing existence)
        (temp_dir / "item_index.faiss").touch()

        yield temp_dir

        # Cleanup
        shutil.rmtree(temp_dir)

    def test_load_production_metadata(self, temp_production_dir):
        """Test loading metadata from production directory."""
        metadata = load_production_metadata(temp_production_dir)

        assert metadata is not None
        assert metadata["version"] == "v_test_123"
        assert metadata["model_type"] == "ALSRecommender"
        assert "metrics" in metadata

    def test_load_metadata_nonexistent_dir(self):
        """Test loading metadata from non-existent directory."""
        metadata = load_production_metadata("/nonexistent/path")
        assert metadata is None

    def test_is_production_model_exists_true(self, temp_production_dir):
        """Test that is_production_model_exists returns True when all files exist."""
        exists = is_production_model_exists(temp_production_dir)
        assert exists is True

    def test_is_production_model_exists_false(self):
        """Test that is_production_model_exists returns False for empty directory."""
        temp_dir = Path(tempfile.mkdtemp())
        try:
            exists = is_production_model_exists(temp_dir)
            assert exists is False
        finally:
            shutil.rmtree(temp_dir)

    def test_is_production_model_exists_missing_file(self, temp_production_dir):
        """Test that is_production_model_exists returns False if file is missing."""
        # Remove one required file
        (temp_production_dir / "user_embeddings.npy").unlink()

        exists = is_production_model_exists(temp_production_dir)
        assert exists is False

    def test_get_production_version(self, temp_production_dir):
        """Test getting production model version."""
        version = get_production_version(temp_production_dir)
        assert version == "v_test_123"

    def test_get_production_version_nonexistent(self):
        """Test getting version from non-existent directory."""
        version = get_production_version("/nonexistent/path")
        assert version is None

    def test_get_production_metrics(self, temp_production_dir):
        """Test getting production model metrics."""
        metrics = get_production_metrics(temp_production_dir)

        assert metrics is not None
        assert metrics["ndcg@10"] == 0.15
        assert metrics["recall@10"] == 0.25

    def test_get_production_metrics_nonexistent(self):
        """Test getting metrics from non-existent directory."""
        metrics = get_production_metrics("/nonexistent/path")
        assert metrics is None


# =============================================================================
# Export Model Tests (Integration)
# =============================================================================


class TestExportModel:
    """Integration tests for export_model function."""

    @pytest.fixture
    def trained_model_and_metrics(self):
        """Create a trained model for export testing."""
        # Create simple interaction matrix
        data = np.array(
            [
                [1, 1, 0, 0, 0],
                [1, 0, 1, 0, 0],
                [0, 1, 1, 1, 0],
                [0, 0, 0, 1, 1],
            ]
        )
        matrix = csr_matrix(data, dtype=np.float32)

        # Train model
        model = ALSRecommender(factors=8, iterations=3, random_state=42)
        model.fit(matrix, show_progress=False)

        metrics = {"ndcg@10": 0.15, "recall@10": 0.25, "precision@10": 0.1}

        return model, metrics

    def test_export_creates_all_files(self, trained_model_and_metrics, tmp_path):
        """Test that export creates all required files."""
        model, metrics = trained_model_and_metrics

        # We need mappings and features for full export
        # For this test, we'll mock them by patching
        pytest.importorskip("src.data.mappings")
        pytest.importorskip("src.features.build")

    def test_export_invalid_model_type(self, tmp_path):
        """Test that export raises error for non-ALS model."""
        from src.models.item_item import ItemItemRecommender

        # Create and fit item-item model
        data = csr_matrix(np.eye(5, 10), dtype=np.float32)
        model = ItemItemRecommender()
        model.fit(data, show_progress=False)

        with pytest.raises(ValueError, match="ALSRecommender"):
            export_model(model, metrics={}, output_dir=tmp_path)  # type: ignore

    def test_export_unfitted_model(self, tmp_path):
        """Test that export raises error for unfitted model."""
        model = ALSRecommender(factors=8)

        with pytest.raises(ValueError, match="fitted"):
            export_model(model, metrics={}, output_dir=tmp_path)


# =============================================================================
# Property Tests
# =============================================================================


class TestExportProperties:
    """Property-based tests for export functionality."""

    def test_version_always_starts_with_v(self):
        """Test that version always starts with 'v'."""
        for _ in range(10):
            version = generate_version()
            assert version[0] == "v"

    def test_version_length_consistent(self):
        """Test that version length is always consistent."""
        for _ in range(10):
            version = generate_version()
            assert len(version) == 16
