"""
Unit tests for the data pipeline.

Tests cover:
- Configuration loading
- Data download and extraction
- Preprocessing (implicit conversion, filtering)
- Time-based splitting
- Feature building
- ID mappings

Run with: pytest tests/test_data.py -v
"""

from pathlib import Path

import pandas as pd
import pytest

# =============================================================================
# Config Tests
# =============================================================================


class TestConfig:
    """Tests for configuration loading."""

    def test_get_data_config_loads(self):
        """Test that data config loads without error."""
        from src.config import get_data_config

        config = get_data_config()
        assert config is not None

    def test_data_config_has_required_fields(self):
        """Test that data config has all required fields."""
        from src.config import get_data_config

        config = get_data_config()

        # Check source
        assert hasattr(config, "source")
        assert config.source.name == "movielens-100k"
        assert "grouplens" in config.source.url

        # Check paths
        assert hasattr(config, "paths")
        assert config.paths.raw == "data/raw"
        assert config.paths.processed == "data/processed"
        assert config.paths.features == "data/features"

        # Check preprocessing
        assert hasattr(config, "preprocessing")
        assert config.preprocessing.implicit_threshold == 4
        assert config.preprocessing.min_user_interactions == 5
        assert config.preprocessing.min_item_interactions == 10

        # Check splitting
        assert hasattr(config, "splitting")
        assert config.splitting.train_ratio == 0.70
        assert config.splitting.val_ratio == 0.15
        assert config.splitting.test_ratio == 0.15

    def test_split_ratios_sum_to_one(self):
        """Test that train/val/test ratios sum to 1.0."""
        from src.config import get_data_config

        config = get_data_config()
        total = (
            config.splitting.train_ratio
            + config.splitting.val_ratio
            + config.splitting.test_ratio
        )
        assert abs(total - 1.0) < 0.001

    def test_get_paths_return_path_objects(self):
        """Test that path methods return Path objects."""
        from src.config import get_data_config

        config = get_data_config()

        assert isinstance(config.get_raw_path(), Path)
        assert isinstance(config.get_processed_path(), Path)
        assert isinstance(config.get_features_path(), Path)


# =============================================================================
# Download Tests
# =============================================================================


class TestDownload:
    """Tests for data download functionality."""

    def test_is_data_downloaded(self):
        """Test that download check works."""
        from src.data.download import is_data_downloaded

        # This should return True if data exists, False otherwise
        result = is_data_downloaded()
        assert isinstance(result, bool)

    def test_get_expected_data_path(self):
        """Test that expected data path is correct."""
        from src.data.download import get_expected_data_path

        path = get_expected_data_path()
        assert isinstance(path, Path)
        assert "ml-100k" in str(path)

    @pytest.mark.skipif(
        not Path("data/raw/ml-100k").exists(), reason="Data not downloaded yet"
    )
    def test_data_files_exist(self):
        """Test that required data files exist after download."""
        from src.data.download import get_expected_data_path

        data_path = get_expected_data_path()

        # Check required files
        assert (data_path / "u.data").exists(), "Ratings file missing"
        assert (data_path / "u.item").exists(), "Movies file missing"

        # Check files are not empty
        assert (data_path / "u.data").stat().st_size > 0
        assert (data_path / "u.item").stat().st_size > 0


# =============================================================================
# Preprocessing Tests
# =============================================================================


class TestPreprocessing:
    """Tests for data preprocessing."""

    @pytest.mark.skipif(
        not Path("data/raw/ml-100k").exists(), reason="Data not downloaded yet"
    )
    def test_load_ratings(self):
        """Test that ratings load correctly."""
        from src.data.preprocess import load_ratings

        df = load_ratings()

        # Check structure
        assert isinstance(df, pd.DataFrame)
        assert list(df.columns) == ["user_id", "item_id", "rating", "timestamp"]

        # Check data types
        assert df["user_id"].dtype in ["int32", "int64"]
        assert df["item_id"].dtype in ["int32", "int64"]
        assert df["rating"].dtype in ["int8", "int16", "int32", "int64"]

        # Check value ranges
        assert df["rating"].min() >= 1
        assert df["rating"].max() <= 5

        # MovieLens 100K should have 100,000 ratings
        assert len(df) == 100000

    @pytest.mark.skipif(
        not Path("data/raw/ml-100k").exists(), reason="Data not downloaded yet"
    )
    def test_load_movies(self):
        """Test that movies load correctly."""
        from src.data.preprocess import load_movies

        df = load_movies()

        # Check structure
        assert isinstance(df, pd.DataFrame)
        assert "item_id" in df.columns
        assert "title" in df.columns

        # MovieLens 100K should have 1,682 movies
        assert len(df) == 1682

    @pytest.mark.skipif(
        not Path("data/raw/ml-100k").exists(), reason="Data not downloaded yet"
    )
    def test_convert_to_implicit(self):
        """Test implicit feedback conversion."""
        from src.data.preprocess import convert_to_implicit, load_ratings

        ratings_df = load_ratings()
        implicit_df = convert_to_implicit(ratings_df, threshold=4)

        # Should have fewer rows (only ratings >= 4)
        assert len(implicit_df) < len(ratings_df)

        # Should not have rating column
        assert "rating" not in implicit_df.columns

        # Should have user_id, item_id, timestamp
        assert "user_id" in implicit_df.columns
        assert "item_id" in implicit_df.columns
        assert "timestamp" in implicit_df.columns

    @pytest.mark.skipif(
        not Path("data/raw/ml-100k").exists(), reason="Data not downloaded yet"
    )
    def test_filter_by_interactions(self):
        """Test user/item filtering."""
        from src.data.preprocess import (
            convert_to_implicit,
            filter_by_interactions,
            load_ratings,
        )

        ratings_df = load_ratings()
        implicit_df = convert_to_implicit(ratings_df, threshold=4)
        filtered_df = filter_by_interactions(
            implicit_df,
            min_user_interactions=5,
            min_item_interactions=10,
        )

        # Should have fewer rows after filtering
        assert len(filtered_df) <= len(implicit_df)

        # All users should have >= 5 interactions
        user_counts = filtered_df["user_id"].value_counts()
        assert user_counts.min() >= 5

        # All items should have >= 10 interactions
        item_counts = filtered_df["item_id"].value_counts()
        assert item_counts.min() >= 10


# =============================================================================
# Split Tests
# =============================================================================


class TestSplit:
    """Tests for time-based splitting."""

    @pytest.mark.skipif(
        not Path("data/processed/interactions.csv").exists(),
        reason="Preprocessing not run yet",
    )
    def test_load_splits(self):
        """Test that splits load correctly."""
        from src.data.split import load_splits

        train_df, val_df, test_df = load_splits()

        # All should be DataFrames
        assert isinstance(train_df, pd.DataFrame)
        assert isinstance(val_df, pd.DataFrame)
        assert isinstance(test_df, pd.DataFrame)

        # All should have same columns
        assert list(train_df.columns) == list(val_df.columns)
        assert list(val_df.columns) == list(test_df.columns)

    @pytest.mark.skipif(
        not Path("data/processed/train.csv").exists(), reason="Splitting not run yet"
    )
    def test_split_ratios_approximately_correct(self):
        """Test that split sizes match configured ratios."""
        from src.config import get_data_config
        from src.data.split import load_splits

        config = get_data_config()
        train_df, val_df, test_df = load_splits()

        total = len(train_df) + len(val_df) + len(test_df)

        train_ratio = len(train_df) / total
        val_ratio = len(val_df) / total
        test_ratio = len(test_df) / total

        # Allow 1% tolerance
        assert abs(train_ratio - config.splitting.train_ratio) < 0.01
        assert abs(val_ratio - config.splitting.val_ratio) < 0.01
        assert abs(test_ratio - config.splitting.test_ratio) < 0.01

    @pytest.mark.skipif(
        not Path("data/processed/train.csv").exists(), reason="Splitting not run yet"
    )
    def test_no_time_leakage(self):
        """Test that train timestamps are always before val/test."""
        from src.data.split import load_splits

        train_df, val_df, test_df = load_splits()

        # Train max should be <= val min
        assert train_df["timestamp"].max() <= val_df["timestamp"].min()

        # Val max should be <= test min
        assert val_df["timestamp"].max() <= test_df["timestamp"].min()

    @pytest.mark.skipif(
        not Path("data/processed/train.csv").exists(), reason="Splitting not run yet"
    )
    def test_no_duplicate_interactions(self):
        """Test that no interaction appears in multiple splits."""
        from src.data.split import load_splits

        train_df, val_df, test_df = load_splits()

        # Create unique keys for each interaction
        def make_keys(df):
            return set(zip(df["user_id"], df["item_id"], df["timestamp"]))

        train_keys = make_keys(train_df)
        val_keys = make_keys(val_df)
        test_keys = make_keys(test_df)

        # No overlap between sets
        assert len(train_keys & val_keys) == 0
        assert len(train_keys & test_keys) == 0
        assert len(val_keys & test_keys) == 0


# =============================================================================
# Feature Tests
# =============================================================================


class TestFeatures:
    """Tests for feature building."""

    def test_extract_year_from_title(self):
        """Test year extraction from movie titles."""
        from src.features.build import extract_year_from_title

        # Normal cases
        assert extract_year_from_title("Toy Story (1995)") == 1995
        assert extract_year_from_title("GoldenEye (1995)") == 1995
        assert extract_year_from_title("2001: A Space Odyssey (1968)") == 1968

        # Edge cases
        assert extract_year_from_title("Some Movie") is None
        assert extract_year_from_title("Movie (abc)") is None
        assert extract_year_from_title("") is None

    def test_genre_names_constant(self):
        """Test that genre names are defined correctly."""
        from src.features.build import GENRE_NAMES

        assert isinstance(GENRE_NAMES, list)
        assert len(GENRE_NAMES) == 19  # MovieLens 100K has 19 genres
        assert "Action" in GENRE_NAMES
        assert "Comedy" in GENRE_NAMES
        assert "Drama" in GENRE_NAMES

    @pytest.mark.skipif(
        not Path("data/features/item_features.csv").exists(),
        reason="Features not built yet",
    )
    def test_load_item_features(self):
        """Test that item features load correctly."""
        from src.features.build import GENRE_NAMES, load_item_features

        df = load_item_features()

        # Check required columns
        assert "item_id" in df.columns
        assert "title" in df.columns
        assert "year" in df.columns

        # Check genre columns exist
        for genre in GENRE_NAMES:
            assert genre in df.columns

        # Genre columns should be binary (0 or 1)
        for genre in GENRE_NAMES:
            assert df[genre].isin([0, 1]).all()

    @pytest.mark.skipif(
        not Path("data/features/popularity.csv").exists(),
        reason="Features not built yet",
    )
    def test_load_popularity(self):
        """Test that popularity features load correctly."""
        from src.features.build import load_popularity

        df = load_popularity()

        # Check required columns
        assert "item_id" in df.columns
        assert "interaction_count" in df.columns
        assert "popularity_rank" in df.columns

        # Ranks should start at 1
        assert df["popularity_rank"].min() == 1

        # Interaction counts should be positive
        assert (df["interaction_count"] > 0).all()

        # Higher interaction count should mean lower rank number
        top_item = df[df["popularity_rank"] == 1].iloc[0]
        assert top_item["interaction_count"] == df["interaction_count"].max()


# =============================================================================
# Mapping Tests
# =============================================================================


class TestMappings:
    """Tests for ID mappings."""

    @pytest.mark.skipif(
        not Path("data/processed/user_mapping.json").exists(),
        reason="Mappings not built yet",
    )
    def test_load_mappings(self):
        """Test that mappings load correctly."""
        from src.data.mappings import load_mappings

        user_mapping, item_mapping = load_mappings()

        # Should be dictionaries
        assert isinstance(user_mapping, dict)
        assert isinstance(item_mapping, dict)

        # Should not be empty
        assert len(user_mapping) > 0
        assert len(item_mapping) > 0

        # Keys and values should be integers
        for k, v in list(user_mapping.items())[:5]:
            assert isinstance(k, int)
            assert isinstance(v, int)

    @pytest.mark.skipif(
        not Path("data/processed/user_mapping.json").exists(),
        reason="Mappings not built yet",
    )
    def test_mappings_are_contiguous(self):
        """Test that matrix indices are contiguous (0, 1, 2, ...)."""
        from src.data.mappings import load_mappings

        user_mapping, item_mapping = load_mappings()

        # User indices should be 0 to n_users-1
        user_indices = sorted(user_mapping.values())
        assert user_indices == list(range(len(user_mapping)))

        # Item indices should be 0 to n_items-1
        item_indices = sorted(item_mapping.values())
        assert item_indices == list(range(len(item_mapping)))

    @pytest.mark.skipif(
        not Path("data/processed/user_mapping.json").exists(),
        reason="Mappings not built yet",
    )
    def test_id_mapper_class(self):
        """Test the IDMapper utility class."""
        from src.data.mappings import IDMapper

        mapper = IDMapper.from_disk()

        # Test properties
        assert mapper.n_users > 0
        assert mapper.n_items > 0

        # Test forward and reverse mapping consistency
        for user_id in mapper.get_all_user_ids()[:10]:
            idx = mapper.user_to_index(user_id)
            assert mapper.index_to_user(idx) == user_id

        for item_id in mapper.get_all_item_ids()[:10]:
            idx = mapper.item_to_index(item_id)
            assert mapper.index_to_item(idx) == item_id

    @pytest.mark.skipif(
        not Path("data/processed/user_mapping.json").exists(),
        reason="Mappings not built yet",
    )
    def test_mapper_handles_unknown_ids(self):
        """Test that mapper returns None for unknown IDs."""
        from src.data.mappings import IDMapper

        mapper = IDMapper.from_disk()

        # Unknown user
        assert mapper.user_to_index(9999999) is None
        assert mapper.has_user(9999999) is False

        # Unknown item
        assert mapper.item_to_index(9999999) is None
        assert mapper.has_item(9999999) is False


# =============================================================================
# Integration Tests
# =============================================================================


class TestIntegration:
    """Integration tests for the full data pipeline."""

    @pytest.mark.skipif(
        not Path("data/processed/train.csv").exists(),
        reason="Full pipeline not run yet",
    )
    def test_train_users_in_mapping(self):
        """Test that all training users have mappings."""
        from src.data.mappings import IDMapper
        from src.data.split import load_splits

        train_df, _, _ = load_splits()
        mapper = IDMapper.from_disk()

        train_users = set(train_df["user_id"].unique())
        mapped_users = set(mapper.get_all_user_ids())

        # All train users should have mappings
        assert train_users == mapped_users

    @pytest.mark.skipif(
        not Path("data/processed/train.csv").exists(),
        reason="Full pipeline not run yet",
    )
    def test_train_items_in_mapping(self):
        """Test that all training items have mappings."""
        from src.data.mappings import IDMapper
        from src.data.split import load_splits

        train_df, _, _ = load_splits()
        mapper = IDMapper.from_disk()

        train_items = set(train_df["item_id"].unique())
        mapped_items = set(mapper.get_all_item_ids())

        # All train items should have mappings
        assert train_items == mapped_items

    @pytest.mark.skipif(
        not Path("data/features/popularity.csv").exists(),
        reason="Features not built yet",
    )
    def test_popularity_uses_train_only(self):
        """Test that popularity is computed from training data only."""
        from src.data.split import load_splits
        from src.features.build import load_popularity

        train_df, _, _ = load_splits()
        popularity_df = load_popularity()

        # Popularity items should match training items
        train_items = set(train_df["item_id"].unique())
        popularity_items = set(popularity_df["item_id"])

        assert train_items == popularity_items

        # Verify counts match
        train_counts = train_df.groupby("item_id").size()
        for _, row in popularity_df.head(10).iterrows():
            item_id = row["item_id"]
            expected_count = train_counts[item_id]
            assert row["interaction_count"] == expected_count
