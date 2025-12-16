""" This module is the configuration loader for our movie reecommendaion system.
It loads the configuration from the YAML files in the configs directory.
Config are validated using Pydantic models and cached after first load for reuse.
Usage:
    from src.config import get_data_config
    data_config = get_data_config()
    print(data_config.raw_dir) #  path to raw data directory
"""

import os
from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, Field, model_validator

# =============================================================================
# Path Resolution
# =============================================================================


def get_project_root() -> Path:
    current = Path(__file__).resolve().parent
    while current != current.parent:
        if (current / "pyproject.toml").exists():
            return current
        current = current.parent
    # Assume src/ is one level below project root
    return Path(__file__).resolve().parent.parent


def get_config_dir() -> Path:
    env_config_dir = os.getenv("RECOMMENDER_CONFIG_DIR")
    if env_config_dir:
        return Path(env_config_dir)
    return get_project_root() / "configs"


# =============================================================================
# Data Config Models (matching data.yaml structure)
# =============================================================================


class SourceConfig(BaseModel):
    """Dataset source configuration."""

    name: str = Field(description="Dataset identifier (e.g., 'movielens-100k')")
    url: str = Field(description="Download URL for the dataset")


class PathsConfig(BaseModel):
    """Directory paths configuration (relative to project root)."""

    raw: str = Field(description="Directory for raw downloaded data")
    processed: str = Field(description="Directory for processed data")
    features: str = Field(description="Directory for computed features")


class PreprocessingConfig(BaseModel):
    """Preprocessing thresholds configuration."""

    implicit_threshold: int = Field(
        ge=1, le=5, description="Minimum rating to consider as positive interaction"
    )
    min_user_interactions: int = Field(
        ge=1, description="Filter users with fewer interactions"
    )
    min_item_interactions: int = Field(
        ge=1, description="Filter items with fewer interactions"
    )


class SplittingConfig(BaseModel):
    """Train/val/test split configuration."""

    method: str = Field(description="Split method (e.g., 'global_time')")
    train_ratio: float = Field(ge=0.0, le=1.0)
    val_ratio: float = Field(ge=0.0, le=1.0)
    test_ratio: float = Field(ge=0.0, le=1.0)
    seed: int = Field(description="Random seed for reproducibility")

    @model_validator(mode="after")
    def validate_ratios_sum_to_one(self):
        """Ensure train + val + test = 1.0"""
        total = self.train_ratio + self.val_ratio + self.test_ratio
        if abs(total - 1.0) > 0.001:
            raise ValueError(f"Split ratios must sum to 1.0, got {total}")
        return self


class DataConfig(BaseModel):
    """
    Root configuration for data pipeline.

    Args:
        source: Dataset source configuration
        paths: Directory paths configuration
        preprocessing: Preprocessing thresholds configuration
        splitting: Train/val/test split configuration
    """

    source: SourceConfig
    paths: PathsConfig
    preprocessing: PreprocessingConfig
    splitting: SplittingConfig

    # === Convenience methods for absolute paths ===

    def get_raw_path(self) -> Path:
        """Get absolute path to raw data directory."""
        return get_project_root() / self.paths.raw

    def get_processed_path(self) -> Path:
        """Get absolute path to processed data directory."""
        return get_project_root() / self.paths.processed

    def get_features_path(self) -> Path:
        """Get absolute path to features directory."""
        return get_project_root() / self.paths.features


# =============================================================================
# Config Loading Functions
# =============================================================================


def _load_yaml(filename: str) -> dict[str, Any]:
    """Load a YAML file from the config directory."""
    config_path = get_config_dir() / filename
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with open(config_path) as f:
        result = yaml.safe_load(f)

    if not isinstance(result, dict):
        raise ValueError(
            f"Config file {filename} must contain a YAML dictionary, "
            f"got {type(result).__name__}"
        )

    return result


@lru_cache(maxsize=1)
def get_data_config() -> DataConfig:
    """Load and validate data configuration."""
    raw_config = _load_yaml("data.yaml")
    return DataConfig(**raw_config)


def clear_config_cache() -> None:
    """Clear cached configurations.

    Call this if config files have been modified and need to be reloaded.
    Useful for testing.
    """
    get_data_config.cache_clear()
    # Add other config cache clears here as we implement them


# =============================================================================
# CLI: Validate configs when run directly
# =============================================================================

if __name__ == "__main__":
    """Validate config files when run directly: python -m src.config"""
    print(f"Project root: {get_project_root()}")
    print(f"Config directory: {get_config_dir()}")
    print("-" * 50)

    # List of configs to validate (add more as we implement them)
    configs_to_check = [
        ("data.yaml", get_data_config),
    ]

    all_valid = True
    for name, loader in configs_to_check:
        try:
            config = loader()
            print(f"✓ {name}: Valid")
            # Print a sample value to confirm it loaded correctly
            print(f"  └─ Dataset: {config.source.name}")
        except FileNotFoundError:
            print(f"✗ {name}: File not found")
            all_valid = False
        except Exception as e:
            print(f"✗ {name}: {e}")
            all_valid = False

    print("-" * 50)
    if all_valid:
        print("All configurations valid!")
    else:
        print("Some configurations have errors.")
        exit(1)
