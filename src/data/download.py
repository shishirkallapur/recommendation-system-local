"""
Download and extract the MovieLens dataset.

"""

import logging
import shutil
import tempfile
import urllib.request
import zipfile
from pathlib import Path

from src.config import get_data_config

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def get_expected_data_path() -> Path:
    """Get the path where extracted data should exist."""
    config = get_data_config()
    raw_path = config.get_raw_path()
    return raw_path / "ml-100k"


def is_data_downloaded() -> bool:
    """Check if the MovieLens data has already been downloaded and extracted.

    Verifies that the key files we need exist:
    - u.data (ratings)
    - u.item (movie metadata)

    """
    data_path = get_expected_data_path()

    required_files = ["u.data", "u.item"]

    for filename in required_files:
        file_path = data_path / filename
        if not file_path.exists():
            logger.debug(f"Missing required file: {file_path}")
            return False
        # Check file is not empty
        if file_path.stat().st_size == 0:
            logger.warning(f"File exists but is empty: {file_path}")
            return False

    return True


def download_file(url: str, destination: Path) -> None:
    """Download a file from URL to destination.

    Downloads to a temporary file first, then moves to destination.
    This prevents corrupted partial downloads.

    Args:
        url: URL to download from
        destination: Path to save the downloaded file

    Raises:
        urllib.error.URLError: If download fails
        OSError: If file operations fail
    """
    logger.info(f"Downloading from {url}")

    # Create parent directory if needed
    destination.parent.mkdir(parents=True, exist_ok=True)

    # Download to temp file first
    with tempfile.NamedTemporaryFile(delete=False, suffix=".zip") as tmp_file:
        tmp_path = Path(tmp_file.name)

    try:
        # Download with progress indication
        def report_progress(block_num: int, block_size: int, total_size: int) -> None:
            if total_size > 0:
                percent = min(100, (block_num * block_size * 100) // total_size)
                if block_num % 50 == 0:  # Log every 50 blocks
                    logger.info(f"Download progress: {percent}%")

        urllib.request.urlretrieve(url, tmp_path, reporthook=report_progress)

        # Verify download succeeded (file has content)
        if tmp_path.stat().st_size == 0:
            raise ValueError("Downloaded file is empty")

        # Move to final destination
        shutil.move(str(tmp_path), str(destination))
        logger.info(f"Downloaded to {destination}")

    except Exception:
        # Clean up temp file on failure
        if tmp_path.exists():
            tmp_path.unlink()
        raise


def extract_zip(zip_path: Path, extract_to: Path) -> None:
    """Extract a zip file to the specified directory."""
    logger.info(f"Extracting {zip_path} to {extract_to}")

    extract_to.mkdir(parents=True, exist_ok=True)

    with zipfile.ZipFile(zip_path, "r") as zip_ref:
        zip_ref.extractall(extract_to)

    logger.info("Extraction complete")


def download_movielens(force: bool = False) -> Path:
    """Download and extract the MovieLens dataset.

    This is the main entry point for data download. It:
    1. Checks if data already exists (skips if so, unless force=True)
    2. Downloads the zip file from the configured URL
    3. Extracts it to the configured raw data directory
    4. Cleans up the zip file after extraction
    """
    config = get_data_config()

    # Check if already downloaded
    if not force and is_data_downloaded():
        data_path = get_expected_data_path()
        logger.info(f"Data already exists at {data_path}, skipping download")
        return data_path

    # Get paths from config
    raw_path = config.get_raw_path()
    url = config.source.url
    zip_filename = url.split("/")[-1]  # e.g., "ml-100k.zip"
    zip_path = raw_path / zip_filename

    logger.info(f"Starting MovieLens download: {config.source.name}")

    # Download
    download_file(url, zip_path)

    # Extract
    extract_zip(zip_path, raw_path)

    # Clean up zip file
    logger.info(f"Removing zip file: {zip_path}")
    zip_path.unlink()

    # Verify extraction
    data_path = get_expected_data_path()
    if not is_data_downloaded():
        raise RuntimeError(
            f"Download completed but expected files not found at {data_path}"
        )

    logger.info(f"MovieLens data ready at {data_path}")
    return data_path


def get_ratings_path() -> Path:
    """Get the path to the ratings file (u.data)."""
    path = get_expected_data_path() / "u.data"
    if not path.exists():
        raise FileNotFoundError(
            f"Ratings file not found at {path}. Run download_movielens() first."
        )
    return path


def get_movies_path() -> Path:
    """Get the path to the movies file (u.item)."""
    path = get_expected_data_path() / "u.item"
    if not path.exists():
        raise FileNotFoundError(
            f"Movies file not found at {path}. Run download_movielens() first."
        )
    return path


# =============================================================================
# CLI Entry Point
# =============================================================================

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Download and extract the MovieLens dataset"
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force re-download even if data exists",
    )
    args = parser.parse_args()

    try:
        data_path = download_movielens(force=args.force)
        print(f"\n✓ MovieLens data available at: {data_path}")

        # Show what files are available
        print("\nFiles in dataset:")
        for f in sorted(data_path.iterdir()):
            size_kb = f.stat().st_size / 1024
            print(f"  {f.name}: {size_kb:.1f} KB")

    except Exception as e:
        logger.error(f"Download failed: {e}")
        exit(1)
