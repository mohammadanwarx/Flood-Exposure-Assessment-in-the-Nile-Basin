"""
Configuration settings for the flood exposure pipeline.
"""

import os
from pathlib import Path

# Project root directory
PROJECT_ROOT = Path(__file__).parent.parent.parent

# Data directories
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
EXTERNAL_DATA_DIR = DATA_DIR / "external"

# Raster and vector data paths
RAW_RASTER_DIR = RAW_DATA_DIR / "raster"
RAW_VECTOR_DIR = RAW_DATA_DIR / "vector"

# Output directories
REPORT_DIR = PROJECT_ROOT / "report"
FIGURES_DIR = REPORT_DIR / "figures"

# Coordinate Reference Systems
DEFAULT_CRS = "EPSG:4326"  # WGS84
METRIC_CRS = "EPSG:3857"   # Web Mercator (for metric calculations)

# Processing parameters
RASTER_NODATA_VALUE = -9999
BUFFER_DISTANCE = 1000  # meters
ZONAL_STATS_FUNCTIONS = ["min", "max", "mean", "median", "std", "sum", "count"]

# Visualization settings
DEFAULT_FIGSIZE = (12, 8)
DPI = 300
CMAP_FLOOD = "Blues"
CMAP_EXPOSURE = "YlOrRd"

# API Keys (use environment variables in production)
OPENSTREETMAP_USER_AGENT = os.getenv("OSM_USER_AGENT", "flood-exposure-pipeline")

# Tensor computation settings
USE_GPU = True
TORCH_DEVICE = "cuda" if USE_GPU else "cpu"
TF_DEVICE = "/GPU:0" if USE_GPU else "/CPU:0"

# Data download URLs (examples)
SAMPLE_DATA_URLS = {
    "flood_depth": "https://example.com/flood_depth.tif",
    "admin_boundaries": "https://example.com/boundaries.shp",
}

# Logging configuration
LOG_LEVEL = "INFO"
LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"

def ensure_directories():
    """Create necessary directories if they don't exist."""
    dirs = [
        RAW_RASTER_DIR,
        RAW_VECTOR_DIR,
        PROCESSED_DATA_DIR,
        EXTERNAL_DATA_DIR,
        FIGURES_DIR,
    ]
    for directory in dirs:
        directory.mkdir(parents=True, exist_ok=True)

if __name__ == "__main__":
    ensure_directories()
    print("All directories created successfully!")
