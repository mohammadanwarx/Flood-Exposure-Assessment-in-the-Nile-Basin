"""
Flood Exposure Geospatial Pipeline
===================================

A comprehensive pipeline for flood exposure analysis using geospatial data.
"""

__version__ = "0.1.0"
__author__ = "Your Name"

# Import main modules for easier access
from src import io
from src import preprocessing
from src import analysis
from src import visualization
from src import utils

__all__ = [
    "io",
    "preprocessing",
    "analysis",
    "visualization",
    "utils",
]
