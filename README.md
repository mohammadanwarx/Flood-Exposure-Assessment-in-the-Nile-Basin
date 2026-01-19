# Flood Exposure Geospatial Pipeline

A comprehensive geospatial analysis pipeline for assessing flood exposure using raster and vector data, with support for modern tensor operations and data cube analysis.

## Features

- **Multi-format Data I/O**: Load and process raster (GeoTIFF, NetCDF) and vector (Shapefile, GeoJSON) data
- **Preprocessing Pipeline**: Automated masking, reprojection, and data preparation
- **Zonal Statistics**: Calculate exposure metrics within administrative boundaries
- **Tensor Operations**: Leverage NumPy, PyTorch, and TensorFlow for efficient computation
- **Data Cubes**: Multi-dimensional analysis using xarray
- **Visualization**: Generate maps and plots for results

## Project Structure

```
flood-exposure-geospatial-pipeline/
├── data/               # Data storage (not version controlled)
├── src/                # Source code
├── tests/              # Unit tests
├── notebooks/          # Jupyter notebooks for exploration
├── scripts/            # Executable scripts
├── report/             # Output figures and reports
└── docs/               # Documentation
```

## Installation

### Using Conda (Recommended)

```bash
conda env create -f environment.yml
conda activate flood-exposure
```

### Using pip

```bash
pip install -r requirements.txt
```

### Development Installation

```bash
pip install -e .
```

## Quick Start

```python
from src.io import load_raster, load_vector
from src.analysis import zonal_statistics

# Load flood depth raster
flood_raster = load_raster.read_geotiff("data/raw/raster/flood_depth.tif")

# Load administrative boundaries
admin_boundaries = load_vector.read_shapefile("data/raw/vector/boundaries.shp")

# Calculate exposure metrics
stats = zonal_statistics.compute(flood_raster, admin_boundaries)
```

## Usage

See the `notebooks/` directory for detailed examples and tutorials.

## Testing

```bash
pytest tests/
```

## Contributing

Contributions are welcome! Please ensure tests pass before submitting pull requests.

## License

MIT License

## Contact

[Your contact information]
