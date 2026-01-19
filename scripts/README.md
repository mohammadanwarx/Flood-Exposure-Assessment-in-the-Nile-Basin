# Flood Exposure Geospatial Pipeline - Scripts

This directory contains executable scripts for common pipeline tasks.

## Available Scripts

### setup_environment.py
Creates necessary directories and validates environment setup.

```bash
python scripts/setup_environment.py
```

### run_analysis.py
Runs complete flood exposure analysis pipeline.

```bash
python scripts/run_analysis.py --config config.yaml
```

### download_sample_data.py
Downloads sample datasets for testing.

```bash
python scripts/download_sample_data.py
```

## Usage

All scripts should be run from the project root directory:

```bash
cd flood-exposure-geospatial-pipeline
python scripts/script_name.py
```
