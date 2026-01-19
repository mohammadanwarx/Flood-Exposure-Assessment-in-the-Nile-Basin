#!/usr/bin/env python
"""
Setup script to create necessary directories and validate environment.
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from src.config.settings import ensure_directories


def main():
    """Main setup function."""
    print("=" * 60)
    print("Flood Exposure Geospatial Pipeline - Environment Setup")
    print("=" * 60)
    
    print("\n1. Creating directory structure...")
    ensure_directories()
    print("   ✓ Directories created successfully")
    
    print("\n2. Checking Python version...")
    version = sys.version_info
    print(f"   Python {version.major}.{version.minor}.{version.micro}")
    if version.major < 3 or (version.major == 3 and version.minor < 9):
        print("   ⚠ Warning: Python 3.9+ recommended")
    else:
        print("   ✓ Python version OK")
    
    print("\n3. Checking required packages...")
    required_packages = [
        'numpy', 'pandas', 'geopandas', 'rasterio', 
        'xarray', 'matplotlib', 'shapely'
    ]
    
    missing_packages = []
    for package in required_packages:
        try:
            __import__(package)
            print(f"   ✓ {package}")
        except ImportError:
            print(f"   ✗ {package} (missing)")
            missing_packages.append(package)
    
    if missing_packages:
        print(f"\n⚠ Missing packages: {', '.join(missing_packages)}")
        print("\nInstall with:")
        print("   conda env create -f environment.yml")
        print("   or")
        print("   pip install -r requirements.txt")
    else:
        print("\n✓ All required packages installed")
    
    print("\n" + "=" * 60)
    print("Setup complete! Ready to start analyzing flood exposure.")
    print("=" * 60)


if __name__ == "__main__":
    main()
