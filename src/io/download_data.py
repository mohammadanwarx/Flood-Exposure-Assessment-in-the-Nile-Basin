"""
Functions for downloading external geospatial data.
"""

import requests
from pathlib import Path
from typing import Union, Optional
from tqdm import tqdm
import warnings


def download_file(
    url: str,
    output_path: Union[str, Path],
    chunk_size: int = 8192,
    show_progress: bool = True
) -> Path:
    """
    Download a file from a URL with progress bar.
    
    Parameters
    ----------
    url : str
        URL to download from
    output_path : str or Path
        Local path to save the file
    chunk_size : int
        Size of chunks to download (default: 8192 bytes)
    show_progress : bool
        Whether to show progress bar
        
    Returns
    -------
    Path
        Path to the downloaded file
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    response = requests.get(url, stream=True)
    response.raise_for_status()
    
    total_size = int(response.headers.get('content-length', 0))
    
    with open(output_path, 'wb') as f:
        if show_progress:
            with tqdm(total=total_size, unit='B', unit_scale=True, desc=output_path.name) as pbar:
                for chunk in response.iter_content(chunk_size=chunk_size):
                    if chunk:
                        f.write(chunk)
                        pbar.update(len(chunk))
        else:
            for chunk in response.iter_content(chunk_size=chunk_size):
                if chunk:
                    f.write(chunk)
    
    return output_path


def download_sample_data(
    dataset_name: str,
    output_dir: Optional[Union[str, Path]] = None
) -> Path:
    """
    Download sample datasets for testing and examples.
    
    Parameters
    ----------
    dataset_name : str
        Name of the sample dataset to download
    output_dir : str or Path, optional
        Directory to save the data (default: data/external/)
        
    Returns
    -------
    Path
        Path to the downloaded file
    """
    from src.config.settings import EXTERNAL_DATA_DIR, SAMPLE_DATA_URLS
    
    if output_dir is None:
        output_dir = EXTERNAL_DATA_DIR
    
    if dataset_name not in SAMPLE_DATA_URLS:
        available = ', '.join(SAMPLE_DATA_URLS.keys())
        raise ValueError(
            f"Unknown dataset: {dataset_name}. "
            f"Available datasets: {available}"
        )
    
    url = SAMPLE_DATA_URLS[dataset_name]
    output_path = Path(output_dir) / Path(url).name
    
    print(f"Downloading {dataset_name}...")
    return download_file(url, output_path)


def check_url_accessible(url: str) -> bool:
    """
    Check if a URL is accessible.
    
    Parameters
    ----------
    url : str
        URL to check
        
    Returns
    -------
    bool
        True if accessible, False otherwise
    """
    try:
        response = requests.head(url, timeout=5)
        return response.status_code == 200
    except requests.RequestException:
        return False


def download_openstreetmap_data(
    bbox: tuple,
    tags: dict,
    output_path: Union[str, Path]
) -> Path:
    """
    Download OpenStreetMap data for a given bounding box and tags.
    
    Parameters
    ----------
    bbox : tuple
        Bounding box (min_lon, min_lat, max_lon, max_lat)
    tags : dict
        OSM tags to filter (e.g., {'building': True})
    output_path : str or Path
        Path to save the data
        
    Returns
    -------
    Path
        Path to the downloaded file
        
    Notes
    -----
    Requires osmnx package. Install with: pip install osmnx
    """
    try:
        import osmnx as ox
    except ImportError:
        raise ImportError(
            "osmnx is required for downloading OSM data. "
            "Install it with: pip install osmnx"
        )
    
    gdf = ox.geometries_from_bbox(
        bbox[3], bbox[1], bbox[2], bbox[0],
        tags=tags
    )
    
    output_path = Path(output_path)
    gdf.to_file(output_path)
    
    return output_path
