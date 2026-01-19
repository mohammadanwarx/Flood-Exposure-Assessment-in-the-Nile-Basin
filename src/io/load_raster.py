"""
Functions for loading and reading raster data.
"""

import rasterio
import xarray as xr
import rioxarray
from pathlib import Path
from typing import Union, Tuple
import numpy as np


def read_geotiff(filepath: Union[str, Path]) -> Tuple[np.ndarray, dict]:
    """
    Read a GeoTIFF file and return the data array and metadata.
    
    Parameters
    ----------
    filepath : str or Path
        Path to the GeoTIFF file
        
    Returns
    -------
    data : np.ndarray
        Raster data array
    metadata : dict
        Raster metadata including transform, CRS, bounds, etc.
    """
    with rasterio.open(filepath) as src:
        data = src.read()
        metadata = {
            'transform': src.transform,
            'crs': src.crs,
            'bounds': src.bounds,
            'width': src.width,
            'height': src.height,
            'count': src.count,
            'dtype': src.dtypes[0],
            'nodata': src.nodata,
        }
    
    return data, metadata


def read_raster_as_xarray(filepath: Union[str, Path]) -> xr.DataArray:
    """
    Read a raster file into an xarray DataArray using rioxarray.
    
    Parameters
    ----------
    filepath : str or Path
        Path to the raster file
        
    Returns
    -------
    xr.DataArray
        Raster data as xarray DataArray with spatial metadata
    """
    da = rioxarray.open_rasterio(filepath, masked=True)
    return da


def read_netcdf(filepath: Union[str, Path]) -> xr.Dataset:
    """
    Read a NetCDF file into an xarray Dataset.
    
    Parameters
    ----------
    filepath : str or Path
        Path to the NetCDF file
        
    Returns
    -------
    xr.Dataset
        Multi-dimensional dataset
    """
    ds = xr.open_dataset(filepath)
    return ds


def get_raster_info(filepath: Union[str, Path]) -> dict:
    """
    Get basic information about a raster file without loading all data.
    
    Parameters
    ----------
    filepath : str or Path
        Path to the raster file
        
    Returns
    -------
    dict
        Dictionary with raster information
    """
    with rasterio.open(filepath) as src:
        info = {
            'driver': src.driver,
            'width': src.width,
            'height': src.height,
            'count': src.count,
            'crs': str(src.crs),
            'bounds': src.bounds,
            'transform': src.transform,
            'nodata': src.nodata,
            'dtypes': src.dtypes,
        }
    
    return info


def save_geotiff(
    data: np.ndarray,
    filepath: Union[str, Path],
    metadata: dict,
    compress: str = 'lzw'
) -> None:
    """
    Save a numpy array as a GeoTIFF file.
    
    Parameters
    ----------
    data : np.ndarray
        Raster data to save
    filepath : str or Path
        Output file path
    metadata : dict
        Metadata dictionary with transform, CRS, etc.
    compress : str
        Compression method (default: 'lzw')
    """
    # Ensure data is 3D (bands, height, width)
    if data.ndim == 2:
        data = data[np.newaxis, :, :]
    
    with rasterio.open(
        filepath,
        'w',
        driver='GTiff',
        height=data.shape[1],
        width=data.shape[2],
        count=data.shape[0],
        dtype=data.dtype,
        crs=metadata.get('crs'),
        transform=metadata.get('transform'),
        nodata=metadata.get('nodata'),
        compress=compress
    ) as dst:
        dst.write(data)
