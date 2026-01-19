"""
Raster preprocessing operations including reprojection, resampling, and clipping.
"""

import numpy as np
import rasterio
from rasterio.warp import calculate_default_transform, reproject, Resampling
from rasterio.mask import mask
from typing import Union, Tuple, Optional
from pathlib import Path
import geopandas as gpd


def reproject_raster(
    src_path: Union[str, Path],
    dst_path: Union[str, Path],
    dst_crs: str = 'EPSG:4326',
    resampling_method: Resampling = Resampling.bilinear
) -> None:
    """
    Reproject a raster to a different CRS.
    
    Parameters
    ----------
    src_path : str or Path
        Path to source raster
    dst_path : str or Path
        Path to output reprojected raster
    dst_crs : str
        Target CRS (default: EPSG:4326)
    resampling_method : Resampling
        Resampling method (default: bilinear)
    """
    with rasterio.open(src_path) as src:
        transform, width, height = calculate_default_transform(
            src.crs, dst_crs, src.width, src.height, *src.bounds
        )
        
        kwargs = src.meta.copy()
        kwargs.update({
            'crs': dst_crs,
            'transform': transform,
            'width': width,
            'height': height
        })
        
        with rasterio.open(dst_path, 'w', **kwargs) as dst:
            for i in range(1, src.count + 1):
                reproject(
                    source=rasterio.band(src, i),
                    destination=rasterio.band(dst, i),
                    src_transform=src.transform,
                    src_crs=src.crs,
                    dst_transform=transform,
                    dst_crs=dst_crs,
                    resampling=resampling_method
                )


def clip_raster_by_geometry(
    raster_path: Union[str, Path],
    geometry: gpd.GeoDataFrame,
    output_path: Union[str, Path],
    crop: bool = True
) -> None:
    """
    Clip a raster by a vector geometry.
    
    Parameters
    ----------
    raster_path : str or Path
        Path to input raster
    geometry : gpd.GeoDataFrame
        GeoDataFrame with geometry to clip by
    output_path : str or Path
        Path to output clipped raster
    crop : bool
        Whether to crop the raster to the geometry bounds
    """
    with rasterio.open(raster_path) as src:
        # Ensure geometry is in the same CRS as raster
        if geometry.crs != src.crs:
            geometry = geometry.to_crs(src.crs)
        
        # Clip the raster
        out_image, out_transform = mask(
            src, geometry.geometry, crop=crop, nodata=src.nodata
        )
        
        # Update metadata
        out_meta = src.meta.copy()
        out_meta.update({
            "driver": "GTiff",
            "height": out_image.shape[1],
            "width": out_image.shape[2],
            "transform": out_transform
        })
        
        # Write the clipped raster
        with rasterio.open(output_path, "w", **out_meta) as dest:
            dest.write(out_image)


def resample_raster(
    src_path: Union[str, Path],
    dst_path: Union[str, Path],
    scale_factor: float = 0.5,
    resampling_method: Resampling = Resampling.bilinear
) -> None:
    """
    Resample a raster by a scale factor.
    
    Parameters
    ----------
    src_path : str or Path
        Path to source raster
    dst_path : str or Path
        Path to output resampled raster
    scale_factor : float
        Scale factor (0.5 = half resolution, 2.0 = double resolution)
    resampling_method : Resampling
        Resampling method
    """
    with rasterio.open(src_path) as src:
        # Calculate new dimensions
        new_width = int(src.width * scale_factor)
        new_height = int(src.height * scale_factor)
        
        # Calculate new transform
        transform = src.transform * src.transform.scale(
            (src.width / new_width),
            (src.height / new_height)
        )
        
        # Read and resample data
        data = src.read(
            out_shape=(src.count, new_height, new_width),
            resampling=resampling_method
        )
        
        # Update metadata
        kwargs = src.meta.copy()
        kwargs.update({
            'transform': transform,
            'width': new_width,
            'height': new_height
        })
        
        # Write resampled raster
        with rasterio.open(dst_path, 'w', **kwargs) as dst:
            dst.write(data)


def fill_nodata(
    data: np.ndarray,
    nodata_value: Optional[float] = None,
    method: str = 'nearest'
) -> np.ndarray:
    """
    Fill nodata values in a raster array.
    
    Parameters
    ----------
    data : np.ndarray
        Input raster array
    nodata_value : float, optional
        NoData value to fill (if None, fills NaN values)
    method : str
        Fill method: 'nearest', 'mean', 'median', or a numeric value
        
    Returns
    -------
    np.ndarray
        Array with filled nodata values
    """
    filled_data = data.copy()
    
    if nodata_value is not None:
        mask = (data == nodata_value)
    else:
        mask = np.isnan(data)
    
    if method == 'nearest':
        from scipy.ndimage import distance_transform_edt
        indices = distance_transform_edt(mask, return_distances=False, return_indices=True)
        filled_data = data[tuple(indices)]
    elif method == 'mean':
        fill_value = np.nanmean(data[~mask])
        filled_data[mask] = fill_value
    elif method == 'median':
        fill_value = np.nanmedian(data[~mask])
        filled_data[mask] = fill_value
    else:
        # Assume numeric value
        filled_data[mask] = float(method)
    
    return filled_data


def normalize_raster(
    data: np.ndarray,
    method: str = 'minmax',
    feature_range: Tuple[float, float] = (0, 1)
) -> np.ndarray:
    """
    Normalize raster values.
    
    Parameters
    ----------
    data : np.ndarray
        Input raster array
    method : str
        Normalization method: 'minmax' or 'zscore'
    feature_range : tuple
        Range for minmax normalization
        
    Returns
    -------
    np.ndarray
        Normalized array
    """
    if method == 'minmax':
        data_min = np.nanmin(data)
        data_max = np.nanmax(data)
        normalized = (data - data_min) / (data_max - data_min)
        normalized = normalized * (feature_range[1] - feature_range[0]) + feature_range[0]
    elif method == 'zscore':
        mean = np.nanmean(data)
        std = np.nanstd(data)
        normalized = (data - mean) / std
    else:
        raise ValueError(f"Unknown normalization method: {method}")
    
    return normalized
