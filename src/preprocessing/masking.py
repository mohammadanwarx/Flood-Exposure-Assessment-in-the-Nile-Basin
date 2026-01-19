"""
Masking operations for raster data using vector geometries.
"""

import numpy as np
import rasterio
from rasterio.mask import mask as rio_mask
from rasterio.features import geometry_mask
import geopandas as gpd
from typing import Union, Tuple, Optional
from pathlib import Path


def mask_raster_with_vector(
    raster_path: Union[str, Path],
    vector_gdf: gpd.GeoDataFrame,
    output_path: Optional[Union[str, Path]] = None,
    invert: bool = False,
    crop: bool = True
) -> Tuple[np.ndarray, dict]:
    """
    Mask a raster using vector geometries.
    
    Parameters
    ----------
    raster_path : str or Path
        Path to input raster
    vector_gdf : gpd.GeoDataFrame
        GeoDataFrame with masking geometries
    output_path : str or Path, optional
        Path to save masked raster (if None, returns array only)
    invert : bool
        If True, mask outside the geometries instead of inside
    crop : bool
        If True, crop the raster to the geometry bounds
        
    Returns
    -------
    masked_array : np.ndarray
        Masked raster array
    metadata : dict
        Raster metadata
    """
    with rasterio.open(raster_path) as src:
        # Reproject vector to match raster CRS if needed
        if vector_gdf.crs != src.crs:
            vector_gdf = vector_gdf.to_crs(src.crs)
        
        # Mask the raster
        out_image, out_transform = rio_mask(
            src,
            vector_gdf.geometry,
            crop=crop,
            invert=invert,
            nodata=src.nodata
        )
        
        # Update metadata
        out_meta = src.meta.copy()
        out_meta.update({
            "driver": "GTiff",
            "height": out_image.shape[1],
            "width": out_image.shape[2],
            "transform": out_transform
        })
        
        # Optionally save to file
        if output_path is not None:
            with rasterio.open(output_path, "w", **out_meta) as dest:
                dest.write(out_image)
        
        return out_image, out_meta


def create_mask_from_vector(
    vector_gdf: gpd.GeoDataFrame,
    raster_shape: Tuple[int, int],
    raster_transform: rasterio.Affine,
    invert: bool = False
) -> np.ndarray:
    """
    Create a boolean mask array from vector geometries.
    
    Parameters
    ----------
    vector_gdf : gpd.GeoDataFrame
        GeoDataFrame with geometries
    raster_shape : tuple
        Shape of the output mask (height, width)
    raster_transform : rasterio.Affine
        Affine transform for the raster
    invert : bool
        If True, invert the mask
        
    Returns
    -------
    np.ndarray
        Boolean mask array
    """
    mask_array = geometry_mask(
        vector_gdf.geometry,
        out_shape=raster_shape,
        transform=raster_transform,
        invert=invert
    )
    
    return mask_array


def apply_threshold_mask(
    data: np.ndarray,
    threshold: float,
    comparison: str = 'greater'
) -> np.ndarray:
    """
    Create a mask based on a threshold value.
    
    Parameters
    ----------
    data : np.ndarray
        Input raster array
    threshold : float
        Threshold value
    comparison : str
        Comparison operator: 'greater', 'less', 'equal', 'greater_equal', 'less_equal'
        
    Returns
    -------
    np.ndarray
        Boolean mask array
    """
    if comparison == 'greater':
        mask = data > threshold
    elif comparison == 'less':
        mask = data < threshold
    elif comparison == 'equal':
        mask = data == threshold
    elif comparison == 'greater_equal':
        mask = data >= threshold
    elif comparison == 'less_equal':
        mask = data <= threshold
    else:
        raise ValueError(f"Unknown comparison operator: {comparison}")
    
    return mask


def apply_multi_threshold_mask(
    data: np.ndarray,
    thresholds: list,
    logic: str = 'and'
) -> np.ndarray:
    """
    Apply multiple threshold conditions to create a mask.
    
    Parameters
    ----------
    data : np.ndarray
        Input raster array
    thresholds : list of tuples
        List of (threshold_value, comparison) tuples
    logic : str
        Logic operator: 'and' or 'or'
        
    Returns
    -------
    np.ndarray
        Boolean mask array
    """
    masks = [apply_threshold_mask(data, thresh, comp) for thresh, comp in thresholds]
    
    if logic == 'and':
        combined_mask = np.all(masks, axis=0)
    elif logic == 'or':
        combined_mask = np.any(masks, axis=0)
    else:
        raise ValueError(f"Unknown logic operator: {logic}")
    
    return combined_mask


def mask_nodata_values(
    data: np.ndarray,
    nodata_value: Optional[float] = None
) -> np.ma.MaskedArray:
    """
    Create a masked array from nodata values.
    
    Parameters
    ----------
    data : np.ndarray
        Input raster array
    nodata_value : float, optional
        NoData value (if None, masks NaN values)
        
    Returns
    -------
    np.ma.MaskedArray
        Masked array
    """
    if nodata_value is not None:
        mask = (data == nodata_value)
    else:
        mask = np.isnan(data)
    
    return np.ma.masked_array(data, mask=mask)


def combine_masks(
    masks: list,
    logic: str = 'and'
) -> np.ndarray:
    """
    Combine multiple boolean masks.
    
    Parameters
    ----------
    masks : list of np.ndarray
        List of boolean mask arrays
    logic : str
        Logic operator: 'and', 'or', 'xor'
        
    Returns
    -------
    np.ndarray
        Combined boolean mask
    """
    if logic == 'and':
        combined = np.all(masks, axis=0)
    elif logic == 'or':
        combined = np.any(masks, axis=0)
    elif logic == 'xor':
        combined = np.logical_xor.reduce(masks)
    else:
        raise ValueError(f"Unknown logic operator: {logic}")
    
    return combined
