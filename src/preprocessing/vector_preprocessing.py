"""
Vector preprocessing operations including reprojection, buffering, and simplification.
"""

import geopandas as gpd
from typing import Union, Optional
from pathlib import Path


def reproject_vector(
    gdf: gpd.GeoDataFrame,
    target_crs: str = 'EPSG:4326'
) -> gpd.GeoDataFrame:
    """
    Reproject a GeoDataFrame to a target CRS.
    
    Parameters
    ----------
    gdf : gpd.GeoDataFrame
        Input GeoDataFrame
    target_crs : str
        Target CRS
        
    Returns
    -------
    gpd.GeoDataFrame
        Reprojected GeoDataFrame
    """
    return gdf.to_crs(target_crs)


def buffer_geometries(
    gdf: gpd.GeoDataFrame,
    distance: float,
    resolution: int = 16
) -> gpd.GeoDataFrame:
    """
    Create buffers around geometries.
    
    Parameters
    ----------
    gdf : gpd.GeoDataFrame
        Input GeoDataFrame
    distance : float
        Buffer distance (in units of the CRS)
    resolution : int
        Number of segments per quadrant
        
    Returns
    -------
    gpd.GeoDataFrame
        GeoDataFrame with buffered geometries
    """
    buffered = gdf.copy()
    buffered['geometry'] = gdf.geometry.buffer(distance, resolution=resolution)
    return buffered


def simplify_geometries(
    gdf: gpd.GeoDataFrame,
    tolerance: float,
    preserve_topology: bool = True
) -> gpd.GeoDataFrame:
    """
    Simplify geometries using the Douglas-Peucker algorithm.
    
    Parameters
    ----------
    gdf : gpd.GeoDataFrame
        Input GeoDataFrame
    tolerance : float
        Simplification tolerance
    preserve_topology : bool
        Whether to preserve topology
        
    Returns
    -------
    gpd.GeoDataFrame
        GeoDataFrame with simplified geometries
    """
    simplified = gdf.copy()
    simplified['geometry'] = gdf.geometry.simplify(
        tolerance, preserve_topology=preserve_topology
    )
    return simplified


def clip_vector_by_bounds(
    gdf: gpd.GeoDataFrame,
    bounds: tuple
) -> gpd.GeoDataFrame:
    """
    Clip a GeoDataFrame to bounding box coordinates.
    
    Parameters
    ----------
    gdf : gpd.GeoDataFrame
        Input GeoDataFrame
    bounds : tuple
        Bounding box (minx, miny, maxx, maxy)
        
    Returns
    -------
    gpd.GeoDataFrame
        Clipped GeoDataFrame
    """
    return gdf.cx[bounds[0]:bounds[2], bounds[1]:bounds[3]]


def dissolve_by_attribute(
    gdf: gpd.GeoDataFrame,
    attribute: str,
    aggfunc: str = 'first'
) -> gpd.GeoDataFrame:
    """
    Dissolve geometries based on an attribute.
    
    Parameters
    ----------
    gdf : gpd.GeoDataFrame
        Input GeoDataFrame
    attribute : str
        Attribute to dissolve by
    aggfunc : str or dict
        Aggregation function(s) for other attributes
        
    Returns
    -------
    gpd.GeoDataFrame
        Dissolved GeoDataFrame
    """
    return gdf.dissolve(by=attribute, aggfunc=aggfunc)


def remove_invalid_geometries(
    gdf: gpd.GeoDataFrame,
    fix_invalid: bool = True
) -> gpd.GeoDataFrame:
    """
    Remove or fix invalid geometries.
    
    Parameters
    ----------
    gdf : gpd.GeoDataFrame
        Input GeoDataFrame
    fix_invalid : bool
        If True, attempt to fix invalid geometries; if False, remove them
        
    Returns
    -------
    gpd.GeoDataFrame
        GeoDataFrame with valid geometries
    """
    if fix_invalid:
        clean_gdf = gdf.copy()
        invalid_mask = ~clean_gdf.geometry.is_valid
        clean_gdf.loc[invalid_mask, 'geometry'] = clean_gdf.loc[invalid_mask, 'geometry'].buffer(0)
        return clean_gdf
    else:
        return gdf[gdf.geometry.is_valid].copy()


def calculate_area(
    gdf: gpd.GeoDataFrame,
    column_name: str = 'area_m2',
    unit: str = 'm2'
) -> gpd.GeoDataFrame:
    """
    Calculate area of geometries.
    
    Parameters
    ----------
    gdf : gpd.GeoDataFrame
        Input GeoDataFrame
    column_name : str
        Name for the new area column
    unit : str
        Unit for area calculation: 'm2' or 'km2'
        
    Returns
    -------
    gpd.GeoDataFrame
        GeoDataFrame with area column added
    """
    result = gdf.copy()
    
    # Project to metric CRS if needed
    if not result.crs.is_projected:
        original_crs = result.crs
        result = result.to_crs('EPSG:3857')  # Web Mercator
        area = result.geometry.area
        result = result.to_crs(original_crs)
    else:
        area = result.geometry.area
    
    if unit == 'km2':
        area = area / 1_000_000
    
    result[column_name] = area
    return result


def spatial_join(
    left_gdf: gpd.GeoDataFrame,
    right_gdf: gpd.GeoDataFrame,
    how: str = 'inner',
    predicate: str = 'intersects'
) -> gpd.GeoDataFrame:
    """
    Perform a spatial join between two GeoDataFrames.
    
    Parameters
    ----------
    left_gdf : gpd.GeoDataFrame
        Left GeoDataFrame
    right_gdf : gpd.GeoDataFrame
        Right GeoDataFrame
    how : str
        Join type: 'left', 'right', 'inner'
    predicate : str
        Spatial predicate: 'intersects', 'contains', 'within', etc.
        
    Returns
    -------
    gpd.GeoDataFrame
        Joined GeoDataFrame
    """
    # Ensure same CRS
    if left_gdf.crs != right_gdf.crs:
        right_gdf = right_gdf.to_crs(left_gdf.crs)
    
    return gpd.sjoin(left_gdf, right_gdf, how=how, predicate=predicate)
