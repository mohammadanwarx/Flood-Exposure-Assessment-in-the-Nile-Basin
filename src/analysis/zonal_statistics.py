"""
Zonal statistics calculations for raster-vector analysis.
"""

import numpy as np
import pandas as pd
import geopandas as gpd
from rasterstats import zonal_stats as rs_zonal_stats
from typing import Union, List, Optional
from pathlib import Path
import rasterio
from rasterio.features import rasterize


def compute_zonal_statistics(
    raster_path: Union[str, Path],
    zones_gdf: gpd.GeoDataFrame,
    stats: List[str] = ['min', 'max', 'mean', 'median', 'std', 'sum', 'count'],
    categorical: bool = False,
    all_touched: bool = False
) -> gpd.GeoDataFrame:
    """
    Calculate zonal statistics for raster data within vector zones.
    
    Parameters
    ----------
    raster_path : str or Path
        Path to raster file
    zones_gdf : gpd.GeoDataFrame
        GeoDataFrame with zone polygons
    stats : list
        Statistics to calculate
    categorical : bool
        Whether to treat raster as categorical data
    all_touched : bool
        Whether to include all pixels touched by geometry
        
    Returns
    -------
    gpd.GeoDataFrame
        Input GeoDataFrame with statistics columns added
    """
    # Calculate zonal statistics
    zs = rs_zonal_stats(
        zones_gdf,
        str(raster_path),
        stats=stats,
        categorical=categorical,
        all_touched=all_touched,
        geojson_out=False
    )
    
    # Convert to DataFrame and merge with input GeoDataFrame
    stats_df = pd.DataFrame(zs)
    result = zones_gdf.copy()
    result = pd.concat([result, stats_df], axis=1)
    
    return result


def compute_weighted_zonal_statistics(
    value_raster_path: Union[str, Path],
    weight_raster_path: Union[str, Path],
    zones_gdf: gpd.GeoDataFrame
) -> gpd.GeoDataFrame:
    """
    Calculate weighted zonal statistics.
    
    Parameters
    ----------
    value_raster_path : str or Path
        Path to value raster
    weight_raster_path : str or Path
        Path to weight raster
    zones_gdf : gpd.GeoDataFrame
        GeoDataFrame with zone polygons
        
    Returns
    -------
    gpd.GeoDataFrame
        GeoDataFrame with weighted statistics
    """
    result = zones_gdf.copy()
    
    with rasterio.open(value_raster_path) as value_src:
        with rasterio.open(weight_raster_path) as weight_src:
            value_data = value_src.read(1)
            weight_data = weight_src.read(1)
            
            # Ensure same CRS
            if zones_gdf.crs != value_src.crs:
                zones_gdf = zones_gdf.to_crs(value_src.crs)
            
            weighted_means = []
            weighted_sums = []
            
            for idx, geom in enumerate(zones_gdf.geometry):
                # Create mask for this zone
                mask = rasterize(
                    [(geom, 1)],
                    out_shape=value_data.shape,
                    transform=value_src.transform,
                    fill=0,
                    dtype='uint8'
                )
                
                # Extract values within zone
                zone_values = value_data[mask == 1]
                zone_weights = weight_data[mask == 1]
                
                # Calculate weighted statistics
                valid_mask = (~np.isnan(zone_values)) & (~np.isnan(zone_weights))
                
                if valid_mask.sum() > 0:
                    weighted_mean = np.average(
                        zone_values[valid_mask],
                        weights=zone_weights[valid_mask]
                    )
                    weighted_sum = np.sum(zone_values[valid_mask] * zone_weights[valid_mask])
                else:
                    weighted_mean = np.nan
                    weighted_sum = np.nan
                
                weighted_means.append(weighted_mean)
                weighted_sums.append(weighted_sum)
            
            result['weighted_mean'] = weighted_means
            result['weighted_sum'] = weighted_sums
    
    return result


def compute_percentile_stats(
    raster_path: Union[str, Path],
    zones_gdf: gpd.GeoDataFrame,
    percentiles: List[float] = [10, 25, 50, 75, 90]
) -> gpd.GeoDataFrame:
    """
    Calculate percentile statistics for zones.
    
    Parameters
    ----------
    raster_path : str or Path
        Path to raster file
    zones_gdf : gpd.GeoDataFrame
        GeoDataFrame with zone polygons
    percentiles : list
        List of percentiles to calculate
        
    Returns
    -------
    gpd.GeoDataFrame
        GeoDataFrame with percentile columns
    """
    result = zones_gdf.copy()
    
    with rasterio.open(raster_path) as src:
        data = src.read(1)
        
        # Ensure same CRS
        if zones_gdf.crs != src.crs:
            zones_gdf = zones_gdf.to_crs(src.crs)
        
        for percentile in percentiles:
            percentile_values = []
            
            for geom in zones_gdf.geometry:
                # Create mask for this zone
                mask = rasterize(
                    [(geom, 1)],
                    out_shape=data.shape,
                    transform=src.transform,
                    fill=0,
                    dtype='uint8'
                )
                
                # Extract values
                zone_values = data[mask == 1]
                valid_values = zone_values[~np.isnan(zone_values)]
                
                if len(valid_values) > 0:
                    p_value = np.percentile(valid_values, percentile)
                else:
                    p_value = np.nan
                
                percentile_values.append(p_value)
            
            result[f'p{percentile}'] = percentile_values
    
    return result


def compute_majority_class(
    categorical_raster_path: Union[str, Path],
    zones_gdf: gpd.GeoDataFrame
) -> gpd.GeoDataFrame:
    """
    Find the majority (most common) class in each zone.
    
    Parameters
    ----------
    categorical_raster_path : str or Path
        Path to categorical raster
    zones_gdf : gpd.GeoDataFrame
        GeoDataFrame with zone polygons
        
    Returns
    -------
    gpd.GeoDataFrame
        GeoDataFrame with majority class column
    """
    zs = rs_zonal_stats(
        zones_gdf,
        str(categorical_raster_path),
        categorical=True,
        geojson_out=False
    )
    
    majority_classes = []
    for zone_stats in zs:
        if zone_stats:
            majority_class = max(zone_stats, key=zone_stats.get)
        else:
            majority_class = None
        majority_classes.append(majority_class)
    
    result = zones_gdf.copy()
    result['majority_class'] = majority_classes
    
    return result


def compute_area_above_threshold(
    raster_path: Union[str, Path],
    zones_gdf: gpd.GeoDataFrame,
    threshold: float,
    pixel_area: Optional[float] = None
) -> gpd.GeoDataFrame:
    """
    Calculate area where raster values exceed a threshold in each zone.
    
    Parameters
    ----------
    raster_path : str or Path
        Path to raster file
    zones_gdf : gpd.GeoDataFrame
        GeoDataFrame with zone polygons
    threshold : float
        Threshold value
    pixel_area : float, optional
        Area of each pixel (if None, calculated from transform)
        
    Returns
    -------
    gpd.GeoDataFrame
        GeoDataFrame with area_above_threshold column
    """
    result = zones_gdf.copy()
    
    with rasterio.open(raster_path) as src:
        data = src.read(1)
        
        if pixel_area is None:
            # Calculate pixel area from transform
            transform = src.transform
            pixel_area = abs(transform.a * transform.e)
        
        # Ensure same CRS
        if zones_gdf.crs != src.crs:
            zones_gdf = zones_gdf.to_crs(src.crs)
        
        areas_above = []
        
        for geom in zones_gdf.geometry:
            # Create mask for this zone
            mask = rasterize(
                [(geom, 1)],
                out_shape=data.shape,
                transform=src.transform,
                fill=0,
                dtype='uint8'
            )
            
            # Count pixels above threshold
            zone_values = data[mask == 1]
            count_above = np.sum(zone_values > threshold)
            area_above = count_above * pixel_area
            
            areas_above.append(area_above)
        
        result['area_above_threshold'] = areas_above
    
    return result
