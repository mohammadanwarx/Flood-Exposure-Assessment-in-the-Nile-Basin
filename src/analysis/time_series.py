"""
Time series analysis for multi-temporal geospatial data.
"""

import numpy as np
import pandas as pd
import xarray as xr
from typing import Union, List, Optional, Tuple
from pathlib import Path
from datetime import datetime


def load_time_series_rasters(
    file_paths: List[Union[str, Path]],
    timestamps: Optional[List[datetime]] = None,
    variable_name: str = 'data'
) -> xr.DataArray:
    """
    Load multiple rasters as a time series DataArray.
    
    Parameters
    ----------
    file_paths : list
        List of raster file paths
    timestamps : list of datetime, optional
        Timestamps for each raster
    variable_name : str
        Name for the data variable
        
    Returns
    -------
    xr.DataArray
        Time series DataArray
    """
    import rioxarray
    
    # Load all rasters
    das = [rioxarray.open_rasterio(fp, masked=True) for fp in file_paths]
    
    # Create time dimension
    if timestamps is None:
        timestamps = pd.date_range('2000-01-01', periods=len(das), freq='D')
    
    # Concatenate along time dimension
    ts = xr.concat(das, dim='time')
    ts['time'] = timestamps
    ts.name = variable_name
    
    return ts


def calculate_temporal_statistics(
    time_series: xr.DataArray,
    stats: List[str] = ['mean', 'std', 'min', 'max']
) -> xr.Dataset:
    """
    Calculate temporal statistics across the time dimension.
    
    Parameters
    ----------
    time_series : xr.DataArray
        Time series DataArray
    stats : list
        Statistics to calculate
        
    Returns
    -------
    xr.Dataset
        Dataset with temporal statistics
    """
    results = {}
    
    for stat in stats:
        if stat == 'mean':
            results[stat] = time_series.mean(dim='time')
        elif stat == 'std':
            results[stat] = time_series.std(dim='time')
        elif stat == 'min':
            results[stat] = time_series.min(dim='time')
        elif stat == 'max':
            results[stat] = time_series.max(dim='time')
        elif stat == 'median':
            results[stat] = time_series.median(dim='time')
        elif stat == 'range':
            results[stat] = time_series.max(dim='time') - time_series.min(dim='time')
    
    return xr.Dataset(results)


def detect_temporal_changes(
    time_series: xr.DataArray,
    method: str = 'difference',
    threshold: Optional[float] = None
) -> xr.DataArray:
    """
    Detect changes over time.
    
    Parameters
    ----------
    time_series : xr.DataArray
        Time series DataArray
    method : str
        Change detection method: 'difference', 'percent_change', 'trend'
    threshold : float, optional
        Threshold for classifying significant changes
        
    Returns
    -------
    xr.DataArray
        Change detection results
    """
    if method == 'difference':
        # Simple difference between last and first time steps
        change = time_series.isel(time=-1) - time_series.isel(time=0)
    
    elif method == 'percent_change':
        # Percent change from first time step
        first = time_series.isel(time=0)
        last = time_series.isel(time=-1)
        change = ((last - first) / first) * 100
    
    elif method == 'trend':
        # Linear trend over time
        from scipy.stats import linregress
        
        time_numeric = np.arange(len(time_series.time))
        
        def calc_trend(arr):
            if np.all(np.isnan(arr)):
                return np.nan
            valid = ~np.isnan(arr)
            if valid.sum() < 2:
                return np.nan
            slope, _, _, _, _ = linregress(time_numeric[valid], arr[valid])
            return slope
        
        change = xr.apply_ufunc(
            calc_trend,
            time_series,
            input_core_dims=[['time']],
            vectorize=True
        )
    
    else:
        raise ValueError(f"Unknown method: {method}")
    
    # Apply threshold if provided
    if threshold is not None:
        change = xr.where(np.abs(change) > threshold, change, 0)
    
    return change


def calculate_temporal_anomalies(
    time_series: xr.DataArray,
    baseline_period: Optional[Tuple[datetime, datetime]] = None
) -> xr.DataArray:
    """
    Calculate anomalies relative to a baseline period.
    
    Parameters
    ----------
    time_series : xr.DataArray
        Time series DataArray
    baseline_period : tuple of datetime, optional
        Start and end dates for baseline period
        If None, uses entire time series
        
    Returns
    -------
    xr.DataArray
        Anomaly values
    """
    if baseline_period is not None:
        baseline = time_series.sel(time=slice(baseline_period[0], baseline_period[1]))
    else:
        baseline = time_series
    
    baseline_mean = baseline.mean(dim='time')
    baseline_std = baseline.std(dim='time')
    
    # Calculate standardized anomalies
    anomalies = (time_series - baseline_mean) / baseline_std
    
    return anomalies


def aggregate_temporal_data(
    time_series: xr.DataArray,
    freq: str = 'M',
    method: str = 'mean'
) -> xr.DataArray:
    """
    Aggregate time series data to coarser temporal resolution.
    
    Parameters
    ----------
    time_series : xr.DataArray
        Time series DataArray
    freq : str
        Frequency for aggregation ('D', 'W', 'M', 'Y')
    method : str
        Aggregation method: 'mean', 'sum', 'max', 'min'
        
    Returns
    -------
    xr.DataArray
        Aggregated time series
    """
    resampled = time_series.resample(time=freq)
    
    if method == 'mean':
        result = resampled.mean()
    elif method == 'sum':
        result = resampled.sum()
    elif method == 'max':
        result = resampled.max()
    elif method == 'min':
        result = resampled.min()
    else:
        raise ValueError(f"Unknown method: {method}")
    
    return result


def calculate_cumulative_exposure(
    flood_events: List[np.ndarray],
    event_dates: List[datetime],
    population: np.ndarray
) -> pd.DataFrame:
    """
    Calculate cumulative exposure over multiple flood events.
    
    Parameters
    ----------
    flood_events : list of np.ndarray
        List of flood extent rasters (binary)
    event_dates : list of datetime
        Dates of each flood event
    population : np.ndarray
        Population raster
        
    Returns
    -------
    pd.DataFrame
        DataFrame with cumulative exposure metrics
    """
    cumulative_mask = np.zeros_like(flood_events[0], dtype=bool)
    results = []
    
    for i, (event, date) in enumerate(zip(flood_events, event_dates)):
        # Update cumulative mask
        cumulative_mask = cumulative_mask | (event > 0)
        
        # Calculate metrics
        event_pop = np.nansum(population[event > 0])
        cumulative_pop = np.nansum(population[cumulative_mask])
        event_area = np.sum(event > 0)
        cumulative_area = np.sum(cumulative_mask)
        
        results.append({
            'date': date,
            'event_number': i + 1,
            'event_affected_population': event_pop,
            'cumulative_affected_population': cumulative_pop,
            'event_area_pixels': event_area,
            'cumulative_area_pixels': cumulative_area
        })
    
    return pd.DataFrame(results)


def calculate_return_period_exceedance(
    time_series: xr.DataArray,
    threshold_values: List[float]
) -> pd.DataFrame:
    """
    Calculate exceedance frequencies for different threshold values.
    
    Parameters
    ----------
    time_series : xr.DataArray
        Time series DataArray
    threshold_values : list
        List of threshold values
        
    Returns
    -------
    pd.DataFrame
        DataFrame with exceedance statistics
    """
    total_timesteps = len(time_series.time)
    results = []
    
    for threshold in threshold_values:
        # Count exceedances spatially
        exceedance_count = (time_series > threshold).sum(dim='time')
        
        # Calculate average exceedance frequency
        avg_exceedance_freq = float(exceedance_count.mean())
        max_exceedance_count = float(exceedance_count.max())
        
        # Estimate return period (in units of timesteps)
        if avg_exceedance_freq > 0:
            avg_return_period = total_timesteps / avg_exceedance_freq
        else:
            avg_return_period = np.inf
        
        results.append({
            'threshold': threshold,
            'avg_exceedances': avg_exceedance_freq,
            'max_exceedances': max_exceedance_count,
            'avg_return_period': avg_return_period
        })
    
    return pd.DataFrame(results)
