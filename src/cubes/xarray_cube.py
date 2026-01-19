"""
Multi-dimensional data cube operations using xarray.
"""

import xarray as xr
import numpy as np
import rioxarray
from typing import Union, List, Optional, Tuple
from pathlib import Path


def create_datacube_from_rasters(
    file_paths: List[Union[str, Path]],
    timestamps: Optional[List] = None,
    variable_name: str = 'data'
) -> xr.DataArray:
    """
    Create a 3D datacube from multiple 2D rasters.
    
    Parameters
    ----------
    file_paths : list
        List of raster file paths
    timestamps : list, optional
        Time coordinates for each raster
    variable_name : str
        Name for the data variable
        
    Returns
    -------
    xr.DataArray
        3D datacube (time, y, x)
    """
    # Load all rasters
    arrays = []
    for fp in file_paths:
        da = rioxarray.open_rasterio(fp, masked=True)
        if da.ndim == 3:  # Multiple bands
            da = da.isel(band=0)  # Take first band
        arrays.append(da)
    
    # Concatenate along time dimension
    cube = xr.concat(arrays, dim='time')
    
    # Add time coordinates
    if timestamps is not None:
        cube['time'] = timestamps
    
    cube.name = variable_name
    
    return cube


def slice_datacube_by_bbox(
    cube: xr.DataArray,
    bbox: Tuple[float, float, float, float]
) -> xr.DataArray:
    """
    Slice a datacube by bounding box coordinates.
    
    Parameters
    ----------
    cube : xr.DataArray
        Input datacube
    bbox : tuple
        Bounding box (min_x, min_y, max_x, max_y)
        
    Returns
    -------
    xr.DataArray
        Sliced datacube
    """
    min_x, min_y, max_x, max_y = bbox
    
    sliced = cube.rio.clip_box(
        minx=min_x,
        miny=min_y,
        maxx=max_x,
        maxy=max_y
    )
    
    return sliced


def slice_datacube_by_time(
    cube: xr.DataArray,
    start_time,
    end_time
) -> xr.DataArray:
    """
    Slice a datacube by time range.
    
    Parameters
    ----------
    cube : xr.DataArray
        Input datacube
    start_time
        Start time
    end_time
        End time
        
    Returns
    -------
    xr.DataArray
        Time-sliced datacube
    """
    return cube.sel(time=slice(start_time, end_time))


def aggregate_datacube_temporal(
    cube: xr.DataArray,
    method: str = 'mean',
    freq: Optional[str] = None
) -> xr.DataArray:
    """
    Aggregate datacube along time dimension.
    
    Parameters
    ----------
    cube : xr.DataArray
        Input datacube
    method : str
        Aggregation method: 'mean', 'sum', 'max', 'min', 'median', 'std'
    freq : str, optional
        Resampling frequency ('D', 'W', 'M', 'Y')
        If None, aggregates all time steps
        
    Returns
    -------
    xr.DataArray
        Aggregated datacube
    """
    if freq is not None:
        # Resample to frequency
        resampled = cube.resample(time=freq)
        
        if method == 'mean':
            result = resampled.mean()
        elif method == 'sum':
            result = resampled.sum()
        elif method == 'max':
            result = resampled.max()
        elif method == 'min':
            result = resampled.min()
        elif method == 'median':
            result = resampled.median()
        elif method == 'std':
            result = resampled.std()
        else:
            raise ValueError(f"Unknown method: {method}")
    else:
        # Aggregate all time steps
        if method == 'mean':
            result = cube.mean(dim='time')
        elif method == 'sum':
            result = cube.sum(dim='time')
        elif method == 'max':
            result = cube.max(dim='time')
        elif method == 'min':
            result = cube.min(dim='time')
        elif method == 'median':
            result = cube.median(dim='time')
        elif method == 'std':
            result = cube.std(dim='time')
        else:
            raise ValueError(f"Unknown method: {method}")
    
    return result


def aggregate_datacube_spatial(
    cube: xr.DataArray,
    factor: int = 2,
    method: str = 'mean'
) -> xr.DataArray:
    """
    Aggregate datacube spatially (coarsen resolution).
    
    Parameters
    ----------
    cube : xr.DataArray
        Input datacube
    factor : int
        Coarsening factor
    method : str
        Aggregation method: 'mean', 'sum', 'max', 'min', 'median'
        
    Returns
    -------
    xr.DataArray
        Spatially aggregated datacube
    """
    # Identify spatial dimensions
    spatial_dims = [d for d in ['x', 'y'] if d in cube.dims]
    
    coarsen_dict = {dim: factor for dim in spatial_dims}
    coarsened = cube.coarsen(coarsen_dict, boundary='trim')
    
    if method == 'mean':
        result = coarsened.mean()
    elif method == 'sum':
        result = coarsened.sum()
    elif method == 'max':
        result = coarsened.max()
    elif method == 'min':
        result = coarsened.min()
    elif method == 'median':
        result = coarsened.median()
    else:
        raise ValueError(f"Unknown method: {method}")
    
    return result


def apply_mask_to_datacube(
    cube: xr.DataArray,
    mask: Union[xr.DataArray, np.ndarray]
) -> xr.DataArray:
    """
    Apply a mask to a datacube.
    
    Parameters
    ----------
    cube : xr.DataArray
        Input datacube
    mask : xr.DataArray or np.ndarray
        Boolean mask (True = keep, False = mask out)
        
    Returns
    -------
    xr.DataArray
        Masked datacube
    """
    return cube.where(mask)


def calculate_datacube_statistics(
    cube: xr.DataArray,
    dims: Optional[List[str]] = None
) -> xr.Dataset:
    """
    Calculate comprehensive statistics for a datacube.
    
    Parameters
    ----------
    cube : xr.DataArray
        Input datacube
    dims : list, optional
        Dimensions to compute statistics over (if None, uses all)
        
    Returns
    -------
    xr.Dataset
        Dataset with statistical measures
    """
    stats = xr.Dataset()
    
    stats['mean'] = cube.mean(dim=dims)
    stats['std'] = cube.std(dim=dims)
    stats['min'] = cube.min(dim=dims)
    stats['max'] = cube.max(dim=dims)
    stats['median'] = cube.median(dim=dims)
    stats['q25'] = cube.quantile(0.25, dim=dims)
    stats['q75'] = cube.quantile(0.75, dim=dims)
    
    return stats


def interpolate_datacube_temporal(
    cube: xr.DataArray,
    new_time_coords,
    method: str = 'linear'
) -> xr.DataArray:
    """
    Interpolate datacube to new time coordinates.
    
    Parameters
    ----------
    cube : xr.DataArray
        Input datacube
    new_time_coords
        New time coordinates
    method : str
        Interpolation method: 'linear', 'nearest', 'cubic'
        
    Returns
    -------
    xr.DataArray
        Interpolated datacube
    """
    return cube.interp(time=new_time_coords, method=method)


def interpolate_datacube_spatial(
    cube: xr.DataArray,
    new_shape: Tuple[int, int],
    method: str = 'linear'
) -> xr.DataArray:
    """
    Interpolate datacube to new spatial resolution.
    
    Parameters
    ----------
    cube : xr.DataArray
        Input datacube
    new_shape : tuple
        New shape (height, width)
    method : str
        Interpolation method: 'linear', 'nearest', 'cubic'
        
    Returns
    -------
    xr.DataArray
        Interpolated datacube
    """
    # Get spatial dimensions
    y_dim = [d for d in cube.dims if d in ['y', 'lat', 'latitude']][0]
    x_dim = [d for d in cube.dims if d in ['x', 'lon', 'longitude']][0]
    
    # Create new coordinates
    new_y = np.linspace(cube[y_dim][0], cube[y_dim][-1], new_shape[0])
    new_x = np.linspace(cube[x_dim][0], cube[x_dim][-1], new_shape[1])
    
    # Interpolate
    interpolated = cube.interp({y_dim: new_y, x_dim: new_x}, method=method)
    
    return interpolated


def merge_datacubes(
    cubes: List[xr.DataArray],
    dim: str = 'time'
) -> xr.DataArray:
    """
    Merge multiple datacubes along a dimension.
    
    Parameters
    ----------
    cubes : list
        List of datacubes
    dim : str
        Dimension to merge along
        
    Returns
    -------
    xr.DataArray
        Merged datacube
    """
    return xr.concat(cubes, dim=dim)


def save_datacube(
    cube: xr.DataArray,
    output_path: Union[str, Path],
    format: str = 'netcdf'
) -> None:
    """
    Save a datacube to file.
    
    Parameters
    ----------
    cube : xr.DataArray
        Datacube to save
    output_path : str or Path
        Output file path
    format : str
        Output format: 'netcdf', 'zarr'
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    if format == 'netcdf':
        cube.to_netcdf(output_path)
    elif format == 'zarr':
        cube.to_zarr(output_path)
    else:
        raise ValueError(f"Unknown format: {format}")


def load_datacube(
    file_path: Union[str, Path],
    format: str = 'netcdf'
) -> xr.DataArray:
    """
    Load a datacube from file.
    
    Parameters
    ----------
    file_path : str or Path
        Path to datacube file
    format : str
        File format: 'netcdf', 'zarr'
        
    Returns
    -------
    xr.DataArray
        Loaded datacube
    """
    if format == 'netcdf':
        ds = xr.open_dataarray(file_path)
    elif format == 'zarr':
        ds = xr.open_zarr(file_path)
    else:
        raise ValueError(f"Unknown format: {format}")
    
    return ds


def rolling_window_operation(
    cube: xr.DataArray,
    window_size: int,
    dim: str = 'time',
    operation: str = 'mean'
) -> xr.DataArray:
    """
    Apply rolling window operation along a dimension.
    
    Parameters
    ----------
    cube : xr.DataArray
        Input datacube
    window_size : int
        Size of rolling window
    dim : str
        Dimension to roll over
    operation : str
        Operation: 'mean', 'sum', 'max', 'min', 'std'
        
    Returns
    -------
    xr.DataArray
        Result of rolling operation
    """
    rolling = cube.rolling({dim: window_size}, center=True)
    
    if operation == 'mean':
        result = rolling.mean()
    elif operation == 'sum':
        result = rolling.sum()
    elif operation == 'max':
        result = rolling.max()
    elif operation == 'min':
        result = rolling.min()
    elif operation == 'std':
        result = rolling.std()
    else:
        raise ValueError(f"Unknown operation: {operation}")
    
    return result
