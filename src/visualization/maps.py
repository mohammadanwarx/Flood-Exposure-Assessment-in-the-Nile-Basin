"""
Map visualization functions using matplotlib, folium, and contextily.
"""

import numpy as np
import matplotlib.pyplot as plt
import geopandas as gpd
import rasterio
from typing import Union, Optional, Tuple, List
from pathlib import Path
from matplotlib.colors import Normalize
from matplotlib import cm


def plot_raster(
    raster_path: Union[str, Path, np.ndarray],
    title: str = 'Raster Map',
    cmap: str = 'viridis',
    figsize: Tuple[int, int] = (12, 8),
    colorbar_label: Optional[str] = None,
    vmin: Optional[float] = None,
    vmax: Optional[float] = None,
    save_path: Optional[Union[str, Path]] = None,
    dpi: int = 300
) -> plt.Figure:
    """
    Plot a raster with matplotlib.
    
    Parameters
    ----------
    raster_path : str, Path, or np.ndarray
        Path to raster file or numpy array
    title : str
        Plot title
    cmap : str
        Colormap name
    figsize : tuple
        Figure size
    colorbar_label : str, optional
        Label for colorbar
    vmin, vmax : float, optional
        Min and max values for colormap
    save_path : str or Path, optional
        Path to save figure
    dpi : int
        DPI for saving
        
    Returns
    -------
    plt.Figure
        Matplotlib figure
    """
    # Load raster if path provided
    if isinstance(raster_path, (str, Path)):
        with rasterio.open(raster_path) as src:
            data = src.read(1, masked=True)
            extent = [src.bounds.left, src.bounds.right, src.bounds.bottom, src.bounds.top]
    else:
        data = raster_path
        extent = None
    
    # Create figure
    fig, ax = plt.subplots(figsize=figsize)
    
    # Plot raster
    im = ax.imshow(data, cmap=cmap, vmin=vmin, vmax=vmax, extent=extent)
    
    # Add colorbar
    cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    if colorbar_label:
        cbar.set_label(colorbar_label, rotation=270, labelpad=20)
    
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.set_xlabel('Longitude')
    ax.set_ylabel('Latitude')
    
    plt.tight_layout()
    
    # Save if requested
    if save_path:
        plt.savefig(save_path, dpi=dpi, bbox_inches='tight')
    
    return fig


def plot_vector(
    gdf: gpd.GeoDataFrame,
    column: Optional[str] = None,
    title: str = 'Vector Map',
    cmap: str = 'viridis',
    figsize: Tuple[int, int] = (12, 8),
    legend: bool = True,
    save_path: Optional[Union[str, Path]] = None,
    dpi: int = 300
) -> plt.Figure:
    """
    Plot a vector GeoDataFrame.
    
    Parameters
    ----------
    gdf : gpd.GeoDataFrame
        GeoDataFrame to plot
    column : str, optional
        Column to use for coloring
    title : str
        Plot title
    cmap : str
        Colormap name
    figsize : tuple
        Figure size
    legend : bool
        Whether to show legend
    save_path : str or Path, optional
        Path to save figure
    dpi : int
        DPI for saving
        
    Returns
    -------
    plt.Figure
        Matplotlib figure
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    gdf.plot(ax=ax, column=column, cmap=cmap, legend=legend, edgecolor='black', linewidth=0.5)
    
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.set_xlabel('Longitude')
    ax.set_ylabel('Latitude')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=dpi, bbox_inches='tight')
    
    return fig


def plot_raster_with_vector_overlay(
    raster_path: Union[str, Path],
    gdf: gpd.GeoDataFrame,
    title: str = 'Raster with Vector Overlay',
    raster_cmap: str = 'Blues',
    vector_color: str = 'red',
    figsize: Tuple[int, int] = (12, 8),
    save_path: Optional[Union[str, Path]] = None,
    dpi: int = 300
) -> plt.Figure:
    """
    Plot raster with vector overlay.
    
    Parameters
    ----------
    raster_path : str or Path
        Path to raster file
    gdf : gpd.GeoDataFrame
        Vector data to overlay
    title : str
        Plot title
    raster_cmap : str
        Colormap for raster
    vector_color : str
        Color for vector features
    figsize : tuple
        Figure size
    save_path : str or Path, optional
        Path to save figure
    dpi : int
        DPI for saving
        
    Returns
    -------
    plt.Figure
        Matplotlib figure
    """
    with rasterio.open(raster_path) as src:
        data = src.read(1, masked=True)
        extent = [src.bounds.left, src.bounds.right, src.bounds.bottom, src.bounds.top]
        raster_crs = src.crs
    
    # Ensure vector is in same CRS
    if gdf.crs != raster_crs:
        gdf = gdf.to_crs(raster_crs)
    
    fig, ax = plt.subplots(figsize=figsize)
    
    # Plot raster
    ax.imshow(data, cmap=raster_cmap, extent=extent, alpha=0.7)
    
    # Overlay vector
    gdf.plot(ax=ax, facecolor='none', edgecolor=vector_color, linewidth=2)
    
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.set_xlabel('Longitude')
    ax.set_ylabel('Latitude')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=dpi, bbox_inches='tight')
    
    return fig


def create_interactive_map(
    gdf: gpd.GeoDataFrame,
    column: Optional[str] = None,
    cmap: str = 'viridis',
    zoom_start: int = 10,
    save_path: Optional[Union[str, Path]] = None
):
    """
    Create an interactive map using folium.
    
    Parameters
    ----------
    gdf : gpd.GeoDataFrame
        GeoDataFrame to plot
    column : str, optional
        Column to use for choropleth
    cmap : str
        Colormap name
    zoom_start : int
        Initial zoom level
    save_path : str or Path, optional
        Path to save HTML file
        
    Returns
    -------
    folium.Map
        Folium map object
    """
    try:
        import folium
    except ImportError:
        raise ImportError("folium is required. Install with: pip install folium")
    
    # Reproject to WGS84 for folium
    if gdf.crs != 'EPSG:4326':
        gdf = gdf.to_crs('EPSG:4326')
    
    # Calculate center
    center = [gdf.geometry.centroid.y.mean(), gdf.geometry.centroid.x.mean()]
    
    # Create map
    m = folium.Map(location=center, zoom_start=zoom_start)
    
    if column:
        # Choropleth map
        folium.Choropleth(
            geo_data=gdf,
            data=gdf,
            columns=[gdf.index.name or 'index', column],
            key_on='feature.id',
            fill_color=cmap,
            legend_name=column
        ).add_to(m)
    else:
        # Simple geometry overlay
        folium.GeoJson(gdf).add_to(m)
    
    if save_path:
        m.save(str(save_path))
    
    return m


def plot_flood_depth_map(
    flood_raster: Union[str, Path, np.ndarray],
    boundaries: Optional[gpd.GeoDataFrame] = None,
    title: str = 'Flood Depth Map',
    depth_thresholds: List[float] = [0.5, 1.0, 2.0],
    figsize: Tuple[int, int] = (12, 8),
    save_path: Optional[Union[str, Path]] = None,
    dpi: int = 300
) -> plt.Figure:
    """
    Create a specialized flood depth map with depth categories.
    
    Parameters
    ----------
    flood_raster : str, Path, or np.ndarray
        Flood depth raster
    boundaries : gpd.GeoDataFrame, optional
        Administrative boundaries to overlay
    title : str
        Map title
    depth_thresholds : list
        Depth thresholds for classification
    figsize : tuple
        Figure size
    save_path : str or Path, optional
        Path to save figure
    dpi : int
        DPI for saving
        
    Returns
    -------
    plt.Figure
        Matplotlib figure
    """
    # Load raster
    if isinstance(flood_raster, (str, Path)):
        with rasterio.open(flood_raster) as src:
            data = src.read(1, masked=True)
            extent = [src.bounds.left, src.bounds.right, src.bounds.bottom, src.bounds.top]
    else:
        data = flood_raster
        extent = None
    
    fig, ax = plt.subplots(figsize=figsize)
    
    # Create custom colormap with discrete levels
    from matplotlib.colors import BoundaryNorm
    
    levels = [0] + depth_thresholds + [data.max()]
    norm = BoundaryNorm(levels, ncolors=256)
    
    # Plot flood depth
    im = ax.imshow(data, cmap='Blues', norm=norm, extent=extent)
    
    # Add boundaries if provided
    if boundaries is not None:
        boundaries.plot(ax=ax, facecolor='none', edgecolor='black', linewidth=1)
    
    # Colorbar with custom labels
    cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label('Flood Depth (m)', rotation=270, labelpad=20)
    
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.set_xlabel('Longitude')
    ax.set_ylabel('Latitude')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=dpi, bbox_inches='tight')
    
    return fig


def plot_basemap(
    gdf: gpd.GeoDataFrame,
    basemap_source: str = 'OpenStreetMap',
    alpha: float = 0.7,
    figsize: Tuple[int, int] = (12, 8),
    save_path: Optional[Union[str, Path]] = None,
    dpi: int = 300
) -> plt.Figure:
    """
    Plot vector data with a basemap using contextily.
    
    Parameters
    ----------
    gdf : gpd.GeoDataFrame
        GeoDataFrame to plot
    basemap_source : str
        Basemap provider
    alpha : float
        Transparency for overlay
    figsize : tuple
        Figure size
    save_path : str or Path, optional
        Path to save figure
    dpi : int
        DPI for saving
        
    Returns
    -------
    plt.Figure
        Matplotlib figure
    """
    try:
        import contextily as ctx
    except ImportError:
        raise ImportError("contextily is required. Install with: pip install contextily")
    
    # Reproject to Web Mercator for basemap
    gdf_web = gdf.to_crs('EPSG:3857')
    
    fig, ax = plt.subplots(figsize=figsize)
    
    # Plot data
    gdf_web.plot(ax=ax, alpha=alpha, edgecolor='black')
    
    # Add basemap
    ctx.add_basemap(ax, source=basemap_source)
    
    ax.set_axis_off()
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=dpi, bbox_inches='tight')
    
    return fig
