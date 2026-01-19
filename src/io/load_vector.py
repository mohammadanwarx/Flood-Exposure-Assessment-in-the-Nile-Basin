"""
Functions for loading and reading vector data.
"""

import geopandas as gpd
from pathlib import Path
from typing import Union, List, Optional
import fiona


def read_shapefile(filepath: Union[str, Path], **kwargs) -> gpd.GeoDataFrame:
    """
    Read a shapefile into a GeoDataFrame.
    
    Parameters
    ----------
    filepath : str or Path
        Path to the shapefile
    **kwargs
        Additional arguments passed to geopandas.read_file()
        
    Returns
    -------
    gpd.GeoDataFrame
        Vector data as GeoDataFrame
    """
    gdf = gpd.read_file(filepath, **kwargs)
    return gdf


def read_geojson(filepath: Union[str, Path]) -> gpd.GeoDataFrame:
    """
    Read a GeoJSON file into a GeoDataFrame.
    
    Parameters
    ----------
    filepath : str or Path
        Path to the GeoJSON file
        
    Returns
    -------
    gpd.GeoDataFrame
        Vector data as GeoDataFrame
    """
    gdf = gpd.read_file(filepath, driver='GeoJSON')
    return gdf


def read_geopackage(
    filepath: Union[str, Path],
    layer: Optional[str] = None
) -> gpd.GeoDataFrame:
    """
    Read a GeoPackage file into a GeoDataFrame.
    
    Parameters
    ----------
    filepath : str or Path
        Path to the GeoPackage file
    layer : str, optional
        Layer name to read (if None, reads first layer)
        
    Returns
    -------
    gpd.GeoDataFrame
        Vector data as GeoDataFrame
    """
    gdf = gpd.read_file(filepath, layer=layer)
    return gdf


def list_geopackage_layers(filepath: Union[str, Path]) -> List[str]:
    """
    List all layers in a GeoPackage file.
    
    Parameters
    ----------
    filepath : str or Path
        Path to the GeoPackage file
        
    Returns
    -------
    List[str]
        List of layer names
    """
    return fiona.listlayers(filepath)


def get_vector_info(filepath: Union[str, Path]) -> dict:
    """
    Get basic information about a vector file without loading all data.
    
    Parameters
    ----------
    filepath : str or Path
        Path to the vector file
        
    Returns
    -------
    dict
        Dictionary with vector information
    """
    with fiona.open(filepath) as src:
        info = {
            'driver': src.driver,
            'crs': str(src.crs),
            'schema': src.schema,
            'bounds': src.bounds,
            'count': len(src),
        }
    
    return info


def save_shapefile(
    gdf: gpd.GeoDataFrame,
    filepath: Union[str, Path],
    **kwargs
) -> None:
    """
    Save a GeoDataFrame as a shapefile.
    
    Parameters
    ----------
    gdf : gpd.GeoDataFrame
        GeoDataFrame to save
    filepath : str or Path
        Output file path
    **kwargs
        Additional arguments passed to GeoDataFrame.to_file()
    """
    gdf.to_file(filepath, driver='ESRI Shapefile', **kwargs)


def save_geojson(
    gdf: gpd.GeoDataFrame,
    filepath: Union[str, Path],
    **kwargs
) -> None:
    """
    Save a GeoDataFrame as a GeoJSON file.
    
    Parameters
    ----------
    gdf : gpd.GeoDataFrame
        GeoDataFrame to save
    filepath : str or Path
        Output file path
    **kwargs
        Additional arguments passed to GeoDataFrame.to_file()
    """
    gdf.to_file(filepath, driver='GeoJSON', **kwargs)


def save_geopackage(
    gdf: gpd.GeoDataFrame,
    filepath: Union[str, Path],
    layer: str = 'layer1',
    **kwargs
) -> None:
    """
    Save a GeoDataFrame as a GeoPackage file.
    
    Parameters
    ----------
    gdf : gpd.GeoDataFrame
        GeoDataFrame to save
    filepath : str or Path
        Output file path
    layer : str
        Layer name
    **kwargs
        Additional arguments passed to GeoDataFrame.to_file()
    """
    gdf.to_file(filepath, layer=layer, driver='GPKG', **kwargs)
