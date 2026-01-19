"""
Spatial utility functions for common operations.
"""

import numpy as np
from typing import Tuple, List, Optional
import rasterio
from rasterio.features import geometry_mask
from shapely.geometry import box, Point, Polygon
import geopandas as gpd


def create_bounding_box(
    min_x: float,
    min_y: float,
    max_x: float,
    max_y: float,
    crs: str = 'EPSG:4326'
) -> gpd.GeoDataFrame:
    """
    Create a GeoDataFrame with a bounding box.
    
    Parameters
    ----------
    min_x, min_y, max_x, max_y : float
        Bounding box coordinates
    crs : str
        Coordinate reference system
        
    Returns
    -------
    gpd.GeoDataFrame
        GeoDataFrame with bounding box
    """
    bbox_geom = box(min_x, min_y, max_x, max_y)
    gdf = gpd.GeoDataFrame([1], geometry=[bbox_geom], crs=crs)
    return gdf


def calculate_pixel_coordinates(
    transform: rasterio.Affine,
    rows: int,
    cols: int
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Calculate geographic coordinates for each pixel center.
    
    Parameters
    ----------
    transform : rasterio.Affine
        Affine transform from raster
    rows : int
        Number of rows
    cols : int
        Number of columns
        
    Returns
    -------
    x_coords : np.ndarray
        X coordinates array
    y_coords : np.ndarray
        Y coordinates array
    """
    cols_array, rows_array = np.meshgrid(np.arange(cols), np.arange(rows))
    
    # Transform pixel coordinates to geographic coordinates
    x_coords, y_coords = rasterio.transform.xy(
        transform,
        rows_array.flatten(),
        cols_array.flatten()
    )
    
    x_coords = np.array(x_coords).reshape(rows, cols)
    y_coords = np.array(y_coords).reshape(rows, cols)
    
    return x_coords, y_coords


def calculate_distance_to_features(
    raster_shape: Tuple[int, int],
    transform: rasterio.Affine,
    features: gpd.GeoDataFrame
) -> np.ndarray:
    """
    Calculate distance from each pixel to nearest feature.
    
    Parameters
    ----------
    raster_shape : tuple
        Shape of output raster (rows, cols)
    transform : rasterio.Affine
        Affine transform
    features : gpd.GeoDataFrame
        Features to calculate distance to
        
    Returns
    -------
    np.ndarray
        Distance raster
    """
    from scipy.spatial import cKDTree
    
    # Get pixel coordinates
    x_coords, y_coords = calculate_pixel_coordinates(transform, raster_shape[0], raster_shape[1])
    pixel_points = np.column_stack([x_coords.ravel(), y_coords.ravel()])
    
    # Get feature coordinates
    feature_points = []
    for geom in features.geometry:
        if geom.geom_type == 'Point':
            feature_points.append([geom.x, geom.y])
        elif geom.geom_type == 'LineString':
            feature_points.extend([[x, y] for x, y in geom.coords])
        elif geom.geom_type in ['Polygon', 'MultiPolygon']:
            # Use boundary points
            feature_points.extend([[x, y] for x, y in geom.exterior.coords])
    
    feature_points = np.array(feature_points)
    
    # Build KDTree and calculate distances
    tree = cKDTree(feature_points)
    distances, _ = tree.query(pixel_points)
    
    # Reshape to raster
    distance_raster = distances.reshape(raster_shape)
    
    return distance_raster


def calculate_slope_aspect(
    elevation: np.ndarray,
    cell_size: float = 30.0
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Calculate slope and aspect from elevation raster.
    
    Parameters
    ----------
    elevation : np.ndarray
        Elevation raster
    cell_size : float
        Cell size in map units
        
    Returns
    -------
    slope : np.ndarray
        Slope in degrees
    aspect : np.ndarray
        Aspect in degrees (0-360)
    """
    # Calculate gradients
    dy, dx = np.gradient(elevation, cell_size)
    
    # Calculate slope
    slope = np.degrees(np.arctan(np.sqrt(dx**2 + dy**2)))
    
    # Calculate aspect
    aspect = np.degrees(np.arctan2(-dy, dx))
    aspect = np.where(aspect < 0, 360 + aspect, aspect)
    
    return slope, aspect


def create_fishnet_grid(
    bbox: Tuple[float, float, float, float],
    cell_size: float,
    crs: str = 'EPSG:4326'
) -> gpd.GeoDataFrame:
    """
    Create a fishnet grid of polygons.
    
    Parameters
    ----------
    bbox : tuple
        Bounding box (min_x, min_y, max_x, max_y)
    cell_size : float
        Size of each cell
    crs : str
        Coordinate reference system
        
    Returns
    -------
    gpd.GeoDataFrame
        GeoDataFrame with grid cells
    """
    min_x, min_y, max_x, max_y = bbox
    
    # Create grid coordinates
    x_coords = np.arange(min_x, max_x, cell_size)
    y_coords = np.arange(min_y, max_y, cell_size)
    
    # Create grid cells
    grid_cells = []
    for x in x_coords:
        for y in y_coords:
            cell = box(x, y, x + cell_size, y + cell_size)
            grid_cells.append(cell)
    
    # Create GeoDataFrame
    gdf = gpd.GeoDataFrame(geometry=grid_cells, crs=crs)
    gdf['cell_id'] = range(len(gdf))
    
    return gdf


def random_points_in_polygon(
    polygon: Polygon,
    n_points: int,
    crs: str = 'EPSG:4326'
) -> gpd.GeoDataFrame:
    """
    Generate random points within a polygon.
    
    Parameters
    ----------
    polygon : Polygon
        Shapely polygon
    n_points : int
        Number of points to generate
    crs : str
        Coordinate reference system
        
    Returns
    -------
    gpd.GeoDataFrame
        GeoDataFrame with random points
    """
    min_x, min_y, max_x, max_y = polygon.bounds
    
    points = []
    while len(points) < n_points:
        # Generate random point in bounding box
        x = np.random.uniform(min_x, max_x)
        y = np.random.uniform(min_y, max_y)
        point = Point(x, y)
        
        # Check if point is inside polygon
        if polygon.contains(point):
            points.append(point)
    
    gdf = gpd.GeoDataFrame(geometry=points, crs=crs)
    return gdf


def calculate_area_overlap(
    gdf1: gpd.GeoDataFrame,
    gdf2: gpd.GeoDataFrame,
    overlap_col: str = 'overlap_area'
) -> gpd.GeoDataFrame:
    """
    Calculate area of overlap between two GeoDataFrames.
    
    Parameters
    ----------
    gdf1 : gpd.GeoDataFrame
        First GeoDataFrame
    gdf2 : gpd.GeoDataFrame
        Second GeoDataFrame
    overlap_col : str
        Name for overlap area column
        
    Returns
    -------
    gpd.GeoDataFrame
        gdf1 with overlap area column added
    """
    # Ensure same CRS
    if gdf1.crs != gdf2.crs:
        gdf2 = gdf2.to_crs(gdf1.crs)
    
    result = gdf1.copy()
    overlap_areas = []
    
    for idx, geom1 in enumerate(gdf1.geometry):
        # Find intersecting geometries
        intersecting = gdf2[gdf2.intersects(geom1)]
        
        if len(intersecting) > 0:
            # Calculate total overlap area
            overlap = sum([geom1.intersection(geom2).area for geom2 in intersecting.geometry])
        else:
            overlap = 0
        
        overlap_areas.append(overlap)
    
    result[overlap_col] = overlap_areas
    
    return result


def reproject_bounds(
    bounds: Tuple[float, float, float, float],
    src_crs: str,
    dst_crs: str
) -> Tuple[float, float, float, float]:
    """
    Reproject bounding box coordinates.
    
    Parameters
    ----------
    bounds : tuple
        Bounding box (min_x, min_y, max_x, max_y)
    src_crs : str
        Source CRS
    dst_crs : str
        Destination CRS
        
    Returns
    -------
    tuple
        Reprojected bounds
    """
    from pyproj import Transformer
    
    transformer = Transformer.from_crs(src_crs, dst_crs, always_xy=True)
    
    min_x, min_y = transformer.transform(bounds[0], bounds[1])
    max_x, max_y = transformer.transform(bounds[2], bounds[3])
    
    return (min_x, min_y, max_x, max_y)


def calculate_centroid_coordinates(
    gdf: gpd.GeoDataFrame
) -> gpd.GeoDataFrame:
    """
    Add centroid coordinates to GeoDataFrame.
    
    Parameters
    ----------
    gdf : gpd.GeoDataFrame
        Input GeoDataFrame
        
    Returns
    -------
    gpd.GeoDataFrame
        GeoDataFrame with centroid coordinate columns
    """
    result = gdf.copy()
    
    centroids = result.geometry.centroid
    result['centroid_x'] = centroids.x
    result['centroid_y'] = centroids.y
    
    return result


def snap_points_to_line(
    points: gpd.GeoDataFrame,
    lines: gpd.GeoDataFrame,
    max_distance: Optional[float] = None
) -> gpd.GeoDataFrame:
    """
    Snap points to nearest line features.
    
    Parameters
    ----------
    points : gpd.GeoDataFrame
        Point features
    lines : gpd.GeoDataFrame
        Line features
    max_distance : float, optional
        Maximum snap distance
        
    Returns
    -------
    gpd.GeoDataFrame
        Points with snapped coordinates
    """
    from shapely.ops import nearest_points
    
    # Ensure same CRS
    if points.crs != lines.crs:
        lines = lines.to_crs(points.crs)
    
    result = points.copy()
    snapped_geoms = []
    
    for point in points.geometry:
        # Find nearest line
        min_dist = float('inf')
        nearest_point_on_line = None
        
        for line in lines.geometry:
            nearest = nearest_points(point, line)[1]
            dist = point.distance(nearest)
            
            if dist < min_dist:
                min_dist = dist
                nearest_point_on_line = nearest
        
        # Snap if within max_distance
        if max_distance is None or min_dist <= max_distance:
            snapped_geoms.append(nearest_point_on_line)
        else:
            snapped_geoms.append(point)
    
    result.geometry = snapped_geoms
    
    return result


def get_raster_extent_as_polygon(
    raster_path: str,
    crs: Optional[str] = None
) -> gpd.GeoDataFrame:
    """
    Get raster extent as a polygon GeoDataFrame.
    
    Parameters
    ----------
    raster_path : str
        Path to raster file
    crs : str, optional
        Target CRS (if different from raster CRS)
        
    Returns
    -------
    gpd.GeoDataFrame
        GeoDataFrame with extent polygon
    """
    with rasterio.open(raster_path) as src:
        bounds = src.bounds
        raster_crs = src.crs
        
        extent_poly = box(bounds.left, bounds.bottom, bounds.right, bounds.top)
        gdf = gpd.GeoDataFrame([1], geometry=[extent_poly], crs=raster_crs)
        
        if crs and crs != raster_crs:
            gdf = gdf.to_crs(crs)
    
    return gdf
