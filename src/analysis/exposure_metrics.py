"""
Flood exposure metrics and risk assessment calculations.
"""

import numpy as np
import pandas as pd
import geopandas as gpd
from typing import Union, Optional, Dict, List
from pathlib import Path


def calculate_exposure_index(
    flood_depth: np.ndarray,
    population_density: np.ndarray,
    asset_value: Optional[np.ndarray] = None,
    weights: Dict[str, float] = None
) -> np.ndarray:
    """
    Calculate a composite flood exposure index.
    
    Parameters
    ----------
    flood_depth : np.ndarray
        Flood depth raster
    population_density : np.ndarray
        Population density raster
    asset_value : np.ndarray, optional
        Asset value raster
    weights : dict, optional
        Weights for each component (default: equal weights)
        
    Returns
    -------
    np.ndarray
        Exposure index raster
    """
    if weights is None:
        if asset_value is not None:
            weights = {'depth': 0.33, 'population': 0.33, 'assets': 0.34}
        else:
            weights = {'depth': 0.5, 'population': 0.5, 'assets': 0.0}
    
    # Normalize inputs to 0-1 range
    depth_norm = (flood_depth - np.nanmin(flood_depth)) / (np.nanmax(flood_depth) - np.nanmin(flood_depth))
    pop_norm = (population_density - np.nanmin(population_density)) / (np.nanmax(population_density) - np.nanmin(population_density))
    
    # Calculate weighted index
    exposure_index = (
        weights['depth'] * depth_norm +
        weights['population'] * pop_norm
    )
    
    if asset_value is not None:
        asset_norm = (asset_value - np.nanmin(asset_value)) / (np.nanmax(asset_value) - np.nanmin(asset_value))
        exposure_index += weights['assets'] * asset_norm
    
    return exposure_index


def classify_exposure_risk(
    exposure_values: np.ndarray,
    thresholds: List[float] = [0.2, 0.4, 0.6, 0.8]
) -> np.ndarray:
    """
    Classify exposure values into risk categories.
    
    Parameters
    ----------
    exposure_values : np.ndarray
        Exposure values
    thresholds : list
        Threshold values for classification (creates len(thresholds)+1 classes)
        
    Returns
    -------
    np.ndarray
        Risk class array (0 = lowest, len(thresholds) = highest)
    """
    risk_classes = np.zeros_like(exposure_values, dtype=int)
    
    for i, threshold in enumerate(thresholds):
        risk_classes[exposure_values > threshold] = i + 1
    
    return risk_classes


def calculate_affected_population(
    flood_depth: np.ndarray,
    population: np.ndarray,
    depth_threshold: float = 0.5,
    pixel_area: float = 1.0
) -> Dict[str, float]:
    """
    Calculate population affected by flooding above a depth threshold.
    
    Parameters
    ----------
    flood_depth : np.ndarray
        Flood depth raster (in meters)
    population : np.ndarray
        Population count raster
    depth_threshold : float
        Minimum flood depth to consider (meters)
    pixel_area : float
        Area of each pixel (for density calculations)
        
    Returns
    -------
    dict
        Dictionary with affected population metrics
    """
    # Mask for flooded areas
    flooded_mask = flood_depth >= depth_threshold
    
    # Calculate affected population
    affected_pop = np.nansum(population[flooded_mask])
    total_pop = np.nansum(population)
    percent_affected = (affected_pop / total_pop * 100) if total_pop > 0 else 0
    
    # Calculate by depth categories
    categories = {
        'low': (0.5, 1.0),
        'medium': (1.0, 2.0),
        'high': (2.0, np.inf)
    }
    
    pop_by_category = {}
    for cat_name, (min_depth, max_depth) in categories.items():
        cat_mask = (flood_depth >= min_depth) & (flood_depth < max_depth)
        pop_by_category[cat_name] = np.nansum(population[cat_mask])
    
    return {
        'total_affected': affected_pop,
        'percent_affected': percent_affected,
        'population_by_depth': pop_by_category,
        'flooded_area_km2': np.sum(flooded_mask) * pixel_area / 1e6
    }


def calculate_economic_exposure(
    flood_depth: np.ndarray,
    asset_value: np.ndarray,
    depth_damage_curve: Optional[callable] = None
) -> Dict[str, float]:
    """
    Calculate economic exposure and potential damage.
    
    Parameters
    ----------
    flood_depth : np.ndarray
        Flood depth raster (in meters)
    asset_value : np.ndarray
        Asset value raster
    depth_damage_curve : callable, optional
        Function mapping depth to damage ratio (0-1)
        If None, uses a simple linear curve
        
    Returns
    -------
    dict
        Dictionary with economic exposure metrics
    """
    if depth_damage_curve is None:
        # Simple linear damage curve (0% at 0m, 100% at 3m+)
        def depth_damage_curve(depth):
            return np.clip(depth / 3.0, 0, 1)
    
    # Calculate damage ratio for each pixel
    damage_ratio = depth_damage_curve(flood_depth)
    
    # Calculate potential damage
    potential_damage = asset_value * damage_ratio
    
    total_exposure = np.nansum(asset_value[flood_depth > 0])
    total_damage = np.nansum(potential_damage)
    average_damage_ratio = np.nanmean(damage_ratio[flood_depth > 0])
    
    return {
        'total_exposure': total_exposure,
        'total_potential_damage': total_damage,
        'average_damage_ratio': average_damage_ratio,
        'percent_of_total_assets': (total_exposure / np.nansum(asset_value) * 100) if np.nansum(asset_value) > 0 else 0
    }


def calculate_vulnerability_index(
    social_vulnerability: np.ndarray,
    infrastructure_quality: np.ndarray,
    adaptive_capacity: np.ndarray,
    weights: Dict[str, float] = None
) -> np.ndarray:
    """
    Calculate a composite vulnerability index.
    
    Parameters
    ----------
    social_vulnerability : np.ndarray
        Social vulnerability indicator (0-1)
    infrastructure_quality : np.ndarray
        Infrastructure quality indicator (0-1, higher is better)
    adaptive_capacity : np.ndarray
        Adaptive capacity indicator (0-1, higher is better)
    weights : dict, optional
        Weights for each component
        
    Returns
    -------
    np.ndarray
        Vulnerability index (0-1, higher is more vulnerable)
    """
    if weights is None:
        weights = {
            'social': 0.4,
            'infrastructure': 0.3,
            'adaptive': 0.3
        }
    
    # Invert positive indicators so higher values = more vulnerable
    infrastructure_vuln = 1 - infrastructure_quality
    adaptive_vuln = 1 - adaptive_capacity
    
    vulnerability_index = (
        weights['social'] * social_vulnerability +
        weights['infrastructure'] * infrastructure_vuln +
        weights['adaptive'] * adaptive_vuln
    )
    
    return vulnerability_index


def calculate_flood_frequency_impact(
    flood_depths: List[np.ndarray],
    return_periods: List[float],
    population: np.ndarray
) -> pd.DataFrame:
    """
    Calculate impacts for multiple flood scenarios with different return periods.
    
    Parameters
    ----------
    flood_depths : list of np.ndarray
        List of flood depth rasters for different scenarios
    return_periods : list of float
        Return periods (years) for each scenario
    population : np.ndarray
        Population raster
        
    Returns
    -------
    pd.DataFrame
        DataFrame with impacts for each scenario
    """
    results = []
    
    for depth, rp in zip(flood_depths, return_periods):
        impact = calculate_affected_population(depth, population)
        impact['return_period'] = rp
        impact['annual_probability'] = 1 / rp
        results.append(impact)
    
    return pd.DataFrame(results)


def calculate_expected_annual_damage(
    flood_scenarios: List[np.ndarray],
    probabilities: List[float],
    asset_value: np.ndarray,
    depth_damage_curve: Optional[callable] = None
) -> float:
    """
    Calculate Expected Annual Damage (EAD) from multiple flood scenarios.
    
    Parameters
    ----------
    flood_scenarios : list of np.ndarray
        List of flood depth rasters
    probabilities : list of float
        Annual probabilities for each scenario
    asset_value : np.ndarray
        Asset value raster
    depth_damage_curve : callable, optional
        Depth-damage function
        
    Returns
    -------
    float
        Expected annual damage
    """
    if depth_damage_curve is None:
        def depth_damage_curve(depth):
            return np.clip(depth / 3.0, 0, 1)
    
    ead = 0.0
    
    for depth, prob in zip(flood_scenarios, probabilities):
        damage_ratio = depth_damage_curve(depth)
        scenario_damage = np.nansum(asset_value * damage_ratio)
        ead += scenario_damage * prob
    
    return ead
