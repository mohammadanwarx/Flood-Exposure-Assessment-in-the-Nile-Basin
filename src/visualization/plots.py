"""
Statistical plots and charts for analysis results.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Union, Optional, Tuple, List
from pathlib import Path


def plot_histogram(
    data: Union[np.ndarray, pd.Series],
    title: str = 'Histogram',
    xlabel: str = 'Value',
    ylabel: str = 'Frequency',
    bins: int = 50,
    figsize: Tuple[int, int] = (10, 6),
    save_path: Optional[Union[str, Path]] = None,
    dpi: int = 300
) -> plt.Figure:
    """
    Create a histogram plot.
    
    Parameters
    ----------
    data : np.ndarray or pd.Series
        Data to plot
    title : str
        Plot title
    xlabel : str
        X-axis label
    ylabel : str
        Y-axis label
    bins : int
        Number of bins
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
    fig, ax = plt.subplots(figsize=figsize)
    
    if isinstance(data, np.ndarray):
        data = data.flatten()
        data = data[~np.isnan(data)]
    
    ax.hist(data, bins=bins, edgecolor='black', alpha=0.7)
    
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=dpi, bbox_inches='tight')
    
    return fig


def plot_time_series(
    data: pd.DataFrame,
    x_col: str,
    y_cols: Union[str, List[str]],
    title: str = 'Time Series',
    xlabel: str = 'Time',
    ylabel: str = 'Value',
    figsize: Tuple[int, int] = (12, 6),
    save_path: Optional[Union[str, Path]] = None,
    dpi: int = 300
) -> plt.Figure:
    """
    Create a time series plot.
    
    Parameters
    ----------
    data : pd.DataFrame
        Data to plot
    x_col : str
        Column name for x-axis
    y_cols : str or list
        Column name(s) for y-axis
    title : str
        Plot title
    xlabel : str
        X-axis label
    ylabel : str
        Y-axis label
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
    fig, ax = plt.subplots(figsize=figsize)
    
    if isinstance(y_cols, str):
        y_cols = [y_cols]
    
    for col in y_cols:
        ax.plot(data[x_col], data[col], marker='o', label=col)
    
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.xticks(rotation=45)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=dpi, bbox_inches='tight')
    
    return fig


def plot_bar_chart(
    data: pd.DataFrame,
    x_col: str,
    y_col: str,
    title: str = 'Bar Chart',
    xlabel: str = 'Category',
    ylabel: str = 'Value',
    figsize: Tuple[int, int] = (10, 6),
    save_path: Optional[Union[str, Path]] = None,
    dpi: int = 300
) -> plt.Figure:
    """
    Create a bar chart.
    
    Parameters
    ----------
    data : pd.DataFrame
        Data to plot
    x_col : str
        Column for x-axis (categories)
    y_col : str
        Column for y-axis (values)
    title : str
        Plot title
    xlabel : str
        X-axis label
    ylabel : str
        Y-axis label
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
    fig, ax = plt.subplots(figsize=figsize)
    
    ax.bar(data[x_col], data[y_col], edgecolor='black', alpha=0.7)
    
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=dpi, bbox_inches='tight')
    
    return fig


def plot_scatter(
    data: pd.DataFrame,
    x_col: str,
    y_col: str,
    hue_col: Optional[str] = None,
    title: str = 'Scatter Plot',
    xlabel: str = 'X',
    ylabel: str = 'Y',
    figsize: Tuple[int, int] = (10, 8),
    save_path: Optional[Union[str, Path]] = None,
    dpi: int = 300
) -> plt.Figure:
    """
    Create a scatter plot.
    
    Parameters
    ----------
    data : pd.DataFrame
        Data to plot
    x_col : str
        Column for x-axis
    y_col : str
        Column for y-axis
    hue_col : str, optional
        Column for color coding
    title : str
        Plot title
    xlabel : str
        X-axis label
    ylabel : str
        Y-axis label
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
    fig, ax = plt.subplots(figsize=figsize)
    
    if hue_col:
        sns.scatterplot(data=data, x=x_col, y=y_col, hue=hue_col, ax=ax)
    else:
        ax.scatter(data[x_col], data[y_col], alpha=0.6)
    
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=dpi, bbox_inches='tight')
    
    return fig


def plot_heatmap(
    data: Union[pd.DataFrame, np.ndarray],
    title: str = 'Heatmap',
    xlabel: str = 'X',
    ylabel: str = 'Y',
    cmap: str = 'YlOrRd',
    figsize: Tuple[int, int] = (10, 8),
    save_path: Optional[Union[str, Path]] = None,
    dpi: int = 300
) -> plt.Figure:
    """
    Create a heatmap.
    
    Parameters
    ----------
    data : pd.DataFrame or np.ndarray
        Data to plot
    title : str
        Plot title
    xlabel : str
        X-axis label
    ylabel : str
        Y-axis label
    cmap : str
        Colormap
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
    fig, ax = plt.subplots(figsize=figsize)
    
    sns.heatmap(data, cmap=cmap, annot=False, ax=ax, cbar_kws={'label': 'Value'})
    
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=dpi, bbox_inches='tight')
    
    return fig


def plot_exposure_by_category(
    exposure_data: pd.DataFrame,
    category_col: str,
    value_col: str,
    title: str = 'Exposure by Category',
    figsize: Tuple[int, int] = (10, 6),
    save_path: Optional[Union[str, Path]] = None,
    dpi: int = 300
) -> plt.Figure:
    """
    Create a bar chart for exposure metrics by category.
    
    Parameters
    ----------
    exposure_data : pd.DataFrame
        Exposure data
    category_col : str
        Column with categories
    value_col : str
        Column with values
    title : str
        Plot title
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
    fig, ax = plt.subplots(figsize=figsize)
    
    colors = plt.cm.Reds(np.linspace(0.3, 0.9, len(exposure_data)))
    
    ax.bar(exposure_data[category_col], exposure_data[value_col], color=colors, edgecolor='black')
    
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.set_xlabel('Risk Category')
    ax.set_ylabel('Exposed Population / Assets')
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=dpi, bbox_inches='tight')
    
    return fig


def plot_comparison(
    data_list: List[pd.DataFrame],
    labels: List[str],
    value_col: str,
    title: str = 'Comparison Plot',
    ylabel: str = 'Value',
    figsize: Tuple[int, int] = (12, 6),
    save_path: Optional[Union[str, Path]] = None,
    dpi: int = 300
) -> plt.Figure:
    """
    Create a comparison plot for multiple datasets.
    
    Parameters
    ----------
    data_list : list of pd.DataFrame
        List of dataframes to compare
    labels : list of str
        Labels for each dataset
    value_col : str
        Column to plot
    title : str
        Plot title
    ylabel : str
        Y-axis label
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
    fig, ax = plt.subplots(figsize=figsize)
    
    for data, label in zip(data_list, labels):
        ax.plot(data.index, data[value_col], marker='o', label=label, linewidth=2)
    
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.set_ylabel(ylabel)
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=dpi, bbox_inches='tight')
    
    return fig
