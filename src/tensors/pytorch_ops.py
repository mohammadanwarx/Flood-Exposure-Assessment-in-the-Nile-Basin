"""
PyTorch operations for geospatial raster processing.
"""

import numpy as np
from typing import Tuple, Optional


def setup_device():
    """
    Set up PyTorch device (GPU if available).
    
    Returns
    -------
    torch.device
        Device for tensor operations
    """
    try:
        import torch
    except ImportError:
        raise ImportError("PyTorch is not installed. Install with: pip install torch")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    return device


def raster_convolution(
    raster: np.ndarray,
    kernel: np.ndarray,
    padding: str = 'same'
) -> np.ndarray:
    """
    Apply convolution to a raster using PyTorch.
    
    Parameters
    ----------
    raster : np.ndarray
        Input raster (H, W) or (C, H, W)
    kernel : np.ndarray
        Convolution kernel
    padding : str
        Padding mode: 'same', 'valid'
        
    Returns
    -------
    np.ndarray
        Convolved raster
    """
    import torch
    import torch.nn.functional as F
    
    device = setup_device()
    
    # Reshape inputs for PyTorch
    if raster.ndim == 2:
        raster = raster[np.newaxis, np.newaxis, :, :]  # (1, 1, H, W)
    elif raster.ndim == 3:
        raster = raster[np.newaxis, :, :, :]  # (1, C, H, W)
    
    kernel = kernel[np.newaxis, np.newaxis, :, :]  # (1, 1, kH, kW)
    
    # Convert to tensors
    raster_tensor = torch.from_numpy(raster).float().to(device)
    kernel_tensor = torch.from_numpy(kernel).float().to(device)
    
    # Apply convolution
    if padding == 'same':
        pad_h = kernel.shape[2] // 2
        pad_w = kernel.shape[3] // 2
        output = F.conv2d(raster_tensor, kernel_tensor, padding=(pad_h, pad_w))
    else:
        output = F.conv2d(raster_tensor, kernel_tensor)
    
    # Convert back to numpy
    return output.cpu().numpy().squeeze()


def batch_normalize_raster(
    raster: np.ndarray,
    eps: float = 1e-5
) -> np.ndarray:
    """
    Apply batch normalization to a raster.
    
    Parameters
    ----------
    raster : np.ndarray
        Input raster
    eps : float
        Small value for numerical stability
        
    Returns
    -------
    np.ndarray
        Normalized raster
    """
    import torch
    import torch.nn as nn
    
    device = setup_device()
    
    # Reshape for batch norm
    if raster.ndim == 2:
        raster = raster[np.newaxis, np.newaxis, :, :]
    
    tensor = torch.from_numpy(raster).float().to(device)
    
    # Apply batch normalization
    bn = nn.BatchNorm2d(tensor.shape[1], eps=eps).to(device)
    bn.eval()
    
    normalized = bn(tensor)
    
    return normalized.cpu().numpy().squeeze()


def calculate_spatial_gradients(
    raster: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Calculate spatial gradients using PyTorch.
    
    Parameters
    ----------
    raster : np.ndarray
        Input raster
        
    Returns
    -------
    gradient_x : np.ndarray
        Gradient in x direction
    gradient_y : np.ndarray
        Gradient in y direction
    """
    import torch
    
    device = setup_device()
    
    # Sobel kernels for gradient calculation
    sobel_x = np.array([[-1, 0, 1],
                        [-2, 0, 2],
                        [-1, 0, 1]], dtype=np.float32)
    
    sobel_y = np.array([[-1, -2, -1],
                        [ 0,  0,  0],
                        [ 1,  2,  1]], dtype=np.float32)
    
    # Calculate gradients
    grad_x = raster_convolution(raster, sobel_x)
    grad_y = raster_convolution(raster, sobel_y)
    
    return grad_x, grad_y


def focal_statistics_torch(
    raster: np.ndarray,
    window_size: int = 3,
    stat: str = 'mean'
) -> np.ndarray:
    """
    Calculate focal statistics using PyTorch pooling operations.
    
    Parameters
    ----------
    raster : np.ndarray
        Input raster
    window_size : int
        Size of the focal window
    stat : str
        Statistic to calculate: 'mean', 'max', 'min'
        
    Returns
    -------
    np.ndarray
        Raster with focal statistics
    """
    import torch
    import torch.nn.functional as F
    
    device = setup_device()
    
    # Reshape input
    if raster.ndim == 2:
        raster = raster[np.newaxis, np.newaxis, :, :]
    
    tensor = torch.from_numpy(raster).float().to(device)
    
    # Calculate padding to maintain size
    padding = window_size // 2
    
    # Apply pooling operation
    if stat == 'mean':
        result = F.avg_pool2d(tensor, window_size, stride=1, padding=padding)
    elif stat == 'max':
        result = F.max_pool2d(tensor, window_size, stride=1, padding=padding)
    elif stat == 'min':
        result = -F.max_pool2d(-tensor, window_size, stride=1, padding=padding)
    else:
        raise ValueError(f"Unknown statistic: {stat}")
    
    return result.cpu().numpy().squeeze()


def raster_upsampling(
    raster: np.ndarray,
    scale_factor: float = 2.0,
    mode: str = 'bilinear'
) -> np.ndarray:
    """
    Upsample a raster using PyTorch interpolation.
    
    Parameters
    ----------
    raster : np.ndarray
        Input raster
    scale_factor : float
        Upsampling factor
    mode : str
        Interpolation mode: 'bilinear', 'bicubic', 'nearest'
        
    Returns
    -------
    np.ndarray
        Upsampled raster
    """
    import torch
    import torch.nn.functional as F
    
    device = setup_device()
    
    # Reshape input
    if raster.ndim == 2:
        raster = raster[np.newaxis, np.newaxis, :, :]
    
    tensor = torch.from_numpy(raster).float().to(device)
    
    # Upsample
    upsampled = F.interpolate(
        tensor,
        scale_factor=scale_factor,
        mode=mode,
        align_corners=False if mode != 'nearest' else None
    )
    
    return upsampled.cpu().numpy().squeeze()


def batch_process_rasters(
    rasters: list,
    operation: callable,
    batch_size: int = 4
) -> list:
    """
    Process multiple rasters in batches using PyTorch.
    
    Parameters
    ----------
    rasters : list
        List of raster arrays
    operation : callable
        Operation to apply to each batch
    batch_size : int
        Number of rasters per batch
        
    Returns
    -------
    list
        List of processed rasters
    """
    import torch
    
    device = setup_device()
    results = []
    
    for i in range(0, len(rasters), batch_size):
        batch = rasters[i:i + batch_size]
        
        # Stack into batch
        batch_array = np.stack(batch)
        if batch_array.ndim == 3:
            batch_array = batch_array[:, np.newaxis, :, :]
        
        # Convert to tensor
        batch_tensor = torch.from_numpy(batch_array).float().to(device)
        
        # Apply operation
        processed = operation(batch_tensor)
        
        # Convert back and split
        processed_np = processed.cpu().numpy()
        for j in range(processed_np.shape[0]):
            results.append(processed_np[j].squeeze())
    
    return results
