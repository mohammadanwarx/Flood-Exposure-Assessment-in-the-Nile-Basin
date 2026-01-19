"""
TensorFlow operations for geospatial raster processing.
"""

import numpy as np
from typing import Tuple, Optional


def setup_tf():
    """
    Set up TensorFlow environment.
    
    Returns
    -------
    tf
        TensorFlow module
    """
    try:
        import tensorflow as tf
    except ImportError:
        raise ImportError("TensorFlow is not installed. Install with: pip install tensorflow")
    
    # Check for GPU
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        print(f"GPU available: {len(gpus)} device(s)")
    else:
        print("Running on CPU")
    
    return tf


def raster_convolution_tf(
    raster: np.ndarray,
    kernel: np.ndarray,
    padding: str = 'SAME'
) -> np.ndarray:
    """
    Apply convolution to a raster using TensorFlow.
    
    Parameters
    ----------
    raster : np.ndarray
        Input raster (H, W) or (C, H, W)
    kernel : np.ndarray
        Convolution kernel
    padding : str
        Padding mode: 'SAME', 'VALID'
        
    Returns
    -------
    np.ndarray
        Convolved raster
    """
    tf = setup_tf()
    
    # Reshape inputs for TensorFlow
    if raster.ndim == 2:
        raster = raster[np.newaxis, :, :, np.newaxis]  # (1, H, W, 1)
    elif raster.ndim == 3:
        raster = np.transpose(raster, (1, 2, 0))  # (H, W, C)
        raster = raster[np.newaxis, :, :, :]  # (1, H, W, C)
    
    kernel = kernel[:, :, np.newaxis, np.newaxis]  # (kH, kW, 1, 1)
    
    # Convert to tensors
    raster_tensor = tf.convert_to_tensor(raster, dtype=tf.float32)
    kernel_tensor = tf.convert_to_tensor(kernel, dtype=tf.float32)
    
    # Apply convolution
    output = tf.nn.conv2d(raster_tensor, kernel_tensor, strides=[1, 1, 1, 1], padding=padding)
    
    # Convert back to numpy
    return output.numpy().squeeze()


def normalize_raster_tf(
    raster: np.ndarray,
    method: str = 'minmax'
) -> np.ndarray:
    """
    Normalize a raster using TensorFlow operations.
    
    Parameters
    ----------
    raster : np.ndarray
        Input raster
    method : str
        Normalization method: 'minmax', 'zscore'
        
    Returns
    -------
    np.ndarray
        Normalized raster
    """
    tf = setup_tf()
    
    tensor = tf.convert_to_tensor(raster, dtype=tf.float32)
    
    if method == 'minmax':
        min_val = tf.reduce_min(tensor)
        max_val = tf.reduce_max(tensor)
        normalized = (tensor - min_val) / (max_val - min_val)
    elif method == 'zscore':
        mean = tf.reduce_mean(tensor)
        std = tf.math.reduce_std(tensor)
        normalized = (tensor - mean) / std
    else:
        raise ValueError(f"Unknown method: {method}")
    
    return normalized.numpy()


def calculate_gradients_tf(
    raster: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Calculate spatial gradients using TensorFlow.
    
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
    tf = setup_tf()
    
    # Sobel kernels
    sobel_x = np.array([[-1, 0, 1],
                        [-2, 0, 2],
                        [-1, 0, 1]], dtype=np.float32)
    
    sobel_y = np.array([[-1, -2, -1],
                        [ 0,  0,  0],
                        [ 1,  2,  1]], dtype=np.float32)
    
    # Calculate gradients
    grad_x = raster_convolution_tf(raster, sobel_x)
    grad_y = raster_convolution_tf(raster, sobel_y)
    
    return grad_x, grad_y


def apply_activation_function(
    raster: np.ndarray,
    activation: str = 'relu'
) -> np.ndarray:
    """
    Apply activation function to raster values.
    
    Parameters
    ----------
    raster : np.ndarray
        Input raster
    activation : str
        Activation function: 'relu', 'sigmoid', 'tanh', 'softmax'
        
    Returns
    -------
    np.ndarray
        Raster with activation applied
    """
    tf = setup_tf()
    
    tensor = tf.convert_to_tensor(raster, dtype=tf.float32)
    
    if activation == 'relu':
        result = tf.nn.relu(tensor)
    elif activation == 'sigmoid':
        result = tf.nn.sigmoid(tensor)
    elif activation == 'tanh':
        result = tf.nn.tanh(tensor)
    elif activation == 'softmax':
        result = tf.nn.softmax(tensor)
    else:
        raise ValueError(f"Unknown activation: {activation}")
    
    return result.numpy()


def focal_statistics_tf(
    raster: np.ndarray,
    window_size: int = 3,
    stat: str = 'mean'
) -> np.ndarray:
    """
    Calculate focal statistics using TensorFlow pooling operations.
    
    Parameters
    ----------
    raster : np.ndarray
        Input raster
    window_size : int
        Size of the focal window
    stat : str
        Statistic to calculate: 'mean', 'max'
        
    Returns
    -------
    np.ndarray
        Raster with focal statistics
    """
    tf = setup_tf()
    
    # Reshape input
    if raster.ndim == 2:
        raster = raster[np.newaxis, :, :, np.newaxis]
    
    tensor = tf.convert_to_tensor(raster, dtype=tf.float32)
    
    # Apply pooling operation
    if stat == 'mean':
        result = tf.nn.avg_pool2d(
            tensor,
            ksize=window_size,
            strides=1,
            padding='SAME'
        )
    elif stat == 'max':
        result = tf.nn.max_pool2d(
            tensor,
            ksize=window_size,
            strides=1,
            padding='SAME'
        )
    else:
        raise ValueError(f"Unknown statistic: {stat}")
    
    return result.numpy().squeeze()


def raster_resize_tf(
    raster: np.ndarray,
    new_shape: Tuple[int, int],
    method: str = 'bilinear'
) -> np.ndarray:
    """
    Resize a raster using TensorFlow.
    
    Parameters
    ----------
    raster : np.ndarray
        Input raster
    new_shape : tuple
        New shape (height, width)
    method : str
        Interpolation method: 'bilinear', 'bicubic', 'nearest'
        
    Returns
    -------
    np.ndarray
        Resized raster
    """
    tf = setup_tf()
    
    # Reshape input
    if raster.ndim == 2:
        raster = raster[np.newaxis, :, :, np.newaxis]
    
    tensor = tf.convert_to_tensor(raster, dtype=tf.float32)
    
    # Resize
    if method == 'bilinear':
        resized = tf.image.resize(tensor, new_shape, method='bilinear')
    elif method == 'bicubic':
        resized = tf.image.resize(tensor, new_shape, method='bicubic')
    elif method == 'nearest':
        resized = tf.image.resize(tensor, new_shape, method='nearest')
    else:
        raise ValueError(f"Unknown method: {method}")
    
    return resized.numpy().squeeze()


def batch_process_rasters_tf(
    rasters: list,
    operation: callable,
    batch_size: int = 4
) -> list:
    """
    Process multiple rasters in batches using TensorFlow.
    
    Parameters
    ----------
    rasters : list
        List of raster arrays
    operation : callable
        TensorFlow operation to apply to each batch
    batch_size : int
        Number of rasters per batch
        
    Returns
    -------
    list
        List of processed rasters
    """
    tf = setup_tf()
    results = []
    
    for i in range(0, len(rasters), batch_size):
        batch = rasters[i:i + batch_size]
        
        # Stack into batch
        batch_array = np.stack(batch)
        if batch_array.ndim == 3:
            batch_array = batch_array[:, :, :, np.newaxis]
        
        # Convert to tensor
        batch_tensor = tf.convert_to_tensor(batch_array, dtype=tf.float32)
        
        # Apply operation
        processed = operation(batch_tensor)
        
        # Convert back and split
        processed_np = processed.numpy()
        for j in range(processed_np.shape[0]):
            results.append(processed_np[j].squeeze())
    
    return results


def create_distance_transform_tf(
    binary_mask: np.ndarray
) -> np.ndarray:
    """
    Create a distance transform from a binary mask using TensorFlow.
    
    Parameters
    ----------
    binary_mask : np.ndarray
        Binary mask (0s and 1s)
        
    Returns
    -------
    np.ndarray
        Distance transform
    """
    tf = setup_tf()
    
    # Convert to tensor
    mask = tf.convert_to_tensor(binary_mask, dtype=tf.float32)
    
    # Simple approximation using dilation iterations
    # (True distance transform would require scipy or custom implementation)
    distance = tf.identity(mask)
    kernel = tf.ones((3, 3, 1, 1))
    
    for i in range(10):
        dilated = tf.nn.conv2d(
            distance[tf.newaxis, :, :, tf.newaxis],
            kernel,
            strides=[1, 1, 1, 1],
            padding='SAME'
        )
        distance = tf.where(dilated > 0, distance + 1, distance)
    
    return distance.numpy()
