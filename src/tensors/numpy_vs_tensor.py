"""
Comparison and conversion between NumPy arrays and tensors.
"""

import numpy as np
from typing import Union


def numpy_to_torch(array: np.ndarray):
    """
    Convert NumPy array to PyTorch tensor.
    
    Parameters
    ----------
    array : np.ndarray
        NumPy array
        
    Returns
    -------
    torch.Tensor
        PyTorch tensor
    """
    try:
        import torch
    except ImportError:
        raise ImportError("PyTorch is not installed. Install with: pip install torch")
    
    return torch.from_numpy(array)


def torch_to_numpy(tensor) -> np.ndarray:
    """
    Convert PyTorch tensor to NumPy array.
    
    Parameters
    ----------
    tensor : torch.Tensor
        PyTorch tensor
        
    Returns
    -------
    np.ndarray
        NumPy array
    """
    return tensor.detach().cpu().numpy()


def numpy_to_tensorflow(array: np.ndarray):
    """
    Convert NumPy array to TensorFlow tensor.
    
    Parameters
    ----------
    array : np.ndarray
        NumPy array
        
    Returns
    -------
    tf.Tensor
        TensorFlow tensor
    """
    try:
        import tensorflow as tf
    except ImportError:
        raise ImportError("TensorFlow is not installed. Install with: pip install tensorflow")
    
    return tf.convert_to_tensor(array)


def tensorflow_to_numpy(tensor) -> np.ndarray:
    """
    Convert TensorFlow tensor to NumPy array.
    
    Parameters
    ----------
    tensor : tf.Tensor
        TensorFlow tensor
        
    Returns
    -------
    np.ndarray
        NumPy array
    """
    return tensor.numpy()


def benchmark_operations(array: np.ndarray, operation: str = 'matrix_multiply') -> dict:
    """
    Benchmark the same operation across NumPy, PyTorch, and TensorFlow.
    
    Parameters
    ----------
    array : np.ndarray
        Input array
    operation : str
        Operation to benchmark: 'matrix_multiply', 'sum', 'mean', 'std'
        
    Returns
    -------
    dict
        Timing results for each framework
    """
    import time
    
    results = {}
    
    # NumPy
    start = time.time()
    if operation == 'matrix_multiply':
        _ = np.matmul(array, array.T)
    elif operation == 'sum':
        _ = np.sum(array)
    elif operation == 'mean':
        _ = np.mean(array)
    elif operation == 'std':
        _ = np.std(array)
    results['numpy'] = time.time() - start
    
    # PyTorch
    try:
        import torch
        tensor_torch = torch.from_numpy(array)
        if torch.cuda.is_available():
            tensor_torch = tensor_torch.cuda()
        
        start = time.time()
        if operation == 'matrix_multiply':
            _ = torch.matmul(tensor_torch, tensor_torch.T)
        elif operation == 'sum':
            _ = torch.sum(tensor_torch)
        elif operation == 'mean':
            _ = torch.mean(tensor_torch.float())
        elif operation == 'std':
            _ = torch.std(tensor_torch.float())
        
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        
        results['pytorch'] = time.time() - start
    except ImportError:
        results['pytorch'] = None
    
    # TensorFlow
    try:
        import tensorflow as tf
        tensor_tf = tf.convert_to_tensor(array)
        
        start = time.time()
        if operation == 'matrix_multiply':
            _ = tf.matmul(tensor_tf, tf.transpose(tensor_tf))
        elif operation == 'sum':
            _ = tf.reduce_sum(tensor_tf)
        elif operation == 'mean':
            _ = tf.reduce_mean(tf.cast(tensor_tf, tf.float32))
        elif operation == 'std':
            _ = tf.math.reduce_std(tf.cast(tensor_tf, tf.float32))
        
        results['tensorflow'] = time.time() - start
    except ImportError:
        results['tensorflow'] = None
    
    return results


def compare_memory_usage(array: np.ndarray) -> dict:
    """
    Compare memory usage across different tensor representations.
    
    Parameters
    ----------
    array : np.ndarray
        Input array
        
    Returns
    -------
    dict
        Memory usage in bytes for each framework
    """
    results = {}
    
    # NumPy
    results['numpy'] = array.nbytes
    
    # PyTorch
    try:
        import torch
        tensor = torch.from_numpy(array)
        results['pytorch'] = tensor.element_size() * tensor.nelement()
    except ImportError:
        results['pytorch'] = None
    
    # TensorFlow
    try:
        import tensorflow as tf
        tensor = tf.convert_to_tensor(array)
        results['tensorflow'] = tensor.numpy().nbytes
    except ImportError:
        results['tensorflow'] = None
    
    return results


def optimize_dtype(array: np.ndarray, target_dtype: str = 'float32') -> np.ndarray:
    """
    Optimize array dtype for tensor operations.
    
    Parameters
    ----------
    array : np.ndarray
        Input array
    target_dtype : str
        Target data type
        
    Returns
    -------
    np.ndarray
        Array with optimized dtype
    """
    return array.astype(target_dtype)
