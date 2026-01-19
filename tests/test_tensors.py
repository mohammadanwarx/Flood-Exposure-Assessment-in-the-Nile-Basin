"""
Unit tests for tensor operations.
"""

import pytest
import numpy as np


def test_import_tensor_modules():
    """Test that tensor modules can be imported."""
    from src.tensors import numpy_vs_tensor
    assert numpy_vs_tensor is not None


def test_numpy_tensor_conversion():
    """Test conversion between NumPy and tensors."""
    from src.tensors.numpy_vs_tensor import numpy_to_torch, torch_to_numpy
    
    try:
        import torch
        
        # Test conversion
        arr = np.random.rand(3, 3)
        tensor = numpy_to_torch(arr)
        arr_back = torch_to_numpy(tensor)
        
        assert np.allclose(arr, arr_back)
    except ImportError:
        pytest.skip("PyTorch not installed")


def test_numpy_tensorflow_conversion():
    """Test conversion between NumPy and TensorFlow."""
    from src.tensors.numpy_vs_tensor import numpy_to_tensorflow, tensorflow_to_numpy
    
    try:
        import tensorflow as tf
        
        # Test conversion
        arr = np.random.rand(3, 3)
        tensor = numpy_to_tensorflow(arr)
        arr_back = tensorflow_to_numpy(tensor)
        
        assert np.allclose(arr, arr_back)
    except ImportError:
        pytest.skip("TensorFlow not installed")


if __name__ == '__main__':
    pytest.main([__file__])
