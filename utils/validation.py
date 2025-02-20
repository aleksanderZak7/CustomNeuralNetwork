import numpy as np

def validate_same_shape(array1: np.ndarray, array2: np.ndarray) -> None:
    """
    Ensures that two NumPy arrays have the same shape.

    :param array1: First NumPy array.
    :param array2: Second NumPy array.
    :raises AssertionError: If the shapes are different.
    """
    assert array1.shape == array2.shape, f"Arrays must have the same shape, but got {array1.shape} and {array2.shape}."