import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

import numpy as np
import tensorflow as tf
from typing import Generator, Tuple

def load_mnist_data(num_classes: int = 10) -> Tuple[Tuple[np.ndarray, np.ndarray], Tuple[np.ndarray, np.ndarray]]:
    """
    Loads and preprocesses the MNIST dataset.

    :param num_classes: Number of output classes (default: 10).
    :return: A tuple containing training and test datasets in the form:
             ((X_train, y_train), (X_test, y_test)).
    """
    (train_images, train_labels), (test_images, test_labels) = tf.keras.datasets.mnist.load_data()

    train_images = train_images.reshape(-1, 28 * 28).astype(np.float32)
    test_images = test_images.reshape(-1, 28 * 28).astype(np.float32)
    
    std_pixel: float = np.std(train_images)
    mean_pixel: float = np.mean(train_images)

    train_images = (train_images - mean_pixel) / std_pixel
    test_images = (test_images - mean_pixel) / std_pixel

    train_labels_one_hot = np.eye(num_classes)[train_labels]
    test_labels_one_hot = np.eye(num_classes)[test_labels]

    return (train_images, train_labels_one_hot), (test_images, test_labels_one_hot)

def shuffle_data(X: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Shuffles the dataset randomly.

    :param X: Feature matrix.
    :param y: Target values.
    :return: Shuffled (X, y) pair.
    """
    permutation: np.ndarray = np.random.permutation(X.shape[0])
    return X[permutation], y[permutation]

def batch_data_generator(X: np.ndarray, y: np.ndarray, batch_size: int = 32) -> Generator[Tuple[np.ndarray, np.ndarray], None, None]:
    """
    Generates batches of data for training.

    :param X: Feature matrix.
    :param y: Target values.
    :param batch_size: Size of each batch (default is 32).
    :yield: Tuple of (X_batch, y_batch).
    """
    assert X.shape[0] == y.shape[0], f"Features and target must have the same number of rows, but got {X.shape[0]} and {y.shape[0]}."

    num_samples: int = X.shape[0]
    for i in range(0, num_samples, batch_size):
        yield X[i : i + batch_size], y[i : i + batch_size]