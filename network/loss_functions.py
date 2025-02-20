import numpy as np
from scipy.special import logsumexp
from abc import ABC, abstractmethod
from utils import validate_same_shape


class LossBaseFunction(ABC):
    """
    Base class for loss functions in a neural network.
    """
    __slots__ = ("_prediction", "_target", "input_grad")

    def __init__(self) -> None:
        """Pass"""
        pass
    
    @staticmethod
    def check_prediction_shape(n: int) -> None:
        """
        Checks if division by zero occurs, raising an exception if so.

        :param n: Batch size.
        :raises ValueError: If batch size is zero.
        """
        if n == 0:
            raise ValueError("MeanSquaredError: Division by zero (empty batch).")

    @abstractmethod
    def _output(self) -> float:
        """Each subclass must implement this method."""
        pass

    @abstractmethod
    def _input_grad(self) -> np.ndarray:
        """Each subclass must implement this method."""
        pass

    def forward(self, prediction: np.ndarray, target: np.ndarray) -> float:
        """
        Computes the loss value for predicted and actual values.

        :param prediction: Predicted values.
        :param target: True values.
        :return: Computed loss.
        """
        validate_same_shape(prediction, target)
        
        self._prediction = prediction
        self._target = target
        return self._output()

    def backward(self) -> np.ndarray:
        """
        Computes the gradient of the loss with respect to the input.

        :return: Computed gradient.
        """
        self.input_grad = self._input_grad()
        validate_same_shape(self._prediction, self.input_grad)
        return self.input_grad


class BinaryCrossEntropy(LossBaseFunction):
    """
    Binary Cross-Entropy loss function for binary classification.
    """
    __slots__ = ("_eps",)

    def __init__(self, eps: float = 1e-9) -> None:
        """
        Initializes Binary Cross-Entropy loss.

        :param eps: Small value to avoid log(0) (default: 1e-9).
        """
        super().__init__()
        self._eps = eps

    def _output(self) -> float:
        """
        Computes the Binary Cross-Entropy loss.
        """
        self._prediction = np.clip(self._prediction, self._eps, 1 - self._eps)
        
        loss = - (self._target * np.log(self._prediction) + (1 - self._target) * np.log(1 - self._prediction))
        return float(np.mean(loss))

    def _input_grad(self) -> np.ndarray:
        """
        Computes the gradient of the Binary Cross-Entropy loss.

        :return: Computed gradient.
        """
        return (self._prediction - self._target) / (self._prediction * (1 - self._prediction) * self._prediction.shape[0])


class CategoricalCrossEntropy(LossBaseFunction):
    """
    Cross-entropy loss with softmax activation.
    """
    __slots__ = ("_eps", "_softmax_preds")

    def __init__(self, eps: float = 1e-9) -> None:
        """
        Initializes the softmax cross-entropy loss.

        :param eps: Small value to avoid log(0) (default: 1e-9).
        """
        super().__init__()
        self._eps: float = eps
        self._softmax_preds: np.ndarray = np.array([])

    @staticmethod
    def softmax(x: np.ndarray, axis: int = 1) -> np.ndarray:
        """
        Stable softmax function for each sample in a batch.

        :param x: Input array.
        :param axis: Axis along which to apply softmax (default: 1).
        :return: Softmax-transformed array.
        """
        return np.exp(x - logsumexp(x, axis=axis, keepdims=True))

    def _output(self) -> float:
        """
        Computes the cross-entropy loss value.

        :return: Computed loss.
        """
        target: np.ndarray = (
            np.column_stack([self._target, 1 - self._target])
            if self._target.ndim == 1 or self._target.shape[1] == 1 else self._target
        )
        self._softmax_preds = self.softmax(self._prediction)
        self._softmax_preds = np.clip(self._softmax_preds, self._eps, 1 - self._eps)

        loss: np.ndarray = -np.sum(target * np.log(self._softmax_preds), axis=1)
        return np.mean(loss)

    def _input_grad(self) -> np.ndarray:
        """
        Computes the gradient of the loss with respect to the input.

        :return: Computed gradient.
        """
        return (self._softmax_preds - self._target) / self._prediction.shape[0]


class MeanSquaredError(LossBaseFunction):
    """
    Mean Squared Error (MSE) loss function.
    """
    __slots__ = ()

    def __init__(self) -> None:
        """Pass"""
        super().__init__()

    def _output(self) -> float:
        """
        Computes the Mean Squared Error (MSE) loss.
        """
        n: int = self._prediction.shape[0]
        self.check_prediction_shape(n)
        return float(np.sum(np.power(self._prediction - self._target, 2)) / n)

    def _input_grad(self) -> np.ndarray:
        """
        Computes the gradient of the loss with respect to the input.

        :return: Computed gradient.
        """
        n: int = self._prediction.shape[0]
        self.check_prediction_shape(n)
        return 2.0 * (self._prediction - self._target) / n


class MeanAbsoluteError(LossBaseFunction):
    """
    Mean Absolute Error (MAE) loss function.
    """
    __slots__ = ()

    def __init__(self) -> None:
        """Pass"""
        super().__init__()

    def _output(self) -> float:
        """
        Computes the Mean Absolute Error (MAE) loss.
        """
        n: int = self._prediction.shape[0]
        self.check_prediction_shape(n)

        return float(np.sum(np.abs(self._prediction - self._target)) / n)

    def _input_grad(self) -> np.ndarray:
        """
        Computes the gradient of the MAE loss with respect to the input.

        :return: Computed gradient.
        """
        n: int = self._prediction.shape[0]
        self.check_prediction_shape(n)
        
        return np.where(self._prediction > self._target, 1, -1) / n


class HuberLoss(LossBaseFunction):
    """
    Huber loss function, combining MSE and MAE.
    """
    __slots__ = ("_delta",)

    def __init__(self, delta: float = 1.0) -> None:
        """
        Initializes Huber loss.

        :param delta: Threshold between MAE and MSE behavior.
        """
        super().__init__()
        self._delta = delta

    def _output(self) -> float:
        """
        Computes the Huber loss.
        """
        n: int = self._prediction.shape[0]
        self.check_prediction_shape(n)

        diff = self._prediction - self._target
        abs_diff = np.abs(diff)

        quadratic = 0.5 * np.power(diff, 2)
        linear = self._delta * (abs_diff - 0.5 * self._delta)

        return float(np.mean(np.where(abs_diff <= self._delta, quadratic, linear)))

    def _input_grad(self) -> np.ndarray:
        """
        Computes the gradient of the Huber loss with respect to the input.

        :return: Computed gradient.
        """
        diff = self._prediction - self._target
        abs_diff = np.abs(diff)

        return np.where(abs_diff <= self._delta, diff, self._delta * np.sign(diff)) / self._prediction.shape[0]