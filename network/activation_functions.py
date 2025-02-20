import numpy as np

from .operations import OperationBase


class ReLU(OperationBase):
    """
    ReLU activation function.
    """
    __slots__ = ()  

    def __init__(self) -> None:
        """Initializes the activation function."""
        super().__init__()

    def _compute_output(self) -> np.ndarray:
        """
        Computes the activation output.

        :return: The ReLU transformation of the input.
        """
        return np.maximum(0.0, self._input)

    def _input_gradient(self, output_grad: np.ndarray) -> np.ndarray:
        """
        Computes the gradient of the activation function.

        :param output_grad: Gradient of the output.
        :return: Gradient of ReLU, 1 for positive inputs, 0 otherwise.
        """
        return (self._input > 0) * output_grad
    
    
class ELU(OperationBase):
    """
    Exponential Linear Unit (ELU) activation function.
    """
    __slots__ = ("_alpha",)

    def __init__(self, alpha: float = 1.0) -> None:
        """
        Initializes the ELU activation function.

        :param alpha: Scaling factor for negative inputs (default: 1.0).
        """
        super().__init__()
        self._alpha = alpha

    def _compute_output(self) -> np.ndarray:
        """
        Computes the activation output.

        :return: The ELU transformation of the input.
        """
        return np.where(self._input > 0, self._input, self._alpha * (np.exp(self._input) - 1))

    def _input_gradient(self, output_grad: np.ndarray) -> np.ndarray:
        """
        Computes the gradient of the activation function.

        :param output_grad: Gradient of the output.
        :return: Gradient of the ELU function.
        """
        elu_grad = np.where(self._input > 0, 1, self._compute_output() + self._alpha)
        return output_grad * elu_grad
    
    @property
    def alpha(self) -> float:
        """
        Returns the alpha parameter.
        
        :return: The alpha value.
        """
        return self._alpha


class LeakyReLU(OperationBase):
    """
    Leaky Rectified Linear Unit (Leaky ReLU) activation function.
    """
    __slots__ = ("_alpha",)

    def __init__(self, alpha: float = 0.1) -> None:
        """
        Initializes the Leaky ReLU activation function.

        :param alpha: Slope for negative inputs (default: 0.1).
        """
        super().__init__()
        self._alpha = alpha

    def _compute_output(self) -> np.ndarray:
        """
        Computes the activation output.

        :return: The Leaky ReLU transformation of the input.
        """
        return np.where(self._input > 0, self._input, self._alpha * self._input)

    def _input_gradient(self, output_grad: np.ndarray) -> np.ndarray:
        """
        Computes the gradient of the activation function.

        :param output_grad: Gradient of the output.
        :return: Gradient of the Leaky ReLU function.
        """
        leaky_relu_grad = np.where(self._input > 0, 1, self._alpha)
        return output_grad * leaky_relu_grad
    
    @property
    def alpha(self) -> float:
        """
        Returns the alpha parameter.

        :return: The alpha value.
        """
        return self._alpha


class Sigmoid(OperationBase):
    """
    Sigmoid activation function.
    """
    __slots__ = ()  

    def __init__(self) -> None:
        """Initializes the activation function."""
        super().__init__()

    def _compute_output(self) -> np.ndarray:
        """
        Computes the activation output.

        :return: The sigmoid function applied to the input.
        """
        return 1.0 / (1.0 + np.exp(-self._input))

    def _input_gradient(self, output_grad: np.ndarray) -> np.ndarray:
        """
        Computes the gradient of the activation function.

        :param output_grad: Gradient of the output.
        :return: Gradient of the sigmoid function.
        """
        sigmoid_derivative = self._output * (1.0 - self._output)
        return sigmoid_derivative * output_grad


class Tanh(OperationBase):
    """
    Hyperbolic tangent (Tanh) activation function.
    """
    __slots__ = ()  

    def __init__(self) -> None:
        """Initializes the activation function."""
        super().__init__()

    def _compute_output(self) -> np.ndarray:
        """
        Computes the activation output.

        :return: The tanh transformation of the input.
        """
        return np.tanh(self._input)

    def _input_gradient(self, output_grad: np.ndarray) -> np.ndarray:
        """
        Computes the gradient of the activation function.

        :param output_grad: Gradient of the output.
        :return: Gradient of the tanh function.
        """
        tanh_out = np.tanh(self._input)
        tanh_out = np.clip(tanh_out, -1 + 1e-6, 1 - 1e-6)
        return output_grad * (1 - tanh_out**2)
    

class Softmax(OperationBase):
    """
    Softmax activation function.
    """
    __slots__ = ("_softmax_output",)  

    def __init__(self) -> None:
        """Initializes the activation function."""
        super().__init__()

    def _compute_output(self) -> np.ndarray:
        """
        Computes the activation output.

        :return: The softmax-transformed input.
        """
        exps = np.exp(self._input - np.max(self._input, axis=1, keepdims=True))
        self._softmax_output = exps / np.sum(exps, axis=1, keepdims=True)
        return self._softmax_output

    def _input_gradient(self, output_grad: np.ndarray) -> np.ndarray:
        """
        Computes the gradient of the activation function.

        :param output_grad: Gradient of the output.
        :return: Gradient of the softmax function.
        """
        batch_size = output_grad.shape[0]
        input_grad = np.zeros_like(output_grad)

        for i in range(batch_size):
            jacobian_matrix = np.diag(self._softmax_output[i]) - np.outer(self._softmax_output[i], self._softmax_output[i])
            input_grad[i] = np.dot(output_grad[i], jacobian_matrix)

        return input_grad