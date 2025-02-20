import numpy as np
from abc import ABC, abstractmethod
from utils import validate_same_shape


class OperationBase(ABC):
    """
    Base class for all operations in the neural network.
    """
    __slots__ = ("_input", "_output")

    def __init__(self) -> None:
        """Empty constructor for the abstract base class."""
        pass

    @abstractmethod
    def _compute_output(self) -> np.ndarray:
        """
        Each operation must implement this method to compute its output.

        :return: Output of the operation.
        """
        pass

    @abstractmethod
    def _input_gradient(self, output_grad: np.ndarray) -> np.ndarray:
        """
        Each operation must implement this method to compute the gradient with respect to the input.

        :param output_grad: Gradient of the output.
        :return: Gradient of the input.
        """
        pass

    def forward(self, input: np.ndarray) -> np.ndarray:
        """
        Stores the input and computes the output.

        :param input: Input array.
        :return: Computed output.
        """
        self._input = input
        self._output = self._compute_output()
        return self._output

    def backward(self, output_grad: np.ndarray) -> np.ndarray:
        """
        Validates gradient shapes and computes the input gradient.

        :param output_grad: Gradient of the output.
        :return: Computed input gradient.
        """
        validate_same_shape(self._output, output_grad)
        input_grad: np.ndarray = self._input_gradient(output_grad)
        validate_same_shape(input_grad, self._input)
        return input_grad


class ParamOpBase(OperationBase):
    """
    Base class for operations with trainable parameters.
    """
    __slots__ = ("_param", "_param_grad")

    def __init__(self, param: np.ndarray) -> None:
        """
        Initializes the operation with trainable parameters.

        :param param: Parameter array.
        """
        super().__init__()
        self._param: np.ndarray = param
        self._param_grad: np.ndarray = np.zeros_like(param)
    
    @abstractmethod
    def _compute_param_gradient(self, output_grad: np.ndarray) -> np.ndarray:
        """
        Each subclass must implement this method to compute the gradient with respect to the parameter.

        :param output_grad: Gradient of the output.
        :return: Gradient of the parameter.
        """
        pass
    
    def backward(self, output_grad: np.ndarray) -> np.ndarray:
        """
        Validates shapes and computes both input and parameter gradients.

        :param output_grad: Gradient of the output.
        :return: Computed input gradient.
        """
        validate_same_shape(self._output, output_grad)
        input_grad: np.ndarray = self._input_gradient(output_grad)
        self._param_grad = self._compute_param_gradient(output_grad)
        
        validate_same_shape(self._param, self._param_grad)
        validate_same_shape(self._input, input_grad)
        return input_grad
    
    @property
    def param(self) -> np.ndarray:
        """Returns the parameter."""
        return self._param
    
    @property
    def param_gradient(self) -> np.ndarray:
        """Returns the gradient of the parameter."""
        return self._param_grad


class MatrixMultiplyOp(ParamOpBase):
    """
    Weight multiplication operation in a neural network.
    """
    __slots__ = ()

    def __init__(self, W: np.ndarray) -> None:
        """
        Initializes the operation with weight parameters.

        :param W: Weight matrix.
        """
        super().__init__(W)

    def _compute_output(self) -> np.ndarray:
        """
        Computes the output of the weight multiplication.

        :return: Computed output.
        """
        return np.dot(self._input, self._param)

    def _input_gradient(self, output_grad: np.ndarray) -> np.ndarray:
        """
        Computes the gradient with respect to the input.

        :param output_grad: Gradient of the output.
        :return: Computed input gradient.
        """
        return np.dot(output_grad, self._param.T)

    def _compute_param_gradient(self, output_grad: np.ndarray) -> np.ndarray:
        """
        Computes the gradient with respect to the weight parameters.

        :param output_grad: Gradient of the output.
        :return: Computed parameter gradient.
        """
        return np.dot(self._input.T, output_grad)


class AddBiasOp(ParamOpBase):
    """
    Bias addition operation in a neural network.
    """
    __slots__ = ()

    def __init__(self, B: np.ndarray) -> None:
        """
        Validates shape and initializes the operation with bias parameters.

        :param B: Bias vector.
        """
        assert B.shape[0] == 1, "Bias must have shape (1, n)."
        super().__init__(B)

    def _compute_output(self) -> np.ndarray:
        """
        Computes the output of the bias addition.

        :return: Computed output.
        """
        return self._input + self._param

    def _input_gradient(self, output_grad: np.ndarray) -> np.ndarray:
        """
        Computes the gradient with respect to the input.

        :param output_grad: Gradient of the output.
        :return: Computed input gradient.
        """
        return np.ones_like(self._input) * output_grad

    def _compute_param_gradient(self, output_grad: np.ndarray) -> np.ndarray:
        """
        Computes the gradient with respect to the bias parameters.

        :param output_grad: Gradient of the output.
        :return: Computed parameter gradient.
        """
        return np.sum(output_grad, axis=0, keepdims=True)