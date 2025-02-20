import numpy as np
from abc import ABC, abstractmethod
from utils import validate_same_shape
from typing import List, Union, Optional, Literal

from .dropout import Dropout
from . import operations as op
from . import activation_functions as act


class Layer(ABC):
    """
    Base class for a neural network layer.
    """
    __slots__ = ("_first", "_neurons", "_dropout", "_weight_init", "_param_grads", "_operations", "_activation")
    
    VALID_INITS = ("glorot", "he", "lecun", "zero", "standard")
    ActivationFunctions = Literal["ReLU", "ELU", "LeakyReLU", "Sigmoid", "Tanh", "Softmax"]

    def __init__(self, neurons_: int, dropout_: float = 1.0, 
                 activation_: Union[op.OperationBase, ActivationFunctions, None] = None,
                 weight_init_: Literal["glorot", "he", "lecun", "zero", "standard"] = "standard", **activation_params_: float) -> None:
        """ 
        Initializes the neural network layer.
        
        :param neurons_: Number of neurons in the layer.
        :param dropout_: Dropout probability (between 0 and 1, default is 1.0, meaning no dropout).
        :param activation_: Activation function instance, name, or None (default is None, must be specified).
        :param weight_init_: Weight initialization method (default is "standard").
        :param activation_params_: Additional parameters for activation functions (e.g., alpha for ELU/LeakyReLU).
        """
        if not (0.0 <= dropout_ <= 1.0):
            raise ValueError("Dropout probability must be between 0 and 1.")

        if activation_ is None:
            raise ValueError("Activation function cannot be None.")
        
        if weight_init_ not in Layer.VALID_INITS:
            raise ValueError(f"Invalid weight initialization method: {weight_init_}")

        self._first: bool = True
        self._neurons: int = neurons_
        self._dropout: float = dropout_
        self._weight_init: str = weight_init_
        self._param_grads: List[np.ndarray] = []
        self._operations: List[op.OperationBase] = []
        self._activation: Optional[op.OperationBase] = self._get_activation(activation_, **activation_params_)

    @abstractmethod
    def _setup_layer(self, num_in: int) -> None:
        """
        Abstract method to define layer-specific setup logic.
        
        :param num_in: Number of input features.
        """
        pass
    
    def _get_activation(self, activation: Union[op.OperationBase, "Layer.ActivationFunctions"], **kwargs) -> op.OperationBase:
        """
        Retrieves the activation function instance.
        
        :param activation: Activation function name or instance.
        :param kwargs: Additional parameters for activation function.
        :return: Activation function instance.
        """
        activations = {
            "ReLU":  act.ReLU,
            "ELU":  act.ELU,
            "LeakyReLU":  act.LeakyReLU,
            "Sigmoid":  act.Sigmoid,
            "Tanh":  act.Tanh,
            "Softmax":  act.Softmax
        }
        if activation in activations:
            activation_class = activations[activation]
            return activation_class(**kwargs) if kwargs else activation_class()
        
        raise ValueError(f"Unknown activation function: {activation}")

    def forward(self, input: np.ndarray, train: bool) -> np.ndarray:
        """
        Passes input forward through the layer.

        :param input: Input data.
        :param train: Whether the model is in training mode.
        :return: Output after passing through the layer.
        """
        if self._first:
            self._setup_layer(input)
            self._first = False

        if not train and self._dropout < 1.0:
            self._operations[-1].train = False

        self.output = input
        for operation in self._operations:
            self.output = operation.forward(self.output)
        return self.output

    def backward(self, output_grad: np.ndarray) -> np.ndarray:
        """
        Checks shape and propagates gradient backward.

        :param output_grad: Gradient from the next layer.
        :return: Gradient with respect to input.
        """
        validate_same_shape(self.output, output_grad)

        for operation in reversed(self._operations):
            output_grad = operation.backward(output_grad)
            
        return output_grad

    @property
    def param_grads(self) -> List[np.ndarray]:
        """
        Collects gradients from each operation with parameters.

        :return: List of parameter gradients.
        """
        return [operation.param_gradient for operation in self._operations if isinstance(operation, op.ParamOpBase)]

    @property
    def params(self) -> List[np.ndarray]:
        """
        Collects parameters from each operation with parameters.

        :return: List of parameter arrays.
        """
        return [operation.param for operation in self._operations if isinstance(operation, op.ParamOpBase)]
    
    @property
    def first(self) -> bool:
        """
        Returns whether the layer is being initialized for the first time.

        :return: True if the layer is being initialized, otherwise False.
        """
        return self._first

    @first.setter
    def first(self, value: bool) -> None:
        """
        Sets the flag indicating whether the layer is being initialized for the first time.

        :param value: Boolean flag indicating whether this is the first initialization.
        :raises TypeError: If the provided value is not a boolean.
        """
        if not isinstance(value, bool):
            raise TypeError("First flag must be a boolean.")
        self._first = value


class Dense(Layer):
    """
    Fully connected (Dense) layer.
    """

    def __init__(self, neurons: int, dropout: float = 1.0,
                 activation: Union[op.OperationBase, Layer.ActivationFunctions, None] = None, 
                 weight_init: Literal["glorot", "he", "lecun", "zero", "standard"] = "standard", 
                 **activation_params: float) -> None:
        """
        Initializes the dense layer.
        
        :param neurons: Number of neurons in the layer.
        :param dropout: Dropout probability (between 0 and 1). Default is 1.0 (no dropout).
        :param activation: Activation function instance, name, or None. Must be specified.
        :param weight_init: Weight initialization method. Default is "standard".
                            - "glorot": Xavier/Glorot initialization.
                            - "he": He initialization.
                            - "lecun": LeCun initialization.
                            - "zero": All weights initialized to zero.
                            - "standard": Default standard normal initialization.
        :param activation_params: Additional parameters for activation functions (e.g., alpha for ELU/LeakyReLU).
        :raises ValueError: If dropout is not in range [0, 1] or activation is None.
        """
        super().__init__(neurons_=neurons, dropout_=dropout, activation_=activation, weight_init_=weight_init, **activation_params)

    def _setup_layer(self, input_: np.ndarray) -> None:
        """
        Defines the layer operations and initializes parameters.
        
        :param input_: Input data to determine input size (features).
        """
        num_inputs: int = input_.shape[1] if input_.ndim > 1 else input_.shape[0]

        if self._weight_init == "glorot":
            std_dev: float = np.sqrt(2 / (num_inputs + self._neurons))
        elif self._weight_init == "he":
            std_dev: float = np.sqrt(2 / num_inputs)
        elif self._weight_init == "lecun":
            std_dev: float = np.sqrt(1 / num_inputs)
        elif self._weight_init == "zero":
            std_dev: float = 0.0
        else:
            std_dev: float = 1.0

        weight: np.ndarray = np.random.normal(loc=0, scale=std_dev, size=(num_inputs, self._neurons))
        bias: np.ndarray = np.random.normal(loc=0, scale=std_dev, size=(1, self._neurons))
        self._operations = [op.MatrixMultiplyOp(weight), op.AddBiasOp(bias), self._activation]
        
        if self._dropout < 1.0:
            self._operations.append(Dropout(self._dropout, True))