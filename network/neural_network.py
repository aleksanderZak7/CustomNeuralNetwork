import numpy as np
from typing import List

from .layers import Layer
from .loss_functions import LossBaseFunction


class NeuralNetwork:
    """
    Represents a neural network.
    """
    __slots__ = ("_train", "_layers", "_loss_function")

    def __init__(self, layers: List[Layer], loss_function: LossBaseFunction) -> None:
        """
        Initializes the neural network with layers and a loss function.

        :param layers: List of layers in the network.
        :param loss_function: Loss function used for training.
        """
        self._train: bool = True
        self._layers: List[Layer] = layers
        self._loss_function: LossBaseFunction = loss_function
            
    def eval(self) -> None:
        """
        Switches the network to evaluation mode.
        """
        self._train = False

    def train(self) -> None:
        """
        Switches the network back to training mode.
        """
        self._train = True
    
    def reset(self) -> None:
        """
        Resets the network by clearing the gradients.
        """
        for layer in self._layers:
            layer.first = True

    def predict(self, input: np.ndarray) -> np.ndarray:
        """
        Passes the input data forward through the network.

        :param input: Input data.
        :return: Processed output from the network.
        """
        output: np.ndarray = input
        for layer in self._layers:
            output = layer.forward(output, self._train)
        return output

    def backward(self, loss_grad: np.ndarray) -> None:
        """
        Passes the gradient backward through the network.

        :param loss_grad: Gradient of the loss function with respect to the output.
        """
        grad: np.ndarray = loss_grad
        for layer in reversed(self._layers):
            grad = layer.backward(grad)
            
    def fit(self, X: np.ndarray, y: np.ndarray) -> float:
        """
        Performs a forward pass, computes the loss, and propagates the gradient backward.

        :param x_batch: Input data batch.
        :param y_batch: Expected output values.
        :return: Loss value for the given batch.
        """
        y_pred: np.ndarray = self.predict(X)
        loss: float = self._loss_function.forward(y_pred, y)
        self.backward(self._loss_function.backward())
        return loss
    
    def params(self) -> List[np.ndarray]:
        """
        Retrieves the parameters from each layer.

        :return: List of parameters from all layers.
        """
        return [param for layer in self._layers for param in layer.params]

    def param_gradients(self) -> List[np.ndarray]:
        """
        Retrieves the gradients of parameters from each layer.

        :return: List of parameter gradients from all layers.
        """
        return [grad for layer in self._layers for grad in layer.param_grads]
    
    @property
    def layers(self) -> List[Layer]:
        """
        Gets the layers of the neural network.

        :return: List of layers.
        """
        return self._layers

    @layers.setter
    def layers(self, layers: List[Layer]) -> None:
        """
        Sets the layers of the neural network.

        :param layers: New list of layers.
        """
        if not isinstance(layers, list) or not all(isinstance(layer, Layer) for layer in layers):
            raise TypeError("Layers must be a list of Layer instances.")
        self._layers = layers

    @property
    def loss_function(self) -> LossBaseFunction:
        """
        Gets the loss function of the neural network.

        :return: Loss function.
        """
        return self._loss_function

    @loss_function.setter
    def loss_function(self, loss_function: LossBaseFunction) -> None:
        """
        Sets the loss function of the neural network.

        :param loss_function: New loss function.
        """
        if not isinstance(loss_function, LossBaseFunction):
            raise TypeError("Loss function must be an instance of LossBaseFunction.")
        self._loss_function = loss_function