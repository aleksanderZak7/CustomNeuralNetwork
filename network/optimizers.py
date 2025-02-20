import numpy as np
from typing import Literal, List
from abc import ABC, abstractmethod

from .neural_network import NeuralNetwork


class Optimizer(ABC):
    """
    Base class for neural network optimizers.
    """
    __slots__ = ("_epochs", "_decay_type", "_net", "_step_interval", "_learning_rate", "_decay_per_epoch", 
                 "_initial_learning_rate", "_final_learning_rate")
    
    VALID_DECAY_TYPES = ("step", "cosine", "linear", "polynomial", "exponential")

    def __init__(self, learning_rate_: float = 0.01, final_learning_rate_: float = 0.0, 
                 decay_type_: Literal["step", "cosine", "linear", "polynomial", "exponential"] | None = None,
                 power_: float | None = None, step_: float | None = None, step_interval_: int = 10) -> None:
        """
        Initializes the optimizer.

        :param learning_rate_: Initial learning rate (default: 0.01).
        :param final_learning_rate_: Learning rate at the end of training (default: 0.0).
        :param decay_type_: Type of learning rate decay.
        :param power_: Power factor for polynomial decay.
        :param step_: Step size for step decay.
        :param step_interval_: Number of epochs after which step decay occurs (default: 10).
        """
        if learning_rate_ <= 0:
            raise ValueError("Initial learning rate must be positive.")

        if final_learning_rate_ < 0:
            raise ValueError("Final learning rate must be non-negative.")

        if decay_type_ is not None and decay_type_ not in Optimizer.VALID_DECAY_TYPES:
            raise ValueError(f"Invalid decay type: {decay_type_}")

        if power_ is not None and step_ is not None:
            raise ValueError("Only one of 'power' or 'step' can be specified.")

        self._epochs: int | None = None
        self._decay_type: str = decay_type_
        self._net: NeuralNetwork | None = None
        self._step_interval: int = step_interval_
        self._learning_rate: float = learning_rate_
        self._initial_learning_rate: float = learning_rate_
        self._final_learning_rate: float = final_learning_rate_
        self._decay_per_epoch: float | None = power_ if power_ is not None else step_
    
    @abstractmethod
    def step(self) -> None:
        """
        Performs a single optimization step. Must be implemented by subclasses.
        """
        pass
    
    @abstractmethod
    def reset(self) -> None:
        """
        Resets optimizer parameters to their initial values.
        """
        pass
    
    def decay_learning(self, epoch: int) -> None:
        """
        Applies learning rate decay.
        """
        match self._decay_type:
            case "step":
                if epoch % self._step_interval == 0:
                    self._learning_rate *= self._decay_per_epoch
            case "cosine":
                self._learning_rate = self._final_learning_rate + (self._learning_rate - self._final_learning_rate) \
                    * (1 + np.cos(epoch * self._decay_per_epoch)) / 2
            case "linear":
                self._learning_rate -= self._decay_per_epoch
            case "polynomial":
                decay_factor = (1 - epoch / self._epochs) ** self._decay_per_epoch
                self._learning_rate = self._final_learning_rate + (self._learning_rate - self._final_learning_rate) * decay_factor
            case "exponential":
                self._learning_rate *= self._decay_per_epoch
    
    def _set_decay_per_epoch(self) -> None:
        """
        Sets the decay rate based on the selected decay type.
        """
        if self._epochs == 1 or self._final_learning_rate == 0.0:
            return

        match self._decay_type:
            case "step":
                if self._decay_per_epoch is None or self._decay_per_epoch <= 0:
                    raise ValueError("Step for step decay must be a positive number and cannot be None.")
            case "cosine":
                self._decay_per_epoch = np.pi / self._epochs
            case "linear":
                self._decay_per_epoch = (self._learning_rate - self._final_learning_rate) / (self._epochs - 1)
            case "polynomial":
                if self._decay_per_epoch is None or self._decay_per_epoch <= 0:
                    raise ValueError("Power for polynomial decay must be a positive number and cannot be None.")
            case "exponential":
                self._decay_per_epoch = np.power(self._final_learning_rate / self._learning_rate, 1.0 / (self._epochs - 1))

    @property
    def net(self) -> NeuralNetwork:
        """
        Returns the network assigned to the optimizer.
        """
        return self._net
    
    @net.setter
    def net(self, net_: NeuralNetwork) -> None:
        """
        Assigns a network to the optimizer.
        """
        self._net = net_
    
    @property
    def final_learning_rate(self) -> float:
        """
        Returns the final learning rate (default: 0.0).
        """
        return self._final_learning_rate
    
    @property
    def epochs(self) -> int | None:
        """
        Returns the number of epochs.
        """
        return self._epochs
    
    @epochs.setter
    def epochs(self, value: int) -> None:
        """
        Sets the number of training epochs and updates the decay rate if applicable.

        If a final learning rate is specified, this method recalculates the learning rate 
        decay factor based on the selected decay type.
        """
        if value < 1:
            raise ValueError("Number of epochs must be at least 1.")
        self._epochs = value
        if self._final_learning_rate is not None:
            self._set_decay_per_epoch()


class Adam(Optimizer):
    """
    Adam optimizer.
    """
    __slots__ = ("_beta1", "_beta2", "_epsilon", "_first_moment", "_second_moment", "_time_step")

    def __init__(self, learning_rate: float = 0.001, final_learning_rate: float = 0.0, 
                 decay_type: Literal["step", "cosine", "linear", "polynomial", "exponential"] | None = None, 
                 power: float | None = None, step: float | None = None, step_interval: int = 10,
                 beta1: float = 0.9, beta2: float = 0.999, epsilon: float = 1e-8) -> None:
        """
        Initializes the Adam optimizer.

        :param learning_rate: Initial learning rate (default: 0.001).
        :param final_learning_rate: Learning rate at the end of training (default: 0.0).
        :param decay_type: Type of learning rate decay.
        :param power: Power factor for polynomial decay.
        :param step: Step size for step decay.
        :param step_interval: Number of epochs after which step decay occurs (default: 10).
        :param beta1: Exponential decay rate for the first moment estimates (default: 0.9).
        :param beta2: Exponential decay rate for the second moment estimates (default: 0.999).
        :param epsilon: Small constant for numerical stability (default: 1e-8).
        """
        if not (0 < beta1 < 1):
            raise ValueError("beta1 must be in (0, 1).")
        if not (0 < beta2 < 1):
            raise ValueError("beta2 must be in (0, 1).")
            
        super().__init__(learning_rate_=learning_rate, final_learning_rate_=final_learning_rate,
                         decay_type_=decay_type, power_=power, step_=step, step_interval_=step_interval)
        self._time_step: int = 0
        self._beta1: float = beta1
        self._beta2: float = beta2
        self._epsilon: float = epsilon
        self._first_moment: List[np.ndarray] | None = None
        self._second_moment: List[np.ndarray] | None = None

    def step(self) -> None:
        """
        Updates network parameters using the Adam optimization algorithm.
        """
        if self._epochs is None or self._net is None:
            raise ValueError("The optimizer must have both a network and a defined number of epochs.")

        if self._first_moment is None or self._second_moment is None:
            self._first_moment = [np.zeros_like(p) for p in self._net.params()]
            self._second_moment = [np.zeros_like(p) for p in self._net.params()]

        self._time_step += 1
        for i, (param, grad) in enumerate(zip(self._net.params(), self._net.param_gradients())):
            self._first_moment[i] = self._beta1 * self._first_moment[i] + (1 - self._beta1) * grad
            self._second_moment[i] = self._beta2 * self._second_moment[i] + (1 - self._beta2) * (grad ** 2)
            first_moment_hat = self._first_moment[i] / (1 - self._beta1 ** self._time_step)
            second_moment_hat = self._second_moment[i] / (1 - self._beta2 ** self._time_step)
            param -= self._learning_rate * first_moment_hat / (np.sqrt(second_moment_hat) + self._epsilon)
    
    def reset(self) -> None:
        """
        Resets optimizer parameters to their initial values.
        """
        self._time_step = 0
        self._first_moment = None
        self._second_moment = None
        self._learning_rate = self._initial_learning_rate


class SGD(Optimizer):
    """
    Stochastic Gradient Descent (SGD) optimizer with momentum.
    """
    __slots__ = ("_momentum", "_velocities")

    def __init__(self, learning_rate: float = 0.001, final_learning_rate: float = 0.0, 
                 decay_type: Literal["step", "cosine", "linear", "polynomial", "exponential"] | None = None,
                 power: float | None = None, step: float | None = None, step_interval: int = 10, momentum: float = 0.9) -> None:
        """
        Initializes the SGD optimizer.

        :param learning_rate: Initial learning rate (default: 0.001).
        :param final_learning_rate: Learning rate at the end of training (default: 0.0).
        :param decay_type: Type of learning rate decay.
        :param power: Power factor for polynomial decay.
        :param step: Step size for step decay.
        :param step_interval: Number of epochs after which step decay occurs (default: 10).
        :param momentum: Momentum factor (default: 0.9).
        """
        if momentum < 0 or momentum > 1:
            raise ValueError("Momentum must be in the range [0, 1].")

        super().__init__(learning_rate_=learning_rate, final_learning_rate_=final_learning_rate, 
                         decay_type_=decay_type, power_=power, step_=step, step_interval_=step_interval)
        self._momentum: float = momentum
        self._velocities: List[np.ndarray] | None = None

    def step(self) -> None:
        """
        Updates network parameters using SGD with momentum.
        """
        if self._epochs is None or self._net is None:
            raise ValueError("The optimizer must have both a network and a defined number of epochs.")

        if self._velocities is None:
            self._velocities = [np.zeros_like(p) for p in self._net.params()]

        for param, grad, velocity in zip(self._net.params(), self._net.param_gradients(), self._velocities):
            velocity *= self._momentum
            velocity += self._learning_rate * grad
            param -= velocity
    
    def reset(self) -> None:
        """
        Resets optimizer parameters to their initial values.
        """
        self._velocities = None
        self._learning_rate = self._initial_learning_rate


class RMSProp(Optimizer):
    """
    RMSProp optimizer.
    """
    __slots__ = ("_decay_rate", "_epsilon", "_cache")

    def __init__(self, learning_rate: float = 0.001, final_learning_rate: float = 0.0, 
                 decay_type: Literal["step", "cosine", "linear", "polynomial", "exponential"] | None = None,
                 power: float | None = None, step: float | None = None, step_interval: int = 10,
                 decay_rate: float = 0.99, epsilon: float = 1e-8) -> None:
        """
        Initializes the RMSProp optimizer.

        :param learning_rate_: Initial learning rate (default: 0.001).
        :param final_learning_rate_: Learning rate at the end of training (default: 0.0).
        :param decay_type_: Type of learning rate decay.
        :param power: Power factor for polynomial decay.
        :param step: Step size for step decay.
        :param step_interval: Number of epochs after which step decay occurs (default: 10).
        :param decay_rate: Decay rate for the moving average of squared gradients (default: 0.99).
        :param epsilon: Small constant for numerical stability (default: 1e-8).
        """
        super().__init__(learning_rate_=learning_rate, final_learning_rate_=final_learning_rate,
                         decay_type_=decay_type, power_=power, step_=step, step_interval_=step_interval)
        self._epsilon: float = epsilon
        self._cache: List[np.ndarray] = []
        self._decay_rate: float = decay_rate

    def step(self) -> None:
        """
        Updates network parameters using the RMSProp algorithm.
        """
        if self._epochs is None or self._net is None:
            raise ValueError("The optimizer must have both a network and a defined number of epochs.")

        if not self._cache:
            self._cache = [np.zeros_like(p) for p in self._net.params()]

        for i, (param, grad) in enumerate(zip(self._net.params(), self._net.param_gradients())):
            self._cache[i] = self._decay_rate * self._cache[i] + (1 - self._decay_rate) * (grad ** 2)
            param -= self._learning_rate * grad / (np.sqrt(self._cache[i]) + self._epsilon)
    
    def reset(self) -> None:
        """
        Resets optimizer parameters to their initial values.
        """
        self._cache.clear()
        self._learning_rate = self._initial_learning_rate