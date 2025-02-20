import numpy as np

from .operations import OperationBase


class Dropout(OperationBase):
    """
    Implementation of the Dropout layer.
    """
    __slots__ = ("_train", "_keep_prob", "_mask")

    def __init__(self, keep_prob: float, train: bool = True) -> None:
        """
        Initializes the Dropout layer.

        :param keep_prob: Probability of keeping a unit active.
        :param train: Whether the model is in train mode (default: True).
        """
        super().__init__()
        self._train = train
        self._keep_prob = keep_prob
        self._mask: np.ndarray | None = None

    def _compute_output(self) -> np.ndarray:
        """
        Computes the output of the Dropout layer.

        :return: The input scaled by the dropout mask.
        """
        if self._train:
            self._mask = np.random.binomial(1, self._keep_prob, size=self._input.shape).astype(np.float32)
            
            if np.any(np.isnan(self._mask)):
                raise ValueError("Dropout mask contains NaN values!")
            return self._input * self._mask
        else:
            return self._input * self._keep_prob


    def _input_gradient(self, output_grad: np.ndarray) -> np.ndarray:
        """
        Computes the gradient of the Dropout layer.

        :param output_grad: Gradient from the next layer.
        :return: Gradient after applying the dropout mask.
        """
        if self._mask is None:
            raise ValueError("Mask is not initialized. Forward pass must be performed before backward pass.")
        return output_grad * self._mask
    
    @property
    def train(self) -> bool:
        """
        Returns whether the model is in train mode.

        :return: True if train mode is enabled, otherwise False.
        """
        return self._train

    @train.setter
    def train(self, value: bool) -> None:
        """
        Sets the train mode.

        :param value: Boolean flag for train mode.
        :raises ValueError: If value is not a boolean.
        """
        if not isinstance(value, bool):
            raise ValueError("Train mode must be either True or False.")
        self._train = value