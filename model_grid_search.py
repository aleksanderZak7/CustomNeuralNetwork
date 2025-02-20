import numpy as np
from train_model import train
import matplotlib.pyplot as plt

from network.layers import Dense
from network.neural_network import NeuralNetwork
from network.optimizers import Optimizer, Adam, SGD
from network.loss_functions import CategoricalCrossEntropy


class ModelGridSearch:
    """
    Class for performing grid search over different sample models, optimizers, and hyperparameters.
    """
    __slots__ = ("_X_test", "_y_test", "_X_train", "_y_train", "_epochs", "_eval_every", "_batch_size", "_accuracies", "_models", "_optimizers")
    
    def __init__(self, X_train: np.ndarray, y_train: np.ndarray, X_test: np.ndarray, y_test: np.ndarray) -> None:
        """
        Initializes the grid search with training and testing data.
        
        :param X_test: Test features.
        :param y_test: Test labels.
        :param X_train: Training features.
        :param y_train: Training labels.
        """
        self._X_test = X_test
        self._y_test = y_test
        self._X_train = X_train
        self._y_train = y_train
        self._epochs: int | None = None
        self._eval_every: int | None = None
        self._batch_size: int | None = None
        self._accuracies: dict[str, dict[int, float]] = {}

        self._models = [
            NeuralNetwork(
                layers=[
                    Dense(neurons=256, activation="ReLU", weight_init="he", dropout=0.8),
                    Dense(neurons=64, activation="ReLU", weight_init="he", dropout=0.8),
                    Dense(neurons=10, activation="ReLU", weight_init="he")
                ],
                loss_function=CategoricalCrossEntropy()
            ),
            NeuralNetwork(
                layers=[
                    Dense(neurons=256, activation="Tanh", weight_init="lecun", dropout=0.8),
                    Dense(neurons=64, activation="Tanh", weight_init="lecun", dropout=0.8),
                    Dense(neurons=10, activation="Softmax", weight_init="he")
                ],
                loss_function=CategoricalCrossEntropy()
            )
        ]

        self._optimizers: dict[str, Optimizer] = {
            "3 times ReLU with Adam optimizer": Adam(learning_rate=0.0005, final_learning_rate=0.00005, decay_type="cosine",
                                                beta1=0.9, beta2=0.999, epsilon=1e-7),
            "3 times ReLU with SGD optimizer": SGD(learning_rate=0.005, final_learning_rate=0.0005, momentum=0.95, 
                                                   decay_type='cosine'),
            "2 times Tanh and Softmax with Adam optimizer": Adam(learning_rate=0.001, final_learning_rate=0.0001, 
                                                    decay_type="cosine", beta1=0.95, beta2=0.98, epsilon=1e-7),
            "2 times Tanh and Softmax with SGD optimizer": SGD(learning_rate=0.005, final_learning_rate=0.0005, momentum=0.94, decay_type='cosine')
        }

    def run_search(self, epochs: int = 100, eval_every: int = 10, batch_size: int = 60) -> None:
        """
        Performs grid search over all models and optimizers.

        :param epochs: Number of epochs for training.
        :param eval_every: Frequency of evaluation during training.
        :param batch_size: Batch size for training.
        """
        self._epochs = epochs
        self._eval_every = eval_every
        self._batch_size = batch_size
        
        for i, model in enumerate(self._models):
            optimizers_to_use = list(self._optimizers.items())[i * 2:(i + 1) * 2]

            for opt_name, optimizer in optimizers_to_use:
                print(f"Training model: {opt_name}")
                self._accuracies[opt_name] = train(
                    model, optimizer, self._X_train, self._y_train, self._X_test, self._y_test,
                    epochs=self._epochs, eval_every=self._eval_every, batch_size=self._batch_size, grid_search=True
                )
                model.reset()
                optimizer.reset()
    
    def generate_plot(self) -> None:
        """
        Generates a plot showing accuracy trends over epochs for different optimizers.
        """
        with plt.style.context('seaborn-v0_8-whitegrid'):
            for opt_name, acc_values in self._accuracies.items():
                epochs, accs = zip(*sorted(acc_values.items()))
                plt.plot(epochs, accs, marker="o", linestyle="-", label=opt_name)

            plt.xlabel("Epoch")
            plt.ylabel("Accuracy (%)")
            plt.title("Model Accuracy Over Epochs")
            plt.grid(True, linestyle="--", alpha=0.6)
            plt.legend(frameon=True, fancybox=True, framealpha=1, shadow=True, borderpad=1, loc="lower right")
            plt.show()
        
    def best_model(self) -> NeuralNetwork:
        """
        Returns the best model based on highest accuracy.

        :return: Best performing NeuralNetwork model.
        """
        if self._epochs is None or self._eval_every is None or self._batch_size is None:
            raise ValueError("Grid search has not been run yet. Please run the grid search first.")

        best_optimizer = max(self._accuracies, key=lambda opt: max(self._accuracies[opt].values()))
        best_epoch = max(self._accuracies[best_optimizer], key=self._accuracies[best_optimizer].get)
        best_accuracy = max(self._accuracies[best_optimizer].values())

        best_model_idx = 0 if "ReLU" in best_optimizer else 1
        best_model = self._models[best_model_idx]
        
        print(f"Training model: {best_optimizer} with accuracy: {best_accuracy}% after {best_epoch} epochs.")
        train(best_model, self._optimizers[best_optimizer], self._X_train, 
              self._y_train, self._X_test, self._y_test, epochs=self._epochs, best_epoch_from_grid=best_epoch,
              eval_every=self._eval_every, batch_size=self._batch_size)
        
        return best_model
    
    @property
    def epochs(self) -> int | None:
        """
        Returns the number of epochs used in grid search.
        """
        return self._epochs

    @property
    def eval_every(self) -> int | None:
        """
        Returns the frequency of evaluation during training.
        """
        return self._eval_every

    @property
    def batch_size(self) -> int | None:
        """
        Returns the batch size used for training.
        """
        return self._batch_size