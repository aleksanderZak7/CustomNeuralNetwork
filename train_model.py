import numpy as np
from copy import deepcopy
from collections import defaultdict
from utils import shuffle_data, batch_data_generator, evaluate_model_accuracy

from network.optimizers import Optimizer
from network.neural_network import NeuralNetwork


def train(
    model: NeuralNetwork, optim: Optimizer,
    X_train: np.ndarray, y_train: np.ndarray,
    X_test: np.ndarray, y_test: np.ndarray,
    epochs: int = 100, best_epoch_from_grid: int = -1,
    eval_every: int = 10, batch_size: int = 60, seed: int = 42,
    stop_on_loss_increase: bool = False,
    grid_search: bool = False) -> dict[int, float] | None:
    """
    Trains a neural network using the given optimizer.

    If `grid_search` is enabled, stores accuracy values for evaluation.
    If `stop_on_loss_increase` is enabled, reverts to the last best model if loss increases.

    :param model: Neural network model to be trained.
    :param optim: Optimizer used for updating model parameters.
    :param X_train: Training dataset inputs.
    :param y_train: Training dataset labels.
    :param X_test: Test dataset inputs.
    :param y_test: Test dataset labels.
    :param epochs: Total number of training epochs (default: 100).
    :param best_epoch_from_grid: Best number of epochs determined by grid search (default: -1).
    :param eval_every: Interval (in epochs) for model evaluation (default: 10).
    :param batch_size: Number of samples per training batch (default: 60).
    :param seed: Random seed for reproducibility (default: 42).
    :param stop_on_loss_increase: If True, stops training when validation loss increases (default: False).
    :param grid_search: If True, stores accuracy values for different epochs (default: False).
    :return: A dictionary mapping epochs to accuracy values if `grid_search` is True, otherwise None.
    """
    optim.net = model
    optim.epochs = epochs

    np.random.seed(seed)
    best_loss: float = 1e9
    results: dict[int, float] = defaultdict(dict) if grid_search else None

    for epoch in range(epochs):
        if stop_on_loss_increase and (epoch + 1) % eval_every == 0:
            last_layers = deepcopy(model.layers)
            last_loss_function = deepcopy(model.loss_function)
            
        model.train()
        X_train, y_train = shuffle_data(X_train, y_train)
        batch_data = batch_data_generator(X_train, y_train, batch_size)

        for _, (X_batch, y_batch) in enumerate(batch_data):
            model.fit(X_batch, y_batch)
            optim.step()
            
        if (epoch + 1) % eval_every == 0:
            model.eval()
            y_pred: np.ndarray = model.predict(X_test)
            loss: float = model.loss_function.forward(y_pred, y_test)
            
            if stop_on_loss_increase:
                if loss < best_loss:
                    print(f"Loss after {epoch + 1} epochs: {loss:.4f}")
                    best_loss = loss
                else:
                    print(f"Loss increased after epoch {epoch + 1}. "
                          f"Reverting to model from epoch {epoch + 1 - eval_every} with loss {best_loss:.4f}.")
                    model.layers = last_layers
                    model.loss_function = last_loss_function
                    break
            elif grid_search:
                accuracy: float = evaluate_model_accuracy(y_pred, y_test)
                results[epoch + 1] = accuracy
                print(f"Model accuracy after {epoch + 1} epochs: {accuracy:.2f}%, with loss {loss:.4f}")
            else:
                print(f"Epoch {epoch + 1}/{epochs} completed with loss: {loss:.4f}")
                
                if (epoch + 1) == best_epoch_from_grid:
                    print(f"Training stopped after {epoch + 1} epochs due to best result found during grid search.")
                    break

        if optim.final_learning_rate:
            optim.decay_learning(epoch)
    print("Training completed.\n")
    return results