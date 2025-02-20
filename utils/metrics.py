import numpy as np

def mean_absolute_error(y_pred: np.ndarray, y_true: np.ndarray) -> float:
    """
    Computes the Mean Absolute Error (MAE).

    :param y_pred: Predicted values.
    :param y_true: Actual target values.
    :return: MAE score.
    """ 
    return float(np.mean(np.abs(y_true - y_pred)))

def root_mean_squared_error(y_pred: np.ndarray, y_true: np.ndarray) -> float:
    """
    Computes the Root Mean Squared Error (RMSE).

    :param y_pred: Predicted values.
    :param y_true: Actual target values.
    :return: RMSE score.
    """
    return float(np.sqrt(np.mean(np.power(y_true - y_pred, 2))))

def eval_regression_model(y_pred: np.ndarray, y_true: np.ndarray) -> None:
    """
    Evaluates a regression model using Mean Absolute Error (MAE) and Root Mean Squared Error (RMSE).

    :param y_pred: Predicted values.
    :param y_true: Actual target values.
    """
    y_pred = y_pred.reshape(-1, 1)
    mae: float = mean_absolute_error(y_pred, y_true)
    rmse: float = root_mean_squared_error(y_pred, y_true)

    print(f"Mean Absolute Error (MAE): {mae:.2f}")
    print(f"Root Mean Squared Error (RMSE): {rmse:.2f}\n")

def evaluate_model_accuracy(y_predict: np.ndarray, y_test: np.ndarray) -> float:
    """
    Evaluates the model's accuracy on the test set.

    :param y_predict: Predicted probabilities or logits.
    :param y_test: True labels, either as indices or one-hot encoded.
    :return: Model accuracy formatted to two decimal places as a float.
    """
    if y_test.ndim > 1:
        y_test = np.argmax(y_test, axis=1)

    accuracy: float = (np.equal(np.argmax(y_predict, axis=1), y_test).sum() * 100.0 / y_test.shape[0])
    return np.round(accuracy, 2)