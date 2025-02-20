from model_grid_search import ModelGridSearch
from utils import load_mnist_data, evaluate_model_accuracy

from network.neural_network import NeuralNetwork

def main() -> None:
    """
    Main function for training and evaluating a neural network on the MNIST dataset.
    """
    (X_train, y_train), (X_test, y_test) = load_mnist_data()  # MNIST images are 28x28 = 784 features

    grid_search = ModelGridSearch(X_train, y_train, X_test, y_test)
    grid_search.run_search(epochs= 100, eval_every = 10, batch_size = 60)
    grid_search.generate_plot()
    
    model: NeuralNetwork = grid_search.best_model()
    
    y_predict = model.predict(X_test)
    accuracy = evaluate_model_accuracy(y_predict, y_test)

    # Final accuracy of the sample model on test data: 98.58%
    print(f"Final accuracy of the sample model on test data: {accuracy:.2f}%")

if __name__ == "__main__":
    main()