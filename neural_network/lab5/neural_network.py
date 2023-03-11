import numpy as np
import pandas as pd
from sklearn import datasets
from sklearn.model_selection import train_test_split

TRAIN_SIZE = 0.7
TEST_SIZE = 0.3

class NeuralNetwork:
    def __init__(self, num_layers :int, learn_rate :float, input_layer_size: int=64, output_layer_size: int=10):
        self.num_layers = num_layers
        self.learn_rate = learn_rate
        self.input_layer_size = input_layer_size
        self.output_layer_size = output_layer_size
        self.layer_sizes = self.init_layer_sizes()
        self.weights = self.init_random_weights()
        self.biases = self.init_random_biases()

    def sum_up_neurons_init(self, weights: np.array, input_values :np.array, biases: np.array):
        return np.dot(weights, input_values) + biases

    def ReLU(self, z):
        return np.maximum(0, z)

    def ReLU_derivative(self, z):
        return z > 0

    def sigmoid(self, z):
        return 1.0/(1.0 + np.exp(-z))

    def sigmoid_derivative(self, z):
        return self.sigmoid(z)*(1-self.sigmoid(z))

    def forward_propagation(self, inputs :np.array, activation_function :callable):
        list_of_inputs = [inputs]
        list_of_sumators = []
        for weights, biases in zip(self.weights[:-1], self.biases[:-1]):
            inputs = self.sum_up_neurons_init(weights, inputs, biases)
            list_of_sumators.append(inputs)
            inputs = activation_function(inputs)
            list_of_inputs.append(inputs)

        inputs = self.sum_up_neurons_init(self.weights[-1], inputs, self.biases[-1])
        list_of_sumators.append(inputs)
        inputs = self.sigmoid(inputs)
        list_of_inputs.append(inputs)
        return list_of_sumators, list_of_inputs

    def back_propagation(self, y :np.array, inputs :np.array):
        gradient = [np.zeros(weights.shape) for weights in self.weights]
        new_biases = [np.zeros(biases.shape) for biases in self.biases]
        sumators_array, inputs_array = self.forward_propagation(inputs, self.sigmoid)
        error = (inputs_array[-1] - y) * self.sigmoid_derivative(sumators_array[-1])
        new_biases[-1] = error
        gradient[-1] = np.dot(error, inputs_array[-2].T)
        for i in range(2, self.num_layers):
            z = sumators_array[-i]
            derivative = self.sigmoid_derivative(z)
            error = np.dot(self.weights[-i + 1].T, error) * derivative
            new_biases[-i] = error
            gradient[-i] = np.dot(error, inputs_array[-i-1].T)
        return gradient, new_biases

    def update_weights(self, gradient :np.array, new_biases :np.array):
        for i, (nw, nb) in enumerate(zip(gradient, new_biases)):
            self.weights[i] = self.weights[i] - self.learn_rate * nw
            self.biases[i] = new_biases[i] - self.learn_rate * nb

    def learn(self, x_train :np.array, y_train_encoded :np.array,  num_epochs :int):
        for i in range(num_epochs):
            for image, label in zip(x_train, y_train_encoded):
                gradient, new_biases = self.back_propagation(label, image)
                self.update_weights(gradient, new_biases)

    def calculate_label(self, image: np.array):
        for weights, biases in zip(self.weights, self.biases):
            image = self.sigmoid(self.sum_up_neurons_init(weights, image, biases))
        return image

    def test_accuracy(self, x: np.array, y: np.array):
        x_results = [self.calculate_label(image) for image in x]
        x_results = np.argmax(x_results, axis=1)
        y_labels = np.argmax(y, axis=1)
        accuracy = np.sum(x_results == y_labels)/len(x)
        return accuracy

    def init_layer_sizes(self):
        self.layer_sizes = [self.input_layer_size]
        for i in range(1, self.num_layers-1):
            self.layer_sizes.append(max(min(int(1/3 * self.layer_sizes[i-1]+self.output_layer_size), self.input_layer_size), self.output_layer_size))
        self.layer_sizes.append(self.output_layer_size)
        return self.layer_sizes

    def parse_random_matrix(self, size_x :int, size_y :int):
        weight = np.random.uniform(-1/np.sqrt(self.input_layer_size), 1/np.sqrt(self.input_layer_size), (size_x, size_y))
        return weight

    def init_random_weights(self):
        random_weights = []
        sizes = self.layer_sizes
        for i, size in enumerate(sizes):
            try:
                weight = self.parse_random_matrix(sizes[i+1], sizes[i])
                random_weights.append(weight)
            except IndexError:
                return random_weights

    def init_random_biases(self):
        sizes = self.layer_sizes
        random_biases = [self.parse_random_matrix(y, 1) for y in sizes[1:]]
        return random_biases



def split_data(dataset, digits):
    x_train, x_rem, y_train, y_rem = train_test_split(dataset, digits.target, train_size=TRAIN_SIZE)
    x_valid, x_test, y_valid, y_test = train_test_split(x_rem, y_rem, test_size=TEST_SIZE)
    return x_train, y_train, x_valid, y_valid, x_test, y_test


def flatten_images(digits):
    data = digits.images.reshape(len(digits.images), -1)
    return data


def encode(Y):
    y_encoded = np.zeros((Y.size, Y.max()+ 1))
    y_encoded[np.arange(Y.size), Y] = 1
    return y_encoded


def error_function(y, yd):
    vec_difference = y - yd
    return np.dot(vec_difference, vec_difference)


def prepare_data(data):
    images_flattened = flatten_images(data)
    x_train, y_train, x_valid, y_valid, x_test, y_test = split_data(images_flattened, data)
    x_train = [x.reshape(len(x_train[0]), 1) for x in x_train]
    x_valid = [x.reshape(len(x_valid[0]), 1) for x in x_valid]
    x_test = [x.reshape(len(x_test[0]), 1) for x in x_test]
    y_train = encode(y_train)
    y_valid = encode(y_valid)
    y_test = encode(y_test)
    y_train = [y.reshape(len(y_train[0]), 1) for y in y_train]
    y_valid = [y.reshape(len(y_valid[0]), 1) for y in y_valid]
    y_test = [y.reshape(len(y_test[0]), 1) for y in y_test]
    return x_train, y_train, x_valid, y_valid, x_test, y_test

def build_and_validate(x_train, y_train, x_valid, y_valid, iterations, learning_rates):
    accuracies_iterations = []
    accuracies_learning_rates = []
    for i in iterations:
        network = NeuralNetwork(4, 0.07)
        network.learn(x_train, y_train, i)
        accuracies_iterations.append(network.test_accuracy(x_valid, y_valid))
    print(accuracies_iterations)
    for l in learning_rates:
        network = NeuralNetwork(4, l)
        network.learn(x_train, y_train, 200)
        accuracies_learning_rates.append(network.test_accuracy(x_valid, y_valid))
    print(accuracies_learning_rates)


def main():
    data = datasets.load_digits()
    x_train, y_train, x_valid, y_valid, x_test, y_test = prepare_data(data)
    iterations = [1, 10, 50, 100, 200, 500, 1000]
    learning_rates = [0.0001, 0.0005, 0.001, 0.005, 0.01, 0.05, 0.1, 0.3, 0.5]
    # build_and_validate(x_train, y_train, x_valid, y_valid, iterations, learning_rates)
    network = NeuralNetwork(4, 0.07)
    network.learn(x_train, y_train, 100)
    print(f"Accuracy: {network.test_accuracy(x_test, y_test)}")


if __name__ == "__main__":
    main()

