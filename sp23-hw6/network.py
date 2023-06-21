import random
from tqdm import tqdm
import numpy as np


class Network(object):
    def __init__(self, sizes):
        """
        Args:
            sizes (List[int]): Contains the size of each layer in the network.
        """
        self.num_layers = len(sizes)
        self.sizes = sizes
        self.biases = [np.random.randn(y, 1) for y in sizes[1:]]
        self.weights = [np.random.randn(y, x) for x, y in zip(sizes[:-1], sizes[1:])]

    """
    4.1 Feed forward the input x through the network.
    """

    def feedforward(self, x):
        """
        Args:
            x (npt.array): Input to the network.
        Returns:
            List[npt.array]: List of weighted input values to each node
            List[npt.array]: List of activation output values of each node
        """
        z_values = []
        a_values = [x]
        for i in range(self.num_layers-1):
            z = np.dot(self.weights[i], a_values[-1]) + self.biases[i]
            a = sigmoid(z)
            z_values.append(z)
            a_values.append(a)

        return z_values, a_values

    """
    4.2 Backpropagation to compute gradients.
    """

    def backprop(self, x, y, zs, activations):
        """
        Args:
            x (npt.array): Input vector.
            y (float): Target value.
            zs (List[npt.array]): List of weighted input values to each node.
            activations (List[npt.array]): List of activation output values of each node.
        Returns:
            List[npt.array]: List of gradients of bias parameters.
            List[npt.array]: List of gradients of weight parameters.
        """
        delta_b = [np.zeros(b.shape) for b in self.biases]
        delta_w = [np.zeros(w.shape) for w in self.weights]

        delta = (activations[-1] - y) * sigmoid_prime(zs[-1])
        delta_b[-1] = delta
        delta_w[-1] = np.dot(delta, activations[-2].T)
        for l in range(2, self.num_layers):
            delta = np.dot(self.weights[-l + 1].T, delta) * sigmoid_prime(zs[-l])
            delta_b[-l] = delta
            delta_w[-l] = np.dot(delta, activations[-l - 1].T)

        return delta_b, delta_w

    """
    4.3 Update the network's biases and weights after processing a single mini-batch.
    """

    def update_mini_batch(self, mini_batch, alpha):
        """
        Args:
            mini_batch (List[Tuple]): List of (input vector, output value) pairs.
            alpha: Learning rate.
        Returns:
            float: Average loss on the mini-batch.
        """
        grad_b = [np.zeros(b.shape) for b in self.biases]
        grad_w = [np.zeros(w.shape) for w in self.weights]
        total_loss = 0.0
        for x, y in mini_batch:
            z_list, a_list = self.feedforward(x)
            delta_b, delta_w = self.backprop(x, y, z_list, a_list)
            grad_b = [gb + db for gb, db in zip(grad_b, delta_b)]
            grad_w = [gw + dw for gw, dw in zip(grad_w, delta_w)]
            loss = self.loss_function(a_list[-1], y)
            total_loss += loss
        self.weights = [w - (alpha / len(mini_batch)) * gw for w, gw in zip(self.weights, grad_w)]
        self.biases = [b - (alpha / len(mini_batch)) * gb for b, gb in zip(self.biases, grad_b)]
        avg_loss = total_loss / len(mini_batch)
        return avg_loss
    """
    Train the neural network using mini-batch stochastic gradient descent.
    """

    def SGD(self, data, epochs, alpha, decay, batch_size=32, test=None):
        n = len(data)
        losses = []
        for j in range(epochs):
            print(f"training epoch {j+1}/{epochs}")
            random.shuffle(data)
            for k in tqdm(range(n // batch_size)):
                mini_batch = data[k * batch_size : (k + 1) * batch_size]
                loss = self.update_mini_batch(mini_batch, alpha)
                losses.append(loss)
            alpha *= decay
            if test:
                print(f"Epoch {j+1}: eval accuracy: {self.evaluate(test)}")
            else:
                print(f"Epoch {j+1} complete")
        return losses

    """
    Returns classification accuracy of network on test_data.
    """

    def evaluate(self, test_data):
        test_results = [
            (np.argmax(self.feedforward(x)[1][-1]), y) for (x, y) in test_data
        ]
        return sum(int(x == y) for (x, y) in test_results) / len(test_data)

    def loss_function(self, y, y_prime):
        return 0.5 * np.sum((y - y_prime) ** 2)

    """
    Returns the gradient of the squared error loss function.
    """

    def loss_derivative(self, output_activations, y):
        return output_activations - y


def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))


def sigmoid_prime(z):
    return sigmoid(z) * (1 - sigmoid(z))
    
