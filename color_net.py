import numpy as np


def squish_color(color_value):
    return np.around(color_value /2.55, decimals=2) / 100


class NeuralNetwork():

    def __init__(self):
        np.random.seed(1)

        self.synaptic_weights = 2 * np.random.random((3,1)) -1

    def sigmoid(self, x):
        return 1 / (1 + np.exp(x))

    def sigmoid_derivative(self, x):
        return x * (1 - x)

    def train(self, training_inputs, training_outputs, training_iterations):

        for iteration in range(training_iterations):

            output = self.think(training_inputs)

            error = training_outputs - output

            adjustments = np.dot(training_inputs.T, error * self.sigmoid_derivative(output))

            self.synaptic_weights += adjustments

    def think(self, inputs):

        inputs = squish_color(inputs.astype(float))
        output = self.sigmoid(np.dot(inputs, self.synaptic_weights))

        return output


if __name__ == '__main__':

    neural_network = NeuralNetwork()

    print('Random synaptic weights : ')
    print(neural_network.synaptic_weights)

    training_inputs = np.array([[0,0,0],
                            [255,255,255],
                            [255,255,255],
                            [0, 0, 0],
                            [255, 204, 153],
                            [51, 51, 255],
                            [0,255, 255],
                            [102, 0, 102],
                            ])

    np.random.seed(1)

    #training_inputs = squish_color(training_inputs)

    training_outputs = np.array([[0,1,1,0,0,1,0,1]]).T

    neural_network.train(training_inputs, training_outputs, 50000)

    print("Synaptic weights after training : ")
    print(neural_network.synaptic_weights)

    A = str(input('Input 1 : '))
    B = str(input('Input 2 : '))
    C = str(input('Input 3 : '))

    print("New situation : input data = ", A, B, C)
    print("Output data : ")
    print(np.around(neural_network.think(np.array([A, B, C])), decimals=2))

