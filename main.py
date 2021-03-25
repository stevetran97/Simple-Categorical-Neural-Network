import numpy as np


class NeuralNetwork():
    def __init__(self):
        # Random Seed if needed for debugging
        # np.random.seed(1)
        # Set and Get Synaptic Weights
        self.synaptic_weights = 2 * np.random.random((3, 1)) - 1

    # Helpers
    def CalcSigmoid(self, x):
        """
        Takes in x
        Returns normalized x using the Sigmoid Function
        """
        return 1 / (1 + np.exp(-x))

    def TakeSigmoidDeriv(self, x):
        """
        Takes in x
        Evaluates x through the derivative of the Sigmoid
        Used for Weight Adjustment
        """
        return x*(1-x)

    def Train(self, x_training, y_training, training_iterations):
        """
        Model takes results of the last round of calculations
        Calculates and makes adjustments to for the next round
        Repeats process until the model accurately predicts the training set
        """
        for iteration in range(training_iterations):
            # Calculate y_output
            y_model_outputs = self.Think(
                x_training)

            # Calculate Error
            error = y_training - y_model_outputs

            # Adjustments
            adjustments = np.dot(x_training.T, error *
                                 self.TakeSigmoidDeriv(y_model_outputs))
            self.synaptic_weights += adjustments

    def Think(self, inputs):
        """
        The process/Mechanism in which the model evaluates its information
        Uses a combination of weighted x's to categorize the input
        """

        y_output = self.CalcSigmoid(
            np.dot(inputs, self.synaptic_weights))
        return y_output


# Main
# Model Inputs
# Variables
# Training Set
x_training = np.array([[0, 0, 1],
                       [1, 1, 1],
                       [1, 0, 1],
                       [0, 1, 1]])
y_training = np.array([[0, 1, 1, 0]]).T
training_iterations = 50000

test_inputs = [[1, 0, 0],
               [1, 1, 0],
               [0, 0, 0],
               [0, 1, 0]]

neural_network = NeuralNetwork()
# Call training
neural_network.Train(x_training, y_training, training_iterations)
# Find values
print("\nFinal Calculated Synaptic Weights are: ")
print(neural_network.synaptic_weights)

print("\nCheck Training Inputs: ")
print(neural_network.Think(x_training))

print("\nTesting the Neural Network with Test Inputs ")
print(neural_network.Think(test_inputs))
