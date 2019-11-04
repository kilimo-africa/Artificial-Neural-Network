from numpy import exp, array, dot

class NeuralNetwork():
    def __init__(self, layer1, layer2):
        self.layer1 = layer1
        self.layer2 = layer2

    # The Sigmoid function, which describes an S shaped curve.
    # We pass the weighted sum of the inputs through this function to
    # normalise them between 0 and 1.
    def __sigmoid(self, x):
        return 1 / (1 + exp(-x))

    # The derivative of the Sigmoid function.
    # This is the gradient of the Sigmoid curve.
    # It indicates how confident we are about the existing weight.
    def __sigmoid_derivative(self, x):
        y = self.__sigmoid(x)
        return y * (1 - y)

    # We train the neural network through a process of trial and error.
    # Adjusting the synaptic weights each time.
    def train(self, training_set_inputs, training_set_outputs, number_of_training_iterations):
        for iteration in range(number_of_training_iterations):
            # Pass the training set through our neural network
            output_from_layer_1, output_from_layer_2 = self.think(training_set_inputs)

            # Calculate the error for layer 2 (The difference between the desired output
            # and the predicted output).
            layer2_error = training_set_outputs - output_from_layer_2
            layer2_delta = layer2_error * self.__sigmoid_derivative(output_from_layer_2)

            # Calculate the error for layer 1 (By looking at the weights in layer 1,
            # we can determine by how much layer 1 contributed to the error in layer 2).
            layer1_error = layer2_delta.dot(self.layer2.T)
            layer1_delta = layer1_error * self.__sigmoid_derivative(output_from_layer_1)

            # Calculate how much to adjust the weights by
            layer1_adjustment = training_set_inputs.T.dot(layer1_delta)
            layer2_adjustment = output_from_layer_1.T.dot(layer2_delta)

            # Adjust the weights.
            self.layer1 += layer1_adjustment
            self.layer2 += layer2_adjustment

    # The neural network thinks.
    def think(self, inputs):
        output_from_layer1 = self.__sigmoid(dot(inputs, self.layer1))
        output_from_layer2 = self.__sigmoid(dot(output_from_layer1, self.layer2))
        return output_from_layer1, output_from_layer2

    # The neural network prints its weights
    def print_weights(self):
        print ("Layer 1 (2 neurons, each with 3 inputs): ")
        print (self.layer1)
        print ("Layer 2 (1 neuron, with 2 inputs):")
        print (self.layer2)

if __name__ == "__main__":

    # Create layer 1 (2 neurons, each with 3 inputs)
    layer1 = array([[0.2, 0.1], [0.3, 0.1], [0.2, 0.1]])

    # Create layer 2 (a single neuron with 2 inputs)
    layer2 = array([[0.5 , 0.1]]).T

    #create an input list

    # Combine the layers to create a neural network
    neural_network = NeuralNetwork(layer1, layer2)

    #print("Weights 1: ", layer1)
    #print("Weights 2: ", layer2)

    #print ("Stage 1) Random starting synaptic weights: ")
    neural_network.print_weights()

    # The training set. We have 7 examples, each consisting of 3 input values
    # and 1 output value.
    #training_set_inputs = array([[0, 0, 1], [0, 1, 1], [1, 0, 1], [0, 1, 0], [1, 0, 0], [1, 1, 1], [0, 0, 0]])
    #training_set_outputs = array([[0, 1, 1, 1, 1, 0, 0]]).T

    training_set_inputs = array([[0.273, 0.5, 0.6], [0.5, 0.6, 0.1], [0.6, 0.1, 0], [0.1, 0, 0.8], [0, 0.8, 1], [0.8, 1, 0.6]])

    training_set_outputs = array([[0.1], [0], [0.8], [1], [0.6], [0.5]])

    # Train the neural network using the training set.
    # Do it 60,000 times and make small adjustments each time.
    neural_network.train(training_set_inputs, training_set_outputs, 60000)

    #Print weights after training
    print("Weights after training")
    print("Weights 1: ", layer1)
    print("Weights 2: ", layer2)

    # Test the neural network with a new situation
    
