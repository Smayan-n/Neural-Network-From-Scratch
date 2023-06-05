from Layer import Layer


class NeuralNetwork:
    def __init__(self, layer_sizes: list[int]):
        # list of layer objects
        self.layers: list[Layer] = [None] * (len(layer_sizes) - 1)

        # depending on the layerSizes passed in, makes the layers needed for the NN
        for i in range(len(layer_sizes) - 1):
            self.layers[i] = Layer(layer_sizes[i], layer_sizes[i + 1])

    def calculate_outputs(self, inputs):
        """passes inputs through the neural network and returns output"""

        layer_activations = inputs
        for layer in self.layers:
            layer_activations = layer.calculate_outputs(layer_activations)

        return layer_activations

    def calculate_cost(self, data_point, expected_output):
        """calculates output layer cost for one data point"""

        # calculate outputs
        outputs = self.calculate_outputs(data_point)
        output_layer = self.layers[len(self.layers) - 1]
        cost = 0

        # get cost for each output node
        for i in range(len(outputs)):
            cost += output_layer.node_cost(outputs[i], expected_output[i])

        return cost

    def calculate_average_cost(self, data_points, expected_outputs):
        """calculates output layer cost for multiple data points"""
        total_cost = 0
        for i in range(len(data_points)):
            total_cost += self.calculate_cost(data_points[i], expected_outputs[i])

        # return average cost
        return total_cost / len(data_points)

    def learn(self, training_data, expected_outputs, learn_rate):
        """uses gradient decent to train the neural network (using finite-difference method)
        1 call to the learn function is 1 learning iteration"""

        h = 0.0001
        original_cost = self.calculate_average_cost(training_data, expected_outputs)

        # loop through whole network to see how each individual weight and bias affects the cost
        for layer in self.layers:

            # for weights
            for i in range(layer.incoming_nodes):
                for j in range(layer.outgoing_nodes):

                    # change weights by a small amount and see how it affects the cost (Δc / Δw - cost gradient)
                    layer.weights[i][j] += h
                    delta_cost = (
                        self.calculate_average_cost(training_data, expected_outputs)
                        - original_cost
                    )
                    # reset weight
                    layer.weights[i][j] -= h
                    layer.weight_cost_gradients[i][j] = delta_cost / h

            # for biases
            for i in range(len(layer.biases)):

                # change biases by a small amount as see how it affects the cost (Δc / Δb - cost gradient)
                layer.biases[i] += h
                delta_cost = (
                    self.calculate_average_cost(training_data, expected_outputs)
                    - original_cost
                )
                # reset bias
                layer.biases[i] -= h
                layer.bias_cost_gradients[i] = delta_cost / h

        # apply all gradients
        for layer in self.layers:
            layer.apply_gradients(learn_rate)

        # return cost after every learning iteration for visualization
        return original_cost
