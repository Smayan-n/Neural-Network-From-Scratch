import math, random

# keep a constant random seed so all network learning tests are the same
random.seed(0)

# layer class
class Layer:
    def __init__(self, incoming_nodes, outgoing_nodes):
        # number of nodes / inputs feeding into this layer
        self.incoming_nodes = incoming_nodes
        # number of nodes / outputs leaving this layer
        self.outgoing_nodes = outgoing_nodes

        # weights and biases (weights are initialized to random values and biases start at 0)
        self.weights = [
            [0 for i in range(outgoing_nodes)] for i in range(incoming_nodes)
        ]  # each row holds weights connecting an incoming node to an outgoing node
        self.biases = [0] * outgoing_nodes  # a bias for each outgoing node
        self.initialize_weights()

        # cost gradients for weights and biases
        self.weight_cost_gradients = [
            [0 for i in range(outgoing_nodes)] for i in range(incoming_nodes)
        ]
        self.bias_cost_gradients = [0] * outgoing_nodes

    def initialize_weights(self):
        """initializes weights to values between -1 and 1"""
        # rows, cols
        # weights
        for r in range(self.incoming_nodes):
            for c in range(self.outgoing_nodes):
                self.weights[r][c] = 2 * random.random() - 1

    def calculate_outputs(self, inputs):
        """calculates and returns output Activations for this layer"""
        # there are self.incoming_nodes inputs to the function
        # there are self.outgoing_nodes weighted outputs calculated

        weighted_inputs = [0] * self.outgoing_nodes

        # calculate weights and biases
        weighted_input = 0
        for j in range(self.outgoing_nodes):
            # add bias
            weighted_input = self.biases[j]
            for i in range(self.incoming_nodes):
                # add weights
                weighted_input += inputs[i] * self.weights[i][j]

            # add calculated output to array
            weighted_inputs[j] = weighted_input

        # return output activations
        return [
            self.activation_function(weighted_input)
            for weighted_input in weighted_inputs
        ]

    def apply_gradients(self, learn_rate):
        """apply cost gradients to weights and biases"""
        # learn rate is to adjust sensitivity of changes to weights and biases

        for j in range(self.outgoing_nodes):
            self.biases[j] -= self.bias_cost_gradients[j] * learn_rate
            for i in range(self.incoming_nodes):
                # print(self.cost_gradient_weights[i][j] * learn_rate)
                self.weights[i][j] -= self.weight_cost_gradients[i][j] * learn_rate

    def node_cost(self, output_activation, expected_output):
        """cost function for layer - needs to be run for all nodes in layer"""
        error = output_activation - expected_output
        return error * error  # to make it positive and exaggerate larger errors

    def activation_function(self, x):
        """returns output activation for a weighted input"""
        # Sigmoid
        # return 1 / (1 + math.exp(-x))

        # RELu
        return max(0, x)
