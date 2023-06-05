from Layer import Layer
from NeuralNetwork import NeuralNetwork
import matplotlib.pyplot as plt
import random


data_points = [
    [0.72, 0.97],
    [0.9, 0.76],
    [0.32, 0.55],
    [0.9, 1.0],
    [0.02, 0.79],
    [0.54, 0.41],
    [0.54, 0.64],
    [0.71, 0.18],
    [0.17, 0.22],
    [0.22, 0.89],
    [0.03, 0.38],
    [0.8, 0.46],
    [0.2, 0.8],
    [0.0, 0.04],
    [0.92, 0.81],
    [0.16, 0.17],
    [0.14, 0.46],
    [0.66, 0.64],
    [0.52, 0.12],
    [0.77, 0.26],
    [0.84, 0.37],
    [0.1, 0.96],
    [0.88, 0.78],
    [0.38, 0.99],
    [0.57, 0.82],
    [0.44, 0.45],
    [0.21, 0.88],
    [0.92, 0.95],
    [0.18, 0.22],
    [0.87, 0.47],
    [0.85, 0.68],
    [0.84, 0.49],
    [0.74, 0.74],
    [0.67, 0.14],
    [0.26, 0.94],
    [0.95, 0.09],
    [0.48, 0.87],
    [0.93, 0.27],
    [0.92, 0.18],
    [0.7, 0.02],
    [0.34, 0.8],
    [0.07, 0.31],
    [0.18, 0.08],
    [0.51, 0.62],
    [0.51, 0.15],
    [0.75, 0.57],
    [0.44, 0.1],
    [0.54, 0.57],
    [0.29, 0.67],
    [0.58, 0.47],
]
expected_outputs = []  # 1 for safe and #0 for unsafe

# produce points and expected outputs
data_points = []
for i in range(250):
    data_points.append([round(random.random(), 2), round(random.random(), 2)])

for point in data_points:
    if point[0] > 0.40 and point[1] < 0.55:
        expected_outputs.append([1, 0])
    else:
        expected_outputs.append([0, 1])


def plot_graph():
    # plot data points
    for i, point in enumerate(data_points):
        if expected_outputs[i] == [1, 0]:
            plt.plot([point[0]], [point[1]], marker="o", color="green")
        else:
            plt.plot([point[0]], [point[1]], marker="o", color="red")

    plt.xlim(0, 1)
    plt.ylim(0, 1)

    # naming the x axis
    plt.xlabel("x - axis")
    # naming the y axis
    plt.ylabel("y - axis")

    # giving a title to my graph
    plt.title("Neural Network Visualization")

    # function to show the plot
    plt.show()


def plot_cost(costs):

    plt.plot([i for i in range(len(costs))], costs, color="red", marker="o")

    plt.xlim(0, len(costs))
    plt.ylim(0, 1)

    # naming the x axis
    plt.xlabel("batch #")
    # naming the y axis
    plt.ylabel("cost")

    # giving a title to my graph
    plt.title("Cost Vs time")

    # function to show the plot
    plt.show()


def train_network():
    # 2 inputs - spot size, spike size
    # 2 outputs - safe,
    #           unsafe
    layerSizes = [2, 5, 2]
    nn = NeuralNetwork(layerSizes)

    costs = []

    # mini-batch gradient descent approach
    learning_iterations = 100
    batches = 10
    for i in range(learning_iterations):
        # NOTE: maybe all gradients should be applied only at the end of one epoch(iteration)
        for batch_index in range(batches):

            start_index = batch_index * (len(data_points) // batches)
            cost = nn.learn(
                data_points[start_index : start_index + (len(data_points) // batches)],
                expected_outputs[
                    start_index : start_index + (len(data_points) // batches)
                ],
                # learning rate
                0.1,
            )
            # print(f"Iteration: {i}    Batch: {batch_index}    Cost: {cost}")
        print(f"Iteration: {i}  Cost: {cost}")

        costs.append(cost)

    # Batch gradient descent approach
    # iterations = 250
    # for i in range(iterations):
    #     cost = nn.learn(data_points, expected_outputs, 0.1)
    #     print(f"Iteration: {i}    Cost: {cost}")
    #     costs.append(cost)

    print(
        f"Final Model Cost: {nn.calculate_average_cost(data_points, expected_outputs)}"
    )

    # test network with unseen data
    print(gauge_network_performance(nn, data_points, expected_outputs))

    new_data_points = []
    new_expected_outputs = []  # 1 for safe and #0 for unsafe

    # produce points and expected outputs
    new_data_points = []
    for i in range(1000):
        new_data_points.append([round(random.random(), 2), round(random.random(), 2)])

    for point in new_data_points:
        if point[0] > 0.40 and point[1] < 0.55:
            new_expected_outputs.append([1, 0])
        else:
            new_expected_outputs.append([0, 1])

    print(
        "New data "
        + gauge_network_performance(nn, new_data_points, new_expected_outputs)
    )

    plot_cost(costs)


def gauge_network_performance(nn, data_points, expected_outputs):
    correct = 0
    for i in range(len(data_points)):
        outputs = nn.calculate_outputs(data_points[i])
        index = outputs.index(max(outputs))

        output = [1, 0] if index == 0 else [0, 1]

        if output == expected_outputs[i]:
            correct += 1

    return f"Predictions Correct: {correct} / {len(data_points)}  ({correct/len(data_points)*100}%)"


train_network()
# plot_graph()
