import random


def function(x):
    return x**2


input_value = 10  # random starting input value


def learn(learn_rate):
    global input_value

    # find the slope between 2 very close points (assumed as the approx slope at a point)
    h = 0.00001  # the higher this value, the more accurate the slope
    # basically (y2 - y1)/(x2 - x1)
    delta_output = function(input_value + h) - function(input_value)
    slope = delta_output / h

    # the slope value tells us whether the output of the function is increasing or decreasing at the current input value,
    # so subtracting the slope from the input value should cause the output of the function to decrease (gradient descent)
    # Note: the slope is only accurate for a tiny region around the current value (if the function has sharp turns for example)
    # so the input should not be changed to drastically (hence the learn_rate parameter)
    input_value -= slope * learn_rate
    return function(input_value)


for i in range(10):
    print(learn(0.1))
