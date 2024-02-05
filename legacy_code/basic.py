"""
This is an implementation of a neural network consisting of a single neuron with
w1, w2 and bias as its parameters. Training data contains truth tables of several basic
logic gates, that the neuron learns to mimic.
There are 2 approaches that can be used for driving the cost function towards zero:
1. Finite differences (approximates the partial derivative value by nudging each of the parameters by <eps>
                       and estimating the resulting difference in the cost)
2. Gradient descent (computes proper partial derivatives for each of the parameters)
"""
import time
import numpy as np
import random
import matplotlib.pyplot as plt


def sigmoid(x):
    """
    Helper function for computing sigmoid(x)
    """
    return 1 / (1 + np.exp(-x))

def cost(w1, w2, b, train):
    """
    Helper function for the finite difference approach. 
    Computes the error value for current <w1>, <w2>, <b> on the <train> data set.
    """
    result = 0
    # iterate through each sample of training data
    for data in train:
        x1 = data[0]
        x2 = data[1]
        # compute an activation value of the neuron
        a = sigmoid(x1 * w1 + x2 * w2 + b)
        # compute the error value as a square of the difference
        # between expected and actual output
        err = a - data[2]
        result += err**2
    # compute an average error value across all samples of data
    result /= len(train)

    return result

def finite_diff(train, n):
    """
    Approximates the parameter values of a single neuron on the <train> data set.

    Uses finite differences approach for estimating the value of the derivative for each
    of parameters.
    Input: training data set, <n> - number of training iterations
    Output: trained state of the neuron (w1, w2, b)
    """
    lower = -10
    upper = 10
    range_width = upper - lower
    w1 = random.random() * range_width + lower
    w2 = random.random() * range_width + lower
    b  = random.random() * range_width + lower

    # comment this out if you want 
    # random initialization of parameters
    w1 = w2 = b = 0

    # constants used for derivatives
    # and adjusting the speed of learning
    eps  = 1e-3
    rate = 1

    # report the initial state of the model
    print("---------------------")
    print("FINITE DIFFERENCES")
    report_state(w1, w2, b, train)
    print("\n<<<TRAINING>>>\n")

    # initialize the graph of
    # Cost / Number of training iterations 
    x_vals = []
    y_vals = []

    # train the model n-times
    start = time.time()
    for iter in range(n):
        # compute the current cost value
        c = cost(w1, w2, b, train)
        # add values to the graph
        x_vals.append(iter)
        y_vals.append(c)
        # approximate derivative values of the cost function 
        # for the current w1, w2 and bias values
        dw1 = (cost(w1 + eps, w2, b, train) - c)/eps
        dw2 = (cost(w1, w2 + eps, b, train) - c)/eps
        db  = (cost(w1, w2, b + eps, train) - c)/eps
        # adjust all of the parameters to minimize the cost function
        w1 -= rate * dw1
        w2 -= rate * dw2
        b  -= rate * db

    end = time.time()
    # report the final state of the model
    report_state(w1, w2, b, train)
    print ("\nElapsed time:", end - start)
    print("---------------------")
    # plot the graph of cost/iterations
    plot_cost(x_vals, y_vals)

    return w1, w2, b

def gcost(w1, w2, b, train):
    """
    Helper function for the gradient descent approach.
    Computes the gradient descent value as an average sum of
    partial derivatives of each parameter for the cost function.
    """
    # initialize the gradient values
    dw1 = dw2 = db = 0
    # iterate through each sample of training data
    for data in train:
        x1 = data[0]
        x2 = data[1]
        exp = data[2]
        # compute an activation value of the neuron
        a = sigmoid(x1 * w1 + x2 * w2 + b)
        # expression for partial derivative computation
        der_expr = 2 * (a - exp) * a * (1 - a)
        # add the derivative values to the gradient
        db += der_expr
        dw1 += der_expr * x1
        dw2 += der_expr * x2
    # compute an average sum of derivative values across all samples
    db /= len(train)
    dw1 /= len(train)
    dw2 /= len(train)

    return dw1, dw2, db

def gradient(train, n):
    """
    Approximates the parameter values of a single neuron on the <train> data set.

    Uses partial derivative approach for estimating the value of the derivative for each
    of parameters.
    Input: training data set, <n> - number of training iterations
    Output: trained state of the neuron (w1, w2, b)
    """
    lower = -10
    upper = 10
    range_width = upper - lower
    w1 = random.random() * range_width + lower
    w2 = random.random() * range_width + lower
    b  = random.random() * range_width + lower

    # comment this out if you want 
    # random initialization of parameters
    w1 = w2 = b = 0

    # constant used for adjusting
    # the steepness of gradient descent
    rate = 1

    # report the initial state of the model
    print("---------------------")
    print("GRADIENT DESCENT")
    report_state(w1, w2, b, train)
    print("\n<<<TRAINING>>>\n")

    # initialize the graph of
    # Cost / Number of training iterations 
    x_vals = []
    y_vals = []

    # train the model n-times
    start = time.time()
    for iter in range(n):
        # compute the current cost value
        c = cost(w1, w2, b, train)
        # add values to the graph
        x_vals.append(iter)
        y_vals.append(c)
        # Compute the gradient descent value of the cost function 
        # for the current <w1>, <w2> and <b> values
        dw1, dw2, db = gcost(w1, w2, b, train)
        # adjust all of the parameters to minimize the cost function
        w1 -= rate * dw1
        w2 -= rate * dw2
        b  -= rate * db
    
    end = time.time()
    # report the final state of the model
    report_state(w1, w2, b, train)
    print ("\nElapsed time:", end - start)
    print("---------------------")
    # plot the graph of cost/iterations
    plot_cost(x_vals, y_vals)

    return w1, w2, b

def report_state(w1, w2, b, train):
    """
    Prints out the current state of the model
    """
    # prints out the parameter states and the cost
    print("Parameters:")
    print("W1:   ", w1, "\n", "W2:   ", w2, "\n", "Bias: ", b, sep='')
    print("\nCost:", cost(w1, w2, b, train))
    # pushes the input data through the neuron and prints out its output
    output = []
    print("\nOutput of the neuron:")
    for i in range(2):
        for j in range(2):
            output.append((i , j, round(sigmoid(i*w1 + j*w2 + b), 2)))
    for row in output:
        print(row)

def plot_cost(x_vals, y_vals):
    """
    Plots the function of cost vs. number of training iterations
    """
    plt.plot(x_vals, y_vals)
    plt.xlabel('Number of iterations')
    plt.ylabel('Cost')
    plt.grid(True)
    plt.show()


def main ():
    # training data sets for OR, AND and NAND
    or_train = {(0, 0, 0),
                (1, 0, 1),
                (0, 1, 1),
                (1, 1, 1)}
    and_train = {(0, 0, 0),
                 (1, 0, 0),
                 (0, 1, 0),
                 (1, 1, 1)}
    nand_train = {(0, 0, 1),
                  (1, 0, 1),
                  (0, 1, 1),
                  (1, 1, 0)}

    # the gradient descent approach is ~85% faster
    # than the finite difference computation
    params = gradient(nand_train, 10000)
    params = finite_diff(nand_train, 10000)

if __name__ == "__main__":
    main()