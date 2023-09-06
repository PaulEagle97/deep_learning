"""
An implementation of the most basic ML algorithm, that approximates the <m> ratio 
for linear functions of the type y = mx.
Either a finite difference approach or a partial derivative computation can be used
for driving the cost function towards zero.
"""

import random
import matplotlib.pyplot as plt

def cost(w, train):
    """
    Helper function for approximating m-parameter of linear functions.
    Computes an error value of the current <w> on the <train> data set.
    """
    result = 0
    for data in train:
        x = data[0]
        y = x * w
        d = y - data[1]
        result += d**2
    
    result /= len(train)

    return result

def dw_cost(w, train):
    """
    Helper function for computing the gradient value of the cost function.
    Computes the partial derivative value for current <w> on the <train> data set.
    """
    result = 0
    for data in train:
        x = data[0]
        y = data[1]
        dval = 2 * (x * w - y) * x
        result += dval
    
    result /= len(train)

    return result

def finite_diff(train, n):
    """
    Approximates the value of <m> for simple linear functions of type:
    y = mx
    Uses finite differences approach for estimating the value of the derivative.
    Input: training data set, <n> - number of training iterations
    Output: approximation of <m>
    """
    # initializing W to some random value 
    # in the range (lower - upper):
    lower = -100
    upper = 100
    range_width = upper - lower
    w = random.random() * range_width + lower

    eps = 1e-3
    rate = 1e-2

    # train the model n-times
    for _ in range(n):
        # approximate derivative value for current W
        dcost = (cost(w + eps, train) - cost(w, train))/eps
        # drive W towards local minima of the cost function
        w -= rate * dcost
    
    return w

def part_deriv(train, n):
    """
    Approximates the value of <m> for simple linear functions of type:
    y = mx
    Uses partial derivative approach for estimating the value of the derivative.
    Input: training data set, <n> - number of training iterations
    Output: approximation of <m>
    """
    # initializing W to some random value 
    # in the range (lower - upper):
    lower = -100
    upper = 100
    range_width = upper - lower
    w = random.random() * range_width + lower

    rate = 1e-2

    # train the model n-times
    for _ in range(n):
        # compute exact derivative value for current W
        dcost = dw_cost(w, train)
        # drive W towards local minima of the cost function
        w -= rate * dcost
    
    return w

def plot_cost(range_w, train):
    """
    Plots the cost vs. a range of W-values
    """
    x_vals = []
    y_vals = []
    for w in range (-range_w, range_w+1):
        x_vals.append(w)
        y_vals.append(cost(w, train))

    plt.plot(x_vals, y_vals)
    plt.xlabel('W')
    plt.ylabel('Cost')
    plt.grid(True)
    plt.show()

def main ():
    # defining the <m> value for the y = mx
    m = 3
    # creating a training set
    train = {(x, m * x) for x in range(5)}
    print (part_deriv(train, 100))
    plot_cost(10, train)


if __name__ == "__main__":
    main()