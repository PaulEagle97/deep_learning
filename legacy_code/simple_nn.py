"""
Contains an implementation of a simple NN that is capable of mimicking the behavior 
of various fundamental logic gates, such as AND, OR or XOR.
Employs a finite difference approach for computing the partial derivative values
for each of parameters.
"""
import random
import numpy as np
import matplotlib.pyplot as plt

class SimpleNN:
    """
    A class implementation of a simple NN consisting of 3 neurons
    with the following configuration:
    2in x 2n x 1n x 1out
    """
    def __init__(self, train, rand_range):
        """
        Constructs the class instance and initializes the parameter/gradient tables.
        """
        self._train = train
        self._pars = dict()
        self._grad = dict()
        # constant for the derivative approximation
        self._eps = 1e-1
        # constant - speed of learning
        self._rate = 1
        
        for param_name in self.name_gen():
            # initialize the parameters randomly
            self._pars[param_name] = random.uniform(-rand_range, rand_range)
            # initialize the table of partial derivative values
            self._grad[param_name] = 0


    def name_gen(self):
        """
        Helper function-generator of strings for iterating through 
        each of parameters.
        """
        neur_nms = ("or", "nand", "and")
        par_nms = ("w1", "w2", "b")
        for neuron_name in neur_nms:
            for parameter_name in par_nms:
                full_name = neuron_name + "_" + parameter_name
                yield full_name

    def __str__(self):
        """
        Returns a string - the current state of the model
        """
        # show the current state of each parameter
        output = "\n1)Parameters:\n"
        for par_name, value in self._pars.items():
            temp = par_name + " = " + str(round(value, 3)) + "\n"
            output += temp
        # show the table of partial derivates for each parameter
        output += "\n2)Gradient:\n"
        for par_name, value in self._grad.items():
            temp = par_name + " = " + str(round(value, 3)) + "\n"
            output += temp

        output += f"\n3)Cost: {round(self.cost(), 3)}\n"
        # show the output of the model for each sample of input
        output += "\n4)Total NN performance:"
        for i in range(2):
            for j in range(2):
                output += f"\n({i}, {j}, {round(self.forward(i, j), 1)})"
        # show isolated output of each neuron
        output += "\n\n5)Output of the neurons:"
        output += self.get_neuron_state()

        return output

    def sigmoid(self, x):
        """
        Helper function for computing sigmoid(x)
        """
        return 1 / (1 + np.exp(-x))

    def forward(self, x1, x2):
        """
        Takes in input values and activates each of 3 neurons of NN.
        """
        #a  =  sigm(w1x1 + w2x2 + b)
        a1  =  self.sigmoid(self._pars["or_w1"]   * x1  +  self._pars["or_w2"]   * x2  +  self._pars["or_b"])
        a2  =  self.sigmoid(self._pars["nand_w1"] * x1  +  self._pars["nand_w2"] * x2  +  self._pars["nand_b"])
        a3  =  self.sigmoid(self._pars["and_w1"]  * a1  +  self._pars["and_w2"]  * a2  +  self._pars["and_b"])
        return a3

    def cost(self):
        """
        Computes the current cost of the NN.
        """
        result = 0
        # iterate through each sample of training data
        for data in self._train:
            x1 = data[0]
            x2 = data[1]
            # push the input values through each layer
            y = self.forward(x1, x2)
            # compute the error value as a square of the difference
            # between expected and actual output
            err = y - data[2]
            result += err**2
        # compute an average error value across all samples of data
        result /= len(self._train)

        return result

    def finite_diff(self):
        """
        Approximates values of partial derivatives of each parameter
        in relation to the cost function (gradient), and stores them in <self._grad>.
        The current parameter values remain unchanged.  
        """
        # compute the initial cost
        c = self.cost()
        # iterate through each of parameters
        for param_name in self.name_gen():
            # save the original parameter value
            saved = self._pars[param_name]
            # tweak the parameter by <eps>
            self._pars[param_name] += self._eps
            # compute the resulting change of the cost function
            # (approximation of the parameter's partial derivative)
            # and store it in a separate table of gradients
            self._grad[param_name] = (self.cost() - c) / self._eps
            # restore the original value of the parameter
            self._pars[param_name] = saved

    def apply_grad(self):
        """
        Modifies the current parameter values 
        by substracting the corresponding values from
        the gradient table.
        """
        for param_name in self.name_gen():
            self._pars[param_name] -= self._rate * self._grad[param_name]

    def learn(self, n):
        """
        Trains the NN on the training data set <n> times,
        prints out NN state and plots the cost function.
        Uses the finite difference approach.
        """
        # lists of values for the plot of cost.
        x_vals = []
        y_vals = []
        # compute the initial gradient table
        self.finite_diff()
        # print out the initial state of NN
        print("\n<<< Initial state >>>", "\n", self, sep='')
        # repeat <n> times:
        for iter in range(n):
            # update the plot values
            x_vals.append(iter)
            y_vals.append(self.cost())
            # train the model
            self.apply_grad()
            self.finite_diff()
        # print out the final state of NN
        print("\n<<< Final state >>>", "\n", self, sep='')
        # plot the cost
        self.plot_cost(x_vals, y_vals)

    def plot_cost(self, x_vals, y_vals):
        """
        Plots the function of cost vs. number of training iterations
        """
        plt.plot(x_vals, y_vals)
        plt.xlabel('Number of iterations')
        plt.ylabel('Cost')
        plt.grid(True)
        plt.show()

    def get_neuron_state(self):
        """
        Pushes the training data through each of the neurons (separately)
        and prints out their outputs.
        Allows to see the type of logic gate behavior per each neuron. 
        """
        new_neuron = True
        output = ''
        # iterate through each parameter
        for param_name in self.name_gen():
            if new_neuron:
                # add the separator
                output += "\n-----------"
                # get the name of the neuron
                name_str = param_name.split('_')[0]
                # add the header with the neuron's name
                output += f"\n'{name_str.upper()}' neuron:\n"
                # reset the flag
                new_neuron = False
                # reset the parameter list
                params = []
            # add parameter to the list
            params.append(param_name)
            # check if the list contains 
            # all 3 parameters (w1, w2, b)
            if len(params) == 3:
                for i in range(2):
                    for j in range(2):
                        # push each sample through the neuron and store the output
                        output += f"({i}, {j}, {round(self.forward_single(params, i, j), 1)})\n"
                # start the new block of parameters
                new_neuron = True
        
        return output

    def forward_single(self, params, x1, x2):
        """
        Pushes input through a single neuron and returns its activation.
        """
        return self.sigmoid(self._pars[params[0]]*x1 + self._pars[params[1]]*x2 + self._pars[params[2]])

def main ():
    # training data sets for OR, AND, NAND and XOR
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
    xor_train = {(0, 0, 0),
                 (1, 0, 1),
                 (0, 1, 1),
                 (1, 1, 0)}

    xor_model = SimpleNN(xor_train, 1)
    xor_model.learn(5000)

if __name__ == "__main__":
    main()