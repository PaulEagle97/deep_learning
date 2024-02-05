"""
Features a flexible class implementation of neural networks from the first principles, as well as
several auxiliary methods for testing and debugging.
"""
import time
import copy
import itertools
import numpy as np
import matplotlib.pyplot as plt
import multiprocessing
from multiprocessing import Process, Queue
np.seterr(under='warn')

class NN:
    """
    A class implementation of a fully-connected neural network with a customizable size and configuration.
    It employs only the core Python and numpy functions. The performance optimization is done via numpy vectorization.
    Multiple activation and loss functions are supported and can be easily extended.
    Provides two plotting methods (static / real-time).
    Allows for storing the image of the network after the training in a file, as well as importing other saved images.
    """
    def __init__(self, nn_layout, rand_range, train_data, name, rate=1):
        """
        The function is the constructor for a `NN` class, which initializes the parameters and the state 
        of the newly created neural network object.
        The `NN` object state is defined by mapping between each layer and the corresponding matrices that 
        contain values for weights and biases plus the name of the activation function. It is represented as 
        the dictionary with the following structure:
        `{layer: ([w, b], 'act_func')}`, where:
            - `w` is the matrix of weights for the layer;
            - `b` is the matrix of biases for the layer;
            - `'act_func'` is the activation function for the layer;
            - `layer` is the number of the layer.

        :param nn_layout: A layout of the neural network. Defined as a tuple of type:
        `((inp, l1, l2, ... , ln), act_map)`, where:
            - `inp` denotes the size of the input;
            - `ln` denotes the number of neurons in the n-th layer.
            - `act_map` denotes types of activation funcs for each layer.
        :param rand_range: A tuple containing the lower and upper limits for random initialization of 
        parameter matrices
        :param train_data: A sequence of training data samples of type:
        `((a1, b1, c1, ...), (a2, b2, c2, ...), (an, bn, cn, ...))`, where:
            - each sample contains `j` input and `k` output values, with `j` defined by `nn_layout[0][0]`
        :param name: The name of the neural network that identifies it
        :param rate: Is used to define the speed of learning in the neural network. It determines how quickly 
        the network adjusts its weights and biases during the training process. Defaults to `1` (optional)
        """
        # to store shape of each parameter mx
        self._param_config = self._mx_size_map(nn_layout[0])
        self._nn_depth = len(self._param_config)
        # to store activation funcs for each layer
        self._act_funcs = nn_layout[1]
        # to map act. funcs and derivatives to each layer
        self._layer_config = self._act_func_map(self._act_funcs)
        # to store training dataset
        self._train = train_data.copy()
        self._inp_size = nn_layout[0][0]
        self._curr_cost = None
        # to store the state of NN
        self._params = dict()
        # to store activations
        self._acts = dict()
        # to store the gradient
        self._grad = dict()
        # constant for finite difference computation
        self._eps = 1e-1
        # to define the speed of learning
        self._rate = rate
        self._name = name
        # initializes the parameters and the gradient
        for layer in range(1, self._nn_depth + 1):
            # gets the sizes of parameter matrixes for each layer
            w_shape = self._param_config[layer][0]
            b_shape = self._param_config[layer][1]
            # creates two random matrixes for weights and biases per each layer
            w = np.random.uniform(rand_range[0], rand_range[1], size=w_shape)
            b = np.random.uniform(rand_range[0], rand_range[1], size=b_shape)
            self._params[layer] = [w, b]
            # initializes gradient table with empty matrixes of the same shape
            w_grad = np.zeros(shape=w_shape)
            b_grad = np.zeros(shape=b_shape)
            self._grad[layer] = [w_grad, b_grad]

    # helper methods for __init__
    def _mx_size_map(self, params):
        """ @public
        The function creates a mapping between each layer in a neural network and the sizes of
        the corresponding weight and bias matrices.
        
        :param params: The `params` parameter is a list that represents the number of neurons in each layer
        of a neural network. For example, if `params = (10, 20, 30)`, it means that the neural network has 3
        layers with 10 neurons in the input layer, 20 in the inner layer, and 30 in the output layer
        :return: a dictionary where the keys are the layer numbers and the values are tuples. Each tuple
        contains the size of the weight matrix and the size of the bias matrix for that layer:
                `{1:(w1_size, b1_size), 
                  2:(w2_size, b2_size),
                  n:(wn_size, bn_size)}`.
        """
        nn_layout = dict()
        for layer in range(1, len(params)):
            nn_layout[layer] = ((params[layer-1], params[layer]), (1, params[layer]))

        return nn_layout
    def _act_func_map(self, act_map):
        """ @public
        The function converts a provided map of activation functions for each layer into a
        dictionary with direct references to the functions and their derivatives for each layer.
        
        :param act_map: The `act_map` parameter is a dictionary that maps layer numbers to activation
        function names. Each key-value pair in the dictionary represents a layer in the neural network,
        where the key is the layer number (an integer) and the value is the name of the activation function
        (a string)
        :return: a new dictionary where each layer number is mapped to a tuple containing the activation
        function and its derivative for that layer.
        """
        # validate the input
        assert len(act_map) == self._nn_depth, "Invalid activation map (length)"
        assert all(key in range(1, self._nn_depth + 1) for key in act_map.keys()), "Invalid layer number(s)"
        assert all(isinstance(func_name, str) for func_name in act_map.values()), "Invalid function name(s)"

        # return a new dict with funcs and their derivatives mapped to each layer
        return {layer: (getattr(self, "_" + func_name), getattr(self, "_" + func_name + "_der")) for layer, func_name in act_map.items()}

    # methods for printing to the console
    def __str__(self, with_full_report=False):
        """
        The function returns a string representation of the current state of the model, including
        the model name, cost, parameters, and performance on training samples.
        
        :param with_full_report: The (optional) `with_full_report` parameter is a boolean flag that determines whether
        to include additional information in the string representation of the model's state. If
        `with_full_report` is set to `True`, the string will include the values of parameters for each
        neuron and the actual output of the model on all training samples. Defaults to `False`
        :return: a string that represents the current state of the model.
        """
        state_str  = f"1) Model: {self._name}\n"
        state_str += f"2) Cost: {self._cost()}\n"

        if with_full_report:
            # prints out parameter matrixes (weights and biases)
            # for each of the layers of the NN
            state_str += "3) State of NN parameters:\n"
            for layer, param_lst in self._params.items():
                state_str += f"Layer: {str(layer)}\n"
                for idx, param_mx in enumerate(param_lst):
                    if idx == 0:
                        state_str += "<<< Weights >>>\n"
                    else:
                        state_str += "\n<<< Biases >>>\n"
                    state_str += str(param_mx)
                state_str += "\n---------------\n"

            # prints out mapping between input values and
            # the actual output of the NN for all data samples
            state_str += "4) Total NN performance:"
            for data in self._train:
                state_str += "\n("
                input = np.array([data[ :self._inp_size]])
                for inp_val in input:
                    state_str += f"{inp_val}, "
                
                output = self._forward(input)
                state_str += f"{output})"

        return state_str
    def _cost(self):
        """
        The function computes the current cost value of the neural network by calculating the mean
        squared error (MSE) between the network's output and the expected output for each training data sample.
        :return: the current cost value of the neural network.
        """
        result = 0
        # iterates through each entry of the training data
        for data in self._train:
            # splits training data array 
            # into input/expected output subarrays
            input     = np.array([data[ :self._inp_size]])
            expected  = np.array([data[self._inp_size: ]])

            # feeds input subarray into the NN
            # collects the output data array
            output = self._forward(input)

            # verifies that there is no mismatch between 
            # the sizes of the expected and actual output arrays
            assert output.shape == expected.shape
            
            # iterates through both arrays and 
            # computes corresponding error values
            err_vals = (output - expected)[0] ** 2

            # adds the accumulated error to the total cost
            result += np.sum(err_vals)

        # computes the mean error value per 1 sample of training data
        result /= len(self._train)

        return result

    # methods for computing activations
    def _sigmoid(self, x):
        """
        The function computes the sigmoid of a given input.
        
        :param x: The parameter `x` is the input value for which we want to compute the sigmoid function
        :return: the value of sigmoid(x).
        """
        return 1 / (1 + np.exp(-x))
    def _relu(self, x):
        """
        The function computes the ReLU (Rectified Linear Unit) of a given input.
        
        :param x: The parameter `x` is a scalar or an array-like object representing the input to the ReLU
        function
        :return: the maximum value between 0 and the input value x.
        """
        return np.maximum(0, x)
    def _softmax(self, x):
        """
        The function computes the softmax in a stable manner of a given input.
        
        :param x: The parameter `x` is a numpy array representing the input values for which we want to
        compute the softmax function
        :return: the softmax of the input array x.
        """
        norm_x = x - np.max(x)
        exp_z = np.exp(norm_x)
        result = exp_z / np.sum(exp_z)

        return result

    # methods for computing activation derivatives
    def _sigmoid_der(self, x):
        """
        Computes the derivative values for f(x)=1/(1+e**(-x)).
        """
        return x * (1 - x)
    def _relu_der(self, x):
        """
        Computes the derivative values for f(x)=max(0, x).
        """
        return x > 0
    def _softmax_der(self, x):
        """
        Computes the derivative values for f(x)=softmax([x]).
        """
        return self._sigmoid_der(x)

    # methods for computing loss derivatives
    def _abs_der(self, x):
        """
        Computes the derivative values for f(x) = |x|.
        """
        # creates an array of ones with the same shape as input arr
        der_arr = np.ones_like(x)
        # replaces elements less than or equal to zero with -1
        der_arr[x <= 0] = -1
        return der_arr
    def _sqr_der(self, x):
        """
        Computes the derivative values for f(x)=x**2.
        """
        return 2 * x
    def _half_sqr_der(self, x):
        """
        Computes the derivative values for f(x)=(x**2)/2.
        """
        return x
    def _lncosh_der(self, x):
        """
        Computes the derivative values for f(x)=ln(cosh(x)).
        """
        return np.tanh(x)

    # core methods for the learning process
    def _forward(self, input_arr):
        """ @public
        The function takes an input array and passes it through each layer of a neural network,
        applying weights, biases, and activation functions to produce the final output.
        
        :param input_arr: The input_arr is a numpy array that represents the input to the neural network. It
        is the input that will be forwarded through each layer of the network
        :return: the activations of the last layer of the neural network.
        """
        # initializes input as the 0-th layer of activations
        a0 = input_arr
        self._acts[0] = a0
        # iterates through each layer of the NN
        for layer, param_lst in self._params.items():
            # gets activation func for current layer
            act_func = self._layer_config[layer][0]
            # retrieves values of weights
            # and biases of the layer
            w = param_lst[0]
            b = param_lst[1]
            # applies weights
            a1 = np.matmul(a0, w)
            # applies biases
            a1 += b
            # applies the activation function
            a1 = act_func(a1)
            # saves the resulting activations
            self._acts[layer] = a1
            # assigns the activations as
            # an input for the next layer
            a0 = a1
        # returns activations of the last layer
        return a0
    def _stochastic_descent(self, batch_ratio):
        """ @public
        The function implements the stochastic gradient descent algorithm for a neural
        network, updating the parameters based on the computed gradients.
        It does the following operations:
            Keeps track of the total accumulated error value and the count of current samples.
            Shuffles the training data randomly and iterates through each data sample.
            For each sample, it 
                1. splits the training data array into input and expected output subarrays;
                2. computes activations for each layer, and saves the last layer activations (actual output);
                3. sums up squares of error values of each activation in the last layer;
                4. performs backpropagation and computes the gradient.
            If the batch size is reached, it applies the accumulated gradient values to the parameters.
            Finally, it updates the current cost of the model as an average error value per one sample.
        
        :param batch_ratio: The `batch_ratio` parameter is a float value that represents the ratio of the
        total number of training samples that should be used in each batch. For example, if `batch_ratio` is
        set to `0.5`, it means that each batch will contain `50%` of the total training samples.
        """
        # keeps the total accumulated error value
        total_err = 0
        # keeps the count of current samples
        sample_counter = 0
        # shuffles the training data randomly
        np.random.shuffle(self._train)
        # iterates through each data sample
        for data in self._train:
            # splits training data array 
            # into input/expected output subarrays
            input     = np.array([data[ :self._inp_size]])
            expected  = np.array([data[self._inp_size: ]])

            # computes activations for each layer
            # and saves the last layer activations
            output = self._forward(input)

            # verifies that there is no mismatch between 
            # the sizes of the expected and actual output arrays
            assert output.shape == expected.shape

            # computes activation derivative values for the last layer
            # as a difference between expected and actual outputs of NN
            da_next = (output - expected)[0]

            # sums up squares of error values 
            # of each activation in the last layer
            total_err += np.sum(da_next ** 2)

            # performs backpropagation and computes the gradient
            self._backprop(da_next)

            sample_counter += 1
            # checks if the batch size is reached
            if sample_counter / len(self._train) >= batch_ratio:
                # applies accumulated gradient values to the parameters
                self._apply_grad(sample_counter)
                sample_counter = 0

        # applies the gradient of the samples
        # that didn't reach the size of the batch
        if sample_counter > 0:
            self._apply_grad(sample_counter)

        # updates the current cost of the model 
        # as an average error value per 1 sample
        self._curr_cost = total_err / len(self._train)    
    def _backprop(self, da_next):
        """ @public
        The function performs backpropagation to compute and accumulate the gradients of the weights and biases for later application to the parameters of a neural network.
        
        :param da_next: da_next is the derivative of the cost function with respect to the activations of
        the next layer. It represents the backpropagated error from the next layer
        """
        # iterates through each layer backwards starting from the last one
        for layer in reversed(range(1, len(self._params)+1)):

            # gets the weights for the current layer
            layer_wghts = self._params[layer][0]
            
            # gets the derivative of activation func for the current layer
            der_func = self._layer_config[layer][1]

            # computes the derivative expression vectorized for all activations
            der_expr = self._lncosh_der(da_next) * der_func(self._acts[layer][0])

            # computes bias derivatives for the entire layer
            db = der_expr

            # computes weights derivatives for the entire layer
            dw = der_expr[:, np.newaxis] * self._acts[layer-1][0]
            dw = dw.reshape(-1, len(self._acts[layer-1][0])).T

            # computes the gradient for current layer
            grad_mxs = [dw, db]

            # adds up the gradient values of the current sample
            # to the gradient sum of previously computed samples
            for idx, mx in enumerate(self._grad[layer]):
                mx += grad_mxs[idx]

            # computes activation derivatives for the previous layer
            da_next = der_expr @ layer_wghts.T
    def _apply_grad(self, num_samples):
        """ @public
        The function applies the stored gradient values to the parameters of a neural network.
        
        :param num_samples: The `num_samples` parameter represents the number of samples used to compute the
        gradient. It is used to normalize the gradient update step by dividing it by the number of samples.
        This helps to ensure that the gradient update is not too large or too small, regardless of the
        number of samples used in the batch.
        """
        # iterates through parameters
        for layer, param_mxs in self._params.items():
            for idx, param_mx in enumerate(param_mxs):
                # gets the gradient values
                grad_mx = self._grad[layer][idx]
                # subtracts the gradient from the parameters
                param_mx -= self._rate * (grad_mx / num_samples)
                # resets the gradient to zero for next training cycle
                grad_mx[:] = 0

    # API method for setting up the learning parameters
    def learn(self, num_epochs, algorithm=_stochastic_descent, rate=1, batch_ratio=1/10, threshold=0.02, stop=True, plot_static=False, plot_dynamic=False, upd_interval=20, with_full_report=False, return_cost = False):
        """
        The function trains a neural network for a specified number of epochs using a chosen
        algorithm, reports the state of the model, plots the cost function, and computes the total run time.
        
        :param num_epochs: The number of epochs, which is the number of times the neural network will be
        trained on the dataset
        :param algorithm: Determines the learning algorithm to be used for training the neural network. 
        The default value is `stochastic_descent`, but you can pass any other function that implements a 
        different learning algorithm
        :param rate: The learning rate of the neural network. It determines how quickly the parameters of the 
        network are updated during training. A higher learning rate can result in faster convergence, but it 
        may also cause the network to overshoot the optimal solution. Defaults to `1` (optional)
        :param batch_ratio: Determines the ratio of the training data used in each iteration of the learning 
        algorithm
        :param threshold: Is used to determine when to stop the learning process. If the current cost of the 
        model falls below the threshold value, the learning process will stop
        :param stop: Determines whether the learning process should stop when the threshold cost is reached. 
        If set to `True`, the learning process will stop when the cost falls below the specified threshold. 
        Otherwise, the learning process will continue until all epochs are completed, regardless of the cost. 
        Defaults to `True` (optional)
        :param plot_static: Determines whether to plot the graph of the cost function after training the neural 
        network. If set to `True`, the graph will be plotted using the `static_plot` method. Defaults to `False` 
        (optional)
        :param plot_dynamic: Determines whether or not to plot the cost function in real-time during the learning 
        process. If set to `True`, a separate process will be created to handle the plotting. Defaults to `False` 
        (optional)
        :param upd_interval: Determines the number of epochs after which the dynamic cost plot is updated during 
        the learning process. Defaults to `20` (optional)
        :param with_full_report: Determines whether to include a full report of the neural network's state after 
        training. If set to `True`, the full report will be printed to the console. Otherwise, only a summary 
        of the neural network's state will be printed. Defaults to `False` (optional)
        :param return_cost: Determines whether the function should return the list of cost values. If set to 
        `True`, the function will return a list of cost values for each epoch. Defaults to `False` (optional)
        :return: If the `return_cost` parameter is set to `True`, it returns a list of cost values. Otherwise it 
        returns `None`
        """
        assert not (plot_dynamic and plot_static), "Incorrect plotting mode setup"
        # initializes threshold flag
        thresh_flag = False
        # sets the learning rate
        self._rate = rate

        # reports the initial state of the NN
        print("\n<<< INITIAL STATE >>>\n")
        print(self.__str__(with_full_report))

        if return_cost:
            # initializes table for cost/epoch
            cost_vals = []
        if plot_static:  
            # initializes lists for plotting
            x_vals = []
            y_vals = []
        if plot_dynamic:
            # creates a queue for communication with the plotting process
            queue = Queue()
            # creates the plotting process
            plotting_process = Process(target=self._dynamic_plot, args=(queue,))
            plotting_process.start()

        # initializes the progress bar
        EMPTY, COMPLETE = ' - ', ' # '
        BAR_LENGTH = 10
        CURSOR_UP = '\033[1A'
        CLEAR = '\x1b[2K'
        CLEAR_LINE = CURSOR_UP + CLEAR
        counter = 1
        ratio = num_epochs / BAR_LENGTH
        progress = f"[{EMPTY * BAR_LENGTH}]"
        print(f"\n<<< LEARNING PROGRESS >>>\n{progress}")

        start_t = time.time()
        # repeats the learning procedure <num_epochs> times
        for epoch in range(num_epochs):
            # executes the selected algorithm
            algorithm(self, batch_ratio)
            # update the progress bar if needed
            if (epoch+1) >= ratio*counter:
                print(CLEAR_LINE, end='')
                print(f"[{COMPLETE * counter}{EMPTY * (BAR_LENGTH-counter)}]")
                counter += 1

            # gets the current cost of the model
            cost = self._curr_cost
            if return_cost:
                # saves the values for the cost table
                cost_vals.append(cost)
            if plot_static:
                # saves the values for the plot
                x_vals.append(epoch + 1)
                y_vals.append(cost)
            if plot_dynamic:
                # updates the cost plot every <upd_interval> cycles
                if (epoch + 1) % upd_interval == 0:
                    # sends data to the plotting process
                    queue.put((epoch + 1, cost))

            # detects the time when the threshold cost is reached
            if cost < threshold and not thresh_flag:
                thresh_t = time.time()
                thresh_iter = epoch
                thresh_flag = True
                # stops learning
                if stop == True:
                    print(CLEAR_LINE, end='')
                    print(f"[{COMPLETE * BAR_LENGTH}]")
                    break
        
        end_t = time.time()

        if plot_dynamic:
            # signals the plot process to terminate
            queue.put(None)

        # reports the final state of the model
        print(self.__str__(with_full_report))
        print(f"\nTime elapsed: {round(end_t - start_t, 3)}s.")
        # prints how much time was needed to pass the cost threshold
        if thresh_flag:
            print(f"{100 * threshold}% of cost was reached in {round(thresh_t - start_t, 3)}s ({thresh_iter} iterations).")

        if plot_static:
            # plots the cost
            self._static_plot(x_vals, y_vals)

        if return_cost:
            return cost_vals

    # methods for plotting the cost
    def _static_plot(self, x_vals, y_vals):
        """ @public
        The function plots the cost of a model against the number of training iterations.
        
        :param x_vals: Is a list or array of values representing the number of learning iterations. 
        These values will be plotted on the x-axis of the graph
        :param y_vals: Represents the values of the cost function for each corresponding value 
        in the `x_vals` parameter. These values are used to plot the cost vs. number of training 
        iterations
        """
        plt.plot(x_vals, y_vals, color='red')
        plt.xlabel('Number of learning iterations')
        plt.ylabel('Cost of the model')
        plt.grid(True)
        plt.show()
    def _dynamic_plot(self, *args):
        """ @public
        The function plots the cost function over the number of learning iterations in real-time as the data comes in through 
        a queue. It updates the plot with each new piece of data received through the queue, allowing for live visualization 
        of the training process. The plot remains open and dynamically updates until a `None` value is received through the 
        queue.

        :param args: The first element of `args` is expected to be a queue that provides tuples of `(iteration, cost)`, where iteration 
        is an integer representing the iteration number, and cost is a float representing the cost value at that iteration
        """
        # gets the queue
        queue = args[0]
        # small constant for scaling of axes
        eps = 1e-3
        # initializes the plot
        plt.figure()
        plt.xlabel('Iterations')
        plt.ylabel('Cost')
        plt.grid(True)
        
        x_vals = []
        y_vals = []

        # creates the line object
        line, = plt.plot(x_vals, y_vals, color='red')
        cost_text = plt.text(1, 1, '', transform=plt.gca().transAxes, ha='right', va='bottom')
        plt.show(block=False)

        try:
            while True:
                # checks if there's data in the queue
                if not queue.empty():
                    # gets values from the queue
                    data = queue.get()
                    if data is None:
                        break
                    
                    # stores the values for plotting
                    iteration, cost = data
                    x_vals.append(iteration)
                    y_vals.append(cost)

                    # updates the xdata and ydata of the line
                    line.set_data(x_vals, y_vals)

                    # sets appropriate plot limits
                    plt.xlim(min(x_vals)-eps, max(x_vals)+eps)
                    plt.ylim(min(y_vals)-eps, max(y_vals)+eps)

                    # updates the cost text with the current value of y
                    cost_text.set_text(f'Current accuracy: {(1-cost)*100:.1f}%')

                    # draws the updated plot
                    plt.draw()
                    plt.pause(0.001)  # pauses for a short duration to display the plot

        except KeyboardInterrupt:
            pass
        
        plt.show()

    # methods for storing a NN object state in memory
    def export_nn(self, data_path):
        """
        The function creates a dictionary, that contains the current parameters and activation
        functions of a neural network, and saves it to a file, which can be later used by the method 
        `import_nn` to restore the state of the neural network.
        
        :param data_path: Is a string that represents the path where the neural network image will be saved
        as a numpy file
        """
        nn_state = dict()
        for layer, params in self._params.items():
            act_func = self._act_funcs[layer]
            nn_state[layer] = (params, act_func)

        np.save(data_path, nn_state)
        print("The image was successfully exported to " + data_path)
    def import_nn(self, data_path):
        """
        The function imports new parameters for a neural network and updates the corresponding class instance 
        attributes. It collaborates with the method `export_nn` to streamline the storage of neural networks.
        
        :param data_path: Is a string that represents the path to the file from which the new parameters for 
        the neural network will be imported
        """
        # read new parameters from the file
        new_state = np.load(data_path, allow_pickle='TRUE').item()
        # creates temp dicts to store new configuration
        act_funcs = dict()
        new_params = dict()
        for layer, config in new_state.items():
            # checks if there is act func
            if isinstance(config[1], str):
                new_params[layer] = config[0]
                # retrieves act func name
                act_funcs [layer] = config[1]
            else:
                new_params[layer] = config
                # assigns the default act func
                act_funcs [layer] = 'sigmoid'
        # updated weights and biases
        self._params = copy.deepcopy(new_params)
        self._nn_depth = len(self._params)
        # maps layers to act funcs
        self._layer_config = self._act_func_map(act_funcs)
        print("The image was successfully imported from " + data_path)

    # LEGACY METHODS (can be used but are outperformed by the current algorithms)
    def _legacy_finite_diff(self, *_):
        """
        The function uses the finite difference method to approximate the partial derivatives of each 
        parameter and construct the gradient of the cost function, and then updates the parameter
        values accordingly.
        """
        # computes the initial cost of the model
        self._curr_cost = self._cost()
        # dictionary that will store 
        # updated parameter values
        upd_vals = dict()
        # iterates through each layer of parameters
        for layer in range(1, len(self._params)+1):
            mx_lst = []
            for mx_idx in range(2):
                param_vals = self._params[layer][mx_idx]
                # creates a copy of the current values
                new_vals = param_vals.copy()
                # iterates through each parameter value
                with np.nditer([param_vals, new_vals], op_flags=['readwrite']) as it:
                    for param_val, new_val in it:
                        # saves the original value
                        saved = np.copy(param_val)
                        # tweaks the parameter by <_eps>
                        param_val[...] += self._eps
                        # computes the derivate of the cost function
                        grad_val = (self._cost() - self._curr_cost) / self._eps
                        # updates the parameter value (in the copy)
                        new_val[...] -= self._rate * grad_val
                        # restores the original value
                        param_val[...] = saved
                # adds the matrix with updated values to the list-layer
                mx_lst.append(new_vals)
            # adds the layer to the final updated dictionary
            upd_vals[layer] = mx_lst
        # the updated dictionary becomes the new parameter dictionary
        self._params = upd_vals
    def _legacy_gradient_descent(self, *_):
        """
        The function implements a backpropagation algorithm for training a neural network by iterating 
        through each layer and instantly updating the weights and biases based on the computed derivatives.
        """
        total_err = 0
        # iterates through each data sample
        for data in self._train:
            # splits training data array 
            # into input/expected output subarrays
            input     = np.array([data[ :self._inp_size]])
            expected  = np.array([data[self._inp_size: ]])

            # computes activations for each layer
            # and saves the last layer activations
            output = self._forward(input)

            # verifies that there is no mismatch between 
            # the sizes of the expected and actual output arrays
            assert expected.shape == output.shape

            # initializes a list that holds partial derivatives of
            # activations of the current layer to the cost of the next layer
            da_next = []
            # computes activation derivative values for the last layer
            # as a difference between expected and actual outputs of NN
            with np.nditer([output, expected]) as it:
                for out_val, exp_val in it:
                    da_next.append(out_val - exp_val)
            da_next = np.array(da_next)
            
            # sums up error values of each activation in the last layer
            for err_val in da_next:
                total_err += err_val ** 2
        
            # iterates in reverse through each layer
            # starting from the last one
            for layer in reversed(range(1, len(self._params)+1)):
                db_layer = []
                dw_layer = []
                da_layer = []
                # gets the biases for the current layer
                layer_bss = self._params[layer][1]
                # gets the weights for the current layer
                layer_wghts = self._params[layer][0]
                # iterates through each activation
                for idx, act in enumerate(self._acts[layer][0]):
                    der_expr = 2 * da_next[idx] * act * (1 - act)
                    # computes bias derivative
                    db = der_expr
                    db_layer.append(db)
                    # computes weights derivatives
                    dw_act = []
                    for prev_act in self._acts[layer-1][0]:
                        dw = der_expr * prev_act
                        dw_act.append(dw)
                    dw_layer.append(dw_act)
                    # computes activation derivatives for the previous layer
                    da_act = []
                    curr_wghts = layer_wghts[:, idx]
                    for wght in curr_wghts:
                        da = der_expr * wght
                        da_act.append(da)
                    da_layer.append(da_act)

                # APPLYING THE GRADIENT
                # substracts bias gradient
                bs_grad = np.array(db_layer)
                layer_bss -= self._rate * bs_grad
                # substracts weight gradient
                wght_grad = np.array(dw_layer)
                wght_grad = wght_grad.T
                layer_wghts -= self._rate * wght_grad
                
                # computes a sum of activation derivatives
                # for each activation of the previous layer
                da_next = np.array(da_layer)
                da_next = np.sum(da_next, axis=0)

        # updates the current cost of the model 
        # as an average error value per 1 sample
        self._curr_cost = total_err / len(self._train)
    def _legacy_backprop(self, da_next):
        """
        The function implements backpropagation by iterating through the layers in reverse order, computing
        the derivatives of the biases and weights, and updating the gradient for later use in updating the
        parameters of the neural network. On average, this backpropagation algorithm reaches 1% of cost 30 
        times faster than the '_legacy_finite_diff' method.
        
        :param da_next: An array of computed error values for the current output
        """
        # iterates through each layer backwards starting from the last one
        for layer in reversed(range(1, len(self._params)+1)):
            db_layer = []
            dw_layer = []
            da_layer = []
            # gets the weights for the current layer
            layer_wghts = self._params[layer][0]
            # iterates through each activation
            for idx, act in enumerate(self._acts[layer][0]):
                der_expr = 2 * da_next[idx] * act * (1 - act)
                # computes bias derivative
                db = der_expr
                db_layer.append(db)
                # computes weights derivatives
                dw_act = []
                for prev_act in self._acts[layer-1][0]:
                    dw = der_expr * prev_act
                    dw_act.append(dw)
                dw_layer.append(dw_act)
                # computes activation derivatives for the previous layer
                da_act = []
                curr_wghts = layer_wghts[:, idx]
                for wght in curr_wghts:
                    da = der_expr * wght
                    da_act.append(da)
                da_layer.append(da_act)

            # computes the gradient for current layer
            b_grad = np.array(db_layer)                
            w_grad = np.array(dw_layer)
            w_grad = w_grad.T
            grad_mxs = [w_grad, b_grad]
            # adds up the gradient values of the current sample
            # to the gradient sum of previously computed samples
            for idx, mx in enumerate(self._grad[layer]):
                mx += grad_mxs[idx]
            
            # computes a sum of activation derivatives
            # for each activation of the previous layer
            da_next = np.array(da_layer)
            da_next = np.sum(da_next, axis=0)

def adder_truth_table(num_bits):
    """ @private
    The function generates a truth table for a given number of bits for a binary adder.
    
    :param num_bits: Represents the number of bits in the adder. It determines the size of the truth 
    table that will be generated
    :return: a numpy array that represents the truth table for a `num_bits` adder.
    """
    truth_table = []
    decimal_range = range(2 ** num_bits)
    all_pairs = itertools.combinations_with_replacement(decimal_range, 2)
    for a_pair in all_pairs:
        a = a_pair[0]
        b = a_pair[1]
        a_sum = a + b
        a_str = bin(a)[2:].zfill(num_bits)
        b_str = bin(b)[2:].zfill(num_bits)
        sum_str = bin(a_sum)[2:].zfill(num_bits + 1)
        sample = list(int(char) for char in (a_str + b_str + sum_str))
        truth_table.append(sample)

    return np.array(truth_table)

def main():
    """
    The main function is used to test the neural network implementation.
    """
    multiprocessing.freeze_support()
    np.set_printoptions(precision=0, suppress=True)
    rand_range = (-1, 1)

    n_bit = 4
    adder_train = adder_truth_table(n_bit)
    act_funcs = {1:"sigmoid", 2:"sigmoid", 3:"sigmoid"}
    adder_layout = ((2*n_bit, 3*n_bit, 2*n_bit, n_bit + 1), act_funcs)

    adder_nn = NN(adder_layout, rand_range, adder_train, f"{n_bit}-bit Adder")
    adder_nn.learn(1000, batch_ratio=1/len(adder_train), rate=1/4, stop=True, plot_dynamic=True)

def legacy_main():
    """ @private
    """
    rand_range = (-1, 1)
    # <<< OR, AND, NAND and XOR >>>
    or_train = ((0, 0, 0),
                (1, 0, 1),
                (0, 1, 1),
                (1, 1, 1))
    and_train = ((0, 0, 0),
                 (1, 0, 0),
                 (0, 1, 0),
                 (1, 1, 1))
    nand_train = ((0, 0, 1),
                  (1, 0, 1),
                  (0, 1, 1),
                  (1, 1, 0))
    xor_train = ((0, 0, 0),
                 (1, 0, 1),
                 (0, 1, 1),
                 (1, 1, 0))
    # activation function for each layer of NN
    act_funcs = {layer:"sigmoid" for layer in range(1, 3)}
    # layout of a fully connected NN of type: [2xinp ---> 2nx1n ---> out]
    xor_layout = ((2, 2, 1), act_funcs)
    xor_nn = NN(xor_layout, rand_range, np.array(xor_train), "XOR gate")

    # <<< MUX and DMUX >>>
    dmux_train = ((0, 0, 0, 0),
                  (0, 1, 0, 0),
                  (1, 0, 1, 0),
                  (1, 1, 0, 1))
    dmux_layout = ((2, 2, 2), act_funcs)
    
    mux_train = ((0, 0, 0, 0),
                 (0, 0, 1, 0),
                 (0, 1, 0, 1),
                 (0, 1, 1, 1),
                 (1, 0, 0, 0),
                 (1, 0, 1, 1),
                 (1, 1, 0, 0),
                 (1, 1, 1, 1),
                 )
    mux_layout = ((3, 2, 1), act_funcs)
    
    dmux_nn = NN(dmux_layout, rand_range, np.array(dmux_train), "Dmux gate")    
    mux_nn = NN(mux_layout, rand_range, np.array(mux_train), "Mux gate")
    
    # <<< ADDERS >>> 
    half_add_train = ((0, 0, 0, 0),
                      (0, 1, 1, 0),
                      (1, 0, 1, 0),
                      (1, 1, 0, 1))
    half_add_layout = ((2, 3, 2), act_funcs)

    full_add_train = ((0, 0, 0, 0, 0),
                      (0, 0, 1, 1, 0),
                      (0, 1, 0, 1, 0),
                      (0, 1, 1, 0, 1), 
                      (1, 0, 0, 1, 0),
                      (1, 0, 1, 0, 1),
                      (1, 1, 0, 0, 1),
                      (1, 1, 1, 1, 1))
    full_add_layout = ((3, 3, 2), act_funcs)

    half_add_nn = NN(half_add_layout, rand_range, np.array(half_add_train), "Half-adder")
    full_add_nn = NN(full_add_layout, rand_range, np.array(full_add_train), "Full-adder")

if __name__ == "__main__":
    main()