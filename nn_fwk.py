"""
Features a flexible class implementation of NNs from the first principles.
"""
import time
import copy
import itertools
import numpy as np
import matplotlib.pyplot as plt
import multiprocessing
from multiprocessing import Process, Queue
np.seterr(under='warn')

# command for profiling (to run in Command Prompt):
# kernprof -l -v nn_fwk.py
class NN:
    """
    A class implementation of a NN with an arbitrary size and configuration.
    """
    def __init__(self, nn_layout, rand_range, train_data, name, rate=1):
        """
        The constructor for NN class.

        Parameters: <nn_layout> - a layout of the NN. Defined as a tuple of type:
                    ((inp, l1, l2, ... , ln), act_map), where:
                        - <inp> denotes the size of the input;
                        - <ln> denotes the number of neurons in the n-th layer.
                        - <act_map> denotes types of activation funcs for each layer.
                    <rand_range> - a tuple containing the lower and upper limits
                    for random initialization of parameter matrixes.
                    <train_data> - a sequence of training data samples of type:
                    ((a1, b1, c1, ...), (a2, b2, c2, ...), (an, bn, cn, ...)), where:
                        - each sample contains <j> input and <k> output values, with <j>
                          defined by <nn_layout[0][0]>.
                    <name> - a string-name of the NN.

        The NN state is defined by mapping between each layer and 
        corresponding matrixes that contain values for weights and biases:
        {1:[mx_w1, mx_b1],
         2:[mx_w2, mx_b2],
         n:[mx_wn, mx_bn]}
        """
        # to store shape of each parameter mx
        self._param_config = self.mx_size_map(nn_layout[0])
        self._nn_depth = len(self._param_config)
        # to store activation funcs for each layer
        self._act_funcs = nn_layout[1]
        # to map act. funcs and derivatives to each layer
        self._layer_config = self.act_func_map(self._act_funcs)
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
    def mx_size_map(self, params):
        """
        Creates mapping between each layer and sizes 
        of corresponding matrixes for weights and biases.
        Returns:
                {1:(w1_size, b1_size), 
                 2:(w2_size, b2_size),
                 n:(wn_size, bn_size)}
        """
        nn_layout = dict()
        for layer in range(1, len(params)):
            nn_layout[layer] = ((params[layer-1], params[layer]), (1, params[layer]))

        return nn_layout
    def act_func_map(self, act_map):
        """
        Converts provided map for activation functions in each layer to a dictionary 
        with direct references to the functions and their derivatives for each layer.
        """
        # validate the input
        assert len(act_map) == self._nn_depth, "Invalid activation map (length)"
        assert all(key in range(1, self._nn_depth + 1) for key in act_map.keys()), "Invalid layer number(s)"
        assert all(isinstance(func_name, str) for func_name in act_map.values()), "Invalid function name(s)"

        # return a new dict with funcs and their derivatives mapped to each layer
        return {layer: (getattr(self, "_" + func_name), getattr(self, "_" + func_name + "_der")) for layer, func_name in act_map.items()}

    def __str__(self, with_full_report=False):
        """
        Returns a string - the current state of the model
        Can be modified by including the values of parameters for each neuron,
        or by printing out an actual output of the model on all training samples.
        """
        state_str  = f"1) Model: {self._name}\n"
        state_str += f"2) Cost: {self.cost()}\n"

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
                
                output = self.forward(input)
                state_str += f"{output})"

        return state_str

    def cost(self):
        """
        Computes the current cost value of the NN.
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
            output = self.forward(input)

            # verifies that there is no mismatch between 
            # the sizes of the expected and actual output arrays
            assert output.shape == expected.shape
            
            # iterates through both arrays and 
            # computes corresponding error values
            
            #------------------------------------------
            err_vals = (output - expected)[0] ** 2
            # err_vals = np.abs((output - expected)[0])
            #------------------------------------------

            # adds the accumulated error to the total cost
            result += np.sum(err_vals)

        # computes the mean error value per 1 sample of training data
        result /= len(self._train)

        return result

    # methods for computing activations
    def _sigmoid(self, x):
        """
        Helper function for computing sigmoid(x)
        """
        return 1 / (1 + np.exp(-x))
    def _relu(self, x):
        """
        Helper function for computing ReLU(x)
        """
        return np.maximum(0, x)
    def _softmax(self, x):
        """
        Helper function for a stable computation of softmax(x)
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

    #@profile
    def forward(self, input_arr):
        """
        Forwards the <input_arr> through each layer of the NN.
        Maps every layer with activation values of each neuron from this layer.
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
    #@profile
    def stochastic_descent(self, batch_ratio):
        """
        An implementation of a backpropagation algorithm via looping through each layer of the NN.
        First, it pushes an input sample through NN and computes activations of each neuron.
        Then it goes backwards through each layer and computes partial derivate values of each
        weight and bias of the layer, and then adds them up to the gradient.
        Only after iterating through all training samples, it applies the resulting gradient to parameter values, 
        thus driving the cost function of the NN towards zero.
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
            output = self.forward(input)

            # verifies that there is no mismatch between 
            # the sizes of the expected and actual output arrays
            assert output.shape == expected.shape

            # computes activation derivative values for the last layer
            # as a difference between expected and actual outputs of NN
            da_next = (output - expected)[0]

            # sums up squares of error values 
            # of each activation in the last layer

            #-------------------------------------
            total_err += np.sum(da_next ** 2)
            # total_err += np.sum(np.abs(da_next))
            #-------------------------------------

            # performs backpropagation and computes the gradient
            self.backprop(da_next)

            sample_counter += 1
            # checks if the batch size is reached
            if sample_counter / len(self._train) >= batch_ratio:
                # applies accumulated gradient values to the parameters
                self.apply_grad(sample_counter)
                sample_counter = 0

        # applies the gradient of the samples
        # that didn't reach the size of the batch
        if sample_counter > 0:
            self.apply_grad(sample_counter)

        # updates the current cost of the model 
        # as an average error value per 1 sample
        self._curr_cost = total_err / len(self._train)    
    #@profile
    def backprop(self, da_next):
        """
        Implementation of a backpropagation via looping through the layers in reverse order.
        Updates <self._grad> for later application to the parameters of NN.
        Input: <da_next> - an array of computed error values for the current output.
        """
        # iterates through each layer backwards starting from the last one
        for layer in reversed(range(1, len(self._params)+1)):

            # gets the weights for the current layer
            layer_wghts = self._params[layer][0]
            
            #---------------------------------------------------------------------------
            # gets the derivative of activation func for the current layer
            der_func = self._layer_config[layer][1]

            # computes the derivative expression vectorized for all activations
            # der_expr = self._half_sqr_der(da_next) * der_func(self._acts[layer][0])
            der_expr = self._lncosh_der(da_next) * der_func(self._acts[layer][0])
            #---------------------------------------------------------------------------

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

    def apply_grad(self, num_samples):
        """
        Applies the stored gradient values to the parameters of NN.
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

    def learn(self, num_epochs, algorithm=stochastic_descent, rate=1, batch_ratio=1/10, threshold=0.02, stop=True, plot_static=False, plot_dynamic=False, upd_interval=20, with_full_report=False, return_cost = False):
        """
        Trains the NN <n> times with the chosen algorithm, reports the state of the model
        plots the graph of the cost function (opt. in real time) and computes the total run time.
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
            plotting_process = Process(target=self.dynamic_plot, args=(queue,))
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
            self.static_plot(x_vals, y_vals)

        # gets the state of NN
        nn_state = self.export_nn()

        if return_cost:
            return nn_state, cost_vals
        return nn_state

    def static_plot(self, x_vals, y_vals):
        """
        Plots the function of cost vs. number of training iterations
        """
        plt.plot(x_vals, y_vals, color='red')
        plt.xlabel('Number of learning iterations')
        plt.ylabel('Cost of the model')
        plt.grid(True)
        plt.show()
    def dynamic_plot(self, *args):
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

    def export_nn(self):
        """
        Creates an object with current parameters of NN, that
        can be saved and passed to the function <self.import_nn()> 
        to restore the state of NN later.
        Output: {layer:([w, b], 'act_func')}
        """
        nn_state = dict()
        for layer, params in self._params.items():
            act_func = self._act_funcs[layer]
            nn_state[layer] = (params, act_func)

        return nn_state
    def import_nn(self, new_state):
        """
        Imports new parameters of NN, and updates the corresponding class instance attributes.
        Works together with the function <self.export_nn()> to facilitate the process of storing NNs.
        """
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
        self._layer_config = self.act_func_map(act_funcs)

    # LEGACY METHODS (can be used but are outperformed by the current algorithms)
    def finite_diff(self, *_):
        """
        Uses finite difference method for approximating the partial
        derivatives of each parameter and constructing the gradient 
        of the cost function.
        Mutates each of the NN parameters by substracting their respective
        derivative values.
        """
        # computes the initial cost of the model
        self._curr_cost = self.cost()
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
                        grad_val = (self.cost() - self._curr_cost) / self._eps
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
    def old_descent(self, *_):
        """
        An implementation of a backpropagation algorithm via looping through each layer of the NN.
        First, it pushes an input sample through NN and computes activations of each neuron.
        Then it goes backwards through each layer and computes partial derivate values of each
        weight and bias of the layer, and applies them to the parameters on the fly.
        Each training sample iteration results in mutation of parameter values towards minimizing
        the cost function of the NN.
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
            output = self.forward(input)

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

                # APPLYING THE GRADIENT (LAYER)
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
    def old_backprop(self, da_next):
        """
        Implementation of a backpropagation via looping through the layers in reverse order.
        Updates <self._grad> for later application to the parameters of NN.
        Input: <da_next> - an array of computed error values for the current output.
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
    """
    Generates a truth table for a <num_bits> adder.
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

    # on average, the backpropagation algorithm reaches 1% of cost 
    # 30 times faster than the finite difference computation
    multiprocessing.freeze_support()
    np.set_printoptions(precision=0, suppress=True)
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

    n_bit = 5
    adder_train = adder_truth_table(n_bit)
    act_funcs = {1:"sigmoid", 2:"sigmoid", 3:"sigmoid"}
    adder_layout = ((2*n_bit, 3*n_bit, 2*n_bit, n_bit + 1), act_funcs)

    adder_nn = NN(adder_layout, rand_range, adder_train, f"{n_bit}-bit Adder")
    adder_nn.learn(500, batch_ratio=1/len(adder_train), rate=1/4, stop=True, plot_static=True)


if __name__ == "__main__":
    main()