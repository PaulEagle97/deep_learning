import time
import copy
import itertools
import numpy as np
import matplotlib.pyplot as plt
import multiprocessing
from multiprocessing import Process, Queue

# command for profiling (to run in Command Prompt):
# kernprof -l -v nn_fwk.py
class NN:
    """
    A class implementation of a NN with an arbitrary size and configuration.
    """
    def __init__(self, nn_layout, rand_range, train_data, name, rate=1):
        """
        The constructor for NN class.

        Parameters: <nn_config> - a layout of the NN. Defined as a tuple of type:
                    (inp, l1, l2, ... , ln), where:
                        - <inp> denotes the size of the input;
                        - <ln> denotes the number of neurons in the n-th layer.
                    <rand_range> - a tuple containing the lower and upper limits
                    for random initialization of parameter matrixes.
                    <train_data> - a sequence of training data samples of type:
                    ((a1, b1, c1, ...), (a2, b2, c2, ...), (an, bn, cn, ...)), where:
                        - each sample contains <j> input and <k> output values, with <j>
                          defined by <nn_config>[0].
                    <name> - a string-name of the NN.

        The NN state is defined by mapping between each layer and 
        corresponding matrixes that contain values for weights and biases:
        {1:[mx_w1, mx_b1],
         2:[mx_w2, mx_b2],
         n:[mx_wn, mx_bn]}
        """
        # gets the size for each matrix of parameters
        self._nn_config = self.mx_config(nn_layout)
        self._train = train_data.copy()
        self._inp_size = nn_layout[0]
        self._curr_cost = None
        self._params = dict()
        self._acts = dict()
        self._grad = dict()
        # constant for finite difference computation
        self._eps = 1e-1
        # constant that defines the speed of learning
        self._rate = rate
        self._name = name
        # initializes the parameters and the gradient
        for layer in range(1, len(self._nn_config)+1):
            # gets the sizes of parameter matrixes for each layer
            w_shape = self._nn_config[layer][0]
            b_shape = self._nn_config[layer][1]
            # creates one random matrix for weights and 
            # another one for biases per each layer of NN
            w = np.random.uniform(rand_range[0], rand_range[1], size=w_shape)
            b = np.random.uniform(rand_range[0], rand_range[1], size=b_shape)
            param_mxs = [w, b]
            self._params[layer] = param_mxs
            # initializes gradient table with empty matrixes 
            # of the same shape as the parameter matrixes
            w_grad = np.zeros(shape=w_shape)
            b_grad = np.zeros(shape=b_shape)
            grad_mxs = [w_grad, b_grad]
            self._grad[layer] = grad_mxs

    def load_params(self, new_params):
        """
        Loads a provided dictionary with values 
        for each weight & bias into the NN.
        """
        try:
            for layer in self._params:
                for mx_idx in range(2):
                    assert new_params[layer][mx_idx].shape == self._params[layer][mx_idx].shape
        except Exception as e:
            print("An error occurred while importing parameters:", e)
            return

        self._params = copy.deepcopy(new_params)

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

    def mx_config(self, params):
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

    def _sigmoid(self, x):
        """
        Helper function for computing sigmoid(x)
        """
        return 1 / (1 + np.exp(-x))

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
            err_vals = (output - expected)[0] ** 2

            # adds the accumulated error to the total cost
            result += np.sum(err_vals)

        # computes the mean error value per 1 sample of training data
        result /= len(self._train)

        return result
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
            # retrieves values of weights
            # and biases of the layer
            w = param_lst[0]
            b = param_lst[1]
            # applies weights
            a1 = np.matmul(a0, w)
            # applies biases
            a1 += b
            # applies the activation function
            a1 = self._sigmoid(a1)
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
            total_err += np.sum(da_next ** 2)

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
            
            # computes the derivative expression vectorized for all activations
            der_expr = 2 * da_next * self._acts[layer][0] * (1 - self._acts[layer][0])

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

    def learn_static (self, n, alg_name='stochastic_descent', rate=1, batch_ratio=1/10, threshold=0.02, with_plot=True,  stop=True, with_full_report=False):
        """
        Trains the NN <n> times with the chosen algorithm, reports the state of the model,
        plots the graph of the cost function and computes the total run time.
        """
        # sets the learning rate
        self._rate = rate

        # initializes lists for plotting
        x_vals = []
        y_vals = []

        # reports the initial state of the NN
        print("\n<<< INITIAL STATE >>>\n")
        print(self.__str__(with_full_report))

        # starts the timer
        start_t = time.time()
        thresh_flag = False
        print("\n<<< LEARNING >>>\n")
        # repeats the learning procedure <n> times
        for iter in range(n):
            # selects an algorithm
            algorithm = getattr(self, alg_name)
            # executes the algorithm
            algorithm(batch_ratio)

            # gets the current cost of the model
            cost = self._curr_cost

            # saves the values for the plot
            x_vals.append(iter + 1)
            y_vals.append(cost)

            # detects the time when the threshold cost is reached
            if cost < threshold and not thresh_flag:
                thresh_t = time.time()
                thresh_iter = iter
                thresh_flag = True
                # stops learning
                if stop == True:
                    break
        
        # stops the timer
        end_t = time.time()
        # prints the trained state of the model
        print(self.__str__(with_full_report))
        # prints the total time
        print(f"\nTime elapsed: {round(end_t - start_t, 3)}s.")
        # prints how much time was needed to pass the cost threshold
        if thresh_flag:
            print(f"{100 * threshold}% of cost was reached in {round(thresh_t - start_t, 3)}s ({thresh_iter} iterations).")

        if with_plot:
            # plots the cost
            self.static_plot(x_vals, y_vals)

        return self._params, y_vals

    def static_plot(self, x_vals, y_vals):
        """
        Plots the function of cost vs. number of training iterations
        """
        plt.plot(x_vals, y_vals, color='red')
        plt.xlabel('Number of learning iterations')
        plt.ylabel('Cost of the model')
        plt.grid(True)
        plt.show()

    def learn_dynamic(self, n, alg_name='stochastic_descent', rate=1, batch_ratio=1/10, threshold=0.02, upd_interval=20, stop=True, with_full_report=False):
        """
        Trains the NN <n> times with the chosen algorithm, reports the state of the model
        plots the graph of the cost function in real time and computes the total run time.
        """
        # sets the learning rate
        self._rate = rate

        # reports the initial state of the NN
        print("\n<<< INITIAL STATE >>>\n")
        print(self.__str__(with_full_report))

        # creates a queue for communication with the plotting process
        queue = Queue()
        
        # creates the plotting process
        plotting_process = Process(target=self.dynamic_plot, args=(queue,))
        plotting_process.start()

        # starts the timer
        start_t = time.time()
        thresh_flag = False
        print("\n<<< LEARNING >>>\n")
        # repeats the learning procedure <n> times
        for iter in range(n):
            # selects an algorithm
            algorithm = getattr(self, alg_name)
            # executes the algorithm
            algorithm(batch_ratio)
            
            # gets the current cost of the model
            cost = self._curr_cost

            # updates the cost plot every <upd_interval> cycles
            if (iter + 1) % upd_interval == 0:
                # sends data to the plotting process
                queue.put((iter + 1, cost))

            # detects when the threshold cost is reached
            if cost < threshold and not thresh_flag:
                thresh_t = time.time()
                thresh_iter = iter
                thresh_flag = True
                # stops learning
                if stop == True:
                    break
        
        # stops the timer
        end_t = time.time()

        # signals the plot process to terminate
        queue.put(None)

        # prints the trained state of the model
        print(self.__str__(with_full_report))
        # prints the total time
        print(f"\nTime elapsed: {round(end_t - start_t, 3)}s.")
        # prints how much time was needed to pass the cost threshold
        if thresh_flag:
            print(f"{100 * threshold}% of cost was reached in {round(thresh_t - start_t, 3)}s ({thresh_iter} iterations).")

        return self._params

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


def gen_truth_table(num_bits):
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
    # layout of a fully connected NN of type: 
    # [2xinp ---> 2nx1n ---> out]
    xor_layout = (2, 2, 1)
    xor_nn = NN(xor_layout, rand_range, np.array(xor_train), "XOR gate")
    # xor_nn.learn_dynamic(500)

    # <<< MUX and DMUX >>>
    dmux_train = ((0, 0, 0, 0),
                  (0, 1, 0, 0),
                  (1, 0, 1, 0),
                  (1, 1, 0, 1))
    dmux_layout = (2, 2, 2)
    
    mux_train = ((0, 0, 0, 0),
                 (0, 0, 1, 0),
                 (0, 1, 0, 1),
                 (0, 1, 1, 1),
                 (1, 0, 0, 0),
                 (1, 0, 1, 1),
                 (1, 1, 0, 0),
                 (1, 1, 1, 1),
                 )
    mux_layout = (3, 2, 1)
    
    dmux_nn = NN(dmux_layout, rand_range, np.array(dmux_train), "Dmux gate")
    # dmux_nn.learn_dynamic(200)
    
    mux_nn = NN(mux_layout, rand_range, np.array(mux_train), "Mux gate")
    # mux_nn.learn_dynamic(100)
    
    # <<< ADDERS >>> 
    half_add_train = ((0, 0, 0, 0),
                      (0, 1, 1, 0),
                      (1, 0, 1, 0),
                      (1, 1, 0, 1))
    half_add_layout = (2, 3, 2)

    full_add_train = ((0, 0, 0, 0, 0),
                      (0, 0, 1, 1, 0),
                      (0, 1, 0, 1, 0),
                      (0, 1, 1, 0, 1), 
                      (1, 0, 0, 1, 0),
                      (1, 0, 1, 0, 1),
                      (1, 1, 0, 0, 1),
                      (1, 1, 1, 1, 1))
    full_add_layout = (3, 3, 2)

    half_add_nn = NN(half_add_layout, rand_range, np.array(half_add_train), "Half-adder")
    # half_add_nn.learn_dynamic(200)

    full_add_nn = NN(full_add_layout, rand_range, np.array(full_add_train), "Full-adder")
    # full_add_nn.learn_dynamic(200)

    n_bit = 4
    adder_train = gen_truth_table(n_bit)
    adder_layout = (2*n_bit, 3*n_bit, 2*n_bit, n_bit + 1)

    adder_nn = NN(adder_layout, rand_range, adder_train, f"{n_bit}-bit Adder")
    adder_nn.learn_static(500, batch_ratio=1/len(adder_train), rate=1/10, stop=False)
    

if __name__ == "__main__":
    main()