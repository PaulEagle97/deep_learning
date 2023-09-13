import numpy as np
import time

# Define a custom error handler function
def underflow_handler(err, flag_obj):
    if err[0] == 'underflow':
        # Get the operand that caused the underflow
        op = flag_obj[1]
        # Replace the operand with 0
        op[...] = 0
        return 0
    else:
        raise FloatingPointError(err)

def softmax(x):
    norm_x = x - np.max(x)
    exp_z = np.exp(norm_x)
    return exp_z / np.sum(exp_z)

# arr = np.array([0.1, 0.3, 0.7, 1, 2, 3, 4, 5, 6, 7])
# print(softmax(arr))
# print(arr - np.mean(arr))
# print(np.linalg.norm(arr))
# start = time.perf_counter()
# for _ in range(50000):
#     new_arr = softmax(arr)
# end = time.perf_counter()

# print(end - start)

# arr1 = np.array([1, 2, 3])
# arr2 = np.array([4, 5, 6, 7])

# Create a higher-dimensional array by stacking arr1 and arr2
# arr3 = np.array([arr1, arr2])

# Iterate through arr3
# for subarray in arr3:
#     print(subarray)

# Set the custom error handler
np.seterr(under='call')  # Set underflow to 'call'
np.seterrcall(underflow_handler)

# Example that may cause an underflow
x = np.array([308, 309])
result = 1 * x + 10 * 1e-4448 # Multiplying by a small number to potentially cause underflow

# Print the result (which will have the operand replaced with 0)
print(result)