import numpy as np
import time
import random

def main():
    rows = 5000
    cols = 5000
    start_lst = time.time()
    a_lst = []
    for i in range(rows):
        a_row = []
        for j in range(cols):
            a_row.append(random.random())
        a_lst.append(a_row)
    end_lst = time.time()
    start_arr = time.time()
    a_arr = np.random.uniform(0, 1, (rows, cols))
    end_arr = time.time()

    print("List time elapsed:", end_lst-start_lst)
    print("Array time elapsed:", end_arr-start_arr)

main()