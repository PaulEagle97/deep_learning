"""
This script features simple Optical Character Recognition application, based on the MNIST set
of handwritten digits.
"""
from mlp_framework import MLP
import numpy as np
import os
import matplotlib.pyplot as plt
from keras.datasets import mnist

def main():
    np.set_printoptions(precision=3, suppress=True)
    curr_dir = os.getcwd()
    file_name = 'mnist_data.npz'
    data_path = os.path.join (curr_dir, "train_data", "MNIST_numpy", file_name)

    # UNCOMMENT TO WRITE DATASET TO A FILE
    #----------------------------------------------
    # mnist_data = mnist.load_data()
    # np.savez(data_path, train_imgs=mnist_data[0][0], 
    #                     train_lbls=mnist_data[0][1],
    #                     test_imgs= mnist_data[1][0], 
    #                     test_lbls= mnist_data[1][1]) 
    #----------------------------------------------

    mnist_data = np.load(data_path)
    train_imgs=mnist_data["train_imgs"]
    train_lbls=mnist_data["train_lbls"]
    test_imgs= mnist_data["test_imgs"]
    test_lbls= mnist_data["test_lbls"]

    file_name = 'OCR_new.npy'
    data_path = os.path.join (curr_dir, "saved_NNs", file_name)

    # UNCOMMENT TO SEE THE TRAINING IMAGES
    # for i in range(9):
    #     plt.subplot(330 + 1 + i)
    #     plt.imshow(train_imgs[i+27], cmap=plt.get_cmap('gray'))
    # plt.show()
    # return

    train_size = len(train_imgs)
    train_data = np.empty([train_size, 794])
    for idx in range(train_size):
        nn_output = np.zeros(10)
        lbl = train_lbls[idx]
        nn_output[lbl] = 1.0
        flat_img = np.ravel(train_imgs[idx])
        merged = np.concatenate((flat_img/255, nn_output))
        train_data[idx] = merged

    rand_range = (-1, 1)
    act_funcs = {1:"sigmoid", 2:"softmax"}
    ocr_layout = ((784, 10, 10), act_funcs)

    # act_funcs = {1:"softmax"}
    # ocr_layout = ((784, 10), act_funcs)
    ocr_nn = MLP(ocr_layout, rand_range, train_data, "Basic OCR")

    # LOAD THE IMAGE FROM FILE
    ocr_nn.import_mlp(data_path)

    # LEARN & PLOT
    # plot_vals = np.array([[], []])
    # tot_it = 0
    # num_it = 10
    # y_vals = ocr_nn.learn(num_it, batch_ratio=600/train_size, rate=1, stop=False, plot_static=False, return_cost=True)
    # new_vals = np.array((np.arange(num_it)+(tot_it+1), np.array(y_vals)))
    # plot_vals = np.hstack((plot_vals, new_vals))
    # tot_it += num_it
    # ocr_nn._static_plot(plot_vals[0], plot_vals[1])

    # STORE THE IMAGE AS FILE
    # file_name = 'OCR_new.npy'
    # data_path = os.path.join (curr_dir, "saved_NNs", file_name)
    # ocr_nn.export_nn(data_path)

    # TEST ON THE TEST SET
    test_size = len(test_imgs)
    errs = np.zeros(test_size, dtype=np.uint8)
    for idx in range(test_size):
        lbl = test_lbls[idx]
        flat_img = np.ravel(test_imgs[idx])
        nn_output = ocr_nn._forward(np.array([flat_img/255]))
        predicted = np.argmax(nn_output)
        errs[idx] = (predicted != lbl)
    err_idxs = np.nonzero(errs)

    # UNCOMMENT TO SEE THE FAILED IMAGES
    # for i in range(9):
    #     plt.subplot(330 + 1 + i)
    #     plt.imshow(test_imgs[err_idxs[0][i]], cmap=plt.get_cmap('gray'))
    # plt.show()

    print(f"Total number of errors ={sum(errs)}")
    print(f"Accuracy = {(test_size - sum(errs))/test_size * 100}%")


if __name__ == "__main__":
    main()