"""
This script permits to process one of 28x28 greyscale images from MNIST dataset and perform the following actions:
1. Save an image as a training data for a NN object
2. Train the NN with a chosen set of parameters
3. Plot the cost of the model
4. Save the trained model to an external file
5. Import a previously trained NN from a file
6. Upscale the image "saved" in NN to any resolution and display the result.
"""
from nn_fwk import NN
import gzip
import os
import numpy as np
import matplotlib.pyplot as plt

def img_encode (image):
    """
    Receives an image represented by a 2D array with integer values (0 .. 255).
    Normalizes coordinates of each pixel and its brightness.
    Returns a new matrix with 3 columns (x-val, y-val, brightness).
    """
    image_norm = image / 255

    # Get the dimensions of the image
    height, width = image.shape

    # Create an array of row indices (normalized)
    y_coords = np.arange(height)
    y_norm = y_coords / (height - 1)

    # Create an array of column indices (normalized)
    x_coords = np.arange(width)
    x_norm = x_coords / (width - 1)

    # Create meshgrid of coordinates
    x_grid, y_grid = np.meshgrid(x_norm, y_norm)

    # Reshape the image into a 1D array
    brightness = image_norm.flatten()

    # Stack x-coordinates, y-coordinates, and brightness values
    encoded_matrix = np.vstack((x_grid.flatten(), y_grid.flatten(), brightness)).T

    return encoded_matrix

def img_decode(encoded_matrix, original_shape):
    """
    Receives an encoded matrix with 3 columns (x-val, y-val, brightness)
    and the original shape of the image.
    Reconstructs the original 2D array-matrix with pixel brightness values.
    Returns the reconstructed image.
    """
    # Extract x-coordinates, y-coordinates, and brightness values
    x_coords = encoded_matrix[:, 0]
    y_coords = encoded_matrix[:, 1]
    brightness = encoded_matrix[:, 2]

    # Calculate the dimensions of the image
    height, width = original_shape

    # Calculate the row indices from normalized y-coordinates
    y_indices = np.round(y_coords * (height - 1)).astype(int)

    # Calculate the column indices from normalized x-coordinates
    x_indices = np.round(x_coords * (width - 1)).astype(int)

    # Initialize an empty image
    decoded_image = np.zeros(original_shape, dtype=int)

    # Fill in the brightness values at the calculated coordinates
    decoded_image[y_indices, x_indices] = brightness * 255

    # Return the reconstructed image
    return decoded_image

def norm_coord(gen_size):
    """
    Generates a matrix with the provided <gen_size>
    and values of coordinates normalized between 0 and 1.
    """
    height = gen_size[0]
    width = gen_size[1]

    # Create an array of row indices (normalized)
    y_coords = np.arange(height)
    y_norm = y_coords / (height - 1)

    # Create an array of column indices (normalized)
    x_coords = np.arange(width)
    x_norm = x_coords / (width - 1)

    # Create meshgrid of coordinates
    x_grid, y_grid = np.meshgrid(x_norm, y_norm)

    # Stack x-coordinates, y-coordinates, and brightness values
    coord_matrix = np.vstack((x_grid.flatten(), y_grid.flatten())).T

    return coord_matrix

def main():
    # constants
    IMAGE_SIZE = 28
    NUM_IMAGES = 50
    IMG_IDX = 30

    # file path to the MNIST dataset
    curr_dir = os.getcwd()
    file_name = 'MNIST-train-images-idx3-ubyte.gz'
    data_path = os.path.join(curr_dir, "train_data", file_name)

    # LOAD IMAGE
    with gzip.open(data_path, 'r') as imgs:
        # skip the header
        imgs.read(16)
        # read image data
        buf = imgs.read(IMAGE_SIZE * IMAGE_SIZE * NUM_IMAGES)
        data = np.frombuffer(buf, dtype=np.uint8)
        data = data.reshape(NUM_IMAGES, IMAGE_SIZE, IMAGE_SIZE, 1)
        # extract and squeeze the selected image
        image = np.asarray(data[IMG_IDX]).squeeze()

    # ENCODE IMAGE
    img_mx = img_encode(image)

    # CREATE NN
    rand_range = (-1, 1)
    act_funcs = {1:"sigmoid", 2:"sigmoid", 3:"sigmoid"}
    nn_layout = ((2, 7, 4, 1), act_funcs)
    img_nn = NN(nn_layout, rand_range, img_mx, "Image Upscaler")

    # LOAD NN PARAMS (OPTIONAL)
    file_name = 'UPSC_3.npy'
    data_path = os.path.join (curr_dir, "saved_NNs", file_name)
    read_params = np.load(data_path, allow_pickle='TRUE').item()
    img_nn.import_nn(read_params)

    # LEARN & PLOT & SAVE (OPTIONAL)
    # plot_vals = np.array([[], []])
    # tot_it = 0
    # num_it = 500
    # nn_state, y_vals = img_nn.learn_static(num_it, batch_ratio=1/(28**2), rate=1, stop=False, with_plot=False)
    # new_vals = np.array((np.arange(num_it)+(tot_it+1), np.array(y_vals)))
    # plot_vals = np.hstack((plot_vals, new_vals))
    # tot_it += num_it
    # img_nn.static_plot(plot_vals[0], plot_vals[1])
    # file_name = 'UPSC_new.npy'
    # data_path = os.path.join (curr_dir, "saved_NNs", file_name)
    # np.save(data_path, nn_state) 

    # FEED INPUT
    RESOLUTION = (128, 128)
    # get norm coordinates for the given resolution
    norm_inp = norm_coord(RESOLUTION)
    # create temp array to hold the image matrix
    gen_img = np.array([])
    # iterate through each pair of coordinates
    for inp_data in norm_inp:
        # compute the brightness of the pixel
        output = img_nn.forward([inp_data])
        # combine coordinates and the brightness
        new_row = np.concatenate([inp_data, output[0]])
        # add new entry to the image matrix
        if len(gen_img) > 0:
            gen_img = np.vstack([gen_img, new_row])
        else:
            gen_img = np.concatenate([gen_img, new_row])

    # DECODE IMAGE
    new_img = img_decode(gen_img, RESOLUTION)

    # SHOW ORIGINAL AND GENERATED IMAGE
    plt.figure(figsize=(10, 5))     # adjusts the image size
    imgs = [image, new_img]
    for i in range(2):
        plt.subplot(1, 2, i + 1)
        plt.imshow(imgs[i], cmap='gray')
    plt.show()


if __name__ == "__main__":
    main()