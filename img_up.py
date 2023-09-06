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
    # LOAD IMAGE
    curr_dir = os.getcwd()
    file_name = 'MNIST-train-images-idx3-ubyte.gz'
    data_path = os.path.join (curr_dir, "train_data", file_name)

    image_size = 28
    num_images = 50

    with gzip.open (data_path,'r') as imgs:
        # magic_number = int.from_bytes(imgs.read(4), 'big')
        # tot_num_images = int.from_bytes(imgs.read(4), 'big')
        
        # print("Total number of images:", tot_num_images)
        imgs.read(16)
        
        buf = imgs.read(image_size * image_size * num_images)
        data = np.frombuffer(buf, dtype=np.uint8)
        data = data.reshape(num_images, image_size, image_size, 1)

    image = np.asarray(data[4]).squeeze()
    gen_size = (120, 120)

    # plt.imshow(image, cmap='gray', vmin=0, vmax=255)
    # plt.colorbar()
    # plt.show()
    # return

    # ENCODE IMAGE
    img_mx = img_encode(image)
    # norm_inp = np.split(img_mx, [2], 1)[0]
    norm_inp = norm_coord(gen_size)

    # CREATE NN
    rand_range = (-1, 1)
    act_funcs = {1:"relu", 2:"sigmoid", 3:"sigmoid"}
    nn_layout = ((2, 7, 4, 1), act_funcs)
    img_nn = NN(nn_layout, rand_range, img_mx, "Image Upscaler")

    # LEARN & PLOT
    plot_vals = np.array([[], []])
    tot_it = 0

    num_it = 100
    nn_state, y_vals = img_nn.learn_static(num_it, batch_ratio=1/(28**2), rate=1/7, stop=False, with_plot=False)
    new_vals = np.array((np.arange(num_it)+(tot_it+1), np.array(y_vals)))

    plot_vals = np.hstack((plot_vals, new_vals))
    tot_it += num_it

    # <<< PLOT COST >>>
    img_nn.static_plot(plot_vals[0], plot_vals[1])

    # SAVE PARAMS
    file_name = 'NN_params_new.npy'
    data_path = os.path.join (curr_dir, "saved_NNs", file_name)
    np.save(data_path, nn_state) 

    # LOAD PARAMS
    # read_params = np.load(data_path, allow_pickle='TRUE').item()
    # img_nn.load_params(read_params)

    # FEED INPUT
    gen_img = np.array([])
    for inp_data in norm_inp:
        output = img_nn.forward([inp_data])
        new_row = np.concatenate([inp_data, output[0]])

        if len(gen_img) > 0:
            gen_img = np.vstack([gen_img, new_row])
        else:
            gen_img = np.concatenate([gen_img, new_row])

    # DECODE IMAGE
    new_img = img_decode(gen_img, gen_size)

    # SHOW RESULT
    plt.imshow(new_img, cmap='gray', vmin=0, vmax=255)
    plt.colorbar()
    plt.show()


if __name__ == "__main__":
    main()