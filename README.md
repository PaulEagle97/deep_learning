# Deep Learning from First Principles

This project aims to explore deep learning by building algorithms from the ground up, using only core Python functionalities, Numpy arrays, and fundamental mathematical principles. 
Without relying on high-level libraries, this initiative seeks to provide an educational insight into the inner workings of deep learning models, without the abstraction layers introduced by comprehensive libraries such as TensorFlow or PyTorch.

## Project Structure

- **mlp_framework.py**: The core of the project, containing the backbone of the implemented algorithms.
- **applications/**: Various scripts, demonstrating the practical application of the developed algorithms.
  - **computer_vision**: Applications, focused on the Computer Vision technologies.
    - **ocr.py**: Work in progress. A simple optical character recognition script based on the MNIST dataset.
    - **img_up.py**: Work in progress. Aims to upscale image resolution of the models, that were previously trained by `ocr.py`.
- **docs/**: Contains HTML-formatted documentation for `mlp_framework.py`, offering detailed insights into its functionality and usage.
- **legacy_code/**: Archives previous versions of the main framework, showcasing the development journey and iterations.
- **misc/**: Includes auxiliary code used for testing and other non-critical functions. This code is never imported by other modules but is essential for development and debugging processes.

## Getting Started

To initiate work on this project, first clone the repository. Then, create a virtual environment and install the required dependencies listed in the `requirements.txt` file to set up your environment. 
