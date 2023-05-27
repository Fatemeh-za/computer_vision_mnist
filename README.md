# computer_vision_mnist

# MNIST Handwritten Digit Recognition

This repository contains an example of a convolutional neural network (CNN) being trained on the MNIST dataset to recognize handwritten digits.

## Requirements

- Python 3
- PyTorch
- Torchvision


## Code Explanation

The code is divided into four main parts:

1. **Define settings**: This section sets up the basic parameters for the training process, such as the batch size, number of epochs, learning rate, and decay rate.

2. **Prepare the data**: This section loads the MNIST dataset and applies transformations to convert the images into tensors and normalize them.

3. **Define the architecture of the neural network**: This section defines the architecture of the CNN using PyTorch's `nn.Module` class. The network consists of two convolutional layers followed by max pooling, and three fully connected layers.

4. **Train the model**: This section trains the model using stochastic gradient descent with momentum. The training process involves iterating over the training data, computing the forward pass to get the outputs from the inputs using the network, computing the loss from the outputs and labels using the loss function, computing the gradients from the loss using backpropagation, and updating the parameters using the optimizer.
