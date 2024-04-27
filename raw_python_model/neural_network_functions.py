import numpy as np

from raw_python_model.neural_network_components import conv_backward, conv_forward, pool_backward, pool_forward, relu, relu_backward

# Function to initialize parameters
def initialize_parameters():
    """
    Initializes parameters for the convolutional neural network.

    Returns:
    parameters -- python dictionary containing initialized parameters:
                    W1 -- weights for the first convolutional layer, shape: (4, 4, 3, 8)
                    b1 -- biases for the first convolutional layer, shape: (1, 1, 1, 8)
                    W2 -- weights for the second convolutional layer, shape: (2, 2, 8, 16)
                    b2 -- biases for the second convolutional layer, shape: (1, 1, 1, 16)
                    W3 -- weights for the fully connected layer, shape: (16*8*8, 6)
                    b3 -- biases for the fully connected layer, shape: (1, 6)
    """
    # Setting the same seed for consistency
    np.random.seed(1)
    
    # CONV layer 1 parameters
    # W1: Randomly initialized weights for the first convolutional layer.
    #     Shape: (4, 4, 3, 8), where 4x4 is the filter size, 3 is the number of input channels (RGB), and 8 is the number of filters.
    #     Weights are scaled by the square root of the number of input units (4*4*3) for better convergence.
    W1 = np.random.randn(4, 4, 3, 8) / np.sqrt(4*4*3)
    # b1: Initialized biases for the first convolutional layer.
    #     Shape: (1, 1, 1, 8), one bias per filter.
    b1 = np.zeros((1, 1, 1, 8))
    
    # CONV layer 2 parameters
    # W2: Randomly initialized weights for the second convolutional layer.
    #     Shape: (2, 2, 8, 16), where 2x2 is the filter size, 8 is the number of input channels (from previous layer), and 16 is the number of filters.
    #     Weights are scaled by the square root of the number of input units (2*2*8) for better convergence.
    W2 = np.random.randn(2, 2, 8, 16) / np.sqrt(2*2*8)
    # b2: Initialized biases for the second convolutional layer.
    #     Shape: (1, 1, 1, 16), one bias per filter.
    b2 = np.zeros((1, 1, 1, 16))
    
    # Dense layer parameters
    # W3: Randomly initialized weights for the fully connected layer.
    #     Shape: (16*8*8, 6), where 16*8*8 is the flattened output size from the previous layer (after convolution and pooling).
    #     Weights are scaled by the square root of the flattened output size for better convergence.
    W3 = np.random.randn(16*8*8, 6) / np.sqrt(16*8*8)
    # b3: Initialized biases for the fully connected layer.
    #     Shape: (1, 6), one bias per class.
    b3 = np.zeros((1, 6))
    
    parameters = {"W1": W1, "b1": b1, "W2": W2, "b2": b2, "W3": W3, "b3": b3}
    
    return parameters

def initialize_hyperparameters():
    """
    Initializes hyperparameters for convolutional and pooling layers.
    
    Returns:
    hparameters -- dictionary containing hyperparameters for each layer
    """
    hparameters = {}
    # The padding is calculated as to keep Same padding through the layers and avoid reducing the image size
    
    # Hyperparameters for Convolutional Layer 1
    hparameters['conv1'] = {"stride": 1, "pad": 2}
    
    # Hyperparameters for Pooling Layer 1
    hparameters['pool1'] = {"f": 8, "stride": 8, "pad": 4}
    
    # Hyperparameters for Convolutional Layer 2
    hparameters['conv2'] = {"stride": 1, "pad": 1}
    
    # Hyperparameters for Pooling Layer 2
    hparameters['pool2'] = {"f": 4, "stride": 4, "pad": 2}
    
    return hparameters

def forward_propagation(X, parameters, hparameters):
    """
    Implements the forward propagation for the model:
    CONV2D -> RELU -> MAXPOOL -> CONV2D -> RELU -> MAXPOOL -> FLATTEN -> DENSE
    
    Arguments:
    X -- input dataset, numpy array of shape (m, n_H_prev, n_W_prev, n_C_prev)
    parameters -- python dictionary containing your parameters "W1", "b1", "W2", "b2", "W3", "b3"
                  the shapes are given in initialize_parameters

    Returns:
    Z3 -- the output of the last LINEAR unit
    cache -- tuple containing "Z1", "cache_conv1", "cache_relu1", "cache_pool1", "Z2", "cache_conv2", "cache_relu2", "cache_pool2", "Z3"
    """
    
    # Retrieve the parameters from the "parameters" dictionary
    W1 = parameters['W1']  # Shape: (4, 4, 3, 8), where 4x4 is the filter size, 3 is the number of input channels (RGB), and 8 is the number of filters.
    b1 = parameters['b1']  # Shape: (1, 1, 1, 8), one bias per filter.
    W2 = parameters['W2']  # Shape: (2, 2, 8, 16), where 2x2 is the filter size, 8 is the number of input channels (from previous layer), and 16 is the number of filters.
    b2 = parameters['b2']  # Shape: (1, 1, 1, 16), one bias per filter.
    W3 = parameters['W3']  # Shape: (16*8*8, 6), flattened output size from the previous layer (after convolution and pooling) by the number of classes.
    b3 = parameters['b3']  # Shape: (1, 6), one bias per class.
    
    # CONV: with 8 4 by 4 filters, stride 1 and the corresponding weights
    Z1, cache_conv1 = conv_forward(X, W1, b1, hparameters['conv1'])
    # RELU
    A1, cache_relu1 = relu(Z1)
    # MAXPOOL: with 8 by 8 filter size and an 8 by 8 stride and with no trainable parametres 
    A1_pool, cache_pool1 = pool_forward(A1, hparameters['pool1'], mode="max")
    # CONV: with 16 2 by 2 filters, stride 1 and the corresponding weights
    Z2, cache_conv2 = conv_forward(A1_pool, W2, b2, hparameters['conv2'])
    # RELU
    A2, cache_relu2 = relu(Z2)
    # MAXPOOL: with a 4 by 4 filter size and a 4 by 4 stride and with no trainable parametres 
    A2_pool, cache_pool2 = pool_forward(A2, hparameters['pool2'], mode="max")
    # FLATTEN
    print("A2_pool.shape",A2_pool.shape)
    A2_flatten = A2_pool.reshape(A2_pool.shape[0], -1)
    print("🚀 ~ A2_flatten.shape:", A2_flatten.shape)
    print("🚀 ~ W3.shape:", W3.shape)
    print("🚀 ~ b3.shape:", b3.shape)
    # DENSE layer
    Z3 = np.dot(A2_flatten, W3) + b3
    
    cache = (Z1, cache_conv1, cache_relu1, cache_pool1, Z2, cache_conv2, cache_relu2, cache_pool2, Z3)
    
    return Z3, cache

def compute_cost(Z3, Y):
    """
    Compute the cross-entropy cost given the output of the last layer and the true labels.
    
    Arguments:
    Z3 -- output of the last linear unit, shape (m, n_classes)
    Y -- true "label" vector, shape (m, n_classes)
    
    Returns:
    cost -- cross-entropy cost
    """
    m = Y.shape[0]  # number of examples
    n_classes = Y.shape[1]  # number of classes
    
    # Compute softmax activation
    softmax = np.exp(Z3) / np.sum(np.exp(Z3), axis=1, keepdims=True)
    
    # Compute cross-entropy loss
    # The cross-entropy loss between predicted and true class labels for each example is summed over all examples
    # and averaged over the number of examples.
    cost = -1/m * np.sum(Y * np.log(softmax + 1e-8))  # Adding a small epsilon to avoid division by zero
    
    return cost

def backward_propagation(Y, cache):
    """
    Implement the backward propagation for the model:
    CONV2D -> RELU -> MAXPOOL -> CONV2D -> RELU -> MAXPOOL -> FLATTEN -> DENSE
    
    Arguments:
    Y -- true "label" vector, of shape (m, n_classes)
    cache -- cache output from forward_propagation()
    
    Returns:
    gradients -- A dictionary with the gradients
    """
    gradients = {}
    (Z1, cache_conv1, cache_relu1, cache_pool1, Z2, cache_conv2, cache_relu2, cache_pool2, Z3) = cache
    
    # Computes the gradient of the loss function with respect to the output of the last linear unit (Z3). 
    # It calculates the difference between the predicted output Z3 and the true labels Y.
    dZ3 = Z3 - Y
    
    # A2_flatten is created by reshaping the output of the second pooling layer (cache_pool2).
    # The reshaping operation transforms the 4D multi-dimensional array output of the pooling layer 
    # into a 2D multi-dimensional array, suitable for further computations.
    A2_flatten = cache_pool2.reshape(cache_pool2.shape[0], -1)
    
    # Calculate the gradient of the loss with respect to the weights (W3) multiplying the flattened activations (A2_flatten) 
    # and the gradient of the loss (dZ3)
    gradients['dW3'] = np.dot(A2_flatten.T, dZ3)
    
    # Calculates the gradient of the loss with respect to the biases (b3) by summing the gradients along the examples axis.
    gradients['db3'] = np.sum(dZ3, axis=0, keepdims=True)
    
    # Computes the gradient of the loss with respect to the output of the flattening layer (dA2_pool). 
    # It's calculated by multiplying the gradient of the loss with respect to the output of the dense layer (dZ3) 
    # by the transpose of the activations from the previous pooling layer (cache_pool2).
    dA2_pool = np.dot(dZ3, cache_pool2.T)
    
    # dA2 is reshaped to match the shape of the output of the second pooling layer (cache_pool2). 
    # This step "unflattens" the pooled gradients back to their original shape.
    dA2 = dA2_pool.reshape(cache_pool2.shape)
    
    # dA2_relu calculates the gradient of the loss with respect to the output of the ReLU activation function 
    # using the cached values from the forward propagation step.
    dA2_relu = relu_backward(dA2, cache_relu2)
    
    # dA2_pool computes the gradient of the loss with respect to the output of the second pooling layer 
    # by applying the backward operation of the pooling layer.
    dA2_pool = pool_backward(dA2_relu, cache_pool2, mode="max")
    
    # dZ2 computes the gradient of the loss with respect to the output of the second convolutional layer 
    # by applying the backward operation of the convolutional layer.
    dZ2 = conv_backward(dA2_pool, cache_conv2)
    
    # gradients['dW2'] calculates the gradient of the loss with respect to the weights (W2) 
    # by summing the gradients along the examples axis.
    gradients['dW2'] = np.sum(dZ2, axis=0, keepdims=True)
    
    # gradients['db2'] calculates the gradient of the loss with respect to the biases (b2) 
    # by summing the gradients along the spatial dimensions.
    gradients['db2'] = np.sum(dZ2, axis=(0,1,2), keepdims=True)
    
    # dA1_relu calculates the gradient of the loss with respect to the output of the 
    # first ReLU activation function using the cached values from the forward propagation step.
    dA1_relu = relu_backward(dZ2, cache_relu1)
    
    # dA1_pool computes the gradient of the loss with respect to the output of the first pooling layer 
    # by applying the backward operation of the pooling layer.
    dA1_pool = pool_backward(dA1_relu, cache_pool1, mode="max")
    
    # dZ1 computes the gradient of the loss with respect to the output of the first convolutional layer 
    # by applying the backward operation of the convolutional layer.
    dZ1 = conv_backward(dA1_pool, cache_conv1)
    
    # gradients['dW1'] calculates the gradient of the loss with respect to the weights (W1) 
    # by summing the gradients along the examples axis.
    gradients['dW1'] = np.sum(dZ1, axis=0, keepdims=True)
    
    # gradients['db1'] calculates the gradient of the loss with respect to the biases (b1) by summing the gradients along the spatial dimensions.
    gradients['db1'] = np.sum(dZ1, axis=(0,1,2), keepdims=True)
    
    return gradients


def update_parameters(parameters, gradients, learning_rate):
    """
    Update parameters using gradient descent
    
    Arguments:
    parameters -- python dictionary containing your parameters 
    gradients -- python dictionary containing your gradients, output of backward_propagation
    
    Returns:
    parameters -- python dictionary containing your updated parameters 
                  parameters['W' + str(l)] = ... 
                  parameters['b' + str(l)] = ...
    """
    L = len(parameters) // 2 # number of layers in the neural network

    # Update rule for each parameter
    for l in range(L):
        parameters["W" + str(l+1)] = parameters["W" + str(l+1)] - learning_rate * gradients["dW" + str(l+1)]
        parameters["b" + str(l+1)] = parameters["b" + str(l+1)] - learning_rate * gradients["db" + str(l+1)]

    return parameters
