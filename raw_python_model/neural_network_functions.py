import numpy as np

from raw_python_model.neural_network_components import conv_backward, conv_forward, pool_backward, pool_forward, relu, relu_backward

def initialize_parameters():
    """
    Initializes parameters for the convolutional neural network with specific shapes for the weights and biases of each layer.
    The weights are initialized using a scaled Gaussian distribution to aid in convergence.

    Returns:
    parameters -- python dictionary containing:
                  'W1' -- weights for the first convolutional layer, shape: (5, 5, 3, 8)
                  'b1' -- biases for the first convolutional layer, shape: (1, 1, 1, 8)
                  'W2' -- weights for the second convolutional layer, shape: (3, 3, 8, 16)
                  'b2' -- biases for the second convolutional layer, shape: (1, 1, 1, 16)
                  'W3' -- weights for the fully connected layer, shape: (16*2*2, 6)
                  'b3' -- biases for the fully connected layer, shape: (1, 6)
    """
    # Setting the same seed to ensure consistency in weight initialization across different runs, useful when testing small hparams changes
    np.random.seed(1)
    
    # Initialize weights and biases for the first convolutional layer, where 5x5 is the filter size, 
    # 3 is the number of input channels (RGB), and 8 is the number of filters.
    # Weights are normalized using He initialization (scaling by the square root of the number of units in the preceding layer)
    W1 = np.random.randn(5, 5, 3, 8) / np.sqrt(5*5*3)
    
    # Initialize biases for the first convolutional layer.
    # Shape: (1, 1, 1, 8), one bias per filter initialized to zero 
    b1 = np.zeros((1, 1, 1, 8))
    
    # Initialize weights and biases for the second convolutional layer where 3x3 is the filter size, 
    # 8 is the number of input channels (from previous layer), and 16 is the number of filters.
    W2 = np.random.randn(3, 3, 8, 16) / np.sqrt(3*3*8)
    
    # Initialize biases for the second convolutional layer.
    # Shape: (1, 1, 1, 16), one bias per filter initialized to zero 
    b2 = np.zeros((1, 1, 1, 16))
    
    # Initialize weights and biases for the fully connected (Dense) layer, 
    # where 16*2*2 is the flattened output size from the previous layer (after convolution and pooling).
    # Using He initialization for the weights to maintain the variance in activations
    W3 = np.random.randn(16*2*2, 6) / np.sqrt(16*2*2)
    
    # Initialize biases for the fully connected layer.
    # Shape: (1, 6), one bias per output class 
    b3 = np.zeros((1, 6))
    
    # Combine all parameters into a dictionary to be returned
    parameters = {"W1": W1, "b1": b1, "W2": W2, "b2": b2, "W3": W3, "b3": b3}
    
    return parameters

def initialize_hyperparameters():
    """
    Initializes hyperparameters for convolutional and pooling layers to specify strides and padding.
    
    Returns:
    hparameters -- dictionary containing hyperparameters:
                   'conv1' -- stride and padding for the first convolutional layer
                   'pool1' -- filter size, stride, and padding for the first pooling layer
                   'conv2' -- stride and padding for the second convolutional layer
                   'pool2' -- filter size, stride, and padding for the second pooling layer
    """
    hparameters = {}
    # The padding is calculated as to keep "Same" padding through the layers and avoid reducing the image size
    
    # Hyperparameters for Convolutional Layer 1
    hparameters['conv1'] = {"stride": 1, "pad": 2}
    
    # Hyperparameters for Pooling Layer 1
    hparameters['pool1'] = {"f": 8, "stride": 8, "pad": 4}
    
    # Hyperparameters for Convolutional Layer 2
    hparameters['conv2'] = {"stride": 1, "pad": 1}
    
    # Hyperparameters for Pooling Layer 2
    hparameters['pool2'] = {"f": 4, "stride": 4, "pad": 2}
    
    return hparameters

def forward_propagation(X, parameters, hparameters, print_shapes=False):
    """
    Implements the forward propagation for the model:
    CONV2D -> RELU -> MAXPOOL -> CONV2D -> RELU -> MAXPOOL -> FLATTEN -> DENSE
    
    Arguments:
    X -- input dataset, numpy array of shape (m, n_H_prev, n_W_prev, n_C_prev)
    parameters -- python dictionary containing parameters "W1", "b1", "W2", "b2", "W3", "b3"
    print_shapes -- boolean, if True, prints the shapes of various tensors throughout the model

    Returns:
    Z3 -- the output of the last LINEAR unit
    cache -- tuple containing the caches from each layer including Z3 and W3
    """
    
    # Retrieve the parameters for the convolutional and fully connected layers
    W1 = parameters['W1']  # Shape: (5, 5, 3, 8), where 5x5 is the filter size, 3 is the number of input channels (RGB), and 8 is the number of filters.
    b1 = parameters['b1']  # Shape: (1, 1, 1, 8), one bias per filter.
    W2 = parameters['W2']  # Shape: (3, 3, 8, 16), where 3x3 is the filter size, 8 is the number of input channels (from previous layer), and 16 is the number of filters.
    b2 = parameters['b2']  # Shape: (1, 1, 1, 16), one bias per filter.
    W3 = parameters['W3']  # Shape: (16*2*2, 6), flattened output size from the previous layer (after convolution and pooling) by the number of classes.
    b3 = parameters['b3']  # Shape: (1, 6), one bias per class.

    
    # First convolutional layer operation
    Z1, cache_conv1 = conv_forward(X, W1, b1, hparameters['conv1'])
    if print_shapes:
        print("Shape of Z1 (after first conv):", Z1.shape)
    
    # Activation function after first convolution
    A1, cache_relu1 = relu(Z1)
    
    # First pooling layer reduces spatial dimensions through MaxPooling (with no trainable parametres)
    A1_pool, cache_pool1 = pool_forward(A1, hparameters['pool1'], mode="max")
    if print_shapes:
        print("Shape of A1_pool (after first pool):", A1_pool.shape)
    
    # Second convolutional layer operation
    Z2, cache_conv2 = conv_forward(A1_pool, W2, b2, hparameters['conv2'])
    if print_shapes:
        print("Shape of Z2 (after second conv):", Z2.shape)

    # Activation function after second convolution
    A2, cache_relu2 = relu(Z2)
    
    
    # Second pooling layer reduces spatial dimensions through MaxPooling (with no trainable parametres)
    A2_pool, cache_pool2 = pool_forward(A2, hparameters['pool2'], mode="max")
    if print_shapes:
        print("Shape of A2_pool (after second pool):", A2_pool.shape)
    
    # Flatten the output from the final pooling layer to prepare for the dense layer
    A2_flatten = A2_pool.reshape(A2_pool.shape[0], -1)
    if print_shapes:
        print("Shape of A2_flatten (after flatten):", A2_flatten.shape)


    # Dense layer operation to produce the final output predictions
    Z3 = np.dot(A2_flatten, W3) + b3
    if print_shapes:
        print("Shape of Z3 (output of dense layer):", Z3.shape)
    
    # Collect all intermediate results and caches for the backward pass
    cache = (Z1, cache_conv1, cache_relu1, cache_pool1, Z2, cache_conv2, cache_relu2, cache_pool2, A2_pool, W3, Z3)
    
    return Z3, cache

def compute_cost(Z3, Y, print_shapes=False):
    """
    Compute the cross-entropy cost using the softmax output of the last linear layer and the true labels.
    
    Arguments:
    Z3 -- output of the last linear unit, shape (m, n_classes), where m is the number of examples
    Y -- true "label" vector, one-hot encoded, shape (m, n_classes)
    print_shapes -- boolean, if True, prints the shapes of Z3 and the computed cost

    
    Returns:
    cost -- cross-entropy cost averaged over all examples
    """
    m = Y.shape[0]  # The number of examples in the batch
    n_classes = Y.shape[1]  # The number of classes (dimensions of softmax output)

    if print_shapes:
        print("Computing cost after forward propagation:")
        print("From Z3.shape:", Z3.shape)
    
    # Step 1: Normalize Z3 to avoid overflow in exp
    # Find the max value in each example to use for normalization
    Z3_max = np.max(Z3, axis=1, keepdims=True)
    
    # Subtract the max value for numerical stability in exp calculations
    Z3_stable = Z3 - Z3_max

    # Step 2: Compute softmax probabilities
    # Exponentiate the stabilized logits
    exp_scores = np.exp(Z3_stable)
    
    # Normalize to get probabilities (knowing it wont be a division by 0 due to exponentiation)
    softmax = exp_scores / np.sum(exp_scores, axis=1, keepdims=True)

    # Step 3: Compute cross-entropy loss
    # Compute log of probabilities, add epsilon to prevent log(0)
    log_probs = np.log(softmax + 1e-8)
    
    # Sum across all examples and classes
    cross_entropy_loss = -np.sum(Y * log_probs)

    # Step 4: Average the loss over all examples
    cost = cross_entropy_loss / m
    
    if print_shapes:
        print("Computed cost:", cost)
    
    return cost

def backward_propagation(Y, cache, print_shapes=False):
    """
    Implement the backward propagation for the model:
    CONV2D -> RELU -> MAXPOOL -> CONV2D -> RELU -> MAXPOOL -> FLATTEN -> DENSE
    
    Arguments:
    Y -- true "label" vector, of shape (m, n_classes)
    cache -- cache output from forward_propagation()
    print_shapes -- boolean, if True, prints the shapes of various tensors throughout the model
    
    Returns:
    gradients -- A dictionary with the gradients
    """
    # Unpack cache from forward propagation
    (Z1, cache_conv1, cache_relu1, cache_pool1, Z2, cache_conv2, cache_relu2, cache_pool2, A2_pool, W3, Z3) = cache
    
    # Initialize gradients dictionary
    gradients = {}
    
    # Number of examples
    m = Y.shape[0]
    
    # Compute gradients for the output layer (DENSE)
    if print_shapes:
        print("Z3 shape:", Z3.shape)
    
    # Apply the log-sum-exp to stabilize the computation of softmax during backpropagation, same as on the cost compute
    # Find max for numerical stability
    Z3_max = np.max(Z3, axis=1, keepdims=True)
    
     # Subtract max from Z3
    Z3_stable = Z3 - Z3_max
    
    # Exponentiate the stabilized scores
    exp_scores = np.exp(Z3_stable)
    
    # Compute softmax probabilities
    softmax = exp_scores / np.sum(exp_scores, axis=1, keepdims=True)
    
    # Calculate the gradient of loss with respect to Z3 (which are the predictions)
    dZ3 = softmax - Y
    if print_shapes:
        print("dZ3 shape:", dZ3.shape)
    
    # Backpropagation through Dense layer
    # Flatten pooled feature map to feed into dense layer
    A2_flatten = A2_pool.reshape(A2_pool.shape[0], -1)
    
    # Compute gradient of weights of dense layer
    gradients['dW3'] = 1./m * np.dot(A2_flatten.T, dZ3)
    
    # Compute gradient of biases of dense layer
    gradients['db3'] = 1./m * np.sum(dZ3, axis=0, keepdims=True)
    
    if print_shapes:
        print("A2_flatten shape:", A2_flatten.shape)
        print("dW3 shape:", gradients['dW3'].shape)
        print("db3 shape:", gradients['db3'].shape)
    
    # Gradient propagation through Flatten layer
    # Map gradients back to the pre-flattened layer
    dA2_flatten = np.dot(dZ3, W3.T)
    
    # Reshape gradients to match the pooling layer's output shape
    dA2_pool = dA2_flatten.reshape(A2_pool.shape)
    
    if print_shapes:
        print("dA2_flatten shape:", dA2_flatten.shape)
        print("dA2_pool shape:", dA2_pool.shape)

    # Backpropagation through Max Pool layer 2
    # Gradient through max pooling layer
    dA2 = pool_backward(dA2_pool, cache_pool2, mode="max")
    
    if print_shapes:
        print("dA2 shape:", dA2.shape)
    
    # Backpropagation through ReLU layer 2
    # Apply gradient of ReLU to get gradient with respect to Z2
    dZ2 = relu_backward(dA2, cache_relu2)
    
    if print_shapes:
        print("dZ2 shape:", dZ2.shape)

    # Backpropagation through Conv layer 2
    # Gradient through convolutional layer
    dA1_pool, gradients['dW2'], gradients['db2'] = conv_backward(dZ2, cache_conv2)
    
    if print_shapes:
        print("dA1_pool shape:", dA1_pool.shape)
        print("dW2 shape:", gradients['dW2'].shape)
        print("db2 shape:", gradients['db2'].shape)

    # Backpropagation through Max Pool layer 1
    # Gradient through max pooling layer
    dA1 = pool_backward(dA1_pool, cache_pool1, mode="max")
    
    if print_shapes:
        print("dA1 shape:", dA1.shape)
    
    # Backpropagation through ReLU layer 1
    # Apply gradient of ReLU to get gradient with respect to Z1
    dZ1 = relu_backward(dA1, cache_relu1)
    
    if print_shapes:
        print("dZ1 shape:", dZ1.shape)

    # Backpropagation through Conv layer 1
    # Gradient through convolutional layer
    _, gradients['dW1'], gradients['db1'] = conv_backward(dZ1, cache_conv1)
    
    if print_shapes:
        print("dW1 shape:", gradients['dW1'].shape)
        print("db1 shape:", gradients['db1'].shape)
    
    return gradients


# # OLD
def old_backward_propagation(Y, cache, print_shapes=False):
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
    (Z1, cache_conv1, cache_relu1, cache_pool1, Z2, cache_conv2, cache_relu2, cache_pool2, A2_pool, Z3) = cache
    
    # Computes the gradient of the loss function with respect to the output of the last linear unit (Z3). 
    # It calculates the difference between the predicted output Z3 and the true labels Y.
    dZ3 = Z3 - Y
    print("dZ3 shape:", dZ3.shape)
    
    # A2_flatten is created by reshaping the output of the second pooling layer (A2_pool).
    # The reshaping operation transforms the 4D multi-dimensional array output of the pooling layer 
    # into a 2D multi-dimensional array, suitable for further computations.
    print("A2_pool shape:", A2_pool.shape) 
    A2_flatten = A2_pool.reshape(A2_pool.shape[0], -1)
    print("A2_flatten shape:", A2_flatten.shape) 
    
    # Calculate the gradient of the loss with respect to the weights (W3) multiplying the flattened activations (A2_flatten) 
    # and the gradient of the loss (dZ3)
    gradients['dW3'] = np.dot(A2_flatten.T, dZ3)
    print("dW3 shape:", gradients['dW3'].shape)
    
    # Calculates the gradient of the loss with respect to the biases (b3) by summing the gradients along the examples axis.
    gradients['db3'] = np.sum(dZ3, axis=0, keepdims=True)
    print("db3 shape:", gradients['db3'].shape) 
    
    # Computes the gradient of the loss with respect to the output of the flattening layer. 
    # It's calculated by multiplying the gradient of the loss with respect to the output of the dense layer 
    # by the transpose of the activations from the previous pooling layer.
    print("dZ3 shape:", dZ3.shape)
    print("A2_pool shape:", A2_pool.shape)
    print("A2_flatten shape:", A2_flatten.shape) 
    dA2_pool = np.dot(dZ3, A2_pool.T)
    print("dA2_pool shape:", dA2_pool.shape)
    
    # dA2 is reshaped to match the shape of the output of the second pooling layer (A2_pool). 
    # This step "unflattens" the pooled gradients back to their original shape.
    print("cache_pool2 A2_pool shape:", A2_pool.shape)
    dA2 = dA2_pool.reshape(A2_pool.shape)
    print("dA2_pool shape:", dA2_pool.shape)
    
    # dA2_relu calculates the gradient of the loss with respect to the output of the ReLU activation function 
    # using the cached values from the forward propagation step.
    dA2_relu = relu_backward(dA2, cache_relu2)
    print("dA2_relu shape:", dA2_relu.shape)
    
    # dA2_pool computes the gradient of the loss with respect to the output of the second pooling layer 
    # by applying the backward operation of the pooling layer.
    dA2_pool = pool_backward(dA2_relu, cache_pool2, mode="max")
    print("dA2_pool shape:", dA2_pool.shape)
    
    # dZ2 computes the gradient of the loss with respect to the output of the second convolutional layer 
    # by applying the backward operation of the convolutional layer.
    dZ2 = conv_backward(dA2_pool, cache_conv2)
    print("dZ2 shape:", dZ2.shape)
    
    # gradients['dW2'] calculates the gradient of the loss with respect to the weights (W2) 
    # by summing the gradients along the examples axis.
    gradients['dW2'] = np.sum(dZ2, axis=0, keepdims=True)
    print("dW2 shape:", gradients['dW2'].shape)
    
    # gradients['db2'] calculates the gradient of the loss with respect to the biases (b2) 
    # by summing the gradients along the spatial dimensions.
    gradients['db2'] = np.sum(dZ2, axis=(0,1,2), keepdims=True)
    print("db2 shape:", gradients['db2'].shape)
    
    # dA1_relu calculates the gradient of the loss with respect to the output of the 
    # first ReLU activation function using the cached values from the forward propagation step.
    dA1_relu = relu_backward(dZ2, cache_relu1)
    print("dA1_relu shape:", dA1_relu.shape)
    
    # dA1_pool computes the gradient of the loss with respect to the output of the first pooling layer 
    # by applying the backward operation of the pooling layer.
    dA1_pool = pool_backward(dA1_relu, cache_pool1, mode="max")
    print("dA1_pool shape:", dA1_pool.shape)
    
    # dZ1 computes the gradient of the loss with respect to the output of the first convolutional layer 
    # by applying the backward operation of the convolutional layer.
    dZ1 = conv_backward(dA1_pool, cache_conv1)
    print("dZ1 shape:", dZ1.shape)
    
    # gradients['dW1'] calculates the gradient of the loss with respect to the weights (W1) 
    # by summing the gradients along the examples axis.
    gradients['dW1'] = np.sum(dZ1, axis=0, keepdims=True)
    print("dW1 shape:", gradients['dW1'].shape)
    
    # gradients['db1'] calculates the gradient of the loss with respect to the biases (b1) by summing the gradients along the spatial dimensions.
    gradients['db1'] = np.sum(dZ1, axis=(0,1,2), keepdims=True)
    print("db1 shape:", gradients['db1'].shape)
    
    return gradients


def update_parameters(parameters, gradients, learning_rate):
    """
    Update parameters using gradient descent
    
    Arguments:
    parameters -- python dictionary containing your parameters 
    gradients -- python dictionary containing your gradients, output of backward_propagation
    learning_rate -- float, the learning rate used in the gradient descent update
    
    Returns:
    parameters -- python dictionary containing your updated parameters 
    """
    L = len(parameters) // 2 # number of layers in the neural network

    print("Updating parameters with gradient descent...")
    # Iterate over each layer to update its parameters
    for l in range(L):
        parameters["W" + str(l+1)] = parameters["W" + str(l+1)] - learning_rate * gradients["dW" + str(l+1)]
        parameters["b" + str(l+1)] = parameters["b" + str(l+1)] - learning_rate * gradients["db" + str(l+1)]

    return parameters


def evaluate_model(X_test, Y_test, parameters, hparameters):
    """
    Evaluates the performance of the neural network model on a test set.
    
    Arguments:
    X_test -- input data for the test set, numpy array of shape (m_test, n_H, n_W, n_C)
    Y_test -- true "label" vectors for the test set, numpy array of shape (m_test, n_classes)
    parameters -- python dictionary containing your parameters "W1", "b1", "W2", "b2", "W3", "b3"
    hparameters -- hyperparameters for the model, including stride and padding information
    
    Returns:
    accuracy -- the accuracy of the model on the test set
    """
    # Conduct forward propagation to predict the test set results
    Z3, _ = forward_propagation(X_test, parameters, hparameters)
    
    # Conduct forward propagation to predict the test set results
    predictions = np.argmax(Z3, axis=1)
    
    # Extract class labels from the true labels for comparison
    labels = np.argmax(Y_test, axis=1)
    
    # Calculate the accuracy as the mean of correct predictions
    accuracy = np.mean(predictions == labels)
    
    return accuracy
