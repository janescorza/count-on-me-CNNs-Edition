
import numpy as np


def zero_pad(X, pad):
    """
    Pad with zeros all images of the dataset X. The padding is applied to both the height and width of an image.
    
    Argument:
    X -- python numpy array of shape (m, n_H, n_W, n_C) representing a batch of m images
    pad -- integer, amount of padding around each image on vertical and horizontal dimensions
    
    Returns:
    X_pad -- padded image of shape (m, n_H + 2 * pad, n_W + 2 * pad, n_C)
    """
    
    # Add padding to only the the height and the widht, leaving the number of channels and examples unchanged
    X_pad = np.pad(X, ((0, 0), (pad, pad), (pad, pad), (0, 0)), mode='constant', constant_values=(0, 0))
    
    return X_pad


def conv_single_step(a_slice_prev, W, b):
    """
    Apply one filter defined by parameters W on a single slice (a_slice_prev) of the output activation 
    of the previous layer.
    
    Arguments:
    a_slice_prev -- slice of input data of shape (f, f, n_C_prev)
    W -- Weight parameters contained in a window - matrix of shape (f, f, n_C_prev)
    b -- Bias parameters contained in a window - matrix of shape (1, 1, 1)
    
    Returns:
    Z -- a scalar value, the result of convolving the sliding window (W, b) on a slice x of the input data
    """
    # Element-wise product between a_slice_prev and W.
    s = a_slice_prev * W
    # Sum over all entries of the volume s and with no axis to sum over both dimensions
    Z = np.sum(s)
    # Adding bias while casting b to a float() so that Z results in a scalar value.
    Z = float(b) + Z
    
    return Z

# FORWARD CONVOLUTION LAYER

def conv_forward(A_prev, W, b, hparameters):
    """
    Implements the forward propagation for a convolution function
    
    Arguments:
    A_prev -- output activations of the previous layer, 
        numpy array of shape (m, n_H_prev, n_W_prev, n_C_prev)
    W -- Weights, numpy array of shape (f, f, n_C_prev, n_C)
    b -- Biases, numpy array of shape (1, 1, 1, n_C)
    hparameters -- python dictionary containing "stride" and "pad"
        
    Returns:
    Z -- conv output, numpy array of shape (m, n_H, n_W, n_C)
    cache -- cache of values needed for the conv_backward() function
    """
    
    # Retrieve dimensions from A_prev's and W (weights that have filter and dimension sizes) 
    (m, n_H_prev, n_W_prev, n_C_prev) = A_prev.shape
    (f, f, n_C_prev, n_C) = W.shape
    stride = hparameters["stride"]
    pad = hparameters["pad"]
    
    # Compute the output dimensions based on the strinde, padding and previous size
    n_H = (int((n_H_prev-f+2*pad)/stride))+1
    n_W = (int((n_W_prev-f+2*pad)/stride))+1
    
    # Initialize output to zeros
    Z = np.zeros((m,n_H, n_W, n_C))
    # Add padding to the input volume
    A_prev_pad = zero_pad(A_prev, pad)
    
    print("🚀 ~ conv_forward")
    print("Input Shape (without padding):", A_prev.shape)
    print("Input Shape (including padding):", A_prev_pad.shape)
    print("Output Shape:", Z.shape)
    
    for i in range(m):
        # Select ith example's padded activation
        a_prev_pad = A_prev_pad[i]
        for h in range(n_H):
            # find vertical start and end based on stride and filter size
            vert_start = h*stride
            vert_end = h*stride + f
            for w in range(n_W):
                # find horizontal start and end based on stride and filter size
                horiz_start = w*stride 
                horiz_end = w*stride + f 
                for c in range(n_C):
                    # Where a_prev_pad has size (n_H_prev, n_W_prev, n_C_prev) with the additional padding
                    a_slice_prev = a_prev_pad[vert_start:vert_end, horiz_start:horiz_end, :]
                    # Select ith filter's weights, as we convolve each filter individually with the whole input 
                    # (that's why we have a : in the third dimensions, getting all n_C_prev dimensions)
                    # Where has size (f, f, n_C_prev, n_C) 
                    weights = W[:,:,:,c]
                    biases = b[:,:,:,c]
                    Z[i, h, w, c] = conv_single_step(a_slice_prev, weights, biases)
        
    # Save forward prop info in "cache" for backprop
    cache = (A_prev, W, b, hparameters)
    
    return Z, cache

def relu(z):
    """
    ReLU of z

    Arguments:
    z -- A scalar or numpy array of any size.

    Return:
    r -- ReLU(z)
    cache -- a tuple containing "Z"
    """
    print("🚀 ~ relu")
    print("Input Shape:", z.shape)
    r = np.maximum(0, z)
    cache = z

    return r, cache

# FORWARD POOLING FUNCTION

def pool_forward(A_prev, hparameters, mode = "max"):
    """
    Implements the forward pass of the pooling layer
    
    Arguments:
    A_prev -- Input data, numpy array of shape (m, n_H_prev, n_W_prev, n_C_prev)
    hparameters -- python dictionary containing "f" and "stride"
    mode -- the pooling mode you would like to use, defined as a string ("max" or "average")
    
    Returns:
    A -- output of the pool layer, a numpy array of shape (m, n_H, n_W, n_C)
    cache -- cache used in the backward pass of the pooling layer, contains the input and hparameters 
    """
    # Retrieve dimensions from the input shape
    (m, n_H_prev, n_W_prev, n_C_prev) = A_prev.shape
    
    f = hparameters["f"]
    stride = hparameters["stride"]
    
    # Define the dimensions of the output based on the filter size and stride
    n_H = int(1 + (n_H_prev - f) / stride)
    n_W = int(1 + (n_W_prev - f) / stride)
    # Dimensions remain unchanged during pooling
    n_C = n_C_prev
    
    
    # Initialize output matrix
    A = np.zeros((m, n_H, n_W, n_C))   
    
    print("🚀 ~ pool_forward")
    print("Input Shape:", A_prev.shape)
    print("Output Shape:", A.shape)
    
    # loop over the examples
    for i in range(m):
        # Select ith example
        a_prev = A_prev[i]
        for h in range(n_H):
            # find vertical start and end based on stride and filter size
            vert_start = h*stride
            vert_end = h*stride + f
            for w in range(n_W):
                # find horizontal start and end based on stride and filter size
                horiz_start = w*stride
                horiz_end = w*stride + f
                # loop over the channels which remain unchaged between input & ouput 
                for c in range(n_C):
                    # Select ith example's slice for each channel, as we do not mix channels on pooling
                    # Where a_prev has size (n_H_prev, n_W_prev, n_C_prev)
                    a_prev_slice = a_prev[vert_start:vert_end, horiz_start:horiz_end, c]
                    # Compute pooling operation and set max or avg of slice, in the output layer position
                    if mode == "max":
                        A[i, h, w, c] = np.max(a_prev_slice)
                    elif mode == "avg":
                        A[i, h, w, c] = np.mean(a_prev_slice)
    
    
    # Store the input and hparameters in cache back prop
    cache = (A_prev, hparameters)

    
    return A, cache


# BACKPROP CONVOLUTION FUNCTION
def conv_backward(dZ, cache):
    """
    Implement the backward propagation for a convolution function
    
    Arguments:
    dZ -- gradient of the cost with respect to the output of the conv layer (Z), numpy array of shape (m, n_H, n_W, n_C)
    cache -- cache of values needed for the conv_backward(), output of conv_forward()
    
    Returns:
    dA_prev -- gradient of the cost with respect to the input of the conv layer (A_prev),
               numpy array of shape (m, n_H_prev, n_W_prev, n_C_prev)
    dW -- gradient of the cost with respect to the weights of the conv layer (W)
          numpy array of shape (f, f, n_C_prev, n_C)
    db -- gradient of the cost with respect to the biases of the conv layer (b)
          numpy array of shape (1, 1, 1, n_C)
    """    
    
    # Retrieve information from "cache"
    (A_prev, W, b, hparameters) = cache
    # Retrieve dimensions from A_prev's shape which will store the dimensions "before" the current layer
    (m, n_H_prev, n_W_prev, n_C_prev) = A_prev.shape
    # Retrieve dimensions from W's shape which help calculate the dimensions/values "before" the current layer
    (f, f, n_C_prev, n_C) = W.shape

    stride = hparameters["stride"]
    pad = hparameters["pad"]
    
    # Retrieve dimensions from dZ's shape which is the "output" we are going "back" from
    (m, n_H, n_W, n_C) = dZ.shape
    
    # Initialize dA_prev with the corresponding shape the "input" of the conv layer had
    dA_prev = np.zeros((m, n_H_prev, n_W_prev, n_C_prev))
    # Initialize dW with the size for the weights that the layer had
    dW = np.zeros((f, f, n_C_prev, n_C))
    # Initialize db to correspond to the layers number of "filters"
    db = np.zeros((1,1, 1, n_C))
    
    # Pad A_prev and dA_prev as to have the correct "input" size
    A_prev_pad = zero_pad(A_prev, pad)
    dA_prev_pad = zero_pad(dA_prev, pad)
    
    # loop over the examples
    for i in range(m):
        # select ith example from A_prev_pad and dA_prev_pad (which is initialized to 0s)
        a_prev_pad = A_prev_pad[i]
        da_prev_pad = dA_prev_pad[i]
        for h in range(n_H):
            # find vertical start and end based on stride and filter size
            vert_start = h*stride
            vert_end = h*stride + f
            for w in range(n_W):               
                # find horizontal start and end based on stride and filter size
                horiz_start = w*stride
                horiz_end = w*stride + f
                for c in range(n_C):
                    # Select ith example's slice for all channels, as all "input" channels are used on the convolution           
                    a_slice = a_prev_pad[vert_start:vert_end, horiz_start:horiz_end, :]

                    # Update gradients, using the following functtions that I will explain for context and understanding:
                    
                    da_prev_pad[vert_start:vert_end, horiz_start:horiz_end, :] += W[:,:,:,c] * dZ[i, h, w, c]
                    # da_prev_pad represents the gradient of the cost function with respect to the padded input activations of the previous layer.
                    # During backprop, we need to update da_prev_pad to pass the gradients backward "through" the padding.
                    # W[:,:,:,c] represents the weights of the filter corresponding to the current neuron of the conv layer.
                    # dZ[i, h, w, c] represents the gradient of the cost function with respect to the output of the current convolutional neuron of the conv layer.
                    # We multiply them to calculate the gradient contribution of the current neuron's output to each element in da_prev_pad (taking to account all positions in the filter per each element).  
                    # We accumulate these contributions from all the elements in the corresponding window in the input feature map that was convolved to produce the current pixel in the output feature map to update the da_prev_pad.
                    
                    # This ensures that each input element of the current neuron contributes to the gradient of the 
                    # cost function with respect to the input activations of the previous layer, weighted by all the filter weights.
                    # And so, this operation results in a matrix (da_prev_pad) that represents how much each input feature contributed to the error in the output layer
                    
                    dW[:,:,:,c] += a_slice * dZ[i, h, w, c]
                    # dW represents the gradient of the cost function with respect to the weights of the filters (effectively measuring how much each weight in the filter contributed to the error in the output layer).
                    # During backprop, we need to update dW to adjust the filter weights based on their impact on the output and the corresponding gradients.
                    # a_slice represents the slice of the input activations of the previous layer that corresponds to the receptive field of the current neuron.
                    # Multiplying a_slice by dZ[i, h, w, c] calculates the gradient contribution of the current neuron's output to each weight in the filter, as we pic a specific w within dZ.
                    # We accumulate these contributions from all the input elements in the receptive field to update the dW for the current position w in the filter.
                    # And so, this operation results in a matrix (dW) that represents how much each weight in the filter contributed to the error in the output layer
                    
                    
                    db[:,:,:,c] += dZ[i, h, w, c]
                    # db represents the gradient of the cost function with respect to the biases of the filters.
                    # During backprop, we need to update db to adjust the biases based on their impact on the output and the corresponding gradients.
                    # dZ[i, h, w, c] represents the gradient of the cost function with respect to the output of the current convolutional neuron.
                    # We accumulate the gradients of the output with respect to the biases for all the input elements in the receptive field to update the biases of the current filter.
                    # And so, this operation results in a matrix (db) that represents the total contribution of each bias to the error in the output layer.
                    
        # Set the ith example's dA_prev to the unpadded da_prev_pad, as that is the true "input" of the conv layer
        dA_prev[i, :, :, :] = da_prev_pad[pad:-pad, pad:-pad, :]


    
    # Making sure your output shape is correct
    assert(dA_prev.shape == (m, n_H_prev, n_W_prev, n_C_prev))
    
    return dA_prev, dW, db

def relu_backward(dA, Z):
    """
    Gradient of the ReLU function

    Arguments:
    dA -- post-activation gradient for current layer l
    Z -- output of the forward propagation of the same layer l

    Returns:
    dZ -- Gradient of the cost with respect to Z
    """
    dZ = np.array(dA, copy=True)    # initialize dZ to be a copy of dA
    dZ[Z <= 0] = 0  # Applying the derivative of the ReLu activation function.

    return dZ

# BACKPROP POOLING FUNCTION

# Creating helper functions for backpropagation of pooling layers
def create_mask_from_window(x):
    """
    Creates a mask from an input matrix x, to identify the max entry of x 
    by setting a True value at that (row, col) location, and False everywhere else.
    The max value is important because this is the input value that ultimately influenced the output, and therefore the cost.
    
    Arguments:
    x -- Array of shape (f, f)
    
    Returns:
    mask -- Array of the same shape as window, contains a True at the position corresponding to the max entry of x.
    """    
    mask = (x == np.max(x))
    
    return mask

def distribute_value(dz, shape):
    """
    Distributes the input value in the matrix accounting for the fact that in average pooling, 
    every element of the input window has equal influence on the output
    
    Arguments:
    dz -- input scalar
    shape -- the shape (n_H, n_W) of the output matrix for which we want to distribute the value of dz
    
    Returns:
    a -- Array of size (n_H, n_W) for which we distributed the value of dz
    """    
    
    # Retrieve dimensions from of the matrix
    (n_H, n_W) = shape
    # Compute the value to distribute throughout the matrix 
    average = dz* (1/(n_H*n_W))
    # Create a matrix where every entry is the "average" value
    a = np.full((n_H,n_W),average)
    
    return a

def pool_backward(dA, cache, mode = "max"):
    """
    Implements the backward pass of the pooling layer
    
    Arguments:
    dA -- gradient of cost with respect to the output of the pooling layer, same shape as A
    cache -- cache output from the forward pass of the pooling layer, contains the layer's input and hparameters 
    mode -- the pooling mode you would like to use, defined as a string ("max" or "average")
    
    Returns:
    dA_prev -- gradient of cost with respect to the input of the pooling layer, same shape as A_prev
    """

    (A_prev, hparameters) = cache
    f = hparameters["f"]
    stride = hparameters["stride"]
    
    # Retrieve dimensions from A_prev's shape (before the pooling) and dA's shape (after the pooling)
    m, n_H_prev, n_W_prev, n_C_prev = A_prev.shape
    m, n_H, n_W, n_C = dA.shape
    
    dA_prev = np.zeros((m, n_H_prev, n_W_prev, n_C_prev))   
    
    for i in range(m):
        # Select ith example's activation
        a_prev = A_prev[i] 
        for h in range(n_H):
            # find vertical start and end based on stride and filter size
            vert_start = h*stride
            vert_end = h*stride + f
            for w in range(n_W):              
                # find horizontal start and end based on stride and filter size
                horiz_start = w*stride
                horiz_end = w*stride + f
                for c in range(n_C): 
                    if mode == "max":
                        # Select ith example's slice for each channel, as we do not mix channels on pooling
                        # Where a_prev has size (n_H_prev, n_W_prev, n_C_prev)           
                        a_prev_slice = a_prev[vert_start:vert_end, horiz_start:horiz_end, c]
                        # Create the mask from a_prev_slice to identify the max
                        mask = create_mask_from_window(a_prev_slice)
                        # We se dA_prev to be dA_prev plus the mask multiplied by the gradient of the output of the pooling layer, so the mask selects the max.
                        # where dA is the gradient of cost with respect to the output of the pooling layer,
                        dA_prev[i, vert_start:vert_end, horiz_start:horiz_end, c] += mask * dA[i, h, w, c]
                        
                        
                        # Set dA_prev to be dA_prev + the mask multiplied to the corresponding input of dA
                    elif mode == "avg":
                        # Get the value da from dA for this poistion
                        da = dA[i,h,w,c]
                        # Define the shape of the filter to calclulate the average
                        shape = (f,f)
                        # We add the distributed value of da to each of the elements of the slice, as they all contributed to the da "cost" in the same distribution. 
                        dA_prev[i, vert_start: vert_end, horiz_start: horiz_end, c] += distribute_value(da, shape)
    
    return dA_prev