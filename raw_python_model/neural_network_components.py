
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
    
    for i in range(m):
        # Select ith training example's padded activation
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
                    # Select ith training example's padded activation slice
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
    r = np.maximum(0, z)
    cache = z

    return r, cache