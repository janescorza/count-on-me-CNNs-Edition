from tensorflow.keras.layers import BatchNormalization, Add, Activation, Conv2D
from tensorflow.keras.initializers import random_uniform, glorot_uniform

def identity_block(X, f, filters, initializer=random_uniform, training=False):
    """
    Implementation of the identity block as defined in ResNet architectures.
    
    The identity block is a standard block used in ResNets when the input activation 
    (say X) has the same dimension as the output activation. This block is "identity" 
    because it skips over some layers and adds the output from an earlier layer to a later layer,
    which helps in training deeper networks without degrading the network accuracy.
    
    Arguments:
    X -- input tensor of shape (m, n_H_prev, n_W_prev, n_C_prev)
    f -- integer, specifying the shape of the middle CONV's window for the main path
    filters -- python list of integers, defining the number of filters in the CONV layers of the main path
    initializer -- to set up the initial weights of a layer. Equals to random uniform initializer
    training -- boolean, specifying whether the call is for training or inference

    
    Returns:
    X -- output of the identity block, tensor of shape (m, n_H, n_W, n_C)
    """
    
    # Retrieve the number of filters for each convolution
    F1, F2, F3 = filters
    
    # Store the input value (identity) to add it to the main path after convolution operations
    X_shortcut = X
    
    # First component of the main path: a 1x1 convolution layer
    # It shrinks the spatial dimensions without changing depth
    X = Conv2D(filters=F1, kernel_size=1, strides=(1,1), padding='valid', kernel_initializer=initializer(seed=0))(X)
    
    # Batch Normalization normalizes the activations from the previous layer
    # Axis=3 normalizes the features axis for 4D tensors
    X = BatchNormalization(axis=3)(X, training=training)
    
    # Apply ReLU activation to introduce non-linearity to the transformation
    X = Activation('relu')(X)
    
    # Second component of the main path: a spatial convolution layer
    # The filter size 'f' is typically 3x3 and 'same' padding keeps spatial dimensions constant
    X = Conv2D(filters=F2, kernel_size=f, strides=(1,1), padding='same', kernel_initializer=initializer(seed=0))(X)
    
    # Another batch normalization step
    X = BatchNormalization(axis=3)(X, training=training)
    
    # ReLU activation for the second main path component
    X = Activation('relu')(X)

    # Third component of the main path: another 1x1 convolution layer
    # This layer increases the depth to match the shortcut path's depth if needed
    X = Conv2D(filters=F3, kernel_size=1, strides=(1,1), padding='valid', kernel_initializer=initializer(seed=0))(X)
    
    # Final batch normalization for the main path before merging with the shortcut path
    X = BatchNormalization(axis=3)(X, training=training)
    
    # Final step of the identity block: Add the shortcut to the main path
    # The shortcut path is element-wise added back to the main path after the convolutions
    X = Add()([X, X_shortcut])
    
    # Apply ReLU activation to the combined output of the Convolutions and the shortcut to finalize the block
    X = Activation('relu')(X)

    return X


def convolutional_block(X, f, filters, s=2, initializer=glorot_uniform, training=False):
    """
    Implements a convolutional block in a ResNet model, which includes a series of convolutional layers
    with batch normalization and ReLU activations. The block includes a skip connection that helps in
    mitigating the vanishing gradient problem in deep networks.

    Arguments:
    X -- input tensor of shape (m, n_H_prev, n_W_prev, n_C_prev), where m is the number of examples
    f -- integer, specifying the kernel size of the middle convolutional layer's window for the main path
    filters -- list of integers, defining the number of filters in the CONV layers of the main path
    s -- integer, specifying the stride to be used in the first convolutional layer in the main path
    initializer -- initializer function for the kernel weights of the layers (default: Glorot uniform)
    training -- boolean, indicating whether the model is being trained or used for inference

    Returns:
    X -- the output tensor of the block, with shape (m, n_H, n_W, n_C)
    """
    
    # Retrieve the number of filters for each convolution layer in the block
    F1, F2, F3 = filters
    
    # Save the input value to use later in a skip connection, helping to maintain the strength of the gradient flow
    X_shortcut = X
    
    # First component of the main path: convolutional layer with F1 filters, a 1x1 kernel size, and stride 's'
    # Reduces the spatial dimensions without changing the depth if stride > 1
    X = Conv2D(filters=F1, kernel_size=1, strides=(s, s), padding='valid', kernel_initializer=initializer(seed=0))(X)
    # Batch normalization to normalize the activations from the convolutional layer
    X = BatchNormalization(axis=3)(X, training=training)
    # ReLU activation function to introduce non-linearity and aid in mitigating the vanishing gradient problem
    X = Activation('relu')(X)

    # Second component of the main path: convolutional layer with F2 filters, a 'f'x'f' kernel size, and stride 1
    # Maintains spatial dimensions while transforming depth to capture more complex features
    X = Conv2D(filters=F2, kernel_size=f, strides=(1, 1), padding='same', kernel_initializer=initializer(seed=0))(X)
    # Batch normalization to normalize the activations from the convolutional layer
    X = BatchNormalization(axis=3)(X, training=training)
    # ReLU activation function for further non-linearity and depth of feature activation
    X = Activation('relu')(X)

    # Third component of the main path: convolutional layer with F3 filters, a 1x1 kernel size, and stride 1
    # Adjusts the depth, preparing the tensor for addition with the shortcut path
    X = Conv2D(filters=F3, kernel_size=1, strides=(1, 1), padding='valid', kernel_initializer=initializer(seed=0))(X)
    # Batch normalization to normalize the activations from the convolutional layer
    X = BatchNormalization(axis=3)(X, training=training)

    # Shortcut path: convolutional layer with F3 filters, a 1x1 kernel size, and stride 's'
    # Directly modifies the shortcut path to match the main path's dimensions, if needed
    X_shortcut = Conv2D(filters=F3, kernel_size=1, strides=(s, s), padding='valid', kernel_initializer=initializer(seed=0))(X_shortcut)
    # Batch normalization to normalize the activations from the convolutional layer
    X_shortcut = BatchNormalization(axis=3)(X_shortcut, training=training)
    
    # Final step: add the shortcut value to the main path and pass it through a ReLU activation
    # This step reintroduces the original input into the output, adding the outputs of the layers bypassed by the shortcut
    X = Add()([X, X_shortcut])
    X = Activation('relu')(X)
    
    return X

