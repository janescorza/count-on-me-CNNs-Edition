import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import BatchNormalization, Input, Dense, Activation, ZeroPadding2D, Flatten, Conv2D, AveragePooling2D, MaxPooling2D
from tensorflow.keras.initializers import glorot_uniform
from tensorflow.keras.models import Model


from models.residual_network_model.neural_network_components import convolutional_block, identity_block

def train_model(model, x_train, y_train, x_test, y_test, epochs, batch_size):
    """
    Train the model with the train set and check accuracy to finally plot the training and validation metrics.
    """

    train_dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train)).batch(batch_size)
    test_dataset = tf.data.Dataset.from_tensor_slices((x_test, y_test)).batch(batch_size)

    history = model.fit(train_dataset, epochs=epochs, validation_data=test_dataset)

    # Plot accuracy
    plt.figure(figsize=(12, 8))
    plt.plot(history.history['accuracy'], label='Training Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.title('Model Accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(loc='lower right')
    plt.show()

    # Plot loss
    plt.figure(figsize=(12, 8))
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Model Loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(loc='upper right')
    plt.show()

def ResNet50(input_shape = (64, 64, 3), classes = 6, training=False):
    """
    Constructs a ResNet50 model with detailed stage-wise implementations, following the standard architecture:
    CONV2D -> BATCHNORM -> RELU -> MAXPOOL -> CONVBLOCK -> IDBLOCK*2 -> CONVBLOCK -> IDBLOCK*3
    -> CONVBLOCK -> IDBLOCK*5 -> CONVBLOCK -> IDBLOCK*2 -> AVGPOOL -> FLATTEN -> DENSE
    Based on the original RreNet50 architecture: https://arxiv.org/abs/1512.03385

    This architecture leverages residual blocks to enable training of very deep neural networks without degradation 
    in performance due to vanishing gradients, as each block includes a skip connection that bypasses non-linear transformations.

    Arguments:
    input_shape -- shape of the images of the dataset, tuple of integers (height, width, channels)
    classes -- integer, number of classes to predict, corresponds to the dimensionality of the output layer
    training -- boolean, specifies whether the model is being trained; affects batch normalization behavior

    Returns:
    model -- a Keras Model instance representing the ResNet50 architecture
    """
    
   # Define the input tensor with the specified shape
    X_input = Input(input_shape)

    # Apply zero-padding for convolution boundary conditions
    X = ZeroPadding2D((3, 3))(X_input)

    # Stage 1: Initial convolution and max-pooling layer
    X = Conv2D(64, (7, 7), strides=(2, 2), kernel_initializer=glorot_uniform(seed=0))(X)
    X = BatchNormalization(axis=3)(X, training=training)
    X = Activation('relu')(X)
    X = MaxPooling2D((3, 3), strides=(2, 2))(X)

    # Stage 2: First set of convolutional and identity blocks
    X = convolutional_block(X, f=3, filters=[64, 64, 256], s=1, training=training)
    X = identity_block(X, 3, [64, 64, 256], training=training)
    X = identity_block(X, 3, [64, 64, 256], training=training)

    # Stage 3: Second set, increasing depth and reducing spatial dimensions
    X = convolutional_block(X, f=3, filters=[128, 128, 512], s=2, training=training)
    for _ in range(3):
        X = identity_block(X, 3, [128, 128, 512], training=training)

    # Stage 4: Third set with further increased depth
    X = convolutional_block(X, f=3, filters=[256, 256, 1024], s=2, training=training)
    for _ in range(5):
        X = identity_block(X, 3, [256, 256, 1024], training=training)

    # Stage 5: Final set before pooling
    X = convolutional_block(X, f=3, filters=[512, 512, 2048], s=2, training=training)
    X = identity_block(X, 3, [512, 512, 2048], training=training)
    X = identity_block(X, 3, [512, 512, 2048], training=training)

    # Average Pooling: Reduces each feature map to a single value, preparing for the dense output layer
    X = AveragePooling2D(pool_size=(2, 2))(X)

    # Output layer: Flatten the feature maps and connect to a dense layer for classification
    X = Flatten()(X)
    X = Dense(classes, activation='softmax', kernel_initializer=glorot_uniform(seed=0))(X)

    # Create and return the model
    model = Model(inputs=X_input, outputs=X, name='ResNet50')
    
    return model