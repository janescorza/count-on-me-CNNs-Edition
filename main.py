import time

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt


from models.raw_python_model.tests import run_raw_python_model_tests
from models.raw_python_model.neural_network_functions import compute_cost, evaluate_model, initialize_parameters, initialize_hyperparameters, forward_propagation, backward_propagation, update_parameters
from models.residual_network_model.neural_network_functions import ResNet50
from models.tensorflow_functional_model.neural_network_functions import functional_convolutional_model, train_model
from utils.prepare_dataset import prepare_dataset
from utils.image_prediction import predict_image_class

def run_custom_cnn(x_train_numpy, y_train_numpy, x_test_numpy, y_test_numpy):
    """
    Execute the custom-built convolutional neural network (CNN) from scratch.
    """
    print("\n🚀 Launching the custom-built CNN from scratch...")
    parameters, hparameters = initialize_parameters(), initialize_hyperparameters()
    
    print("Shape of x_train:", x_train_numpy.shape)
    print("Shape of y_train:", y_train_numpy.shape)    
    
    if input("Do you want to run a test suite for the main convolution functions in the raw Python model? (y/n): ").lower() == 'y':
        print("Keep an eye on the shapes of the inputs and outputs of each function to understand how they work together.")
        run_raw_python_model_tests()
        
    print_shapes = input("Do you want to print the shapes of the arrays during the run? (y/n): ").lower() == 'y'

    print("\n🌟 Training the model...")
    learning_rate = 0.00015
    num_epochs = 100
    start_training = time.time()
    for epoch in range(num_epochs):
        # Forward propagation
        Z3, cache = forward_propagation(x_train_numpy, parameters, hparameters, print_shapes)
        
        # Compute cost
        cost = compute_cost(Z3, y_train_numpy, print_shapes)
        
        # Backward propagation
        gradients = backward_propagation(y_train_numpy, cache, print_shapes)
        
        # Update parameters
        parameters = update_parameters(parameters, gradients, learning_rate)
        
        # Print cost every epoch
        print(f"Cost after epoch {epoch}: {cost}")
        
    end_training = time.time() - start_training
    print(f"Training completed in {end_training:.2f} seconds")

    print("\n🔍 Evaluating the model...")
    accuracy = evaluate_model(x_test_numpy, y_test_numpy, parameters, hparameters)
    print(f"Model accuracy on test set: {accuracy:.5%}")

def run_tensorflow_model(x_train, y_train, x_test, y_test):
    """
    Execute a TensorFlow-based model.
    """
    print("\n🚀 Launching the TensorFlow model... ")
    conv_model = functional_convolutional_model((64, 64, 3))
    conv_model.compile(optimizer='adam',
                    loss='categorical_crossentropy',
                    metrics=['accuracy'])
    
    if input("Do you want to see a summary of the model that has just been compiled? (y/n): ").lower() == 'y':
        conv_model.summary()    

    print("\n🌟 Training the model...")
    train_model(conv_model, x_train, y_train, x_test, y_test, epochs=100, batch_size=64)
    
    print("\n🔎 Model predictions...")
    predict_image_class(conv_model)


def run_resnet50_model(x_train, y_train, x_test, y_test):
    """
    Execute a ResNet50 model (which is also TensorFlow based).
    """
    print("\n🚀 Launching the ResNet50 model...")
    model = ResNet50(input_shape = (64, 64, 3), classes = 6)
    opt = tf.keras.optimizers.Adam(learning_rate=0.00015)
    model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])
    
    if input("Do you want to see a summary of the model? (y/n): ").lower() == 'y':
        print(model.summary())
    
    print("\n🌟 Training the model...")
    # Why 33 epochs you may ask? It has good performance, but more importantly it's a Spanish Formula 1 Meme
    train_model(model, x_train, y_train, x_test, y_test, epochs=33, batch_size=32)
        
    print("\n🔎 Model predictions...")
    predict_image_class(model)


def main():
    """
    Main function to control the flow of neural network model execution based on user input.
    """
    print("Welcome to the Neural Network Playground!")
    while True:
        print("1: Custom CNN from Scratch")
        print("2: TensorFlow Model")
        print("3: ResNet50 Model")
        print("q: to quit")
        choice = input("Select the model to run [1-3] or q to quit: ")
        
        if choice == 'q':
            break
        
        # Prepare dataset
        print("\n🔄 Preparing datasets...")
        x_train, y_train, x_test, y_test, input_features, pure_test_images, pure_test_labels, x_train_numpy, y_train_numpy, x_test_numpy, y_test_numpy = prepare_dataset()
        
        # If you needed to extract images from the test set to use on mode 2 uncomment the following line
        # extract_and_save_images(pure_test_images, pure_test_labels)
        
        if choice == '1':
            run_custom_cnn(x_train_numpy, y_train_numpy, x_test_numpy, y_test_numpy)
        elif choice == '2':
            run_tensorflow_model(x_train, y_train, x_test, y_test)
        elif choice == '3':
            run_resnet50_model(x_train, y_train, x_test, y_test)
        else:
            print("\n❗ Invalid choice. Please select a valid option.")

    print("Thanks for counting on me for learning about hand signs with convolutional neural networks! ;)")
    
if __name__ == "__main__":
    main()
