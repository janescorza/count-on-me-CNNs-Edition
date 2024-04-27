from raw_python_model.tests import run_raw_python_model_tests
from utils.prepare_dataset import prepare_dataset
import numpy as np
from raw_python_model.neural_network_components import conv_forward, pool_forward, relu
from raw_python_model.neural_network_functions import compute_cost, initialize_parameters, initialize_hyperparameters, forward_propagation, backward_propagation, update_parameters


def main():
    x_train, y_train, x_test, y_test, input_features, pure_test_images, pure_test_labels, x_train_numpy, y_train_numpy = prepare_dataset()
    
    print("Shape of x_train:", x_train_numpy.shape)
    print("Shape of y_train:", y_train_numpy.shape)    
    
    # If you needed to extract images from the test set to use on mode 2
    # extract_and_save_images(pure_test_images, pure_test_labels)
    
    print("Do you want to run a test suite for the main convolution functions in the raw Python model? (y/n)")
    user_input = input().lower()
    if user_input == "y":
        print("Keep an eye on the shapes of the inputs and outputs of each function to understand how they work together.")
        run_raw_python_model_tests()
        
        
    
    # Load your dataset
    # Assuming you have your dataset loaded into x_train_numpy and y_train_numpy

    # Initialize parameters and hyperparameters
    parameters = initialize_parameters()
    hparameters = initialize_hyperparameters()

    # Training loop
    learning_rate = 0.00015
    num_epochs = 100
    for epoch in range(num_epochs):
        # Forward propagation
        Z3, cache = forward_propagation(x_train_numpy, parameters, hparameters)
        
        # Compute cost
        # Assuming you have a function to compute the cost, let's call it compute_cost()
        cost = compute_cost(Z3, y_train_numpy)
        
        # Backward propagation
        gradients = backward_propagation(y_train_numpy, cache)
        
        # Update parameters
        parameters = update_parameters(parameters, gradients, learning_rate)
        
        # Print cost every few epochs
        if epoch % 10 == 0:
            print(f"Cost after epoch {epoch}: {cost}")

    # At this point, your model is trained and the parameters are updated.
    # You can use this trained model for prediction or further evaluation.


    print("Thanks for counting on me for learning about hand signs with convolutional neural networks! ;)")
    
if __name__ == "__main__":
    main()
