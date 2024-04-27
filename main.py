import time
from raw_python_model.tests import run_raw_python_model_tests
from utils.prepare_dataset import prepare_dataset
from raw_python_model.neural_network_functions import compute_cost, evaluate_model, initialize_parameters, initialize_hyperparameters, forward_propagation, backward_propagation, update_parameters

def run_custom_cnn(x_train_numpy, y_train_numpy, x_test_numpy, y_test_numpy):
    """
    Execute the custom-built convolutional neural network (CNN) from scratch.
    """
    print("\n🚀 Launching the custom-built CNN from scratch...")
    parameters, hparameters = initialize_parameters(), initialize_hyperparameters()
    
    print("Shape of x_train:", x_train_numpy.shape)
    print("Shape of y_train:", y_train_numpy.shape)    
    
    print("Do you want to run a test suite for the main convolution functions in the raw Python model? (y/n)")
    user_input = input().lower()
    if user_input == "y":
        print("Keep an eye on the shapes of the inputs and outputs of each function to understand how they work together.")
        run_raw_python_model_tests()
        
    print("Do you want to print the shapes of the arrays during the run? (y/n)")
    user_input = input().lower()
    if user_input == "y":
        print_shapes = True
    else:
        print_shapes = False
    

    print("\n🌟 Training the model...")
    learning_rate = 0.00015
    num_epochs = 100
    start_training = time.time()
    for epoch in range(num_epochs):
        # Forward propagation
        Z3, cache = forward_propagation(x_train_numpy, parameters, hparameters, print_shapes)
        
        # Compute cost
        # Assuming you have a function to compute the cost, let's call it compute_cost()
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

def run_tensorflow_model():
    """
    Execute a TensorFlow-based model.
    """
    print("\n🚀 Launching the TensorFlow model... (To be implemented)")

def run_resnet50_model():
    """
    Execute a ResNet50 model.
    """
    print("\n🚀 Launching the ResNet50 model... (To be implemented)")


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
        
        # If you needed to extract images from the test set to use on mode 2
        # extract_and_save_images(pure_test_images, pure_test_labels)
        
        if choice == '1':
            run_custom_cnn(x_train_numpy, y_train_numpy, x_test_numpy, y_test_numpy)
        elif choice == '2':
            run_tensorflow_model()
        elif choice == '3':
            run_resnet50_model()
        else:
            print("\n❗ Invalid choice. Please select a valid option.")
        
        
        # print("Do you want to run a test suite for the main convolution functions in the raw Python model? (y/n)")
        # user_input = input().lower()
        # if user_input == "y":
        #     print("Keep an eye on the shapes of the inputs and outputs of each function to understand how they work together.")
        #     run_raw_python_model_tests()
            
        # print("Do you want to print the shapes of the arrays during the run? (y/n)")
        # user_input = input().lower()
        # if user_input == "y":
        #     print_shapes = True
        # else:
        #     print_shapes = False

        # # Initialize parameters and hyperparameters
        # parameters = initialize_parameters()
        # hparameters = initialize_hyperparameters()

        # # Training loop
        # learning_rate = 0.00015
        # num_epochs = 15
        # start_training = time.time()
        # for epoch in range(num_epochs):
        #     # Forward propagation
        #     Z3, cache = forward_propagation(x_train_numpy, parameters, hparameters, print_shapes)
            
        #     # Compute cost
        #     # Assuming you have a function to compute the cost, let's call it compute_cost()
        #     cost = compute_cost(Z3, y_train_numpy, print_shapes)
            
        #     # Backward propagation
        #     gradients = backward_propagation(y_train_numpy, cache, print_shapes)
            
        #     # Update parameters
        #     parameters = update_parameters(parameters, gradients, learning_rate)
            
        #     # Print cost every epoch
        #     print(f"Cost after epoch {epoch}: {cost}")
            
        # end_training = time.time() - start_training
        # print(f"Training completed in {end_training:.2f} seconds")
        
        # # Evaluating model
        # accuracy = evaluate_model(x_test_numpy, y_test_numpy, parameters, hparameters)
        # print(f"Model accuracy on test set: {accuracy:.5%}")



    print("Thanks for counting on me for learning about hand signs with convolutional neural networks! ;)")
    
if __name__ == "__main__":
    main()
