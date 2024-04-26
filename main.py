from raw_python_model.tests import run_raw_python_model_tests
from utils.prepare_dataset import prepare_dataset


def main():
    # x_train, y_train, x_test, y_test, input_features, pure_test_images, pure_test_labels = prepare_dataset()
    
    # If you needed to extract images from the test set to use on mode 2
    # extract_and_save_images(pure_test_images, pure_test_labels)
    
    print("Do you want to run a test suite for the main convolution functions in the raw Python model? (y/n)")
    user_input = input().lower()
    if user_input == "y":
        print("Keep an eye on the shapes of the inputs and outputs of each function to understand how they work together.")
        run_raw_python_model_tests()

    print("Thanks for counting on me for learning about hand signs with convolutional neural networks! ;)")
    
if __name__ == "__main__":
    main()
