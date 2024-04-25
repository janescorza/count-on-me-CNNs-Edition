from utils.prepare_dataset import prepare_dataset

def main():
    x_train, y_train, x_test, y_test, input_features, pure_test_images, pure_test_labels = prepare_dataset()
    
    # If you needed to extract images from the test set to use on mode 2
    # extract_and_save_images(pure_test_images, pure_test_labels)

    print("Thanks for counting on me for learning about hand signs with convolutional neural networks! ;)")
    
if __name__ == "__main__":
    main()
