
import numpy as np
import matplotlib.pyplot as plt

from models.tensorflow_functional_model.neural_network_functions import functional_convolutional_model
from utils.prepare_dataset import preprocess_image_for_prediction
from utils.train_model import train_model


def mode_identify_number(x_train, y_train, x_test, y_test):
    """
    Conducts a loop where it asks the user to identify numbers based on hand sign images,
    compares the user's guess and neural network's prediction to the actual label, and
    provides feedback on both guesses.
    This is based on the model from Mode 2, to provide quick training and good performance.

    Arguments:
    x_train -- images from the train dataset.
    y_train -- true labels corresponding to the images in the training set.
    x_test -- images from the test dataset.
    y_test -- true labels corresponding to the images in test set.
    """
    
    print("\n🚀 Launching the TensorFlow model you will compete against... ")
    conv_model = functional_convolutional_model((64, 64, 3))
    conv_model.compile(optimizer='adam',
                    loss='categorical_crossentropy',
                    metrics=['accuracy'])
    
    if input("Do you want to see a summary of the model that has just been compiled? (y/n): ").lower() == 'y':
        conv_model.summary()    

    print("\n🌟 Training the model...")
    train_model(conv_model, x_train, y_train, x_test, y_test, epochs=100, batch_size=64)
    
    x_test_list = list(x_test.as_numpy_iterator())
    y_test_list = list(y_test.as_numpy_iterator())
    
    while True:
        idx = np.random.randint(0, len(x_test_list))
        
        image, true_label = x_test_list[idx], y_test_list[idx]
        true_class = np.argmax(true_label) 

        plt.imshow(image)
        plt.title("What number is this hand sign?")
        plt.show()

        print(f"\n")

        user_guess = int(input("\033[1;34mEnter your guess (0-5): \033[0m"))        
    
        image = np.expand_dims(image, axis=0)
        print("preprocessed image shape ready for prediction:", image.shape)
        predictions = conv_model.predict(image)
        prediction = np.argmax(predictions, axis=1)[0]        
        
        print(f"Your guess: {user_guess}")
        print(f"Neural network prediction: {prediction}")
        print("----------------------------")
        print(f"Correct answer: {true_class}")
        print("----------------------------")
        
        user_correct = user_guess == true_class
        nn_correct = prediction == true_class
        print( "Did you guess correctly? ", "\033[1;32mYes\033[0m" if user_correct else "\033[1;31mNo\033[0m")
        print("Did the neural network guess correctly? ", "\033[1;32mYes\033[0m" if nn_correct else "\033[1;31mNo\033[0m")

        continue_choice = input("\nDo you want to try another image? (y/n): ")
        if continue_choice.lower() != 'y':
            break


def mode_perform_sign(x_train, y_train, x_test, y_test):
    """
    Asks the user to perform hand signs for randomly selected numbers, predicts using the neural network,
    and provides feedback on whether the performed sign matches the expected sign.

    Arguments:
    x_train -- images from the train dataset.
    y_train -- true labels corresponding to the images in the training set.
    x_test -- images from the test dataset.
    y_test -- true labels corresponding to the images in test set.

    """ 
    print("\n🚀 Launching the TensorFlow model you will help you learn how to perform hand signst... ")
    conv_model = functional_convolutional_model((64, 64, 3))
    conv_model.compile(optimizer='adam',
                    loss='categorical_crossentropy',
                    metrics=['accuracy'])
    
    if input("Do you want to see a summary of the model that has just been compiled? (y/n): ").lower() == 'y':
        conv_model.summary()    

    print("\n🌟 Training the model...")
    train_model(conv_model, x_train, y_train, x_test, y_test, epochs=100, batch_size=64)
    
    while True:
        target_number = np.random.randint(0, 6)
        print(f"\033[1;34mPlease perform the hand sign for the number \033[1;33m{target_number}\033[0m\033[1;34m and take a photo (or use those in folder hand_per_label to ensure distribution consistency).\033[0m")

        image_path = input("Enter the path to your image: ")
        image = preprocess_image_for_prediction(image_path)
        image = np.expand_dims(image, axis=0)  # Add batch dimension

        predictions = conv_model.predict(image)
        prediction = np.argmax(predictions, axis=1)[0]   

        print(f"You tried to do the hand sign for number: {target_number}")
        if prediction == target_number:
            print("\033[1;32mCongratulations! Your hand sign matches the number.\033[0m")
        else:
            print(f"\033[1;31mOops! Your hand sign did not match according to the ConvNet. You showed {prediction}.\033[0m")

        continue_choice = input("Do you want to try another number? (y/n): ")
        print(f"\n")
        if continue_choice.lower() != 'y':
            break
