import numpy as np
from utils.prepare_dataset import preprocess_image_for_prediction


def predict_image_class(model):
    """
    Allows the user to input an image path and uses the TensorFlow model to predict the class of the image.

    Arguments:
    model -- a trained TensorFlow model ready for making predictions.
    """
    if input("Do you want to predict an image? (y/n): ").lower() == 'y':
        while True:
            image_path = input("Enter the path of the image (from the hand_per_label folder for example) to predict (or enter to exit): ")
            if not image_path:
                break
            image = preprocess_image_for_prediction(image_path)
            print("preprocessed image shape:", image.shape)
            # Expanding the dimensions to match the input shape of the model by making m = 1 as there is one example
            image = np.expand_dims(image, axis=0)
            print("preprocessed image shape ready for prediction:", image.shape)
            predictions = model.predict(image)
            predicted_class = np.argmax(predictions, axis=1)[0]
            print(f"The tensorflow convolutional neural network predicts the image shows number: {predicted_class}")