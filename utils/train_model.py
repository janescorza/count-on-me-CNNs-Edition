import matplotlib.pyplot as plt
import tensorflow as tf


def train_model(conv_model, x_train, y_train, x_test, y_test, epochs=100, batch_size=64):
    """
    Trains a TensorFlow model using provided training data and evaluates its performance on a test set.
    After training, the function also visualizes the model's accuracy and loss progression over epochs if requested by the user.

    Arguments:
    conv_model -- TensorFlow model, a compiled instance ready for training.
    x_train -- tf.data.Dataset, the dataset containing the input features for training.
    y_train -- tf.data.Dataset, the dataset containing the labels corresponding to x_train.
    x_test -- tf.data.Dataset, the dataset containing the input features for testing.
    y_test -- tf.data.Dataset, the dataset containing the labels corresponding to x_test.
    epochs -- int, number of epochs to train the model. Default is 100.
    batch_size -- int, number of samples per batch to load. Default is 64.

    """
    train_dataset = tf.data.Dataset.zip(( x_train, y_train)).batch(batch_size)
    test_dataset = tf.data.Dataset.zip((x_test, y_test)).batch(batch_size)
    history = conv_model.fit(train_dataset, epochs=epochs, validation_data=test_dataset)
    
    if input("Do you want to see a graph of the model accuracy over the training and validation sets? (y/n): ").lower() == 'y':
        # Plot training & validation accuracy values
        plt.figure(figsize=(12, 8))
        plt.plot(history.history['accuracy'], label='Training Accuracy')
        plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
        plt.title('Model Accuracy')
        plt.ylabel('Accuracy')
        plt.xlabel('Epoch')
        plt.legend(loc='lower right')
        plt.show()
        
        # Plot training & validation loss values
        plt.figure(figsize=(12, 8))
        plt.plot(history.history['loss'], label='Training Loss')
        plt.plot(history.history['val_loss'], label='Validation Loss')
        plt.title('Model Loss')
        plt.ylabel('Loss')
        plt.xlabel('Epoch')
        plt.legend(loc='upper right')
        plt.show()