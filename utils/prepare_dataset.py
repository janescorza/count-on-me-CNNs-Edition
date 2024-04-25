import os
import h5py
import tensorflow as tf
import matplotlib.pyplot as plt
import cv2


def plot_sign_examples(x_train, y_train):
    """
    Plots a 5x5 grid of sample images from the dataset.
    
    Arguments:
    x_train -- tf.data.Dataset, training set of sign images.
    y_train -- tf.data.Dataset, training set of sign labels.
    """
    images_iter = iter(x_train)
    labels_iter = iter(y_train)
    plt.figure(figsize=(10, 10))
    for i in range(25):
        ax = plt.subplot(5, 5, i + 1)
        plt.imshow(next(images_iter).numpy().astype("uint8"))
        plt.title(next(labels_iter).numpy().astype("uint8"))
        plt.axis("off")
    plt.show()

def load_dataset():
    """
    Loads sign language datasets from HDF5 files for training and testing purposes.
    
    Returns:
    Tuple of tf.data.Datasets -- Containing training and testing datasets for both images (x) and labels (y).
    """
    train_dataset = h5py.File('data/training_set/train_signs.h5', "r")
    test_dataset = h5py.File('data/test_set/test_signs.h5', "r")
    
    # assuming the datasets have a consistent set of classes
    classes = train_dataset['list_classes'][:]
    
    x_train = tf.data.Dataset.from_tensor_slices(train_dataset['train_set_x'])
    y_train = tf.data.Dataset.from_tensor_slices(train_dataset['train_set_y'])

    x_test = tf.data.Dataset.from_tensor_slices(test_dataset['test_set_x'])
    y_test = tf.data.Dataset.from_tensor_slices(test_dataset['test_set_y'])
    
    return x_train, y_train, x_test, y_test, classes

def normalize(image):
    """
    Transforms an image into a normalized tensor of shape (64 * 64 * 3, ).
    
    Arguments:
    image -- tf.Tensor, image tensor.
    
    Returns:
    result -- tf.Tensor, transformed and normalized tensor.
    """
    image = tf.cast(image, tf.float32) / 255.0
    image = tf.reshape(image, [-1,])
    return image
    
def one_hot_matrix(label, num_classes=6):
    """
    Computes the one hot encoding for a single label.
    
    Arguments:
    label -- int, categorical labels.
    num_classes -- int, number of different classes that label can take.
    
    Returns:
    one_hot -- tf.Tensor, a single-column matrix with the one hot encoding.
    """

    one_hot = tf.reshape(tf.one_hot(label, num_classes, axis=0), shape=[num_classes,])
    return one_hot

def prepare_dataset():
    """
    Prepares and normalizes the dataset for use, optionally displaying sample images.
    
    Returns:
    tuple -- normalized training and testing data and labels, and additional dataset details.
    """
    x_train, y_train, x_test, y_test, classes = load_dataset()
    
    print("Element spec of training dataset:", x_train.element_spec)
    print(f"The dataset contains {classes.size} classes which are the following: {classes}")
    
    show_sample = input("Would you like to see an example of the pictures in the dataset? (y/n)")
    if show_sample.lower() == 'y':
        plot_sign_examples(x_train, y_train)
    
    pure_test_images = x_test
    pure_test_labels = y_test
        
    normalized_train = x_train.map(normalize)
    normalized_test = x_test.map(normalize)
        
    print("Element spec of normalized training dataset:", normalized_train.element_spec)
    
    one_hot_train = y_train.map(lambda label: one_hot_matrix(label, num_classes=classes.size))    
    one_hot_y_test = y_test.map(lambda label: one_hot_matrix(label, num_classes=classes.size))
    
    print("Sample element from one hot element from training dataset labels:", next(iter(one_hot_train)))

    input_sample = next(iter(normalized_train))
    input_features = input_sample.shape[0]
    
    return  normalized_train, one_hot_train, normalized_test, one_hot_y_test, input_features, pure_test_images, pure_test_labels

def preprocess_image_for_prediction(image_path):
    """Reads an image from a file and resizes it."""
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = cv2.resize(image, (64, 64))
        
    
    plt.imshow(image)
    plt.title("This is the image you uploaded:")
    plt.show()
        
    return image

# Function used to extract the images in the hand per label folder
def extract_and_save_images(x_test, y_test, output_folder='data/hand_per_label'):
    """
    Extracts one image for each unique label from the test dataset and saves them into a specified folder.
    
    Arguments:
    x_test -- tf.data.Dataset, test set of sign images.
    y_test -- tf.data.Dataset, test set of sign labels.
    output_folder -- str, folder where the images will be saved.
    
    """
    # Create the directory if it does not exist
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    
    # Initialize a dictionary to track the extraction of images for labels 0 to 5
    found_labels = {}
    images_iter = iter(x_test)
    labels_iter = iter(y_test)
    
    # Loop through the dataset and save the first image found for each label
    while len(found_labels) < 6:
        image = next(images_iter)
        label = int(next(labels_iter).numpy())
        
        if label not in found_labels:
            found_labels[label] = True
            # Save the image using OpenCV
            image_path = os.path.join(output_folder, f'label_{label}.png')
            cv2.imwrite(image_path, cv2.cvtColor(image.numpy().astype('uint8'), cv2.COLOR_RGB2BGR))
            print(f"Saved label {label} image to {image_path}")
