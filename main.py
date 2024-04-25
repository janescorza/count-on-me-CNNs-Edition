import numpy as np
import matplotlib.pyplot as plt
from raw_python_model.neural_network_components import conv_forward, relu, zero_pad
from utils.prepare_dataset import prepare_dataset

def main():
    # x_train, y_train, x_test, y_test, input_features, pure_test_images, pure_test_labels = prepare_dataset()
    
    # If you needed to extract images from the test set to use on mode 2
    # extract_and_save_images(pure_test_images, pure_test_labels)
    
    # Introduce random sizes initalization
    np.random.seed(1)
    # (m, n_H_prev, n_W_prev, n_C_prev)
    A_prev = np.random.randn(2, 7, 7, 4)
    print("🚀 ~ A_prev.shape:", A_prev.shape)
    # (f, f, n_C_prev, n_C) 
    W = np.random.randn(3, 3, 4, 8)
    b = np.random.randn(1, 1, 1, 8)
    hparameters = {"pad" : 1,
                "stride": 2}
    
    Z, cache_conv = conv_forward(A_prev, W, b, hparameters)
    print("🚀 ~ Z.shape:", Z.shape)
    # Apply activation
    A, activation_cache = relu(Z)
    print("🚀 ~ A.shape:", A.shape)
    

    print("Thanks for counting on me for learning about hand signs with convolutional neural networks! ;)")
    
if __name__ == "__main__":
    main()
