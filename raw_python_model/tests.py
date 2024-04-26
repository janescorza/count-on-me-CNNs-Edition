
import numpy as np

from raw_python_model.neural_network_components import conv_backward, conv_forward, create_mask_from_window, distribute_value, pool_backward, pool_forward, relu

# These tests are a simplified and lightly modified version of the tests from an assignment I completed as part of the Convolutional Neural Networks course
# from the Deep Learning Specialization by Andrew Ng. Participating in assignments like these has been invaluable
# for challenging myself and gaining a deeper understanding of neural networks.

def conv_forward_test_1(z_mean, z_0_2_1, cache_0_1_2_3):
    test_count = 0
    z_mean_expected = 0.5511276474566768
    z_0_2_1_expected = [-2.17796037,  8.07171329, -0.5772704,   3.36286738,  4.48113645, -2.89198428, 10.99288867,  3.03171932]
    cache_0_1_2_3_expected = [-1.1191154,   1.9560789,  -0.3264995,  -1.34267579]
    
    if np.isclose(z_mean, z_mean_expected):
        test_count = test_count + 1
    else:
        print("\033[91mFirst Test: Z's mean is incorrect. Expected:", z_mean_expected, "\nYour output:", z_mean, ". Make sure you include stride in your calculation\033[90m\n")
        
    if np.allclose(z_0_2_1, z_0_2_1_expected):
        test_count = test_count + 1
    else:
        print("\033[91mFirst Test: Z[0,2,1] is incorrect. Expected:", z_0_2_1_expected, "\nYour output:", z_0_2_1, "Make sure you include stride in your calculation\033[90m\n")
        
    if np.allclose(cache_0_1_2_3, cache_0_1_2_3_expected):
        test_count = test_count + 1
    else:
        print("\033[91mFirst Test: cache_conv[0][1][2][3] is incorrect. Expected:", cache_0_1_2_3_expected, "\nYour output:",
              cache_0_1_2_3, "\033[90m")
    
    if test_count == 3:
        print("conv_forward_test_1 tests passed")

    
def conv_forward_test_2(target):
    # Test 1
    np.random.seed(3)
    A_prev = np.random.randn(2, 5, 7, 4)
    W = np.random.randn(3, 3, 4, 8)
    b = np.random.randn(1, 1, 1, 8)
    
    Z, cache_conv = target(A_prev, W, b, {"pad" : 3, "stride": 1})
    Z_shape = Z.shape
    assert Z_shape[0] == A_prev.shape[0], f"m is wrong. Current: {Z_shape[0]}.  Expected: {A_prev.shape[0]}"
    assert Z_shape[1] == 9, f"n_H is wrong. Current: {Z_shape[1]}.  Expected: 9"
    assert Z_shape[2] == 11, f"n_W is wrong. Current: {Z_shape[2]}.  Expected: 11"
    assert Z_shape[3] == W.shape[3], f"n_C is wrong. Current: {Z_shape[3]}.  Expected: {W.shape[3]}"

    # Test 2 
    Z, cache_conv = target(A_prev, W, b, {"pad" : 0, "stride": 2})
    assert(Z.shape == (2, 2, 3, 8)), "Wrong shape. Don't hard code the pad and stride values in the function"
    
    # Test 3
    W = np.random.randn(5, 5, 4, 8)
    b = np.random.randn(1, 1, 1, 8)
    Z, cache_conv = target(A_prev, W, b, {"pad" : 6, "stride": 1})
    Z_shape = Z.shape
    assert Z_shape[0] == A_prev.shape[0], f"m is wrong. Current: {Z_shape[0]}.  Expected: {A_prev.shape[0]}"
    assert Z_shape[1] == 13, f"n_H is wrong. Current: {Z_shape[1]}.  Expected: 13"
    assert Z_shape[2] == 15, f"n_W is wrong. Current: {Z_shape[2]}.  Expected: 15"
    assert Z_shape[3] == W.shape[3], f"n_C is wrong. Current: {Z_shape[3]}.  Expected: {W.shape[3]}"

    Z_means = np.mean(Z)
    expected_Z = -0.5384027772160062
    
    expected_conv = np.array([[ 1.98848968,  1.19505834, -0.0952376,  -0.52718778],
                             [-0.32158469,  0.15113037, -0.01862772,  0.48352879],
                             [ 0.76896516,  1.36624284,  1.14726479, -0.11022916],
                             [ 0.38825041, -0.38712718, -0.58722031,  1.91082685],
                             [-0.45984615,  1.99073781, -0.34903539,  0.25282509],
                             [ 1.08940955,  0.02392202,  0.39312528, -0.2413848 ],
                             [-0.47552486, -0.16577702, -0.64971742,  1.63138295]])
    
    assert np.isclose(Z_means, expected_Z), f"Wrong Z mean. Expected: {expected_Z} got: {Z_means}"
    assert np.allclose(cache_conv[0][1, 2], expected_conv), f"Values in Z are wrong"

    print("conv_forward_test_2 tests passed")
    
def pool_forward_test_1(target):
    
    # Test 1
    A_prev = np.random.randn(2, 5, 7, 3)
    A, cache = target(A_prev, {"stride" : 2, "f": 2}, mode = "avg")
    A_shape = A.shape
    assert A_shape[0] == A_prev.shape[0], f"Test 1 - m is wrong. Current: {A_shape[0]}.  Expected: {A_prev.shape[0]}"
    assert A_shape[1] == 2, f"Test 1 - n_H is wrong. Current: {A_shape[1]}.  Expected: 2"
    assert A_shape[2] == 3, f"Test 1 - n_W is wrong. Current: {A_shape[2]}.  Expected: 3"
    assert A_shape[3] == A_prev.shape[3], f"Test 1 - n_C is wrong. Current: {A_shape[3]}.  Expected: {A_prev.shape[3]}"
    
    # Test 2
    A_prev = np.random.randn(4, 5, 7, 4)
    A, cache = target(A_prev, {"stride" : 1, "f": 5}, mode = "max")
    A_shape = A.shape
    assert A_shape[0] == A_prev.shape[0], f"Test 2 - m is wrong. Current: {A_shape[0]}.  Expected: {A_prev.shape[0]}"
    assert A_shape[1] == 1, f"Test 2 - n_H is wrong. Current: {A_shape[1]}.  Expected: 1"
    assert A_shape[2] == 3, f"Test 2 - n_W is wrong. Current: {A_shape[2]}.  Expected: 3"
    assert A_shape[3] == A_prev.shape[3], f"Test 2 - n_C is wrong. Current: {A_shape[3]}.  Expected: {A_prev.shape[3]}"
    
    # Test 3
    np.random.seed(1)
    A_prev = np.random.randn(2, 5, 5, 3)
    
    A, cache = target(A_prev, {"stride" : 1, "f": 2}, mode = "max")
    
    assert np.allclose(A[1, 1], np.array([[1.19891788, 0.74055645, 0.07734007],
                                         [0.31515939, 0.84616065, 0.07734007],
                                         [0.69803203, 0.84616065, 1.2245077 ],
                                         [0.69803203, 1.12141771, 1.2245077 ]])), "Wrong value for A[1, 1]"
                                          
    assert np.allclose(cache[0][1, 2], np.array([[ 0.16938243,  0.74055645, -0.9537006 ],
                                         [-0.26621851,  0.03261455, -1.37311732],
                                         [ 0.31515939,  0.84616065, -0.85951594],
                                         [ 0.35054598, -1.31228341, -0.03869551],
                                         [-1.61577235,  1.12141771,  0.40890054]])), "Wrong value for cache"
    
    A, cache = target(A_prev, {"stride" : 1, "f": 2}, mode = "avg")
    
    assert np.allclose(A[1, 1], np.array([[ 0.11583785,  0.34545544, -0.6561907 ],
                                         [-0.2334108,   0.3364666,  -0.69382351],
                                         [ 0.25497093, -0.21741362, -0.07342615],
                                         [-0.04092568, -0.01110394,  0.12495022]])), "Wrong value for A[1, 1]"



def pool_forward_test_2(target):
    
    np.random.seed(1)
    A_prev = np.random.randn(2, 5, 5, 3)
    
    A, cache = target(A_prev, {"stride" : 2, "f": 3}, mode = "max")
    
    assert np.allclose(A[0], np.array([[[1.74481176, 0.90159072, 1.65980218],
                                        [1.74481176, 1.6924546,  1.65980218]],
                                       [[1.13162939, 1.51981682, 2.18557541],
                                        [1.13162939, 1.6924546,  2.18557541]]])), "Wrong value for A[0] in mode max. Make sure you have included stride in your calculation"
                                          
    A, cache = target(A_prev, {"stride" : 2, "f": 3}, mode = "avg")
    
    assert np.allclose(A[1], np.array([[[-0.17313416,  0.32377198, -0.34317572],
                                        [ 0.02030094,  0.14141479, -0.01231585]],
                                       [[ 0.42944926,  0.08446996, -0.27290905],
                                        [ 0.15077452,  0.28911175,  0.00123239]]])), "Wrong value for A[1] in mode avg. Make sure you have included stride in your calculation"    
    

def run_raw_python_model_tests():
    
    print("Running tests...")
    
    # FORWARD CONVOLUTION FUNCTION
    print("Running tests for ~ FORWARD CONVOLUTION FUNCTION")
    np.random.seed(1)
    A_prev = np.random.randn(2, 5, 7, 4)
    W = np.random.randn(3, 3, 4, 8)
    b = np.random.randn(1, 1, 1, 8)
    hparameters = {"pad" : 1,
                "stride": 2}
    print("A_prev.shape:", A_prev.shape)
    Z, cache_conv = conv_forward(A_prev, W, b, hparameters)
    print("Z.shape:", Z.shape)
    # Apply activation
    A_activation, activation_cache = relu(Z)
    print("A_activation.shape:", A_activation.shape)
    
    z_mean = np.mean(Z)
    z_0_2_1 = Z[0, 2, 1]
    cache_0_1_2_3 = cache_conv[0][1][2][3]
    print("Z's mean =\n", z_mean)
    print("Z[0,2,1] =\n", z_0_2_1)
    print("cache_conv[0][1][2][3] =\n", cache_0_1_2_3)

    conv_forward_test_1(z_mean, z_0_2_1, cache_0_1_2_3)
    conv_forward_test_2(conv_forward)
    
    print("\033[92m All forward convolutions tests passed.\033[0m")
    
    print("Running tests for ~ FORWARD POOLING FUNCTION")
    # Case 1: stride of 1
    print("CASE 1:\n")
    np.random.seed(1)
    A_prev_case_1 = np.random.randn(2, 5, 5, 3)
    hparameters_case_1 = {"stride" : 1, "f": 3}
    
    print("A_prev_case_1.shape:", A_prev_case_1.shape)
    A, cache = pool_forward(A_prev_case_1, hparameters_case_1, mode = "max")
    print("A.shape:", A.shape)
    print("mode = max")
    print("A.shape = " + str(A.shape))
    print("A[1, 1] =\n", A[1, 1])
    A, cache = pool_forward(A_prev_case_1, hparameters_case_1, mode = "avg")
    print("mode = avg")
    print("A.shape = " + str(A.shape))
    print("A[1, 1] =\n", A[1, 1])

    pool_forward_test_1(pool_forward)

    # Case 2: stride of 2
    print("\n\033[0mCASE 2:\n")
    np.random.seed(1)
    A_prev_case_2 = np.random.randn(2, 5, 5, 3)
    hparameters_case_2 = {"stride" : 2, "f": 3}

    A, cache = pool_forward(A_prev_case_2, hparameters_case_2, mode = "max")
    print("mode = max")
    print("A.shape = " + str(A.shape))
    print("A[0] =\n", A[0])
    print()

    A, cache = pool_forward(A_prev_case_2, hparameters_case_2, mode = "avg")
    print("mode = avg")
    print("A.shape = " + str(A.shape))
    print("A[1] =\n", A[1])

    pool_forward_test_2(pool_forward)
    print("\033[92m All Forwarda poolings tests passed.\033[0m")

    # BACKPROP CONVOLUTION FUNCTION
    print("Running tests for ~ BACKPROP CONVOLUTION FUNCTION")
    # Running conv_forward to initialize the 'Z' and 'cache_conv",
    # which we'll use to test the conv_backward function
    np.random.seed(1)
    A_prev = np.random.randn(10, 4, 4, 3)
    W = np.random.randn(2, 2, 3, 8)
    b = np.random.randn(1, 1, 1, 8)
    hparameters = {"pad" : 2,
                "stride": 2}
    Z, cache_conv = conv_forward(A_prev, W, b, hparameters)

    # Test conv_backward
    print("Z.shape:", Z.shape)
    print("W.shape:", W.shape)
    print("b.shape:", b.shape)
    dA, dW, db = conv_backward(Z, cache_conv)
    print("dA.shape:", dA.shape)
    print("dW.shape:", dW.shape)
    print("db.shape:", db.shape)

    print("dA_mean =", np.mean(dA))
    print("dW_mean =", np.mean(dW))
    print("db_mean =", np.mean(db))
    assert dA.shape == (10, 4, 4, 3), f"Wrong shape for dA  {dA.shape} != (10, 4, 4, 3)"
    assert dW.shape == (2, 2, 3, 8), f"Wrong shape for dW {dW.shape} != (2, 2, 3, 8)"
    assert db.shape == (1, 1, 1, 8), f"Wrong shape for db {db.shape} != (1, 1, 1, 8)"
    assert np.isclose(np.mean(dA), 1.4524377), "Wrong values for dA"
    assert np.isclose(np.mean(dW), 1.7269914), "Wrong values for dW"
    assert np.isclose(np.mean(db), 7.8392325), "Wrong values for db"
    print("\033[92m All backwards convolutions tests passed.\033[0m")

    # BACKPROP POOLING FUNCTION
    print("Running tests for ~ BACKPROP POOLING FUNCTION")
    np.random.seed(1)
    x = np.random.randn(2, 3)
    mask = create_mask_from_window(x)
    print('x = ', x)
    print("mask = ", mask)

    x = np.array([[-1, 2, 3],
                [2, -3, 2],
                [1, 5, -2]])

    y = np.array([[False, False, False],
        [False, False, False],
        [False, True, False]])
    mask = create_mask_from_window(x)

    assert type(mask) == np.ndarray, "Output must be a np.ndarray"
    assert mask.shape == x.shape, "Input and output shapes must match"
    assert np.allclose(mask, y), "Wrong output. The True value must be at position (2, 1)"

    print("\033[92m All tests for create_mask_from_window passed.\033[0m")
    
    a = distribute_value(2, (2, 2))
    print('distributed value =', a)


    assert type(a) == np.ndarray, "Output must be a np.ndarray"
    assert a.shape == (2, 2), f"Wrong shape {a.shape} != (2, 2)"
    assert np.sum(a) == 2, "Values must sum to 2"

    a = distribute_value(100, (10, 10))
    assert type(a) == np.ndarray, "Output must be a np.ndarray"
    assert a.shape == (10, 10), f"Wrong shape {a.shape} != (10, 10)"
    assert np.sum(a) == 100, "Values must sum to 100"

    print("\033[92m All tests for distribute_value passed.\033[0m")
    
    np.random.seed(1)
    A_prev = np.random.randn(5, 5, 3, 2)
    hparameters = {"stride" : 1, "f": 2}
    A, cache = pool_forward(A_prev, hparameters)
    print(A.shape)
    print(cache[0].shape)
    dA = np.random.randn(5, 4, 2, 2)

    dA_prev1 = pool_backward(dA, cache, mode = "max")
    print("mode = max")
    print('mean of dA = ', np.mean(dA))
    print('dA_prev1[1,1] = ', dA_prev1[1, 1])  
    print()
    dA_prev2 = pool_backward(dA, cache, mode = "avg")
    print("mode = avg")
    print('mean of dA = ', np.mean(dA))
    print('dA_prev2[1,1] = ', dA_prev2[1, 1]) 

    assert type(dA_prev1) == np.ndarray, "Wrong type"
    assert dA_prev1.shape == (5, 5, 3, 2), f"Wrong shape {dA_prev1.shape} != (5, 5, 3, 2)"
    assert np.allclose(dA_prev1[1, 1], [[0, 0], 
                                        [ 5.05844394, -1.68282702], 
                                        [ 0, 0]]), "Wrong values for mode max"
    assert np.allclose(dA_prev2[1, 1], [[0.08485462,  0.2787552], 
                                        [1.26461098, -0.25749373], 
                                        [1.17975636, -0.53624893]]), "Wrong values for mode avg"        

    print("\033[92m All backwards pooling tests have pased passed.\033[0m")


    print("All tests run successfully!")
    