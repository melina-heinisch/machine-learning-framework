import numpy as np

def categorical_cross_entropy(O, Y):
    m = Y.shape[0]
    O = np.clip(O, a_min=0.000000001, a_max=None)
    return -np.sum(Y * np.log(O)) / m

def cross_entropy(O, Y):
    O = np.clip(O, 0.000000001, 0.99999999)
    return np.mean(- Y * np.log(O) - (1 - Y) * np.log(1 - O) )

def mse(O, Y):
    return ((O - Y)**2).mean()

def mse_derivative(O,Y):
    m = Y.shape[0]
    return 2.0 * (O - Y) / m