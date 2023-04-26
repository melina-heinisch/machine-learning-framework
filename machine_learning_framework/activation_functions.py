import numpy as np

def sigmoid(z,derived = False):
    if derived:
        return sigmoid(z) * (1. - sigmoid(z))
    else:
        return 1 / (1+np.e**-z)

def softmax(O):
    e_x = np.exp(O)
    return e_x / np.sum(e_x, axis=1, keepdims=True)

def relu(z, derived = False):
    if derived:
        z[z<=0] = 0.0
        z[z>0] = 1.0
        return z
    else:
        return np.maximum(0.0, z)

def tanh(z, derived = False):
    if derived:
        return 1 - tanh(z)**2
    else:
        return np.tanh(z)