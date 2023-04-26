import numpy as np
import pandas as pd

def get_one_hot(scalar_labels):
    y_one_hot = pd.get_dummies(scalar_labels)
    return y_one_hot.to_numpy()
