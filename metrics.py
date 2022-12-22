import numpy as np

def get_margin_error(y_hat, y, sign):
    """ Get the margin error of the prediction """
    delta = y_hat*sign >= y*sign
    delta = np.where(delta, 0, y_hat - y)
    return np.abs(delta/y)

def get_relative_margin_error(y_hat, y, sign):
    return np.abs((y_hat-y) / y)