import numpy as np
from time import gmtime, strftime, time


def print_time(msg: str, since: float=0.0):
    '''
    `msg`: message to be printed.
    `since`: a float value representing a particular instant in time. (Can be obtained using time())
    Prints the time elapsed since the time represented by `since`, followed by the string `msg`.
    '''
    print("[" + strftime("%H:%M:%S", gmtime(time()-since)) + "] " + msg)


def make_capital_A(shape: tuple):
    '''
    Generates an image in the shape of a capital A (for testing).
    '''
    (height, width) = shape
    slope = 0.4375 * width / height
    arr_i = np.arange(height).reshape(height, 1)
    arr_j = np.arange(width).reshape(1, width)
    # Exclude top and bottom
    cond_1 = abs(arr_i - 0.5 * height) < 0.4 * height
    # Exclude the outside of each 'leg'
    cond_2 = abs(arr_j - 0.5 * width) < 0.00625 * width + slope * arr_i
    # Make middle bar
    cond_3 = abs(arr_i - 0.6 * height) < 0.05 * height
    # Cut out holes
    cond_4 = abs(arr_j - 0.5 * width) > slope * arr_i - 0.09375 * width
    # Combine conditions
    cond = cond_1 & cond_2 & (cond_3 | cond_4)
    return cond.astype(np.float64)