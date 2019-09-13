import math
import numpy as np


AFFINE_PARAMETER_NAMES = ['scale_x', 'scale_y', 'shear_radians', 'rotate_radians', 'offset_x', 'offset_y']
DEFAULT_PARAMETER_VALUES = [1.0, 1.0, 0.0, 0.0, 0.0, 0.0]
DEFAULT_BOUNDS = {'scale_x': (0.5, 2), 'scale_y': (0.5, 2), 'shear_radians': (-math.pi/6, math.pi/6), 'rotate_radians': (-math.pi/6, math.pi/6), 'offset_x': (-100, 100), 'offset_y': (-100, 100)}


class RandomDisplacementBounds(object):
    """random displacement with bounds"""
    def __init__(self, xmin=0, xmax=1, stepsize=0.5):
        self.xmin = xmin
        self.xmax = xmax
        self.stepsize = stepsize

    def __call__(self, x):
        """take a random step but ensure the new position is within the bounds"""
        return x + np.random.uniform(np.maximum(-self.stepsize, self.xmin - x), np.minimum(self.stepsize, self.xmax - x))

def bounds_dict_to_array(bounds_dict: dict):
    '''
    Takes a set of affine parameter bounds represented by a dictionary and returns the same set represented by an array.
    Missing bounds are filled in with default values taken from `DEFAULT_BOUNDS`.
    '''
    bounds_array = np.array(list(DEFAULT_BOUNDS.values()))
    for i in range(6):
        param_name = AFFINE_PARAMETER_NAMES[i]
        if param_name in bounds_dict:
            param = bounds_dict[param_name]
            assert len(param) == 2
            if float(param[0]) <= float(param[1]):
                bounds_array[i] = (float(param[0]), float(param[1]))
    return bounds_array

def params_dict_to_array(params_dict: dict):
    '''
    Takes a set of affine parameters represented by a dictionary and returns the same set represented by an array.
    Missing parameters are filled in with default values taken from `DEFAULT_PARAMETER_VALUES`.
    '''
    params_array = np.array(DEFAULT_PARAMETER_VALUES)
    for i in range(6):
        param_name = AFFINE_PARAMETER_NAMES[i]
        if param_name in params_dict:
            params_array[i] = params_dict[param_name]
    return params_array
    
def outside_bounds(params: np.array, bounds: np.array):
    '''
    Returns True if the parameter set represented by `params` does not fall entirely within the bounds represented by `bounds`.
    '''
    assert len(params) == 6
    assert len(bounds) == 6
    [mins, maxes] = bounds.T
    if not (np.all(mins <= maxes)):
        print("mins: " + str(mins))
        print("maxes: " + str(maxes))
    assert np.all(mins <= maxes)
    return (np.any(params < mins) or np.any(params > maxes))

def get_parameter_flags(lock_strings: list):
    '''
    Uses `lock_strings` to determine which affine transformation parameters are to be estimated and which are to be "locked" to a default value.
    Returns a boolean array of size 6. Elements are True if the corresponding parameters are to be estimated, or False if they are to be "locked".
    '''
    lock_scale_x = 'lock_scale' in lock_strings or 'lock_scale_x' in lock_strings
    lock_scale_y = 'lock_scale' in lock_strings or 'lock_scale_y' in lock_strings or 'isotropic_scaling' in lock_strings
    lock_shear = 'lock_shear' in lock_strings
    lock_rotation = 'lock_rotation' in lock_strings
    lock_translation_x = 'lock_translation' in lock_strings or 'lock_translation_x' in lock_strings
    lock_translation_y = 'lock_translation' in lock_strings or 'lock_translation_y' in lock_strings
    return np.logical_not([lock_scale_x, lock_scale_y, lock_shear, lock_rotation, lock_translation_x, lock_translation_y])

def fit_to_bounds(params: np.array, bounds: np.array):
    '''
    Returns a modified version of `params` such that all parameter values fall within the bounds specified by `bounds`.
    '''
    assert len(params) == 6
    assert len(bounds) == 6
    params_fitted = np.array(params)
    for i in range(6):
        params_fitted[i] = max(params[i], bounds[i][0])
        params_fitted[i] = min(params_fitted[i], bounds[i][1])
    return params_fitted

def list_free_parameters(params: np.array, lock_strings: list):
    '''
    Returns the subset of parameters in `params` which are to be estimated (as opposed to being "locked" to default values).
    '''
    flags = get_parameter_flags(lock_strings)
    return params[flags]

def list_free_parameters_scaled_to_bounds(params: np.array, bounds: np.array, lock_strings: list):
    '''
    Calculates a value between 0 and 1 for each element of `params`, depending on its value relative to the corresponding range specified in `bounds`, and returns the subset of these values that correspond to the parameters to be estimated.
    '''
    [mins, maxes] = bounds.T
    params_scaled = (params - mins) / (maxes - mins)
    flags = get_parameter_flags(lock_strings)
    return params_scaled[flags]

def scale_parameters(params: np.array, scale_factor: float):
    '''
    Scales the translation parameters in `params` by `scale_factor`.
    '''
    return params * np.array([1, 1, 1, 1, scale_factor, scale_factor])

def recover_parameters_from_scaled_guess(guess: np.array, bounds: np.array, lock_strings: list):
    flags = get_parameter_flags(lock_strings)
    if len(guess) != np.sum(flags):
        print("Mismatch between guessed parameter list length and parameter flags!")
        print("Parameter list: " + str(guess))
        print("(Length = " + str(len(guess)) + ")")
        print("Parameter flags: " + str(flags))
        print("(Total = " + str(np.sum(flags)) + ")")
    assert len(guess) == np.sum(flags)
    [mins, maxes] = bounds.T
    params_scaled = np.empty(6)
    params_scaled[flags] = guess
    params_recovered = params_scaled * (maxes - mins) + mins
    params_recovered[~flags] = np.array([1, 1, 0, 0, 0, 0], dtype=float)[~flags]
    if 'isotropic_scaling' in lock_strings:
        params_recovered[1] = params_recovered[0]
    return params_recovered


def remove_locked_parameters(parameters: np.array, isotropic_scaling: bool, lock_scale: bool, lock_shear: bool, lock_rotation: bool, lock_translation: bool):
    '''
    Removes any elements from `parameters` representing locked parameters that will not be optimised.
    '''
    assert len(parameters) == 6
    parameter_flags = np.logical_not([lock_scale, lock_scale or isotropic_scaling, lock_shear, lock_rotation, lock_translation, lock_translation])
    return parameters[parameter_flags]
    
    
def fill_missing_parameters(params_in: np.array, isotropic_scaling: bool, lock_scale: bool, lock_shear: bool, lock_rotation: bool, lock_translation: bool):
    '''
    Adds elements to `params_in` that were previously removed by `remove_locked_parameters`. Default parameter values are used.
    '''
    parameter_flags = np.logical_not([lock_scale, lock_scale or isotropic_scaling, lock_shear, lock_rotation, lock_translation, lock_translation])
    assert len(params_in) == np.sum(parameter_flags)
    params_out = np.array([1, 1, 0, 0, 0, 0], dtype=float)
    if isotropic_scaling and not lock_scale:
        params_out[1] = params_in[0]
    params_out[parameter_flags] = params_in
    return params_out


def affine_params_to_matrix(params: list):
    '''
    Converts a list of affine transformation parameters to the corresponding 3x3 matrix.
    '''
    assert len(params) == 6
    [scale_x, scale_y, shear, rotation, offset_x, offset_y] = params
    a0 = scale_x * math.cos(rotation)
    a1 = -scale_y * math.sin(rotation + shear)
    a2 = offset_x
    b0 = scale_x * math.sin(rotation)
    b1 = scale_y * math.cos(rotation + shear)
    b2 = offset_y
    return np.array([[a0, a1, a2], [b0, b1, b2], [0, 0, 1]])


def affine_matrix_to_params(matrix: np.array):
    '''
    Converts a 3x3 affine transformation matrix to a list of six affine transformation parameters.
    '''
    assert len(matrix.shape) == 2
    assert matrix.shape[0] == 3
    assert matrix.shape[1] == 3
    a0 = matrix[0][0]
    a1 = matrix[0][1]
    a2 = matrix[0][2]
    b0 = matrix[1][0]
    b1 = matrix[1][1]
    b2 = matrix[1][2]
    scale_x = math.sqrt(a0**2 + b0**2)
    scale_y = math.sqrt(a1**2 + b1**2)
    rotation = math.atan2(b0, a0)
    shear = math.atan2(-a1, b1) - rotation
    offset_x = a2
    offset_y = b2
    return [scale_x, scale_y, shear, rotation, offset_x, offset_y]
    

def combine_affine_params(params_applied_first: list, params_applied_second: list):
    '''
    Returns a single set of affine transformation parameters equivalent to applying `params_applied_first` followed by `params_applied_second`.
    '''
    matrix_applied_first = affine_params_to_matrix(params_applied_first)
    matrix_applied_second = affine_params_to_matrix(params_applied_second)
    matrix_combined = np.matmul(matrix_applied_second, matrix_applied_first)
    return affine_matrix_to_params(matrix_combined)