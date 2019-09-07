import math
import numpy as np


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