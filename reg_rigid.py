import hyperspy.api as hs
import image_metrics as im
import matplotlib.pyplot as plt
import numpy as np
import scipy
import skimage

from image_processing import apply_affine_params_to_signal, get_neighbour_similarity, normalised_average_of_signal, normalised_image, rotate, scale, transform_using_values
from reg_rigid_utils import *
from similarity_measure_methods import *
from time import time
from utils import print_time


def estimate_shift_relative_to_most_representative_image(signal_in: hs.signals.Signal2D, exponent=1, sub_pixel_factor=50):
    '''
    Returns a list of estimated shifts of each image relative to the most representative image in the stack (as determined by highest_mutual_information_index)
    '''
    mi_max_index = highest_mutual_information_index(signal_in, exponent=exponent)
    # signal_before: all elements up to and including mi_max_index, reversed
    signal_before = hs.signals.Signal2D(signal_in.data[:mi_max_index+1][::-1])
    # signal_after: all elements from mi_max_index onwards
    # Note that signal_in.data[mi_max_index] is the first element of both signal_before and signal_after.
    signal_after = hs.signals.Signal2D(signal_in.data[mi_max_index:])
    shifts_before = list(signal_before.estimate_shift2D(reference='current', sub_pixel_factor=sub_pixel_factor))
    shifts_after = list(signal_after.estimate_shift2D(reference='current', sub_pixel_factor=sub_pixel_factor))
    # shifts_all: shifts_before, reversed, concatenated with shifts_after from the second element onwards.
    shifts_all = shifts_before[::-1] + shifts_after[1:]
    #shifts_all = np.concatenate(shifts_before[::-1], shifts_after[1:])
    #shifts_all = list(shifts_before[::-1])
    #shifts_all.extend(shifts_after[1:])
    #shifts_1 = shifts_before[::-1]
    #shifts_2 = shifts_after[1:]
    #shifts_all = shifts_1 + shifts_2
    return np.array(shifts_all)


def skimage_estimate_shift(arr_moving: np.array, arr_ref: np.array, upsample_factor=10):
    return skimage.feature.register_translation(arr_moving, arr_ref, upsample_factor=upsample_factor, space='real', return_error=False)
    

def optimise_scale(arr_moving: np.array, arr_ref: np.array, initial_guess_x=1.0, initial_guess_y=1.0):
    '''
    Uses the Powell local optimisation algorithm to obtain scale factors in the x- and y-directions that maximise the mutual information between `arr_scaled` and `arr_ref`, where `arr_scaled` is the scaled version of `arr_moving`.
    '''
    def inverse_mutual_information_after_scaling(parameters):
        arr_scaled = scale(arr_moving, parameters[0], parameters[1])
        return 1/im.similarity_measure(arr_scaled, arr_ref)
    optimisation_result = scipy.optimize.minimize(inverse_mutual_information_after_scaling, [initial_guess_x, initial_guess_y], method='Powell')
    if optimisation_result.success:
        return optimisation_result.x
    else:
        raise ValueError(result.message)
    

def optimise_rotation(arr_moving: np.array, arr_ref: np.array, initial_guess_radians=0.1):
    '''
    Uses the Powell local optimisation algorithm to obtain a rotation angle (in radians) that maximises the mutual information between `arr_rotated` and `arr_ref`, where `arr_rotated` is the rotated version of `arr_moving`.
    '''
    def inverse_mutual_information_after_rotation(parameters):
        arr_rotated = rotate(arr_moving, parameters[0])
        return 1/im.similarity_measure(arr_rotated, arr_ref)
    optimisation_result = scipy.optimize.minimize(inverse_mutual_information_after_rotation, [initial_guess_radians], method='Powell')
    if optimisation_result.success:
        return optimisation_result.x
    else:
        raise ValueError(result.message)
    

def optimise_rotation_best_of_two(arr_moving: np.array, arr_ref: np.array):
    '''
    Uses the Nelder-Mead local optimisation algorithm to obtain a rotation angle (in radians) that maximises the mutual information between `arr_rotated` and `arr_ref`, where `arr_rotated` is the rotated version of `arr_moving`.
    '''
    result_1 = optimise_rotation(arr_moving, arr_ref, initial_guess_radians=0.2)
    result_2 = optimise_rotation(arr_moving, arr_ref, initial_guess_radians=-0.2)
    rotated_1 = rotate(arr_moving, float(result_1))
    rotated_2 = rotate(arr_moving, float(result_2))
    if im.similarity_measure(rotated_1, arr_ref) > im.similarity_measure(rotated_2, arr_ref):
        return result_1
    else:
        return result_2
    

def optimise_affine(arr_moving: np.array, arr_ref: np.array, scale_x=1.0, scale_y=1.0, shear_radians=0.0, rotate_radians=0.0, offset_x=0.0, offset_y=0.0, method='Powell', bounds=None, isotropic_scaling=False, lock_scale=False, lock_shear=False, lock_rotation=False, lock_translation=False, debug=False):
    '''
    Uses a local optimisation algorithm to obtain a set of affine transform parameters that maximises the mutual information between `arr_transformed` and `arr_ref`, where `arr_transformed` is the transformed version of `arr_moving`.
    '''
    params = np.array([scale_x, scale_y, shear_radians, rotate_radians, offset_x, offset_y])
    (height, width) = arr_moving.shape
    if bounds is None:
        bounds = np.array([(0.5, 2), (0.5, 2), (-math.pi/6, math.pi/6), (-math.pi/6, math.pi/6), (-height*0.2, height*0.2), (-width*0.2, width*0.2)])
    
    def inverse_mutual_information_after_transform(free_params):
        def _outside_limits(x: np.array):
            [xmin, xmax] = np.array(bounds).T
            return (np.any(x < xmin) or np.any(x > xmax))
        def _fit_params_to_bounds(params):
            assert len(params) == 6
            params_out = np.array(params)
            for i in range(6):
                params_out[i] = max(params[i], bounds[i][0])
                params_out[i] = min(params_out[i], bounds[i][1])
            return params_out
        transform_params = fill_missing_parameters(free_params, isotropic_scaling, lock_scale, lock_shear, lock_rotation, lock_translation)
        arr_transformed = transform_using_values(arr_moving, transform_params)
        mi = im.similarity_measure(arr_transformed, arr_ref)
        mi_scaled = mi/(arr_ref.size)
        outside = _outside_limits(transform_params)
        metric = 1/(mi_scaled + 1)
        if outside:
            fitted_params = _fit_params_to_bounds(transform_params)
            arr_transformed_fitted = transform_using_values(arr_moving, fitted_params)
            mi_fitted = im.similarity_measure(arr_transformed_fitted, arr_ref)/(arr_ref.size)
            metric = 1/mi_fitted
        if debug:
            print((1/metric, free_params))
        assert (outside and 1/metric <= 1) or ((not outside) and 1/metric >= 1)
        return metric
    
    initial_guess = remove_locked_parameters(params, isotropic_scaling, lock_scale, lock_shear, lock_rotation, lock_translation)
    
    optimisation_result = scipy.optimize.minimize(inverse_mutual_information_after_transform, initial_guess, method=method)
    if optimisation_result.success:
        optimised_parameters = optimisation_result.x
        # If there is only one optimised parameter, optimisation_parameters will be of the form np.array(param), which has zero length.
        # Otherwise, it will be of the form np.array([param1, param2, ...]).
        # np.array(param) should therefore be converted to the form np.array([param]), which has length 1.
        if len(optimised_parameters.shape) == 0:
            optimised_parameters = np.array([optimised_parameters])
        return fill_missing_parameters(optimised_parameters, isotropic_scaling, lock_scale, lock_shear, lock_rotation, lock_translation)
    else:
        raise ValueError(optimisation_result.message)
    
    
def optimise_affine_v2(arr_moving: np.array, arr_ref: np.array, scale_x=1.0, scale_y=1.0, shear_radians=0.0, rotate_radians=0.0, offset_x=0.0, offset_y=0.0, method='Powell', bounds=None, isotropic_scaling=False, lock_scale=False, lock_shear=False, lock_rotation=False, lock_translation=False, debug=False, basinhopping=False, scale_bounds=False, basinhopping_kwargs={'T': 0.5, 'minimizer_kwargs': {'method': 'Powell'}}):
    '''
    Uses a local optimisation algorithm to obtain a set of affine transform parameters that maximises the mutual information between `arr_transformed` and `arr_ref`, where `arr_transformed` is the transformed version of `arr_moving`.
    '''
    params = np.array([scale_x, scale_y, shear_radians, rotate_radians, offset_x, offset_y])
    (height, width) = arr_moving.shape
    if bounds is None:
        bounds = np.array([(0.5, 2), (0.5, 2), (-math.pi/6, math.pi/6), (-math.pi/6, math.pi/6), (-height*0.2, height*0.2), (-width*0.2, width*0.2)])
    arr_ref_ns = get_neighbour_similarity(arr_ref)
        
    def free_params_to_guess(params):
        assert len(params) <= 6
        if not scale_bounds:
            return params
        bounds_reduced = remove_locked_parameters(bounds, isotropic_scaling, lock_scale, lock_shear, lock_rotation, lock_translation)
        params_scaled = []
        for p in range(len(params)):
            p_min = bounds_reduced[p][0]
            p_max = bounds_reduced[p][1]
            p_avg = (p_min + p_max)/2
            p_range = p_max - p_min
            scaled = (params[p] - p_avg)/(p_range/2)
            params_scaled.append(scaled)
        assert len(params_scaled) <= 6
        return params_scaled
    
    def guess_to_free_params(params_scaled):
        if len(params_scaled.shape) == 0:
            params_scaled = np.array([params_scaled])
        if len(params_scaled) > 6:
            print("params_scaled is too long!")
            print(params_scaled)
        assert len(params_scaled) <= 6
        if not scale_bounds:
            return params_scaled
        bounds_reduced = remove_locked_parameters(bounds, isotropic_scaling, lock_scale, lock_shear, lock_rotation, lock_translation)
        params = []
        for p in range(len(params_scaled)):
            p_min = bounds_reduced[p][0]
            p_max = bounds_reduced[p][1]
            p_avg = (p_min + p_max)/2
            p_range = p_max - p_min
            param = p_avg + params_scaled[p] * p_range/2
            params.append(param)
        assert len(params) <= 6
        return params
    
    def inverse_mutual_information_after_transform(params_scaled):
        def _outside_limits(x: np.array):
            [xmin, xmax] = np.array(bounds).T
            return (np.any(x < xmin) or np.any(x > xmax))
        def _fit_params_to_bounds(params):
            assert len(params) == 6
            params_out = np.array(params)
            for i in range(6):
                params_out[i] = max(params[i], bounds[i][0])
                params_out[i] = min(params_out[i], bounds[i][1])
            return params_out
        free_params = guess_to_free_params(params_scaled)
        transform_params = fill_missing_parameters(free_params, isotropic_scaling, lock_scale, lock_shear, lock_rotation, lock_translation)
        #arr_transformed = transform_using_values(arr_moving, transform_params)
        #mi = im.similarity_measure(arr_transformed, arr_ref)
        #mi = similarity_measure_after_transform(arr_ref, arr_moving, transform_params)
        #mi_scaled = mi/(arr_ref.size)
        #mi_scaled = similarity_measure_area_of_overlap(arr_ref, arr_moving, transform_params)
        mi = similarity_measure_using_neighbour_similarity(arr_moving, arr_ref, arr_ref_ns, transform_params, debug=debug, max_groups=6)
        mi_scaled = mi
        outside = _outside_limits(transform_params)
        metric = 1/(mi_scaled + 1)
        if outside:
            fitted_params = _fit_params_to_bounds(transform_params)
            fitted_params_scaled = free_params_to_guess(remove_locked_parameters(fitted_params, isotropic_scaling, lock_scale, lock_shear, lock_rotation, lock_translation))
            #arr_transformed_fitted = transform_using_values(arr_moving, fitted_params)
            #mi_fitted = im.similarity_measure(arr_transformed_fitted, arr_ref)/(arr_ref.size)
            #mi_fitted = similarity_measure_area_of_overlap(arr_ref, arr_moving, fitted_params)
            #mi_fitted = similarity_measure_after_transform(arr_ref, arr_moving, transform_params)
            #mi_fitted_scaled = mi_fitted/(arr_ref.size)
            mi_fitted = similarity_measure_using_neighbour_similarity(arr_moving, arr_ref, arr_ref_ns, fitted_params, debug=debug, max_groups=6)
            mi_fitted_scaled = mi_fitted
            metric = np.float_(np.finfo(np.float_).max)
            if mi_fitted_scaled > 0:
                metric = 1/mi_fitted_scaled
        if debug:
            print((1/metric, free_params))
        if outside:
            assert 1/metric <= 1
        else:
            assert 1/metric >= 1
        return metric
    
    #initial_guess = remove_locked_parameters(params, isotropic_scaling, lock_scale, lock_shear, lock_rotation, lock_translation)
    initial_guess = free_params_to_guess(remove_locked_parameters(params, isotropic_scaling, lock_scale, lock_shear, lock_rotation, lock_translation))
    
    if basinhopping:
        #optimisation_result = scipy.optimize.basinhopping(inverse_mutual_information_after_transform, initial_guess, T=0.5, minimizer_kwargs={'method': 'Powell'})
        optimisation_result = scipy.optimize.basinhopping(inverse_mutual_information_after_transform, initial_guess, **basinhopping_kwargs)
    else:
        optimisation_result = scipy.optimize.minimize(inverse_mutual_information_after_transform, initial_guess, method=method)
    if debug:
        print(optimisation_result)
    """if optimisation_result.success:
        #optimised_parameters = optimisation_result.x
        optimised_parameters = guess_to_free_params(optimisation_result.x)
        # If there is only one optimised parameter, optimisation_parameters will be of the form np.array(param), which has zero length.
        # Otherwise, it will be of the form np.array([param1, param2, ...]).
        # np.array(param) should therefore be converted to the form np.array([param]), which has length 1.
        if len(optimised_parameters.shape) == 0:
            optimised_parameters = np.array([optimised_parameters])
        return fill_missing_parameters(optimised_parameters, isotropic_scaling, lock_scale, lock_shear, lock_rotation, lock_translation)
    else:
        raise ValueError(optimisation_result.message)"""
    optimised_parameters = guess_to_free_params(optimisation_result.x)
    # If there is only one optimised parameter, optimisation_parameters will be of the form np.array(param), which has zero length.
    # Otherwise, it will be of the form np.array([param1, param2, ...]).
    # np.array(param) should therefore be converted to the form np.array([param]), which has length 1.
    """if not basinhopping:
        if len(optimised_parameters.shape) == 0:
            optimised_parameters = np.array([optimised_parameters])"""
    return fill_missing_parameters(optimised_parameters, isotropic_scaling, lock_scale, lock_shear, lock_rotation, lock_translation)

        
def optimise_affine_no_shear(arr_moving: np.array, arr_ref: np.array, scale_x=1.0, scale_y=1.0, rotate_radians=0.0, offset_x=0.0, offset_y=0.0, method='Powell', bounds=None, debug=False):
    '''
    Uses a local optimisation algorithm to obtain a set of affine transform parameters that maximises the mutual information between `arr_transformed` and `arr_ref`, where `arr_transformed` is the transformed version of `arr_moving`. The shear parameter is always zero.
    '''
    (height, width) = arr_moving.shape
    if bounds is None and (method == 'L-BFGS-B' or method == 'TNC' or method == 'SLSQP'):
        #bounds = [(0.5, 2), (0.5, 2), (-math.pi/3, math.pi/3), (-height, height), (-width, width)]
        bounds = [(0.7, 1.5), (0.7, 1.5), (-math.pi/6, math.pi/6), (-height*0.2, height*0.2), (-width*0.2, width*0.2)]
    def inverse_mutual_information_after_transform(parameters):
        #arr_transformed = transform_using_values(arr_moving, [parameters[0], parameters[1], 0, parameters[2], parameters[3], parameters[4]])
        #arr_transformed = affine_from_list(arr_moving, [parameters[2], parameters[0], parameters[1], 0, 0, parameters[3], parameters[4]])
        #metric = 1/im.similarity_measure(arr_transformed, arr_ref)
        #metric = 1/similarity_measure_area_of_overlap(arr_ref, arr_moving, [parameters[0], parameters[1], 0, parameters[2], parameters[3], parameters[4]])
        metric = 1/similarity_measure_after_transform(arr_ref, arr_moving, [parameters[0], parameters[1], 0, parameters[2], parameters[3], parameters[4]])
        if debug:
            print((1/metric, [parameters[0], parameters[1], 0, parameters[2], parameters[3], parameters[4]]))
        return metric
    optimisation_result = scipy.optimize.minimize(inverse_mutual_information_after_transform, [scale_x, scale_y, rotate_radians, offset_x, offset_y], method=method, bounds=bounds)
    if optimisation_result.success:
        return optimisation_result.x
    else:
        raise ValueError(optimisation_result.message)
    

def optimise_scale_and_rotation(arr_moving: np.array, arr_ref: np.array, scale_x=1.0, scale_y=1.0, rotate_radians=0.0, method='Powell', bounds=None):
    '''
    Uses a local optimisation algorithm to obtain a set of affine transform parameters that maximises the mutual information between `arr_transformed` and `arr_ref`, where `arr_transformed` is the transformed version of `arr_moving`. The shear parameter is always zero.
    '''
    if bounds is None and (method == 'L-BFGS-B' or method == 'TNC' or method == 'SLSQP'):
        bounds = [(0.5, 2), (0.5, 2), (-math.pi/3, math.pi/3)]
    def inverse_mutual_information_after_transform(parameters):
        #arr_transformed = transform_using_values(arr_moving, [parameters[0], parameters[1], 0, parameters[2], 0, 0])
        #arr_transformed = affine_from_list(arr_moving, [parameters[2], parameters[0], parameters[1], 0, 0, 0, 0])
        #return 1/im.similarity_measure(arr_transformed, arr_ref)
        #return 1/similarity_measure_area_of_overlap(arr_ref, arr_moving, [parameters[0], parameters[1], 0, parameters[2], 0, 0])
        return 1/similarity_measure_after_transform(arr_ref, arr_moving, [parameters[0], parameters[1], 0, parameters[2], 0, 0])
    optimisation_result = scipy.optimize.minimize(inverse_mutual_information_after_transform, [scale_x, scale_y, rotate_radians], method=method, bounds=bounds)
    if optimisation_result.success:
        return optimisation_result.x
    else:
        raise ValueError(optimisation_result.message)
    

def optimise_scale_and_rotation_best_of_two(arr_moving: np.array, arr_ref: np.array, method='Powell', bounds=None):
    '''
    Uses a local optimisation algorithm to obtain a set of affine transform parameters that maximises the mutual information between `arr_transformed` and `arr_ref`, where `arr_transformed` is the transformed version of `arr_moving`. The shear parameter is always zero.
    '''
    if bounds is None and (method == 'L-BFGS-B' or method == 'TNC' or method == 'SLSQP'):
        bounds = [(0.5, 2), (0.5, 2), (-math.pi/3, math.pi/3)]
    result_1 = optimise_scale_and_rotation(arr_moving, arr_ref, rotate_radians=0.2, method=method, bounds=bounds)
    result_2 = optimise_scale_and_rotation(arr_moving, arr_ref, rotate_radians=-0.2, method=method, bounds=bounds)
    #transformed_1 = transform_using_values(arr_moving, [result_1[0], result_1[1], 0, result_1[2], 0, 0])
    #transformed_2 = transform_using_values(arr_moving, [result_2[0], result_2[1], 0, result_2[2], 0, 0])
    #if im.similarity_measure(transformed_1, arr_ref) > im.similarity_measure(transformed_2, arr_ref):
    sm_1 = similarity_measure_after_transform(arr_ref, arr_moving, [result_1[0], result_1[1], 0, result_1[2], 0, 0])
    sm_2 = similarity_measure_after_transform(arr_ref, arr_moving, [result_2[0], result_2[1], 0, result_2[2], 0, 0])
    if sm_1 > sm_2:
        return result_1
    else:
        return result_2


def optimise_affine_by_differential_evolution(arr_moving: np.array, arr_ref: np.array, bounds=None, maxiter=1000):
    '''
    Uses a global optimisation algorithm, specifically differential evolution, to obtain a set of affine transform parameters that maximises the mutual information between `arr_transformed` and `arr_ref`, where `arr_transformed` is the transformed version of `arr_moving`.
    '''
    def inverse_mutual_information_after_transform(parameters):
        #return 1/im.similarity_measure(transform_using_values(arr_moving, parameters), arr_ref)
        #return 1/similarity_measure_area_of_overlap(arr_ref, arr_moving, parameters)
        return 1/similarity_measure_after_transform(arr_ref, arr_moving, parameters)
    if bounds is None:
        max_scale_factor = 2
        max_translate_factor = 0.2
        bounds = [
            (1/max_scale_factor, max_scale_factor), # scale_x
            (1/max_scale_factor, max_scale_factor), # scale_y
            (-math.pi/3, math.pi/3), # shear
            (-math.pi/3, math.pi/3), # rotation
            (-arr_moving.shape[0]*max_translate_factor, arr_moving.shape[0]*max_translate_factor), # offset_x
            (-arr_moving.shape[1]*max_translate_factor, arr_moving.shape[1]*max_translate_factor)] # offset_y
    assert bounds[0][0] > 0
    #assert bounds[0][1] >= max(bounds[0][0], 1)
    assert bounds[0][1] >= bounds[0][0]
    assert bounds[1][0] > 0
    #assert bounds[1][1] >= max(bounds[1][0], 1)
    assert bounds[1][1] >= bounds[1][0]
    assert abs(bounds[2][0]) < math.pi/2
    assert abs(bounds[2][1]) < math.pi/2 and bounds[2][1] >= bounds[2][0]
    assert abs(bounds[3][0]) <= math.pi
    assert abs(bounds[3][1]) <= math.pi and bounds[3][1] >= bounds[3][0]
    assert abs(bounds[4][0]) <= arr_moving.shape[0]
    assert abs(bounds[4][1]) <= arr_moving.shape[0] and bounds[4][1] >= bounds[4][0]
    assert abs(bounds[5][0]) <= arr_moving.shape[1]
    assert abs(bounds[5][1]) <= arr_moving.shape[1] and bounds[5][1] >= bounds[5][0]
    
    de_result = scipy.optimize.differential_evolution(inverse_mutual_information_after_transform, bounds, maxiter=maxiter).x
    inv_mi_no_transform = inverse_mutual_information_after_transform([1, 1, 0, 0, 0, 0])
    inv_mi = inverse_mutual_information_after_transform(de_result)
    if inv_mi_no_transform <= inv_mi:
        de_result = [1, 1, 0, 0, 0, 0]
    return de_result


def pyramid_affine(arr_moving: np.array, arr_ref: np.array, num_levels=3, registration_method=optimise_affine_by_differential_evolution, reg_args=None, reg_kwargs={}):
    '''
    Uses a pyramid strategy to apply `registration_method` to downsampled versions of `arr_moving`, eventually returning a list of affine transformation parameters estimated to transform arr_moving as nearly as possible to arr_ref.
    '''
    (height, width) = arr_moving.data.shape
    params = [1, 1, 0, 0, 0, 0]
    for n in range(num_levels):
        power_of_2 = num_levels - 1 - n
        (new_height, new_width) = (height//(2**power_of_2), width//(2**power_of_2))
        if new_height < 2 or new_width < 2:
            continue
        (arr_moving_downsampled, arr_ref_downsampled) = (transform_using_values(arr_moving, params), arr_ref)
        if power_of_2 > 0:
            arr_moving_downsampled = skimage.transform.resize(arr_moving_downsampled, (new_height, new_width), mode='reflect', anti_aliasing=True)
            arr_ref_downsampled = skimage.transform.resize(arr_ref_downsampled, (new_height, new_width), mode='reflect', anti_aliasing=True)
        new_params = [1, 1, 0, 0, 0, 0]
        if reg_args is None:
            new_params = registration_method(arr_moving_downsampled, arr_ref_downsampled, **reg_kwargs)
        else:
            new_params = registration_method(arr_moving_downsampled, arr_ref_downsampled, *reg_args, **reg_kwargs)
        if power_of_2 > 0:
            # Computed offsets must be scaled up
            new_params[4] *= height/new_height
            new_params[5] *= width/new_width
        combined_params = combine_affine_params(params, new_params)
        sm_1 = similarity_measure_after_transform(arr_ref, arr_moving, params)
        sm_2 = similarity_measure_after_transform(arr_ref, arr_moving, combined_params)
        if (sm_2 > sm_1):
            params = combined_params
    return params


def pyramid_scale_and_translation(arr_moving: np.array, arr_ref: np.array, num_levels=3, registration_method=optimise_affine_by_differential_evolution):
    '''
    Uses a pyramid strategy to apply `registration_method` to downsampled versions of `arr_moving`, eventually returning a list of affine transformation parameters estimated to transform arr_moving as nearly as possible to arr_ref. Shear and rotation parameters are ignored.
    '''
    (height, width) = arr_moving.data.shape
    params = [1, 1, 0, 0, 0, 0]
    for n in range(num_levels):
        power_of_2 = num_levels - 1 - n
        (new_height, new_width) = (height//(2**power_of_2), width//(2**power_of_2))
        if new_height < 2 or new_width < 2:
            continue
        (arr_moving_downsampled, arr_ref_downsampled) = (transform_using_values(arr_moving, params), arr_ref)
        if power_of_2 > 0:
            arr_moving_downsampled = skimage.transform.resize(arr_moving_downsampled, (new_height, new_width), mode='reflect', anti_aliasing=True)
            arr_ref_downsampled = skimage.transform.resize(arr_ref_downsampled, (new_height, new_width), mode='reflect', anti_aliasing=True)
        new_params = registration_method(arr_moving_downsampled, arr_ref_downsampled, bounds=[(0.8, 1.2), (0.8, 1.2), (0, 0), (0, 0), (-height*0.2, height*0.2), (-width*0.2, width*0.2)])
        if power_of_2 > 0:
            # Computed offsets must be scaled up
            new_params[4] *= height/new_height
            new_params[5] *= width/new_width
        combined_params = combine_affine_params(params, new_params)
        sm_1 = similarity_measure_after_transform(arr_ref, arr_moving, params)
        sm_2 = similarity_measure_after_transform(arr_ref, arr_moving, combined_params)
        if (sm_2 > sm_1):
            params = combined_params
    return params


def scale_and_translation_signal_params(signal_in: hs.signals.Signal2D, num_levels=3, registration_method=optimise_affine_by_differential_evolution):
    '''
    Applies `pyramid_scale_and_translation` to an entire image stack. Returns an array of parameter sets, one per image.
    '''
    num_images = signal_in.data.shape[0]
    params = np.empty((num_images, 6))
    mi_max_index = highest_mutual_information_index(signal_in)
    for t in range(num_images):
        print("Estimating scale and translation parameters for frame " + str(t+1) + " of " + str(num_images))
        params[t] = pyramid_scale_and_translation(signal_in.data[t], signal_in.data[mi_max_index], num_levels=num_levels, registration_method=registration_method)
    return params


def affine_signal_params(signal_in: hs.signals.Signal2D, arr_ref=None, use_normalised_average_as_reference=False, reuse_estimates=False, max_image_dimension=-1, continue_until_no_improvement=False, improvement_threshold=0.1, significant_improvement=0.05, max_num_passes=20, num_levels=1, registration_method=optimise_affine_by_differential_evolution, reg_args=None, reg_kwargs={}):
    '''
    Applies `registration_method` to an entire image stack. Returns an array of parameter sets, one per image.
    '''
    assert len(signal_in.data.shape) == 3
    start_time = time()
    (num_images, height, width) = signal_in.data.shape
    params = np.array([1, 1, 0, 0, 0, 0], dtype=float) * np.ones((num_images, 1))
    # Ensure a reference image is set.
    mi_max_index = -1
    mi_max = 0
    if use_normalised_average_as_reference:
        arr_ref = normalised_average_of_signal(signal_in)
    if arr_ref is None:
        print("Obtaining arr_ref...")
        mi_max_index = highest_mutual_information_index(signal_in)
        arr_ref = signal_in.data[mi_max_index]
        mi_max = im.similarity_measure(normalised_image(arr_ref), normalised_image(arr_ref), measure="NMI")
        print("arr_ref obtained (frame " + str(mi_max_index+1) + " of " + str(num_images) + ").")
    # Validate improvement_threshold
    assert improvement_threshold >= 0
    assert improvement_threshold < 1
    # Ensure the maximum dimension of images to be processed is no greater than max_image_dimension.
    height_resized = height
    width_resized = width
    arr_ref_resized = arr_ref
    if max_image_dimension == -1:
        max_image_dimension = max(height, width)
    else:
        assert max_image_dimension > 0
    resize_image = height > max_image_dimension or width > max_image_dimension
    if resize_image:
        if height > width:
            height_resized = max_image_dimension
            width_resized = int(max_image_dimension * width / height)
        else:
            width_resized = max_image_dimension
            height_resized = int(max_image_dimension * height / width)
        print("height_resized = " + str(height_resized) + ", width_resized = " + str(width_resized))
        # Resize reference image to conform to max_image_dimension.
        arr_ref_resized = skimage.transform.resize(arr_ref, (height_resized, width_resized), mode='reflect', anti_aliasing=True)
    arr_ref_resized_ns = get_neighbour_similarity(arr_ref_resized)
    def _get_initial_guess_kwargs(initial_guess):
        initial_guess_kwargs = {}
        if len(initial_guess) == 6:
            initial_guess_kwargs['scale_x'] = initial_guess[0]
            initial_guess_kwargs['scale_y'] = initial_guess[1]
            initial_guess_kwargs['shear_radians'] = initial_guess[2]
            initial_guess_kwargs['rotate_radians'] = initial_guess[3]
            initial_guess_kwargs['offset_x'] = initial_guess[4]
            initial_guess_kwargs['offset_y'] = initial_guess[5]
        return initial_guess_kwargs
    def _get_random_initial_guess_kwargs(initial_guess):
        initial_guess_kwargs = {}
        initial_guess_kwargs['scale_x'] = np.random.rand()
        initial_guess_kwargs['scale_y'] = np.random.rand()
        initial_guess_kwargs['shear_radians'] = np.random.rand()
        initial_guess_kwargs['rotate_radians'] = np.random.rand()
        initial_guess_kwargs['offset_x'] = np.random.rand()
        initial_guess_kwargs['offset_y'] = np.random.rand()
        if 'bounds' in reg_kwargs:
            bounds = reg_kwargs['bounds']
            initial_guess_kwargs['scale_x'] = initial_guess_kwargs['scale_x'] * (bounds[0][1] - bounds[0][0]) + bounds[0][0]
            initial_guess_kwargs['scale_y'] = initial_guess_kwargs['scale_y'] * (bounds[1][1] - bounds[1][0]) + bounds[1][0]
            initial_guess_kwargs['shear_radians'] = initial_guess_kwargs['shear_radians'] * (bounds[2][1] - bounds[2][0]) + bounds[2][0]
            initial_guess_kwargs['rotate_radians'] = initial_guess_kwargs['rotate_radians'] * (bounds[3][1] - bounds[3][0]) + bounds[3][0]
            initial_guess_kwargs['offset_x'] = initial_guess_kwargs['offset_x'] * (bounds[4][1] - bounds[4][0]) + bounds[4][0]
            initial_guess_kwargs['offset_y'] = initial_guess_kwargs['offset_y'] * (bounds[5][1] - bounds[5][0]) + bounds[5][0]
        initial_guess_kwargs['scale_x'] = 0.5 * (initial_guess[0] + initial_guess_kwargs['scale_x'])
        initial_guess_kwargs['scale_y'] = 0.5 * (initial_guess[1] + initial_guess_kwargs['scale_y'])
        initial_guess_kwargs['shear_radians'] = 0.5 * (initial_guess[2] + initial_guess_kwargs['shear_radians'])
        initial_guess_kwargs['rotate_radians'] = 0.5 * (initial_guess[3] + initial_guess_kwargs['rotate_radians'])
        initial_guess_kwargs['offset_x'] = 0.5 * (initial_guess[4] + initial_guess_kwargs['offset_x'])
        initial_guess_kwargs['offset_y'] = 0.5 * (initial_guess[5] + initial_guess_kwargs['offset_y'])
        return initial_guess_kwargs
    initial_guess_kwargs = {}
    
    count_down = False
    def _i_to_t(i):
        t = min(num_images-1, max(0, i))
        if count_down:
            t = num_images - 1 - t
        return t
    
    # Track worst, average and best similarity measures.
    sm_worst_list_list = []
    sm_avg_list_list = []
    sm_best_list_list = []
    
    # Outer (pyramid) loop
    for n in range(num_levels):
        power_of_2 = num_levels - 1 - n
        print_time("power_of_2 = " + str(power_of_2), start_time)
        (new_height, new_width) = (height_resized//(2**power_of_2), width_resized//(2**power_of_2))
        if new_height < 2 or new_width < 2:
            continue
            
        # Downsample reference image if necessary.
        arr_ref_downsampled = arr_ref_resized
        if power_of_2 > 0:
            arr_ref_downsampled = skimage.transform.resize(arr_ref_resized, (new_height, new_width), mode='reflect', anti_aliasing=True)
            
        num_not_improved = 0
        num_passes = 0
        not_improved_counts = np.zeros(num_images)
        most_significant_improvement = 1
        
        #print_messages = power_of_2 == 0
        #print_messages = False
        print_messages = True
    
        # Track worst, average and best similarity measures.
        sm_worst_list = []
        sm_best_list = []
        sm_avg_list = []
            
        while num_not_improved < (1-improvement_threshold)*num_images and num_passes < max_num_passes and most_significant_improvement > significant_improvement:
            num_not_improved = num_images
            if continue_until_no_improvement:
                num_not_improved = 0
            if use_normalised_average_as_reference:
                print_time("Updating reference image...", start_time)
                signal_transformed = apply_affine_params_to_signal(signal_in, params)
                arr_ref = normalised_average_of_signal(signal_transformed)
                print("Reference image updated.")
                arr_ref_resized = arr_ref
                if resize_image:
                    # Resize reference image to conform to max_image_dimension.
                    arr_ref_resized = skimage.transform.resize(arr_ref, (height_resized, width_resized), mode='reflect', anti_aliasing=True)
                plt.matshow(np.hstack((arr_ref_resized, arr_ref_resized_ns)))
                arr_ref_resized_ns = get_neighbour_similarity(arr_ref_resized)
                print("Reference image neighbour similarity updated.")
                # Downsample reference image if necessary.
                arr_ref_downsampled = arr_ref_resized
                if power_of_2 > 0:
                    arr_ref_downsampled = skimage.transform.resize(arr_ref_resized, (new_height, new_width), mode='reflect', anti_aliasing=True)
                    
            most_significant_improvement = 0
            sm_worst = 1
            sm_best = 0
            sm_total = 0
            for i in range(num_images):
                t = _i_to_t(i)
                t_prev = _i_to_t(i-1)
                if print_messages:
                    print_time("Estimating affine parameters for frame " + str(t+1) + " of " + str(num_images), start_time)

                # Resize moving image to conform to max_image_dimension.
                image_resized = signal_in.data[t]
                if resize_image:
                    image_resized = skimage.transform.resize(signal_in.data[t], (height_resized, width_resized), mode='reflect', anti_aliasing=True)

                # If reusing estimates is unlikely to help, introduce some randomness to the initial guess.
                used_randomisation = False
                if reuse_estimates and not_improved_counts[t_prev] > 1:
                    if not_improved_counts[t] < 5:
                        if print_messages:
                            print("Randomising initial guess.")
                        initial_guess_kwargs = _get_random_initial_guess_kwargs(params[t])
                        used_randomisation = True
                    else:
                        not_improved_counts[t] += 1
                        num_not_improved += 1
                        print("Frame " + str(t+1) + " will be skipped.")
                        if print_messages:
                            print("num_not_improved = " + str(num_not_improved))
                        continue
                        
                    
                """if continue_until_no_improvement:
                    not_improved_counts[t] += 1
                    num_not_improved += 1
                    if print_messages:
                        print("num_not_improved = " + str(num_not_improved))"""

                # Save current best estimate and allocate space for a new estimate.
                old_params = params[t]
                new_params = np.empty(6, dtype=float)
                
                # Downsample moving image if necessary.
                arr_moving_downsampled = image_resized
                if power_of_2 > 0:
                    arr_moving_downsampled = skimage.transform.resize(image_resized, (new_height, new_width), mode='reflect', anti_aliasing=True)
                    
                # Estimate parameters
                if reg_args is None:
                    new_params = registration_method(arr_moving_downsampled, arr_ref_downsampled, **initial_guess_kwargs, **reg_kwargs)
                else:
                    new_params = registration_method(arr_moving_downsampled, arr_ref_downsampled, *reg_args, **initial_guess_kwargs, **reg_kwargs)

                # Computed offsets must be scaled up
                if power_of_2 > 0:
                    new_params[4] *= height_resized/new_height
                    new_params[5] *= width_resized/new_width
                
                # Compare similarity measure before and after
                #sm_old = similarity_measure_after_transform(arr_ref_resized, image_resized, old_params)
                #sm_new = similarity_measure_after_transform(arr_ref_resized, image_resized, new_params)
                sm_old = similarity_measure_using_neighbour_similarity(image_resized, arr_ref_resized, arr_ref_resized_ns, old_params, debug=False, max_groups=6)
                sm_new = similarity_measure_using_neighbour_similarity(image_resized, arr_ref_resized, arr_ref_resized_ns, new_params, debug=False, max_groups=6)
                if print_messages:
                    print("Before: sm_old = " + str(sm_old) + ", old_params = " + str(old_params))
                    print("After:  sm_new = " + str(sm_new) + ", new_params = " + str(new_params))

                # Save best result
                if sm_new > sm_old:
                    if print_messages:
                        print("***New parameters will be saved.***")
                        if used_randomisation:
                            print("***Randomisation helped!***")
                    params[t] = new_params
                    not_improved_counts[t] = 0
                    improvement = (sm_new - sm_old)/sm_old
                    if improvement > most_significant_improvement:
                        most_significant_improvement = improvement
                        if print_messages:
                            print("Most significant improvement: " + str(most_significant_improvement * 100) + "%")
                elif continue_until_no_improvement:
                    not_improved_counts[t] += 1
                    num_not_improved += 1
                    if print_messages:
                        print("num_not_improved = " + str(num_not_improved))
                
                # Update sm_worst, sm_best and sm_total
                sm = max(sm_old, sm_new)
                sm_worst = min(sm_worst, sm)
                sm_best = max(sm_best, sm)
                sm_total += sm

                if reuse_estimates:
                    initial_guess_kwargs = _get_initial_guess_kwargs(params[t])
                    #print(initial_guess_kwargs)
            count_down = not count_down
            num_passes += 1
            sm_worst_list.append(sm_worst)
            sm_best_list.append(sm_best)
            sm_avg_list.append(sm_total/num_images)
            if continue_until_no_improvement:
                print_time("", start_time)
                print("num_passes = " + str(num_passes))
                #print("num_not_improved = " + str(num_not_improved))
                print("Estimates improved: " + str(num_images - num_not_improved) + "/" + str(num_images))
                print("Most significant improvement: " + str(most_significant_improvement * 100) + "%")
        sm_worst_list_list.append(sm_worst_list)
        sm_best_list_list.append(sm_best_list)
        sm_avg_list_list.append(sm_avg_list)
    # Computed offsets must be scaled up
    if resize_image:
        params *= np.array([[1, 1, 1, 1, height/height_resized, width/width_resized]])
    for i in range(len(sm_avg_list_list)):
        print(sm_avg_list_list[i], sm_best_list_list[i], sm_worst_list_list[i])
        plt.plot(sm_avg_list_list[i])
        plt.plot(sm_worst_list_list[i])
        plt.plot(sm_best_list_list[i])
        plt.show()
    print_time("DONE", start_time)
    return params


def pyramid_affine_signal_params(signal_in: hs.signals.Signal2D, arr_ref=None, num_levels=3, registration_method=optimise_affine_by_differential_evolution, reg_args=None, reg_kwargs={}):
    '''
    Applies `pyramid_affine` to an entire image stack. Returns an array of parameter sets, one per image.
    '''
    num_images = signal_in.data.shape[0]
    params = np.empty((num_images, 6))
    if arr_ref is None:
        mi_max_index = highest_mutual_information_index(signal_in)
        arr_ref = signal_in.data[mi_max_index]
    for t in range(num_images):
        print("Estimating affine parameters for frame " + str(t+1) + " of " + str(num_images))
        params[t] = pyramid_affine(signal_in.data[t], arr_ref, num_levels=num_levels, registration_method=registration_method, reg_args=reg_args, reg_kwargs=reg_kwargs)
    return params


def faster_pyramid_affine_signal_params(signal_in: hs.signals.Signal2D, arr_ref=None, interpolate=True, reuse_estimates=True, max_num_images=20, max_image_dimension=64, num_levels=3, polynomial_degree=3, registration_method=optimise_affine_by_differential_evolution, reg_args=None, reg_kwargs={}):
    # Ensure a reference image is set.
    mi_max_index = -1
    mi_max = 0
    if arr_ref is None:
        print("Obtaining arr_ref...")
        mi_max_index = highest_mutual_information_index(signal_in)
        arr_ref = signal_in.data[mi_max_index]
        mi_max = im.similarity_measure(normalised_image(arr_ref), normalised_image(arr_ref), measure="NMI")
        print("arr_ref obtained (frame " + str(mi_max_index+1) + " of " + str(num_images) + ").")
    assert len(signal_in.data.shape) == 3
    (num_images, height, width) = signal_in.data.shape
    # If interpolation does not take place, num_images_inspected must equal num_images.
    num_images_inspected = num_images
    if interpolate:
        num_images_inspected = min(num_images, max_num_images)
    # Ensure the maximum dimension of images to be processed is no greater than max_image_dimension.
    height_resized = height
    width_resized = width
    arr_ref_resized = arr_ref
    resize_image = height > max_image_dimension or width > max_image_dimension
    if resize_image:
        if height > width:
            height_resized = max_image_dimension
            width_resized = int(max_image_dimension * width / height)
        else:
            width_resized = max_image_dimension
            height_resized = int(max_image_dimension * height / width)
        print("height_resized = " + str(height_resized) + ", width_resized = " + str(width_resized))
        # Resize reference image to conform to max_image_dimension.
        arr_ref_resized = skimage.transform.resize(arr_ref, (height_resized, width_resized), mode='reflect', anti_aliasing=True)
    # Select the images for which the parameters will be estimated.
    image_indices = np.arange(num_images_inspected, dtype=float)
    if num_images_inspected < num_images:
        image_indices *= num_images/num_images_inspected
    params = np.array([1, 1, 0, 0, 0, 0], dtype=float) * np.ones((num_images, 1))
    params_found_explicitly = np.array([1, 1, 0, 0, 0, 0], dtype=float) * np.ones((num_images_inspected, 1))
    similarity_measures_for_params_found_explicitly = np.zeros(num_images_inspected, dtype=float)
    # Outer (pyramid) loop
    for n in range(num_levels):
        power_of_2 = num_levels - 1 - n
        print("power_of_2 = " + str(power_of_2))
        (new_height, new_width) = (height_resized//(2**power_of_2), width_resized//(2**power_of_2))
        if new_height < 2 or new_width < 2:
            continue
        # Downsample reference image.
        arr_ref_downsampled = arr_ref_resized
        if power_of_2 > 0:
            arr_ref_downsampled = skimage.transform.resize(arr_ref_resized, (new_height, new_width), mode='reflect', anti_aliasing=True)
        initial_guess_kwargs_prev = {
            'scale_x': 1,
            'scale_y': 1,
            'shear_radians': 0,
            'rotate_radians': 0,
            'offset_x': 0,
            'offset_y': 0
        }
        # Inner (parameter estimation) loop
        for j in range(num_images_inspected):
            i = j
            if n%2 == 1 and reuse_estimates:
                i = num_images_inspected-1-j
            t = int(image_indices[i])
            print("Estimating affine parameters for frame " + str(t+1) + " of " + str(num_images))
            image_resized = signal_in.data[t]
            # Resize moving image to conform to max_image_dimension.
            if resize_image:
                image_resized = skimage.transform.resize(signal_in.data[t], (height_resized, width_resized), mode='reflect', anti_aliasing=True)
            old_params = params[t]
            new_params = np.empty(6, dtype=float)
            # Transform and downsample moving image.
            arr_moving_downsampled = transform_using_values(image_resized, old_params)
            if power_of_2 > 0:
                arr_moving_downsampled = skimage.transform.resize(arr_moving_downsampled, (new_height, new_width), mode='reflect', anti_aliasing=True)
            # Initial guess for new_params produces no change, such that combined_params == old_params.
            initial_guess_kwargs = initial_guess_kwargs_prev
            if not reuse_estimates:
                initial_guess_kwargs = {
                    'scale_x': 1,
                    'scale_y': 1,
                    'shear_radians': 0,
                    'rotate_radians': 0,
                    'offset_x': 0,
                    'offset_y': 0
                }
            # Estimate new_params
            if reg_args is None:
                new_params = registration_method(arr_moving_downsampled, arr_ref_downsampled, **initial_guess_kwargs, **reg_kwargs)
            else:
                new_params = registration_method(arr_moving_downsampled, arr_ref_downsampled, *reg_args, **initial_guess_kwargs, **reg_kwargs)
            initial_guess_kwargs_prev = {
                'scale_x': 1,
                'scale_y': 1,
                'shear_radians': 0,
                'rotate_radians': 0,
                'offset_x': 0,
                'offset_y': 0
            }
            initial_guess_kwargs_prev_temp = {
                'scale_x': new_params[0],
                'scale_y': new_params[1],
                'shear_radians': new_params[2],
                'rotate_radians': new_params[3],
                'offset_x': new_params[4],
                'offset_y': new_params[5]
            }
            # Computed offsets must be scaled up
            if power_of_2 > 0:
                new_params[4] *= height_resized/new_height
                new_params[5] *= width_resized/new_width
            # Combine old_params and new_params
            combined_params = combine_affine_params(old_params, new_params)
            # Compare similarity measure before and after
            sm_old = similarity_measure_after_transform(arr_ref_resized, image_resized, old_params)
            sm_new = similarity_measure_after_transform(arr_ref_resized, image_resized, combined_params)
            print("Before: sm_old = " + str(sm_old) + ", old_params = " + str(old_params))
            print("After: sm_new = " + str(sm_new) + ", combined_params = " + str(combined_params))
            # Save best result
            similarity_measures_for_params_found_explicitly[i] = max(sm_old, sm_new)
            if sm_new > sm_old:
                print("New parameters will be saved.")
                params_found_explicitly[i] = combined_params
                initial_guess_kwargs_prev = initial_guess_kwargs_prev_temp
        if mi_max_index >= 0 and interpolate and not reuse_estimates:
            print("Adding most representative image to sample...")
            mi_min_index = np.argmin(similarity_measures_for_params_found_explicitly)
            similarity_measures_for_params_found_explicitly[mi_min_index] = mi_max
            image_indices[mi_min_index] = mi_max_index
            params_found_explicitly[mi_min_index] = np.array([1, 1, 0, 0, 0, 0])
        # Interpolate
        if interpolate:
            for i in range(6):
                coeffs = np.polyfit(image_indices, params_found_explicitly.T[i], polynomial_degree, w=similarity_measures_for_params_found_explicitly)
                print("Parameter " + str(i) + ": coeffs = " + str(coeffs))
                params.T[i] = np.poly1d(coeffs)(np.arange(num_images))
            image_indices = (image_indices + 1) % num_images
        else:
            params = params_found_explicitly * 1
    # Computed offsets must be scaled up
    if resize_image:
        params *= np.array([[1, 1, 1, 1, height/height_resized, width/width_resized]])
    return params