import cv2
import hyperspy.api as hs
import math
import mutual_information as minf
import numpy as np
import scipy
import SimpleITK as sitk
import skimage


def get_deltas(signal_in: hs.signals.Signal2D, absolute=False, normalised=True):
    '''
    Difference between previous and current intensity of each pixel.
    '''
    new_shape = list(signal_in.data.shape)
    new_shape[0] -= 1
    arr_out = np.empty(tuple(new_shape))
    for i in range(0, len(arr_out)):
        arr_out[i] = np.subtract(signal_in.data[i+1], signal_in.data[i])
        if absolute:
            arr_out[i] = abs(arr_out[i])
    if normalised:
        arr_out -= arr_out.min()
        arr_out *= (1/arr_out.max())
    return hs.signals.Signal2D(arr_out)


def get_signal_average_difference(signal_in: hs.signals.Signal2D):
    '''
    Average intensity difference across whole image from current pixel.
    '''
    data_shape = signal_in.data.shape
    difference_data = np.empty(data_shape)
    for t in range(data_shape[0]):
        for i in range(data_shape[1]):
            for j in range(data_shape[2]):
                difference_data[t][i][j] = (abs(signal_in.data[t] - signal_in.data[t][i][j])).mean()
    return hs.signals.Signal2D(difference_data)


def get_signal_average_rank_difference(signal_in: hs.signals.Signal2D, weighted=False):
    '''
    Average intensity rank difference across whole image from current pixel.
    If `weighted` is True, the differences are each multiplied by the mean value of the absolute difference between the current image and the intensity of the current pixel.
    '''
    data_shape = signal_in.data.shape
    difference_data = np.empty(data_shape)
    for t in range(data_shape[0]):
        data_t = signal_in.data[t]
        argsort_t = data_t.reshape(1, data_t.size).argsort()[0]
        ranks_t = np.empty_like(argsort_t)
        ranks_t[argsort_t] = np.arange(len(argsort_t))
        ranks_t = ranks_t.reshape(data_t.shape)
        for i in range(data_shape[1]):
            for j in range(data_shape[2]):
                difference_data[t][i][j] = (abs(ranks_t - ranks_t[i][j])).mean()
                if weighted:
                    difference_data[t][i][j] *= (abs(signal_in.data[t] - signal_in.data[t][i][j])).mean()
    return hs.signals.Signal2D(difference_data)


def get_neighbour_similarity(arr_in: np.array, exponent=2):
    '''
    Similarity to neighbouring pixels relative to whole image.
    
    The purpose of this method is to obtain a measure of confidence in the intensity value of each pixel. High confidence values are given to pixels that are similar to their neighbours and dissimilar from the average pixel.

The optional `exponent` parameter controls the strength with which distance between pixels is penalised. If `exponent` is large (say, greater than 2), only pixels very close to the pixel under consideration may be considered 'neighbours'.

Ultimately, this or a similar method may be used in conjunction with an optical flow algorithm such as Horn-Schunck to determine the strength of the smoothness constraint at each point.

The current implementation of this method is quite crude, but nonetheless produces qualitatively sensible results.
    '''
    data_shape = arr_in.data.shape
    similarity_data = np.empty(data_shape)
    normalised_data = (arr_in - arr_in.min())/(arr_in.max() - arr_in.min())
    mean_intensity = np.mean(normalised_data)
    for i in range(data_shape[0]):
        for j in range(data_shape[1]):
            current_intensity = normalised_data[i][j]
            similarities = 1 - abs(normalised_data - current_intensity)
            weights = np.fromfunction(lambda i2, j2: (abs(i2 - i)**2 + abs(j2 - j)**2)**(-exponent/2), data_shape)
            weights[i][j] = 0
            weights_sum = np.sum(weights)
            weighted_similarity_sum = np.sum(similarities * weights)
            weighted_intensity_sum = np.sum(normalised_data * weights)
            absolute_similarity = weighted_similarity_sum / weights_sum
            local_intensity = weighted_intensity_sum / weights_sum
            local_abnormality = abs(local_intensity - mean_intensity)
            relative_similarity = absolute_similarity * local_abnormality
            similarity_data[i][j] = relative_similarity
    return similarity_data


def get_signal_neighbour_similarity(signal_in: hs.signals.Signal2D, exponent=2):
    '''
    Applies get_neighbour_similarity to all the images in a stack.
    Returns a new image stack.
    '''
    data_shape = signal_in.data.shape
    similarity_data = np.empty(data_shape)
    for i in range(data_shape[0]):
        similarity_data[i] = get_neighbour_similarity(signal_in.data[i], exponent=exponent)
    return hs.signals.Signal2D(similarity_data)


def apply_displacement_field(displacements: np.array, arr_2d_in: np.array, debug=False):
    '''
    Applies a displacement field to an image.
    
    `displacements` (numpy array of shape (2, height, width)) is a pair of displacement fields: the first is vertical and the second is horizontal.
    `arr_2d_in` (numpy array of shape (height, width)) is the image to which the displacement field will be applied.

    A displaced image (numpy array of shape (height, width)) is returned.
    '''
    data_shape = arr_2d_in.shape
    arr_2d_out = np.zeros(data_shape)
    sum_of_weights = np.zeros(data_shape)
    it = np.nditer(arr_2d_in, flags=['multi_index'])
    while not it.finished:
        (i,j) = it.multi_index
        #(j,i) = it.multi_index
        if debug:
            print("(i,j) = " + str((i,j)))
            print("arr_2d_in[" + str(i) + "][" + str(j) + "] = " + str(arr_2d_in[i][j]))
        u = displacements[0][i][j]
        v = displacements[1][i][j]
        if debug:
            print("(u,v) = " + str((u,v)))
        i2_floor = math.floor(i+u)
        j2_floor = math.floor(j+v)
        i2_part = i + u - i2_floor
        j2_part = j + v - j2_floor
        if i2_floor >= 0 and j2_floor >= 0 and i2_floor < data_shape[0] and j2_floor < data_shape[1]:
            weight = (1 - i2_part) * (1 - j2_part)
            arr_2d_out[i2_floor][j2_floor] += weight * arr_2d_in[i][j]
            sum_of_weights[i2_floor][j2_floor] += weight
            if debug:
                print("arr_2d_out[" + str(i2_floor) + "][" + str(j2_floor) + "] += " + str(weight) + " * " + str(arr_2d_in[i][j]))
                print("sum_of_weights[" + str(i2_floor) + "][" + str(j2_floor) + "] += " + str(weight))
        if i2_floor + 1 >= 0 and j2_floor >= 0 and i2_floor + 1 < data_shape[0] and j2_floor < data_shape[1]:
            weight = i2_part * (1 - j2_part)
            arr_2d_out[i2_floor + 1][j2_floor] += weight * arr_2d_in[i][j]
            sum_of_weights[i2_floor + 1][j2_floor] += weight
            if debug:
                print("arr_2d_out[" + str(i2_floor + 1) + "][" + str(j2_floor) + "] += " + str(weight) + " * " + str(arr_2d_in[i][j]))
                print("sum_of_weights[" + str(i2_floor + 1) + "][" + str(j2_floor) + "] += " + str(weight))
        if i2_floor >= 0 and j2_floor + 1 >= 0 and i2_floor < data_shape[0] and j2_floor + 1 < data_shape[1]:
            weight = (1 - i2_part) * j2_part
            arr_2d_out[i2_floor][j2_floor + 1] += weight * arr_2d_in[i][j]
            sum_of_weights[i2_floor][j2_floor + 1] += weight
            if debug:
                print("arr_2d_out[" + str(i2_floor) + "][" + str(j2_floor + 1) + "] += " + str(weight) + " * " + str(arr_2d_in[i][j]))
                print("sum_of_weights[" + str(i2_floor) + "][" + str(j2_floor + 1) + "] += " + str(weight))
        if i2_floor + 1 >= 0 and j2_floor + 1 >= 0 and i2_floor + 1 < data_shape[0] and j2_floor + 1 < data_shape[1]:
            weight = i2_part * j2_part
            arr_2d_out[i2_floor + 1][j2_floor + 1] += weight * arr_2d_in[i][j]
            sum_of_weights[i2_floor + 1][j2_floor + 1] += weight
            if debug:
                print("arr_2d_out[" + str(i2_floor + 1) + "][" + str(j2_floor + 1) + "] += " + str(weight) + " * " + str(arr_2d_in[i][j]))
                print("sum_of_weights[" + str(i2_floor + 1) + "][" + str(j2_floor + 1) + "] += " + str(weight))
        it.iternext()
    cleaned_sum_of_weights = np.where(sum_of_weights > 0, sum_of_weights, 1)
    arr_2d_out /= cleaned_sum_of_weights
    mean_over_zeroes = np.where(sum_of_weights > 0, 0, arr_2d_in.mean())
    arr_2d_out += mean_over_zeroes
    #return arr_2d_out
    return [arr_2d_out, sum_of_weights]


def apply_displacement_field_sitk(displacements: np.array, arr_2d_in: np.array, rearrange=True):
    '''
    Applies a displacement field using SimpleITK.
    '''
    displacements_rearranged = displacements
    if rearrange:
        displacements_uv_swapped = np.array([-displacements[1], -displacements[0]])
        displacements_rearranged = np.moveaxis(displacements_uv_swapped, 0, 2)
        
    outTx = sitk.DisplacementFieldTransform( sitk.GetImageFromArray(displacements_rearranged, isVector=True) )

    moving = sitk.GetImageFromArray(np.float32(arr_2d_in))

    resampler = sitk.ResampleImageFilter()
    resampler.SetReferenceImage(sitk.GetImageFromArray(arr_2d_in));
    resampler.SetInterpolator(sitk.sitkLinear)
    resampler.SetDefaultPixelValue(arr_2d_in.mean())
    resampler.SetTransform(outTx)

    shifted = sitk.GetArrayFromImage(resampler.Execute(moving))
    return shifted


def make_capital_A(shape: tuple):
    '''
    Generates an image in the shape of a capital A (for testing).
    '''
    (height, width) = shape
    slope = 0.4375 * width / height
    arr_i = np.arange(height).reshape(height, 1)
    arr_j = np.arange(width).reshape(1, width)
    # Exclude top and bottom
    # cond_1.shape is (height, 1)
    cond_1 = abs(arr_i - 0.5 * height) < 0.4 * height
    # Exclude the outside of each 'leg'
    # cond_2.shape is (height, width)
    cond_2 = abs(arr_j - 0.5 * width) < 0.00625 * width + slope * arr_i
    # Make middle bar
    # cond_3.shape is (height, 1)
    cond_3 = abs(arr_i - 0.6 * height) < 0.05 * height
    # Cut out holes
    # cond_4.shape is (height, width)
    cond_4 = abs(arr_j - 0.5 * width) > slope * arr_i - 0.09375 * width
    # Combine conditions
    cond = cond_1 & cond_2 & (cond_3 | cond_4)
    return cond.astype(np.float64)


def horn_schunck(signal_in: hs.signals.Signal2D, alpha_squared: float, num_iterations: int, use_average=False):
    '''
    Horn-Schunck method for optical flow.
    Outputs a series of (vertical, horizontal) displacement field pairs obtained from a corresponding series of 2D images.
    
    The larger `alpha_squared` is, the more strongly departure from smoothness is penalised relative to rate of change of brightness.
    When `alpha_squared` = 1, both kinds of error are weighted equally.
    When `use_average` is True, each displacement field has the same (u,v) values for all pixels. The value used is the mean of the individual values.
    '''
    num_images = signal_in.data.shape[0]
    image_shape = (signal_in.data.shape[1], signal_in.data.shape[2])
    # Estimate derivatives of intensity wrt x, y and t
    I_x = ndi.filters.convolve(signal_in.data, np.array([[[-1,-1],[1,1]],[[-1,-1],[1,1]]])*0.25)
    I_y = ndi.filters.convolve(signal_in.data, np.array([[[-1,1],[-1,1]],[[-1,1],[-1,1]]])*0.25)
    I_t = ndi.filters.convolve(signal_in.data, np.array([[[-1,-1],[-1,-1]],[[1,1],[1,1]]])*0.25)               
    u = np.zeros(signal_in.data.shape)
    v = np.zeros(signal_in.data.shape)
    kernel = np.array([[1,2,1],[2,0,2],[1,2,1]])/12
    for t in range(num_images):
        u_t = np.zeros(signal_in.data[t].shape)
        v_t = np.zeros(signal_in.data[t].shape)
        for _ in range(num_iterations):
            u_mean = ndi.filters.convolve(u_t, kernel)
            v_mean = ndi.filters.convolve(v_t, kernel)
            I_coefficient = (I_x[t] * u_mean + I_y[t] * v_mean + I_t[t]) / (alpha_squared + I_x[t]**2 + I_y[t]**2)
            u_t = u_mean - I_x[t] * I_coefficient
            v_t = v_mean - I_y[t] * I_coefficient
            if (np.array_equal(u_t, u[t]) and np.array_equal(v_t, v[t])):
                break
            u[t] = u_t
            v[t] = v_t
        if (use_average):
            u_avg = u[t].mean()
            v_avg = v[t].mean()
            u[t] = np.ones(u[t].shape) * u_avg
            v[t] = np.ones(v[t].shape) * v_avg
    return (u,v)


def displace_signal_using_horn_schunck_repeatedly(signal_in: hs.signals.Signal2D, alpha_squared=1.0, num_iterations_outer=5, num_iterations_inner=50, use_average=False, use_sitk=True):
    '''
    Applies the Horn-Schunck optical flow method repeatedly to an image stack. The number of times Horn-Schunck is repeated is determined by the num_iterations_outer parameter. The num_iterations_inner parameter of this method corresponds to the num_iterations parameter of horn_schunck.
    '''
    signal_out = signal_in
    for _ in range(num_iterations_outer):
        [us, vs] = horn_schunck(signal_out, alpha_squared, num_iterations_inner, use_average)
        data_out = np.empty(signal_in.data.shape)
        for t in range(signal_in.data.shape[0]):
            data_out[t] = signal_out.data[t]
            if (t < signal_in.data.shape[0] - 1):
                for t2 in range(t, signal_in.data.shape[0]-1):
                    if use_sitk:
                        data_out[t] = apply_displacement_field_sitk(np.array([us[t2], vs[t2]]), data_out[t])
                    else:
                        [data_out[t], _] = apply_displacement_field(np.array([us[t2], vs[t2]]), data_out[t])
        signal_out = hs.signals.Signal2D(data_out)
    return signal_out


def nonrigid(im_stack, demons_it = 20, filter_size = 5.0, max_it = 3, default_pixel_value = 100):
    '''
    Nonrigid optical flow method based on SmartAlign.
    Author: qzo13262
    Modified by fqj69741
    '''
    
    demons = sitk.DemonsRegistrationFilter()
    demons.SetNumberOfIterations( demons_it )
    # Standard deviation for Gaussian smoothing of displacement field
    demons.SetStandardDeviations( filter_size )
    
    for j in range(max_it):
        #Get stack average
        # This will be treated as a reference image. Each 'moving' image will be transformed to align with av_im.
        # The reference image is recalculated max_it times.
        av_im = sitk.GetImageFromArray(np.float32(sum(im_stack)/len(im_stack))) #Faster than numpy.mean for small arrays?
        
        #Get gradient images of stack average
        '''avgrad = np.gradient(av_im)
        normavgradx = avgrad[0]/np.sqrt(np.square(avgrad[0])+np.square(avgrad[1]))
        normavgrady = avgrad[1]/np.sqrt(np.square(avgrad[0])+np.square(avgrad[1]))'''
        
        out_stack = []
        
        for i in range(len(im_stack)):
            
            #print(im_stack[i])
            
            # Load up the next 'moving' image from the image stack.
            moving = sitk.GetImageFromArray(np.float32(im_stack[i]))
            
            # Use the 'demons' method to obtain a displacement field that transforms the moving image
            # to align with the reference image.
            displacementField = demons.Execute( av_im, moving )
            
            dispfield = sitk.GetArrayFromImage(displacementField)
            #print(dispfield)
            
            '''if dispconstraint == 'rwo-locked':
                disp_contrained = '''
            
            # Convert displacementField to a transformation that can be appplied to the moving image via a ResampleImageFilter.
            outTx = sitk.DisplacementFieldTransform( displacementField )
            
            resampler = sitk.ResampleImageFilter()
            resampler.SetReferenceImage(av_im);
            resampler.SetInterpolator(sitk.sitkLinear)
            resampler.SetDefaultPixelValue(default_pixel_value)
            resampler.SetTransform(outTx)
            
            # Append the transformed image to out_stack.
            out_stack.append(sitk.GetArrayFromImage(resampler.Execute(moving)))
            
            '''grad = np.gradient(im_stack[i])
            normgradx = grad[0]/np.sqrt(np.square(grad[0])+np.square(grad[1]))
            normgrady = grad[1]/np.sqrt(np.square(grad[0])+np.square(grad[1]))
            
            transform_x = (av_im-im_stack[i])*(normgradx+normavgradx)
            transform_y = (av_im-im_stack[i])*(normgrady+normavgrady)
            
            tx_filtered = ndimage.filters.gaussian_filter(transform_x,sigma=10)
            ty_filtered = ndimage.filters.gaussian_filter(transform_y,sigma=10)
            
            t_filtered = np.stack((ty_filtered,tx_filtered))
            
            im_t = transform.warp(im_stack[i],t_filtered)'''
            
        im_stack = out_stack
        
        #dispfield = sitk.GetArrayFromImage(displacementField)
        #print(dispfield)
        
        # dispfield is out of scope here, but its most recent value is retained.
        max_disp = np.max(dispfield)
            
        # Terminate early if no part of the (last) image has been displaced by more than 1 pixel.
        # (Is this wise?)
        #if max_disp < 1.0:
        #    print("NRR stopped after "+str(j)+" iterations.")
        #    break
    
    return(out_stack)


def mutual_information(arr_1: np.array, arr_2: np.array):
    '''
    Mutual information between arr_1[i][j] and arr_2[i][j] for all i, j.
    Returns the mutual information between arr_1 and arr_2.
    '''
    return minf.mutual_information(arr_1.flatten(), arr_2.flatten())


def split_by_mutual_information(signal_in: hs.signals.Signal2D, threshold=0.3):
    '''
    Split a signal into a list of numpy arrays of images. Each numpy array in the list contains a subsection of the original signal. For each image in a subsection, the mutual information between that image and the first image in the subsection must be at least some value given by the `threshold` parameter.
    '''
    list_out = []
    sublist_current = []
    t = 0
    while t < signal_in.data.shape[0] - 1:
        t2_loop_interrupted = False
        sublist_current.append(signal_in.data[t])
        for t2 in range(t+1, signal_in.data.shape[0]):
            mi = mutual_information(signal_in.data[t], signal_in.data[t2])
            if mi < threshold:
                list_out.append(np.array(sublist_current))
                sublist_current = []
                t = t2
                t2_loop_interrupted = True
                break
            else:
                sublist_current.append(signal_in.data[t2])
        if not t2_loop_interrupted:
            break
    return list_out


def highest_mutual_information_index(signal_in: hs.signals.Signal2D, exponent=1):
    '''
    Iterates over an image stack. For each image, the sum of the mutual information between that image and each other image (raised to the power `exponent`) is calculated. The index of the image for which this value is greatest is returned.
    For `exponent` values less than 1, low mutual information pairs are more heavily penalised: the index returned will correspond to an image that is moderately representative of the whole stack.
    For `exponent` values greater than 1, high mutual information pairs are more strongly rewarded: the index returned will correspond to an image that is highly representative of some subset of the stack.
    '''
    mi_max = 0.0
    mi_max_index = 0
    for t in range(signal_in.data.shape[0]):
        mi_total = 0.0
        for t2 in range(signal_in.data.shape[0]):
            if t2 == t:
                continue
            mi_total += mutual_information(signal_in.data[t], signal_in.data[t2])**exponent
        if mi_total > mi_max:
            mi_max = mi_total
            mi_max_index = t
    return mi_max_index


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

def transform_using_values(arr_in: np.array, values: list):
    assert len(values) == 6
    delta_x = -0.5 * arr_in.shape[1]
    delta_y = -0.5 * arr_in.shape[0]
    # Image must be shifted by minus half each dimension, then transformed, then shifted back.
    # This way, rotations and shears will be about the centre of the image rather than the top-left corner.
    a0 = values[0] * math.cos(values[3])
    a1 = -values[1] * math.sin(values[3] + values[2])
    a2 = a0 * delta_x + a1 * delta_y + values[4] - delta_x
    b0 = values[0] * math.sin(values[3])
    b1 = values[1] * math.cos(values[3] + values[2])
    b2 = b0 * delta_x + b1 * delta_y + values[5] - delta_y
    tform = skimage.transform.AffineTransform(matrix=np.array([[a0, a1, a2], [b0, b1, b2], [0, 0, 1]]))
    # For some reason, arr_in must be normalised first and scaled back up afterwards, or this method won't work on matrices with large entries.
    arr_out = skimage.transform.warp(arr_in/arr_in.max(), tform.inverse, cval=arr_in.mean()/arr_in.max())*arr_in.max()
    return arr_out


def shift(arr_in: np.array, offset_x, offset_y):
    return transform_using_values(arr_in, [1, 1, 0, 0, offset_x, offset_y])


def shift_signal(signal_in: hs.signals.Signal2D, shifts: np.array):
    # Shifts are in horizontal, vertical order.
    assert shifts.shape == (signal_in.data.shape[0], 2)
    signal_out = hs.signals.Signal2D(np.empty_like(signal_in.data))
    for t in range(signal_in.data.shape[0]):
        signal_out.data[t] = shift(signal_in.data[t], shifts[t][0], shifts[t][1])
    return signal_out


def correct_shifts_vh(signal_in: hs.signals.Signal2D, shifts: np.array):
    # Shifts are in vertical, horizontal order.
    assert shifts.shape == (signal_in.data.shape[0], 2)
    return shift_signal(signal_in, -np.flip(shifts, 1))
    


def cartesian_to_log_polar(arr_in: np.array):
    '''
    Converts the image represented by arr_in from Cartesian to log-polar coordinates.
    '''
    (height, width) = arr_in.shape
    diagonal_length = np.sqrt(((height/2.0)**2.0)+((width/2.0)**2.0))
    arr_out = cv2.linearPolar(arr_in, (height/2, width/2), diagonal_length, cv2.WARP_FILL_OUTLIERS)
    #arr_out = np.empty_like(arr_in)
    #cv2.logPolar(arr_in, arr_out, (height/2, width/2), diagonal_length, cv2.WARP_FILL_OUTLIERS)
    return arr_out


def log_polar_to_cartesian(arr_in: np.array):
    '''
    Converts the image represented by arr_in from log-polar to Cartesian coordinates.
    '''
    (height, width) = arr_in.shape
    diagonal_length = np.sqrt(((height/2.0)**2.0)+((width/2.0)**2.0))
    arr_out = cv2.linearPolar(arr_in, (height/2, width/2), diagonal_length, cv2.WARP_FILL_OUTLIERS + cv2.WARP_INVERSE_MAP)
    #arr_out = np.empty_like(arr_in)
    #cv2.logPolar(arr_in, arr_out, (height/2, width/2), diagonal_length, cv2.WARP_INVERSE_MAP)
    return arr_out


def log_polar_signal(signal_in: hs.signals.Signal2D):
    '''
    Converts signal_in from Cartesian to log-polar coordinates.
    '''
    signal_out = hs.signals.Signal2D(np.empty_like(signal_in.data))
    for t in range(signal_in.data.shape[0]):
        signal_out.data[t] = cartesian_to_log_polar(signal_in.data[t])
    return signal_out


def cartesian_signal(signal_in: hs.signals.Signal2D):
    '''
    Converts signal_in from Cartesian to log-polar coordinates.
    '''
    signal_out = hs.signals.Signal2D(np.empty_like(signal_in.data))
    for t in range(signal_in.data.shape[0]):
        signal_out.data[t] = log_polar_to_cartesian(signal_in.data[t])
    return signal_out


def resample(im_in, transform, default_value=1.0):
    im_ref = im_in
    interpolator = sitk.sitkLinear
    return sitk.Resample(im_in, im_ref, transform, interpolator, default_value)

def affine(arr_in: np.array, rotate_radians=0.0, scale_x=1.0, scale_y=1.0, shear_x=0.0, shear_y=0.0, offset_x=0.0, offset_y=0.0):
    (height, width) = arr_in.shape
    im_in = sitk.GetImageFromArray(arr_in)
    affine = sitk.AffineTransform(2)
    affine.SetCenter((width/2, height/2))
    # Rotate
    affine.Rotate(axis1=0, axis2=1, angle=rotate_radians)
    # Scale
    affine.Scale((1/scale_x, 1/scale_y))
    # Shear
    affine.Shear(axis1=0, axis2=1, coef=shear_x)
    affine.Shear(axis1=1, axis2=0, coef=shear_y)
    # Translate
    affine.SetTranslation((-offset_x, -offset_y))
    # Resample
    im_out = resample(im_in, affine, default_value=arr_in.mean())
    return sitk.GetArrayFromImage(im_out)

def affine_from_list(arr_in: np.array, values: list):
    assert len(values) == 7
    return affine(
        arr_in, 
        rotate_radians=values[0], 
        scale_x=values[1], 
        scale_y=values[2], 
        shear_x=values[3], 
        shear_y=values[4], 
        offset_x=values[5], 
        offset_y=values[6])

def scale(arr_in: np.array, scale_factor_x, scale_factor_y):
    return transform_using_values(arr_in, [scale_factor_x, scale_factor_y, 0.0, 0.0, 0.0, 0.0])


def rotate(arr_in: np.array, rotate_radians):
    return transform_using_values(arr_in, [1.0, 1.0, 0.0, rotate_radians, 0.0, 0.0])
    


def optimise_scale(arr_moving: np.array, arr_ref: np.array, initial_guess_x=1.0, initial_guess_y=1.0):
    '''
    Uses the Powell local optimisation algorithm to obtain scale factors in the x- and y-directions that maximise the mutual information between `arr_scaled` and `arr_ref`, where `arr_scaled` is the scaled version of `arr_moving`.
    '''
    def inverse_mutual_information_after_scaling(parameters):
        arr_scaled = scale(arr_moving, parameters[0], parameters[1])
        return 1/mutual_information(arr_scaled, arr_ref)
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
        return 1/mutual_information(arr_rotated, arr_ref)
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
    if mutual_information(rotated_1, arr_ref) > mutual_information(rotated_2, arr_ref):
        return result_1
    else:
        return result_2
    


def optimise_affine(arr_moving: np.array, arr_ref: np.array, scale_x=1.0, scale_y=1.0, shear_radians=0.0, rotate_radians=0.0, offset_x=0.0, offset_y=0.0, method='Powell', bounds=None):
    '''
    Uses a local optimisation algorithm to obtain a set of affine transform parameters that maximises the mutual information between `arr_transformed` and `arr_ref`, where `arr_transformed` is the transformed version of `arr_moving`.
    '''
    (height, width) = arr_moving.shape
    if bounds is None and (method == 'L-BFGS-B' or method == 'TNC' or method == 'SLSQP'):
        bounds = [(0.5, 2), (0.5, 2), (-math.pi/3, math.pi/3), (-math.pi/3, math.pi/3), (-height, height), (-width, width)]
    def inverse_mutual_information_after_transform(parameters):
        arr_transformed = transform_using_values(arr_moving, parameters)
        return 1/mutual_information(arr_transformed, arr_ref)
    optimisation_result = scipy.optimize.minimize(inverse_mutual_information_after_transform, [scale_x, scale_y, shear_radians, rotate_radians, offset_x, offset_y], method=method, bounds=bounds)
    if optimisation_result.success:
        return optimisation_result.x
    else:
        raise ValueError(result.message)
    


def optimise_affine_no_shear(arr_moving: np.array, arr_ref: np.array, scale_x=1.0, scale_y=1.0, rotate_radians=0.0, offset_x=0.0, offset_y=0.0, method='Powell', bounds=None):
    '''
    Uses a local optimisation algorithm to obtain a set of affine transform parameters that maximises the mutual information between `arr_transformed` and `arr_ref`, where `arr_transformed` is the transformed version of `arr_moving`. The shear parameter is always zero.
    '''
    (height, width) = arr_moving.shape
    if bounds is None and (method == 'L-BFGS-B' or method == 'TNC' or method == 'SLSQP'):
        bounds = [(0.5, 2), (0.5, 2), (-math.pi/3, math.pi/3), (-height, height), (-width, width)]
    def inverse_mutual_information_after_transform(parameters):
        arr_transformed = transform_using_values(arr_moving, [parameters[0], parameters[1], 0, parameters[2], parameters[3], parameters[4]])
        #arr_transformed = affine_from_list(arr_moving, [parameters[2], parameters[0], parameters[1], 0, 0, parameters[3], parameters[4]])
        return 1/mutual_information(arr_transformed, arr_ref)
    optimisation_result = scipy.optimize.minimize(inverse_mutual_information_after_transform, [scale_x, scale_y, rotate_radians, offset_x, offset_y], method=method, bounds=bounds)
    if optimisation_result.success:
        return optimisation_result.x
    else:
        raise ValueError(result.message)
    


def optimise_scale_and_rotation(arr_moving: np.array, arr_ref: np.array, scale_x=1.0, scale_y=1.0, rotate_radians=0.0, method='Powell', bounds=None):
    '''
    Uses a local optimisation algorithm to obtain a set of affine transform parameters that maximises the mutual information between `arr_transformed` and `arr_ref`, where `arr_transformed` is the transformed version of `arr_moving`. The shear parameter is always zero.
    '''
    if bounds is None and (method == 'L-BFGS-B' or method == 'TNC' or method == 'SLSQP'):
        bounds = [(0.5, 2), (0.5, 2), (-math.pi/3, math.pi/3)]
    def inverse_mutual_information_after_transform(parameters):
        arr_transformed = transform_using_values(arr_moving, [parameters[0], parameters[1], 0, parameters[2], 0, 0])
        #arr_transformed = affine_from_list(arr_moving, [parameters[2], parameters[0], parameters[1], 0, 0, 0, 0])
        return 1/mutual_information(arr_transformed, arr_ref)
    optimisation_result = scipy.optimize.minimize(inverse_mutual_information_after_transform, [scale_x, scale_y, rotate_radians], method=method, bounds=bounds)
    if optimisation_result.success:
        return optimisation_result.x
    else:
        raise ValueError(result.message)
    


def optimise_scale_and_rotation_best_of_two(arr_moving: np.array, arr_ref: np.array, method='Powell', bounds=None):
    '''
    Uses a local optimisation algorithm to obtain a set of affine transform parameters that maximises the mutual information between `arr_transformed` and `arr_ref`, where `arr_transformed` is the transformed version of `arr_moving`. The shear parameter is always zero.
    '''
    if bounds is None and (method == 'L-BFGS-B' or method == 'TNC' or method == 'SLSQP'):
        bounds = [(0.5, 2), (0.5, 2), (-math.pi/3, math.pi/3)]
    result_1 = optimise_scale_and_rotation(arr_moving, arr_ref, rotate_radians=0.2, method=method, bounds=bounds)
    result_2 = optimise_scale_and_rotation(arr_moving, arr_ref, rotate_radians=-0.2, method=method, bounds=bounds)
    transformed_1 = transform_using_values(arr_moving, [result_1[0], result_1[1], 0, result_1[2], 0, 0])
    transformed_2 = transform_using_values(arr_moving, [result_2[0], result_2[1], 0, result_2[2], 0, 0])
    if mutual_information(transformed_1, arr_ref) > mutual_information(transformed_2, arr_ref):
        return result_1
    else:
        return result_2
        


def transform_using_matrix(arr_in: np.array, matrix):
    tform = skimage.transform.AffineTransform(matrix=matrix)
    return skimage.transform.warp(arr_in/arr_in.max(), tform.inverse, cval=arr_in.mean()/arr_in.max())*arr_in.max()


def optimise_affine_by_differential_evolution(arr_moving: np.array, arr_ref: np.array, bounds=None, maxiter=1000):
    def inverse_mutual_information_after_transform(parameters):
        return 1/mutual_information(transform_using_values(arr_moving, parameters), arr_ref)
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


def affine_params_to_matrix(params: list):
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
    matrix_applied_first = affine_params_to_matrix(params_applied_first)
    matrix_applied_second = affine_params_to_matrix(params_applied_second)
    matrix_combined = np.matmul(matrix_applied_second, matrix_applied_first)
    return affine_matrix_to_params(matrix_combined)


def pyramid_affine(arr_moving: np.array, arr_ref: np.array, num_levels=3, registration_method=optimise_affine_by_differential_evolution, reg_args=None):
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
            new_params = registration_method(arr_moving_downsampled, arr_ref_downsampled)
        else:
            new_params = registration_method(arr_moving_downsampled, arr_ref_downsampled, *reg_args)
        if power_of_2 > 0:
            # Computed offsets must be scaled up
            new_params[4] *= height/new_height
            new_params[5] *= width/new_width
        combined_params = combine_affine_params(params, new_params)
        if (mutual_information(transform_using_values(arr_moving, combined_params), arr_ref) > mutual_information(transform_using_values(arr_moving, params), arr_ref)):
            params = combined_params
    return params


def pyramid_scale_and_translation(arr_moving: np.array, arr_ref: np.array, num_levels=3, registration_method=optimise_affine_by_differential_evolution):
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
        if (mutual_information(transform_using_values(arr_moving, combined_params), arr_ref) > mutual_information(transform_using_values(arr_moving, params), arr_ref)):
            params = combined_params
    return params


def scale_and_translation_signal_params(signal_in: hs.signals.Signal2D, num_levels=3, registration_method=optimise_affine_by_differential_evolution):
    num_images = signal_in.data.shape[0]
    params = np.empty((num_images, 6))
    mi_max_index = highest_mutual_information_index(signal_in)
    for t in range(num_images):
        print("Estimating affine parameters for frame " + str(t+1) + " of " + str(num_images))
        params[t] = pyramid_scale_and_translation(signal_in.data[t], signal_in.data[mi_max_index], num_levels=num_levels, registration_method=registration_method)
    return params
    return params


def affine_signal_params(signal_in: hs.signals.Signal2D, num_levels=3, registration_method=optimise_affine_by_differential_evolution):
    num_images = signal_in.data.shape[0]
    params = np.empty((num_images, 6))
    mi_max_index = highest_mutual_information_index(signal_in)
    for t in range(num_images):
        print("Estimating affine parameters for frame " + str(t+1) + " of " + str(num_images))
        params[t] = pyramid_affine(signal_in.data[t], signal_in.data[mi_max_index], num_levels=num_levels, registration_method=registration_method)
    return params


def apply_affine_params_to_signal(signal_in: hs.signals.Signal2D, params: np.array):
    num_images = signal_in.data.shape[0]
    assert len(params.shape) == 2
    assert params.shape[0] == num_images
    assert params.shape[1] == 6
    signal_out = hs.signals.Signal2D(np.empty_like(signal_in.data))
    for t in range(num_images):
        signal_out.data[t] = transform_using_values(signal_in.data[t], params[t])
    return signal_out
