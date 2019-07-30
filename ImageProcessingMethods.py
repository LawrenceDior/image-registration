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
    
    
def mutual_information_old(arr_1: np.array, arr_2: np.array):
    '''
    Mutual information between arr_1[i][j] and arr_2[i][j] for all i, j.
    Returns the mutual information between arr_1 and arr_2.
    Obtained from https://matthew-brett.github.io/teaching/mutual_information.html
    '''
    # TODO: decide how to optimally position bin edges
    hgram, x_edges, y_edges = np.histogram2d(arr_1.ravel(), arr_2.ravel(), bins=(20,20))
    # Convert bin counts to probability values
    pxy = hgram / float(np.sum(hgram)) # joint probability mass function
    px = np.sum(pxy, axis=1) # marginal probability mass function for x over y
    py = np.sum(pxy, axis=0) # marginal probability mass function for y over x
    px_py = px[:, None] * py[None, :] # Broadcast to multiply marginals
    # Now we can do the calculation using the pxy, px_py 2D arrays
    nzs = pxy > 0 # Only non-zero pxy values contribute to the sum
    return np.sum(pxy[nzs] * np.log(pxy[nzs] / px_py[nzs]))


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
    affine.SetCenter((height/2, width/2))
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
    

def scale(arr_in: np.array, scale_factor_x=1.0, scale_factor_y=1.0):
    (height, width) = arr_in.shape
    im_in = sitk.GetImageFromArray(arr_in)
    affine = sitk.AffineTransform(2)
    affine.SetCenter((height/2, width/2))
    # Scale
    affine.Scale((1/scale_factor_x, 1/scale_factor_y))
    # Resample
    im_out = resample(im_in, affine, default_value=arr_in.mean())
    return sitk.GetArrayFromImage(im_out)


def rotate(arr_in: np.array, rotate_radians=0.0):
    (height, width) = arr_in.shape
    im_in = sitk.GetImageFromArray(arr_in)
    affine = sitk.AffineTransform(2)
    affine.SetCenter((height/2, width/2))
    # Scale
    affine.Rotate(axis1=0, axis2=1, angle=rotate_radians)
    # Resample
    im_out = resample(im_in, affine, default_value=arr_in.mean())
    return sitk.GetArrayFromImage(im_out)
    


def optimise_scale(arr_moving: np.array, arr_ref: np.array, initial_guess_x=1.0, initial_guess_y=1.0):
    def inverse_mutual_information_after_scaling(parameters):
        arr_scaled = scale(arr_moving, parameters[0], parameters[1])
        return 1/mutual_information(arr_scaled, arr_ref)
    optimisation_result = scipy.optimize.minimize(inverse_mutual_information_after_scaling, [initial_guess_x, initial_guess_y], method='Nelder-Mead')
    return optimisation_result
    


def optimise_rotation(arr_moving: np.array, arr_ref: np.array, initial_guess_radians=0.1):
    def inverse_mutual_information_after_rotation(parameters):
        arr_rotated = rotate(arr_moving, parameters[0])
        return 1/mutual_information(arr_rotated, arr_ref)
    optimisation_result = scipy.optimize.minimize(inverse_mutual_information_after_rotation, [initial_guess_radians], method='Nelder-Mead')
    return optimisation_result


def optimise_affine(arr_moving: np.array, arr_ref: np.array, rotate_radians=0.1, scale_x=1.0, scale_y=1.0, shear_x=0.0, shear_y=0.0, offset_x=0.0, offset_y=0.0):
    def inverse_mutual_information_after_transform(parameters):
        arr_transformed = affine(arr_moving, rotate_radians=parameters[0], scale_x=parameters[1], scale_y=parameters[2], shear_x=parameters[3], shear_y=parameters[4], offset_x=parameters[5], offset_y=parameters[6])
        return 1/mutual_information(arr_transformed, arr_ref)
    optimisation_result = scipy.optimize.minimize(inverse_mutual_information_after_transform, [rotate_radians, scale_x, scale_y, shear_x, shear_y, offset_x, offset_y], method='Nelder-Mead')
    return optimisation_result


def optimise_affine_v2(arr_moving: np.array, arr_ref: np.array, scale_x=1.0, scale_y=1.0, shear_radians=0.0, rotate_radians=0.0, offset_x=0.0, offset_y=0.0):
    def inverse_mutual_information_after_transform(parameters):
        transform = skimage.transform.AffineTransform(matrix=np.array([[parameters[0], parameters[1], parameters[2]], [parameters[3], parameters[4], parameters[5]], [0, 0, 1]]))
        arr_transformed = skimage.transform.warp(arr_moving, transform.inverse)
        return 1/mutual_information(arr_transformed, arr_ref)
    a0 = scale_x * math.cos(rotate_radians)
    a1 = -scale_y * math.sin(rotate_radians + shear_radians)
    a2 = offset_x
    b0 = scale_x * math.sin(rotate_radians)
    b1 = scale_y * math.cos(rotate_radians + shear_radians)
    b2 = offset_y
    #optimisation_result = scipy.optimize.minimize(inverse_mutual_information_after_transform, [a0, a1, a2, b0, b1, b2], method='Nelder-Mead')
    optimisation_result = scipy.optimize.basinhopping(inverse_mutual_information_after_transform, [a0, a1, a2, b0, b1, b2], niter=1000000, niter_success=5, T=10.0)
    return optimisation_result