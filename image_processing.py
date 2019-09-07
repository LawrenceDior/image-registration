import cv2
import hyperspy.api as hs
import math
import image_metrics as im
import matplotlib.pyplot as plt
import numpy as np
import scipy
import SimpleITK as sitk
import skimage
from time import gmtime, strftime, time


def print_time(msg: str, since: float=0.0):
    print("[" + strftime("%H:%M:%S", gmtime(time()-since)) + "] " + msg)


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


def get_delta_signs(signal_in: hs.signals.Signal2D, normalised=True):
    '''
    Sign of difference between previous and current intensity of each pixel.
    '''
    num_images = signal_in.data.shape[0]
    arr_out = np.zeros(signal_in.data.shape)
    for t in range(num_images):
        index_prev = max(0, t-1)
        index_next = min(t+1, num_images-1)
        image_prev = signal_in.data[index_prev]
        image_next = signal_in.data[index_next]
        if normalised:
            image_prev = normalised_image(image_prev)
            image_next = normalised_image(image_next)
        arr_out[t] = np.sign(np.subtract(image_next, image_prev))
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


def normalised_image(arr_in: np.array):
    '''
    Scale the contents of an array to values between 0 and 1.
    '''
    if arr_in.max() == arr_in.min():
        return np.ones_like(arr_in) * max(0, min(1, arr_in.max()))
    return (arr_in - arr_in.min()) / (arr_in.max() - arr_in.min())


def normalised_average_of_signal(signal_in: hs.signals.Signal2D):
    '''
    Normalise all images in a stack and return the average of the normalised images.
    '''
    assert len(signal_in.data.shape) == 3
    (num_images, height, width) = signal_in.data.shape
    data_normalised_sum = np.zeros((height, width), dtype=float)
    for t in range(num_images):
        data_normalised_sum += normalised_image(signal_in.data[t])
    return data_normalised_sum / num_images
    


def get_neighbour_similarity(arr_in: np.array, exponent=2, print_progress=False):
    '''
    Similarity to neighbouring pixels relative to whole image.
    
    The purpose of this method is to obtain a measure of confidence in the intensity value of each pixel. High confidence values are given to pixels that are similar to their neighbours and dissimilar from the average pixel.

    The optional `exponent` parameter controls the strength with which distance between pixels is penalised. If `exponent` is large (say, greater than 2), only pixels very close to the pixel under consideration may be considered 'neighbours'.

    Ultimately, this or a similar method may be used in conjunction with an optical flow algorithm such as Horn-Schunck to determine the strength of the smoothness constraint at each point.

    The current implementation of this method is quite crude, but nonetheless produces qualitatively sensible results.
    '''
    (h, w) = arr_in.shape
    similarity_data = np.empty(arr_in.shape, dtype=float)
    normalised_data = normalised_image(arr_in)
    mean_intensity = np.mean(normalised_data)
    
    # Construct a matrix of weights such that weights_all.shape = (2*h-1, 2*w-1) and weights_all[i][j] = ((i-h+1)**2 + (j-w+1)**2)**(-exponent/2)
    indices_i = np.arange(2 * h - 1).reshape(2 * h - 1, 1)
    indices_j = np.arange(2 * w - 1).reshape(1, 2 * w - 1)
    term_i = (indices_i - h + 1)**2
    term_j = (indices_j - w + 1)**2
    terms_ij = term_i + term_j
    # Set central element to 1 to avoid dividing by zero
    terms_ij[h-1][w-1] = 1
    weights_all = terms_ij ** (-exponent/2)
    # Set central element to 0
    weights_all[h-1][w-1] = 0
    if print_progress:
        print("weights_all constructed")
    assert weights_all.shape == (2*h-1, 2*w-1)
    
    def get_weights(weights_all, i, j):
        (h_all, w_all) = weights_all.shape
        assert h_all % 2 == 1
        assert w_all % 2 == 1
        h = (h_all+1)//2
        w = (w_all+1)//2
        assert i < h
        assert j < w
        if i == 0:
            if j == 0:
                return weights_all[h-i-1:, w-j-1:]
            else:
                return weights_all[h-i-1:, w-j-1:-j]
        elif j == 0:
            return weights_all[h-i-1:-i, w-j-1:]
        else:
            return weights_all[h-i-1:-i, w-j-1:-j]
    
    for i in range(h):
        if print_progress:
            print("Row " + str(i+1) + " of " + str(h))
        for j in range(w):
            # Extract weights relevant to element i, j
            weights = get_weights(weights_all, i, j)
            assert weights.shape == (h, w)
            # weights_sum: used to keep all pixels on the same scale
            weights_sum = np.sum(weights)
            
            current_intensity = normalised_data[i][j]
            # similarities: pixels similar to i, j have high intensities
            similarities = 1 - abs(normalised_data - current_intensity)
            
            # weighted_similarity_sum: sum of similarity matrix weighted by spatial proximity to i, j
            # local_similarity: same value scaled by 1/weights_sum 
            weighted_similarity_sum = np.sum(similarities * weights)
            local_similarity = weighted_similarity_sum / weights_sum
            
            # weighted_intensity_sum: sum of intensity matrix weighted by spatial proximity to i, j
            # local_intensity: same value scaled by 1/weights_sum
            # local_abnormality: measure of how atypical the local area is compared to the image as a whole
            weighted_intensity_sum = np.sum(normalised_data * weights)
            local_intensity = weighted_intensity_sum / weights_sum
            local_abnormality = abs(local_intensity - mean_intensity)
            
            # relative_similarity: highest for pixels in an atypical region that are typical for that region
            relative_similarity = local_similarity * local_abnormality
            similarity_data[i][j] = relative_similarity
    return similarity_data


def get_neighbour_similarity_approx_v2(arr_in: np.array, feature_length=0, print_progress=False):
    (h, w) = arr_in.shape
    if (feature_length < 1):
        feature_length = max(1, max(h, w)/64)
    #exponent = 2/(1 + math.log2(feature_length))
    exponent = 2
    if print_progress:
        print("feature_length = " + str(feature_length) + ", exponent = " + str(exponent))
    
    h_small = int(h/feature_length)
    w_small = int(w/feature_length)
    if print_progress:
        print("(h, w) = " + str((h, w)) + ", (h_small, w_small) = " + str((h_small, w_small)))
    similarity_data = np.empty(arr_in.shape, dtype=float)
    normalised_data = normalised_image(arr_in)
    mean_intensity = np.mean(normalised_data)
    
    # Construct a matrix of weights such that weights_all.shape = (2*h-1, 2*w-1) and weights_all[i][j] = ((i-h+1)**2 + (j-w+1)**2)**(-exponent/2)
    indices_i = np.arange(2 * h - 1).reshape(2 * h - 1, 1)
    indices_j = np.arange(2 * w - 1).reshape(1, 2 * w - 1)
    term_i = (indices_i - h + 1)**2
    term_j = (indices_j - w + 1)**2
    terms_ij = term_i + term_j
    # Set central element to 1 to avoid dividing by zero
    terms_ij[h-1][w-1] = 1
    weights_all = terms_ij ** (-exponent/2)
    # Set central element to 0
    weights_all[h-1][w-1] = 0
    if print_progress:
        print("weights_all constructed")
    assert weights_all.shape == (2*h-1, 2*w-1)

    def get_local_square(arr, i, j, multiplier=3):
        (h, w) = arr.shape
        delta = int(feature_length * multiplier)
        top_left_i = max(i-delta,0)
        top_left_j = max(j-delta,0)
        bottom_right_i = min(i+1+delta,h)
        bottom_right_j = min(j+1+delta,w)
        return arr[top_left_i:bottom_right_i, top_left_j:bottom_right_j]
    
    def get_weights(weights_all, i, j):
        (h_all, w_all) = weights_all.shape
        assert h_all % 2 == 1
        assert w_all % 2 == 1
        h = (h_all+1)//2
        w = (w_all+1)//2
        assert i < h
        assert j < w
        weights = weights_all[h-i-1:2*h-i-1, w-j-1:2*w-j-1]
        return get_local_square(weights, i, j)
    
    for i in range(h):
        if print_progress:
            print("Row " + str(i+1) + " of " + str(h))
        for j in range(w):
            # Extract weights relevant to element i, j
            weights = get_weights(weights_all, i, j)
            # weights_sum: used to keep all pixels on the same scale
            weights_sum = np.sum(weights)
            
            current_intensity = normalised_data[i][j]
            # similarities: pixels similar to i, j have high intensities
            normalised_data_local = get_local_square(normalised_data, i, j)
            similarities = 1 - abs(normalised_data_local - current_intensity)
            weighted_similarity = similarities * weights
            
            # weighted_similarity_sum: sum of similarity matrix weighted by spatial proximity to i, j
            # local_similarity: same value scaled by 1/weights_sum 
            weighted_similarity_sum = np.sum(weighted_similarity)
            local_similarity = weighted_similarity_sum / weights_sum
            
            # weighted_intensity_sum: sum of intensity matrix weighted by spatial proximity to i, j
            # local_intensity: same value scaled by 1/weights_sum
            # local_abnormality: measure of how atypical the local area is compared to the image as a whole
            weighted_intensity = normalised_data_local * weights
            weighted_intensity_sum = np.sum(weighted_intensity)
            local_intensity = weighted_intensity_sum / weights_sum
            local_abnormality = abs(local_intensity - mean_intensity)
            
            # relative_similarity: highest for pixels in an atypical region that are typical for that region
            relative_similarity = local_similarity * local_abnormality
            similarity_data[i][j] = relative_similarity
            """if j == 50 and print_progress:
                print("weights_sum = " + str(weights_sum) + ", current_intensity = " + str(current_intensity) + ", weighted_similarity_sum = " + str(weighted_similarity_sum) + ", local_similarity = " + str(local_similarity) + ", weighted_intensity_sum = " + str(weighted_intensity_sum) + ", local_intensity = " + str(local_intensity) + ", local_abnormality = " + str(local_abnormality) + ", relative_similarity = " + str(relative_similarity))"""
    return similarity_data


def get_neighbour_similarity_faster(arr_in: np.array, feature_length=0, print_progress=False):
    '''
    Faster version of `get_neighbour_similarity` that uses downsampling. Currently does not work as intended.
    '''
    (h, w) = arr_in.shape
    if (feature_length < 1):
        feature_length = max(1, max(h, w)/64)
    #exponent = 2/(1 + math.log2(feature_length))
    exponent = 2
    if print_progress:
        print("feature_length = " + str(feature_length) + ", exponent = " + str(exponent))
    
    h_small = int(h/feature_length)
    w_small = int(w/feature_length)
    if print_progress:
        print("(h, w) = " + str((h, w)) + ", (h_small, w_small) = " + str((h_small, w_small)))
    arr_downsampled = skimage.transform.resize(arr_in, (h_small, w_small), mode='reflect', anti_aliasing=True)
    
    similarity_data = np.empty(arr_in.shape, dtype=float)
    normalised_data = normalised_image(arr_in)
    mean_intensity = np.mean(normalised_data)
    
    #normalised_data_downsampled = normalised_image(arr_downsampled)
    normalised_data_downsampled = skimage.transform.resize(normalised_data, (h_small, w_small), mode='reflect', anti_aliasing=True)
    
    # Construct a matrix of weights such that weights_all.shape = (2*h_small-1, 2*w-1) and weights_all[i][j] = ((i-h_small+1)**2 + (j-w+1)**2)**(-exponent/2)
    indices_i = np.arange(2 * h_small - 1).reshape(2 * h_small - 1, 1)
    indices_j = np.arange(2 * w_small - 1).reshape(1, 2 * w_small - 1)
    term_i = (indices_i - h_small + 1)**2
    term_j = (indices_j - w_small + 1)**2
    terms_ij = term_i + term_j
    # Set central element to 1 to avoid dividing by zero
    terms_ij[h_small-1][w_small-1] = 1
    weights_all = terms_ij ** (-exponent/2)
    # Set central element to 0
    #weights_all[h_small-1][w_small-1] = 0
    weights_all[h_small-1][w_small-1] = 2
    assert weights_all.shape == (2*h_small-1, 2*w_small-1)
    
    if print_progress:
        print("weights_all constructed")
    
    def get_weights(weights_all, i, j):
        (h_all, w_all) = weights_all.shape
        assert h_all % 2 == 1
        assert w_all % 2 == 1
        h = (h_all+1)//2
        w = (w_all+1)//2
        assert i < h
        assert j < w
        return weights_all[h-i-1:2*h-i-1, w-j-1:2*w-j-1]
    
    for i in range(h):
        i_small = int(i * (h_small/h))
        if print_progress:
            print("Row " + str(i+1) + " of " + str(h) + ": i_small = " + str(i_small))
        for j in range(w):
            j_small = int(j * (w_small/w))
            # Extract weights relevant to element i, j
            weights = get_weights(weights_all, i_small, j_small)
            assert weights.shape == (h_small, w_small)
            # weights_sum: used to keep all pixels on the same scale
            weights_sum = np.sum(weights)
            
            current_intensity = normalised_data[i][j]
            # similarities: pixels similar to i, j have high intensities
            similarities = 1 - abs(normalised_data_downsampled - current_intensity)
            
            # weighted_similarity_sum: sum of similarity matrix weighted by spatial proximity to i, j
            # local_similarity: same value scaled by 1/weights_sum 
            weighted_similarity_sum = np.sum(similarities * weights)
            local_similarity = weighted_similarity_sum / weights_sum
            
            # weighted_intensity_sum: sum of intensity matrix weighted by spatial proximity to i, j
            # local_intensity: same value scaled by 1/weights_sum
            # local_abnormality: measure of how atypical the local area is compared to the image as a whole
            weighted_intensity_sum = np.sum(normalised_data_downsampled * weights)
            local_intensity = weighted_intensity_sum / weights_sum
            local_abnormality = abs(local_intensity - mean_intensity)
            
            # relative_similarity: highest for pixels in an atypical region that are typical for that region
            relative_similarity = local_similarity * local_abnormality
            similarity_data[i][j] = relative_similarity
    return similarity_data


def get_signal_neighbour_similarity(signal_in: hs.signals.Signal2D, exponent=2, print_progress=False):
    '''
    Applies `get_neighbour_similarity` to all the images in a stack.
    Returns a new image stack.
    '''
    data_shape = signal_in.data.shape
    similarity_data = np.empty(data_shape)
    for i in range(data_shape[0]):
        if (print_progress):
            print("Calculating pixel-wise neighbour similarity for frame " + str(i+1) + " of " + str(data_shape[0]))
        similarity_data[i] = get_neighbour_similarity(signal_in.data[i], exponent=exponent, print_progress=print_progress)
    return hs.signals.Signal2D(similarity_data)


def get_signal_neighbour_similarity_approx_v2(signal_in: hs.signals.Signal2D, feature_length=0, print_progress=False):
    '''
    Applies `get_neighbour_similarity_approx_v2` to all the images in a stack.
    Returns a new image stack.
    '''
    data_shape = signal_in.data.shape
    similarity_data = np.empty(data_shape)
    for i in range(data_shape[0]):
        if (print_progress):
            print("Calculating pixel-wise neighbour similarity for frame " + str(i+1) + " of " + str(data_shape[0]))
        similarity_data[i] = get_neighbour_similarity_approx_v2(signal_in.data[i], feature_length=feature_length, print_progress=print_progress)
    return hs.signals.Signal2D(similarity_data)


def get_signal_neighbour_similarity_faster(signal_in: hs.signals.Signal2D, feature_length=0, print_progress=False):
    '''
    Applies `get_neighbour_similarity_faster` to all the images in a stack.
    Returns a new image stack.
    '''
    data_shape = signal_in.data.shape
    similarity_data = np.empty(data_shape)
    for i in range(data_shape[0]):
        if (print_progress):
            print("Calculating pixel-wise neighbour similarity for frame " + str(i+1) + " of " + str(data_shape[0]))
        similarity_data[i] = get_neighbour_similarity_faster(signal_in.data[i], feature_length=feature_length, print_progress=print_progress)
    return hs.signals.Signal2D(similarity_data)


def scale_intensity_to_data(arr_in: np.array, intensity_data: np.array):
    '''
    Shifts `arr_in` values such that the mean is zero, multiplies each element of the result with `intensity_data`, then shifts it back up again and returns the result.
    '''
    assert arr_in.shape == intensity_data.shape
    mean_intensity = arr_in.mean()
    arr_out = (arr_in - mean_intensity) * intensity_data + mean_intensity
    return arr_out


def scale_signal_to_data(signal_in: hs.signals.Signal2D, intensity_signal: hs.signals.Signal2D):
    '''
    Applies `scale_intensity_to_data` to each pair of arrays `(signal_in.data[t], intensity_signal.data[t])`
    '''
    assert signal_in.data.shape == intensity_signal.data.shape
    data_shape = signal_in.data.shape
    data_out = np.empty(data_shape)
    for t in range(data_shape[0]):
        data_out[t] = scale_intensity_to_data(signal_in.data[t], intensity_signal.data[t])
    return hs.signals.Signal2D(data_out)
    

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


def horn_schunck(data_in: np.array, alpha_squared: float, num_iterations: int, use_average=False):
    '''
    Horn-Schunck method for optical flow.
    Outputs a series of (vertical, horizontal) displacement field pairs obtained from a corresponding series of 2D images.
    
    The larger `alpha_squared` is, the more strongly departure from smoothness is penalised relative to rate of change of brightness.
    When `alpha_squared` = 1, both kinds of error are weighted equally.
    When `use_average` is True, each displacement field has the same (u,v) values for all pixels. The value used is the mean of the individual values.
    '''
    assert len(data_in.shape) == 3
    (num_images, height, width) = data_in.shape
    # Estimate derivatives of intensity wrt x, y and t
    I_x = ndi.filters.convolve(data_in, np.array([[[-1,-1],[1,1]],[[-1,-1],[1,1]]])*0.25)
    I_y = ndi.filters.convolve(data_in, np.array([[[-1,1],[-1,1]],[[-1,1],[-1,1]]])*0.25)
    I_t = ndi.filters.convolve(data_in, np.array([[[-1,-1],[-1,-1]],[[1,1],[1,1]]])*0.25)
    u = np.zeros(data_in.shape)
    v = np.zeros(data_in.shape)
    kernel = np.array([[1,2,1],[2,0,2],[1,2,1]])/12
    for t in range(num_images):
        u_t = np.zeros((height, width))
        v_t = np.zeros((height, width))
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


def horn_schunck_signal(signal_in: hs.signals.Signal2D, alpha_squared: float, num_iterations: int, use_average=False):
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


def nonrigid_pyramid(signal_in: hs.signals.Signal2D, num_levels=3, max_dimension=64, registration_method=horn_schunck, reg_args=None, reg_kwargs={}):
    assert len(signal_in.data.shape) == 3
    (num_images, height, width) = signal_in.data.shape
    
    # Ensure the maximum dimension of images to be processed is no greater than max_image_dimension.
    height_resized = height
    width_resized = width
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
        
    us = np.zeros(data_in.shape)
    vs = np.zeros(data_in.shape)
    
    images_resized = np.zeros((num_images, height_resized, width_resized))
    if resize_image:
        for t in range(num_images):
            images_resized[t] += skimage.transform.resize(signal_in.data[t], (height_resized, width_resized), mode='reflect', anti_aliasing=True)
    else:
        for t in range(num_images):
            images_resized[t] += signal_in.data[t]
    
    # Pyramid loop
    for n in range(num_levels):
        power_of_2 = num_levels - 1 - n
        print("power_of_2 = " + str(power_of_2))
        (new_height, new_width) = (height_resized//(2**power_of_2), width_resized//(2**power_of_2))
        if new_height < 2 or new_width < 2:
            continue
        new_us = np.zeros((num_images, new_height, new_width))
        new_vs = np.zeros((num_images, new_height, new_width))
        
        images_downsampled = np.zeros((num_images, new_height, new_width))
        # TODO: apply displacement field (us, vs) to signal_in.data
        if power_of_2 > 0:
            for t in range(num_images):
                images_downsampled[t] += skimage.transform.resize(signal_in.data[t], (new_height, new_width), mode='reflect', anti_aliasing=True)
        else:
            for t in range(num_images):
                images_downsampled[t] += images_resized[t]
        
        # Estimate displacement fields
        if reg_args is None:
            (new_us, new_vs) = registration_method(images_downsampled, **reg_kwargs)
        else:
            (new_us, new_vs) = registration_method(images_downsampled, *reg_args, **reg_kwargs)
        
        # TODO: interpolate (new_us, new_vs) to give (us, vs)

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
        #if max_disp < 1.0:
        #    print("NRR stopped after "+str(j)+" iterations.")
        #    break
    
    return(out_stack)


def mutual_information(arr_1: np.array, arr_2: np.array):
    '''
    Mutual information between arr_1[i][j] and arr_2[i][j] for all i, j.
    Returns the mutual information between arr_1 and arr_2.
    '''
    return im.mutual_information(arr_1.flatten(), arr_2.flatten())


def get_percentile_masks(arr_in: np.array, max_groups=3):
    # arr_in will probably be a neighbour similarity array based on a reference image.
    assert max_groups >= 1
    # Each group must have at least 4 elements
    num_groups = min(max_groups, arr_in.size//4)
    group_list = []
    for i in range(num_groups):
        lower_limit = np.percentile(arr_in, 100 * i/num_groups)
        upper_limit = np.percentile(arr_in, 100 * (i+1)/num_groups)
        if i == num_groups-1:
            upper_limit += 1
        current_group = np.logical_and(arr_in >= lower_limit, arr_in < upper_limit).astype(np.int32)
        merge_with_previous_group = False
        if i > 0:
            current_group_size = current_group.sum()
            previous_group_size = group_list[-1].sum()
            if previous_group_size < 4 or current_group_size < 4:
                merge_with_previous_group = True
        if merge_with_previous_group:
            group_list[-1] += current_group
            assert group_list[-1].max() == 1
        else:
            group_list.append(current_group)
    return group_list


def similarity_measure_using_neighbour_similarity(arr_moving: np.array, arr_ref: np.array, arr_ref_ns: np.array, values: list, debug=False, max_groups=3):
    assert arr_moving.shape == arr_ref.shape
    assert arr_ref_ns.shape == arr_ref.shape
    arr_transformed = transform_using_values(arr_moving, values, cval=float('-inf'))
    transform_mask = (arr_transformed >= arr_moving.min()-1).astype(np.int32)
    if transform_mask.max() != 1:
        #print("transform_mask.max(): ", transform_mask.max())
        #print("values: ", values)
        return 0
    #assert transform_mask.max() == 1
    #arr_ref_ns = get_neighbour_similarity(arr_ref)
    group_list = get_percentile_masks(arr_ref_ns, max_groups=max_groups)
    percentile_masks = []
    max_num_masks = len(group_list)
    for i in range(max_num_masks):
        current_group = group_list[i] * transform_mask
        merge_with_previous_group = False
        if i > 0:
            current_group_size = current_group.sum()
            previous_group_size = percentile_masks[-1].sum()
            if previous_group_size < 4 or current_group_size < 4:
                merge_with_previous_group = True
        if merge_with_previous_group:
            percentile_masks[-1] += current_group
        else:
            percentile_masks.append(current_group)
    num_masks = len(percentile_masks)
    percentile_averages = np.empty(num_masks)
    similarity_measures = np.zeros(num_masks)
    for i in range(num_masks):
        mask = percentile_masks[i]
        denominator = max(1, mask.sum())
        percentile_averages[i] = (arr_ref_ns * mask).sum() / denominator
        if mask.sum() >= 4:
            arr_transformed_reduced = arr_transformed.ravel()[np.where(mask.ravel() == 1)]
            arr_ref_reduced = arr_ref.ravel()[np.where(mask.ravel() == 1)]
            similarity_measures[i] = im.similarity_measure(np.array(arr_transformed_reduced), np.array(arr_ref_reduced), measure="NMI")
    denominator = max(1, percentile_averages.sum())
    similarity_measures *= percentile_averages/denominator
    sm = similarity_measures.sum()
    assert sm >= 0
    assert sm <= 1
    return sm


def similarity_measure_area_of_overlap(arr_fixed: np.array, arr_to_transform: np.array, values: list, debug=False):
    assert arr_fixed.shape == arr_to_transform.shape
    arr_transformed = transform_using_values(arr_to_transform, values, cval=float('-inf'))
    arr_1 = arr_fixed.ravel()
    arr_2 = arr_transformed.ravel()
    arr_1_reduced = []
    arr_2_reduced = []
    assert len(arr_1) >= 3
    if debug:
        print("min = " + str(arr_to_transform.min()))
        print(values)
    for i in range(len(arr_1)):
        #if i % (len(arr_1)//10) == 0:
        #    print("arr_2[" + str(i) + "] = " + str(arr_2[i]))
        if arr_2[i] >= arr_to_transform.min():
            arr_1_reduced.append(arr_1[i])
            arr_2_reduced.append(arr_2[i])
    if debug:
        print("Length = " + str(len(arr_2_reduced)))
    #assert len(arr_1_reduced) >= 3
    if len(arr_1_reduced) < 3:
        return 0
    else:
        #sm = im.mutual_information(arr_1_reduced, arr_2_reduced) * len(arr_1_reduced) / len(arr_1)
        sm = im.similarity_measure(np.array(arr_1_reduced), np.array(arr_2_reduced), measure="NMI")
        if debug:
            mi = im.mutual_information(arr_1, transform_using_values(arr_to_transform, values, cval_mean=True).ravel())
            print((mi, sm))
        return sm

def similarity_measure_after_transform(arr_fixed: np.array, arr_to_transform: np.array, values: list, debug=False):
    assert arr_fixed.shape == arr_to_transform.shape
    arr_transformed = transform_using_values(arr_to_transform, values, cval=float('-1'))
    sm = im.similarity_measure(arr_fixed, arr_transformed, measure="NMI")
    if debug:
        sm2 = im.similarity_measure(normalised_image(arr_fixed), normalised_image(arr_transformed), measure="NMI")
        print("Similarity measure comparison: ", (sm, sm2))
    return sm


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
            mi = im.similarity_measure(signal_in.data[t], signal_in.data[t2])
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
            mi_total += im.similarity_measure(signal_in.data[t], signal_in.data[t2])**exponent
        if mi_total > mi_max:
            mi_max = mi_total
            mi_max_index = t
    return mi_max_index


def highest_mutual_information_index_faster(signal_in: hs.signals.Signal2D):
    '''
    Iterates over an image stack. For each image, the the mutual information between that image and the average image is calculated. The index of the image for which this value is greatest is returned.
    '''
    mi_max = 0.0
    mi_max_index = 0
    arr_avg = normalised_average_of_signal(signal_in)
    for t in range(signal_in.data.shape[0]):
        mi = im.similarity_measure(signal_in.data[t], arr_avg, measure="NMI")
        if mi > mi_max:
            mi_max = mi
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


def transform_using_values(arr_in: np.array, values: list, cval=-1, cval_mean=False):
    '''
    Applies an affine transformation to `arr_in` using the parameter values in `values`.
    '''
    assert len(values) == 6
    scale_x = values[0]
    scale_y = values[1]
    shear_radians = values[2]
    rotate_radians = values[3]
    offset_x = values[4]
    offset_y = values[5]
    # Image must be shifted by minus half each dimension, then transformed, then shifted back.
    # This way, rotations and shears will be about the centre of the image rather than the top-left corner.
    shift_x = -0.5 * arr_in.shape[1]
    shift_y = -0.5 * arr_in.shape[0]
    a0 = scale_x * math.cos(rotate_radians)
    a1 = -scale_y * math.sin(rotate_radians + shear_radians)
    a2 = a0 * shift_x + a1 * shift_y + offset_x - shift_x
    b0 = scale_x * math.sin(rotate_radians)
    b1 = scale_y * math.cos(rotate_radians + shear_radians)
    b2 = b0 * shift_x + b1 * shift_y + offset_y - shift_y
    tform = skimage.transform.AffineTransform(matrix=np.array([[a0, a1, a2], [b0, b1, b2], [0, 0, 1]]))
    if cval_mean:
        cval = arr_in.mean()
    arr_out = skimage.transform.warp(arr_in.astype(float), tform.inverse, cval=cval)
    return arr_out


def skimage_estimate_shift(arr_moving: np.array, arr_ref: np.array, upsample_factor=10):
    return skimage.feature.register_translation(arr_moving, arr_ref, upsample_factor=upsample_factor, space='real', return_error=False)


def shift(arr_in: np.array, offset_x, offset_y, cval=-1, cval_mean=False):
    '''
    Shifts `arr_in` down by `offset_x` pixels and right by `offset_y` pixels.
    '''
    return transform_using_values(arr_in, [1, 1, 0, 0, offset_x, offset_y], cval=cval, cval_mean=cval_mean)


def shift_signal(signal_in: hs.signals.Signal2D, shifts: np.array, cval=-1, cval_mean=False):
    '''
    Shifts each image in `signal_in` by an amount specified by the corresponding element of `shifts`.
    Shifts are in horizontal, vertical order.
    '''
    assert shifts.shape == (signal_in.data.shape[0], 2)
    signal_out = hs.signals.Signal2D(np.empty_like(signal_in.data))
    for t in range(signal_in.data.shape[0]):
        signal_out.data[t] = shift(signal_in.data[t], shifts[t][0], shifts[t][1], cval=cval, cval_mean=cval_mean)
    return signal_out


def correct_shifts_vh(signal_in: hs.signals.Signal2D, shifts: np.array, cval=-1, cval_mean=False):
    '''
    Corrects for shifts in `signal_in`, specified by `shifts`, by applying the shifts in reverse.
    Shifts are in vertical, horizontal order.
    '''
    assert shifts.shape == (signal_in.data.shape[0], 2)
    return shift_signal(signal_in, -np.flip(shifts, 1), cval=cval, cval_mean=cval_mean)
    

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


def scale(arr_in: np.array, scale_factor_x, scale_factor_y):
    '''
    Scales the image represented by `arr_in` by scale factors `scale_factor_x` and `scale_factor_y`.
    '''
    return transform_using_values(arr_in, [scale_factor_x, scale_factor_y, 0.0, 0.0, 0.0, 0.0])


def rotate(arr_in: np.array, rotate_radians):
    '''
    Rotates the image represented by `arr_in` by `rotate_radians` radians in the clockwise direction.
    '''
    return transform_using_values(arr_in, [1.0, 1.0, 0.0, rotate_radians, 0.0, 0.0])
    

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
        

def transform_using_matrix(arr_in: np.array, matrix):
    '''
    Applies an affine transformation specified by `matrix` to `arr_in`.
    '''
    tform = skimage.transform.AffineTransform(matrix=matrix)
    return skimage.transform.warp(arr_in/arr_in.max(), tform.inverse, cval=arr_in.mean()/arr_in.max())*arr_in.max()


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


def apply_affine_params_to_signal(signal_in: hs.signals.Signal2D, params: np.array, cval_mean=True):
    '''
    Takes an image stack and a corresponding list of affine parameter sets. Returns a signal representing the result of applying each of these parameter sets to the corresponding image in `signal_in`.
    '''
    num_images = signal_in.data.shape[0]
    assert len(params.shape) == 2
    assert params.shape[0] == num_images
    assert params.shape[1] == 6
    signal_out = hs.signals.Signal2D(np.empty_like(signal_in.data))
    for t in range(num_images):
        signal_out.data[t] = transform_using_values(signal_in.data[t], params[t], cval_mean=cval_mean)
    return signal_out
