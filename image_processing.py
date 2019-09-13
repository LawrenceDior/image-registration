import cv2
import hyperspy.api as hs
import math
import numpy as np
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


def get_percentile_masks(arr_in: np.array, max_groups=3):
    '''
    Splits the pixels in `arr_in` into no more than `max_groups` groups of roughly equal size. Group assignment depends on the intensity of the corresponding pixel in `arr_in`.
    Returns a list of groups. Each group is represented by an array of the same size as `arr_in`. Pixels in a given group are set to 1 and the rest are set to 0.
    '''
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
        

def transform_using_matrix(arr_in: np.array, matrix):
    '''
    Applies an affine transformation specified by `matrix` to `arr_in`.
    '''
    tform = skimage.transform.AffineTransform(matrix=matrix)
    return skimage.transform.warp(arr_in/arr_in.max(), tform.inverse, cval=arr_in.mean()/arr_in.max())*arr_in.max()


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
