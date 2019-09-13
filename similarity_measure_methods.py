import hyperspy.api as hs
import image_metrics as im
import numpy as np

from image_processing import get_neighbour_similarity, get_percentile_masks, normalised_average_of_signal, normalised_image, transform_using_values


def mutual_information(arr_1: np.array, arr_2: np.array):
    '''
    Mutual information between arr_1[i][j] and arr_2[i][j] for all i, j.
    Returns the mutual information between arr_1 and arr_2.
    '''
    return im.mutual_information(arr_1.flatten(), arr_2.flatten())


def similarity_measure_using_neighbour_similarity(arr_moving: np.array, arr_ref: np.array, arr_ref_ns: np.array, values: np.array, debug=False, max_groups=3):
    '''
    Transforms `arr_moving` using affine transformation parameters in `values`, then returns a similarity measure between the result and `arr_ref`.
    The pixels in `arr_ref` are split into no more than `max_groups` groups of roughly equal size. Group assignment depends on the intensity of the corresponding pixel in `arr_ref_ns`.
    For each group, the normalised mutual information between the corresponding pixels in `arr_ref` and the transformed array are calculated. These values are weighted by the average intensity of the corresponding pixels in `arr_ref_ns` and the weighted average is returned as the overall similarity measure. Low-uncertainty regions therefore have the greatest influence over the value returned.
    '''
    assert arr_moving.shape == arr_ref.shape
    assert arr_ref_ns.shape == arr_ref.shape
    arr_transformed = transform_using_values(arr_moving, values, cval=float('-inf'))
    transform_mask = (arr_transformed >= arr_moving.min()-1).astype(np.int32)
    if transform_mask.max() != 1:
        return 0
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
            similarity_measures[i] = max(0, im.similarity_measure(np.array(arr_transformed_reduced), np.array(arr_ref_reduced), measure="NMI"))
    denominator = max(1, percentile_averages.sum())
    similarity_measures *= percentile_averages/denominator
    sm = similarity_measures.sum()
    assert sm >= 0
    assert sm <= 1
    return sm


def similarity_measure_area_of_overlap(arr_fixed: np.array, arr_to_transform: np.array, values: np.array, debug=False):
    '''
    Transforms `arr_to_transform` using affine transformation parameters in `values`, then returns the normalised mutual information between the result and `arr_fixed`.
    Only the pixels covered by both arrays are considered when calculating the mutual information.
    '''
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
        if arr_2[i] >= arr_to_transform.min():
            arr_1_reduced.append(arr_1[i])
            arr_2_reduced.append(arr_2[i])
    if debug:
        print("Length = " + str(len(arr_2_reduced)))
    if len(arr_1_reduced) < 3:
        return 0
    else:
        sm = im.similarity_measure(np.array(arr_1_reduced), np.array(arr_2_reduced), measure="NMI")
        return sm


def similarity_measure_after_transform(arr_fixed: np.array, arr_to_transform: np.array, values: np.array, debug=False):
    '''
    Transforms `arr_to_transform` using affine transformation parameters in `values`, then returns the normalised mutual information between the result and `arr_fixed`.
    '''
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