import cv2
import h5py  # hdf5 reader/writer
import hyperspy.api as hs  # hyperspy
import image_processing as ip
import math
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import reg_nonrigid as rn
import reg_rigid as rr
import reg_rigid_utils as rru
import scipy
import similarity_measure_methods as smm
import SimpleITK as sitk
import skimage
import utils

from scipy import ndimage as ndi
from scipy import signal
from scipy.ndimage.filters import gaussian_filter

# Load signal data from files

fin_1 = h5py.File("data/89109_16_Fe_mantis_norm.hdf5")
data_1 = fin_1["/exchange/data"]
signal_1 = hs.signals.Signal2D(data_1)
signal_1 = signal_1.transpose(signal_axes=(0,2))

fin_2 = h5py.File("data/mantis_55510_55660.hdf5")
data_2 = fin_2["/exchange/data"]
signal_2 = hs.signals.Signal2D(data_2)
signal_2 = signal_2.transpose(signal_axes=(0,2))

fin_3 = h5py.File("data/mantis_raw_55499_55509.hdf5")
data_3 = fin_3["/exchange/data"]
signal_3 = hs.signals.Signal2D(data_3)
signal_3 = signal_3.transpose(signal_axes=(0,2))

signal_4 = hs.load("data/0005-RotSTEM90 ADF1.dm3")
signal_5 = hs.load("data/20_Aligned 20-Stack-5MxHAADF STACK(20).dm3")

signal_1_reduced = hs.signals.Signal2D(signal_1.data[:115][::5])
signal_2_reduced = hs.signals.Signal2D(signal_2.data[::5])

# Generate a very simple synthetic signal for demonstration purposes

height = 38
width = 46

vfield_1 = np.array([np.ones((height, width)) * -2, np.ones((height, width)) * +1])
vfield_2 = np.array([np.ones((height, width)) * -2.7, np.ones((height, width)) * +1.2])

arr_A = utils.make_capital_A((height, width))
signal_A = hs.signals.Signal2D(np.array([arr_A, ip.apply_displacement_field_sitk(vfield_1, arr_A), ip.apply_displacement_field_sitk(vfield_2, arr_A)]))

arr_i = np.arange(height).reshape(height, 1)
arr_j = np.arange(width).reshape(1, width)
# i_plus, i_minus: vertical coordinates of +ve and -ve 'charges'
i_plus = height * 0.75
i_minus = height * 0.25
# j_plus, j_minus: horizontal coordinates of +ve and -ve 'charges'
j_plus = width * 0.25
j_minus = width * 0.75
# dsquared_plus[i][j] = distance squared between +ve 'charge' and pixel i,j
# dsquared_minus[i][j] = distance squared between -ve 'charge' and pixel i,j
dsquared_plus = (arr_i - i_plus)**2 + (arr_j - j_plus)**2
dsquared_minus = (arr_i - i_minus)**2 + (arr_j - j_minus)**2
scale = 2
power = 2

# vfield_3 represents a shift similar to the force experienced by a small test charge under the influence of two fixed point charges: one positive, one negative. Unlike vfield_1 and vfield_2, it is non-uniform.
# vfield_3[i][j] = scale * sum[r_hat * 'charge' * (1/distance from 'charge')^power]
# r_hat = (i - i_plus)/(dsquared^0.5)
# therefore vfield_3[i][j] = scale * sum[(i - i_plus) * 'charge' * (1/distance from 'charge')^(power+1)]
factor_plus = scale * (dsquared_plus**(-0.5*(1+power)))
factor_minus = scale * (dsquared_minus**(-0.5*(1+power)))
vfield_3_i = (arr_i - i_plus) * factor_plus + (i_minus - arr_i) * factor_minus
vfield_3_j = (arr_j - j_plus) * factor_plus + (j_minus - arr_j) * factor_minus
vfield_3 = np.array([vfield_3_i, vfield_3_j])

# Generate another synthetic signal for demonstration purposes

num_keyframes = 3
num_between_keyframes = 6
num_frames = (num_between_keyframes + 1) * num_keyframes
key_indices = np.arange(num_keyframes + 1) * (num_between_keyframes+1) # 0, 3, 6, 9
np.random.seed(0)
key_scale_x = np.random.rand(num_keyframes + 1) * 0.4 + 0.8 # 0.8-1.2
key_scale_y = np.random.rand(num_keyframes + 1) * 0.4 + 0.8 # 0.8-1.2
key_shear = np.random.rand(num_keyframes + 1) * (2*math.pi/12) - math.pi/12 # -pi/12 to +pi/12
key_rotation = np.random.rand(num_keyframes + 1) * (2*math.pi/6) - math.pi/6 # -pi/6 to +pi/6
key_offset_x = np.random.rand(num_keyframes + 1) * (0.4 * height) - (0.2 * height) # -height/5 to +height/5
key_offset_y = np.random.rand(num_keyframes + 1) * (0.4 * width) - (0.2 * width) # -width/5 to +width/5
key_scale_x[-1] = key_scale_x[0]
key_scale_y[-1] = key_scale_y[0]
key_shear[-1] = key_shear[0]
key_rotation[-1] = key_rotation[0]
key_offset_x[-1] = key_offset_x[0]
key_offset_y[-1] = key_offset_y[0]
spline_scale_x = scipy.interpolate.InterpolatedUnivariateSpline(key_indices, key_scale_x)
spline_scale_y = scipy.interpolate.InterpolatedUnivariateSpline(key_indices, key_scale_y)
spline_shear = scipy.interpolate.InterpolatedUnivariateSpline(key_indices, key_shear)
spline_rotation = scipy.interpolate.InterpolatedUnivariateSpline(key_indices, key_rotation)
spline_offset_x = scipy.interpolate.InterpolatedUnivariateSpline(key_indices, key_offset_x)
spline_offset_y = scipy.interpolate.InterpolatedUnivariateSpline(key_indices, key_offset_y)
all_scale_x = spline_scale_x(np.arange(0, num_frames))
all_scale_y = spline_scale_y(np.arange(0, num_frames))
all_shear= spline_shear(np.arange(0, num_frames))
all_rotation = spline_rotation(np.arange(0, num_frames))
all_offset_x = spline_offset_x(np.arange(0, num_frames))
all_offset_y = spline_offset_y(np.arange(0, num_frames))
signal_A_2 = hs.signals.Signal2D(np.empty((num_frames, arr_A.shape[0], arr_A.shape[1])))
for t in range(num_frames):
    signal_A_2.data[t] = ip.transform_using_values(arr_A, [all_scale_x[t], all_scale_y[t], all_shear[t], all_rotation[t], all_offset_x[t], all_offset_y[t]], cval_mean=True)

# Synthetic signal with less variation than signal_A_2
signal_A_3 = hs.signals.Signal2D(np.empty((num_frames, arr_A.shape[0], arr_A.shape[1])))
for t in range(num_frames):
    scale_x = (all_scale_x[t] - 1) * 0.25 + 1 # 0.95-1.05
    scale_y = (all_scale_y[t] - 1) * 0.25 + 1 # 0.95-1.05
    shear = all_shear[t] # -pi/12 to +pi/12
    rotation = all_rotation[t] / 2 # -pi/12 to +pi/12
    offset_x = all_offset_x[t] / 2 # -height/10 to +height/10
    offset_y = all_offset_y[t] / 2 # -width/10 to +width/10
    signal_A_3.data[t] = ip.transform_using_values(arr_A, [scale_x, scale_y, shear, rotation, offset_x, offset_y], cval_mean=True)