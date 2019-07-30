import cv2
import h5py  # hdf5 reader/writer
import hyperspy.api as hs  # hyperspy
import ImageProcessingMethods as ipm
import math
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import scipy
import SimpleITK as sitk
import skimage

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

# Generate a very simple synthetic signal for demonstration purposes

height = 38
width = 46

vfield_1 = np.array([np.ones((height, width)) * -2, np.ones((height, width)) * +1])
vfield_2 = np.array([np.ones((height, width)) * -2.7, np.ones((height, width)) * +1.2])

arr_A = ipm.make_capital_A((height, width))
signal_A = hs.signals.Signal2D(np.array([arr_A, ipm.apply_displacement_field_sitk(vfield_1, arr_A), ipm.apply_displacement_field_sitk(vfield_2, arr_A)]))

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