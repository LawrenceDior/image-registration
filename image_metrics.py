# -*- coding: utf-8 -*-
"""
Created on Fri Aug  2 17:27:58 2019

@author: pq67
"""


import numpy as np

from scipy.ndimage import gaussian_filter


TINY = float(np.finfo(np.double).tiny)
SIGMA_FACTOR = 0.05
# A lambda function to force positive values
nonzero = lambda x: np.maximum(x, TINY)


import numpy as np

def scott_bin_width(data, return_bins=False):
    r"""Return the optimal histogram bin width using Scott's rule
    Scott's rule is a normal reference rule: it minimizes the integrated
    mean squared error in the bin approximation under the assumption that the
    data is approximately Gaussian.
    Parameters
    ----------
    data : array-like, ndim=1
        observed (one-dimensional) data
    return_bins : bool (optional)
        if True, then return the bin edges
    Returns
    -------
    width : float
        optimal bin width using Scott's rule
    bins : ndarray
        bin edges: returned if ``return_bins`` is True
    Notes
    -----
    The optimal bin width is
    .. math::
        \Delta_b = \frac{3.5\sigma}{n^{1/3}}
    where :math:`\sigma` is the standard deviation of the data, and
    :math:`n` is the number of data points [1]_.
    References
    ----------
    .. [1] Scott, David W. (1979). "On optimal and data-based histograms".
       Biometricka 66 (3): 605-610
    See Also
    --------
    knuth_bin_width
    freedman_bin_width
    bayesian_blocks
    histogram
    """
    data = np.asarray(data)
    if data.ndim != 1:
        raise ValueError("data should be one-dimensional")

    n = data.size
    sigma = np.std(data)

    dx = 3.5 * sigma / (n ** (1 / 3))
    Nbins = np.ceil((data.max() - data.min()) / dx)
    Nbins = max(1, Nbins)

    if return_bins:
        bins = data.min() + dx * np.arange(Nbins + 1)
        return dx, Nbins,bins
    else:
        return dx,Nbins


    
def freedman_bin_width(data, return_bins=False):
    r"""Return the optimal histogram bin width using the Freedman-Diaconis rule
    The Freedman-Diaconis rule is a normal reference rule like Scott's
    rule, but uses rank-based statistics for results which are more robust
    to deviations from a normal distribution.
    Parameters
    ----------
    data : array-like, ndim=1
        observed (one-dimensional) data
    return_bins : bool (optional)
        if True, then return the bin edges
    Returns
    -------
    width : float
        optimal bin width using the Freedman-Diaconis rule
    bins : ndarray
        bin edges: returned if ``return_bins`` is True
    Notes
    -----
    The optimal bin width is
    .. math::
        \Delta_b = \frac{2(q_{75} - q_{25})}{n^{1/3}}
    where :math:`q_{N}` is the :math:`N` percent quartile of the data, and
    :math:`n` is the number of data points [1]_.
    References
    ----------
    .. [1] D. Freedman & P. Diaconis (1981)
       "On the histogram as a density estimator: L2 theory".
       Probability Theory and Related Fields 57 (4): 453-476
    See Also
    --------
    knuth_bin_width
    scott_bin_width
    bayesian_blocks
    histogram
    """
    data = np.asarray(data)
    if data.ndim != 1:
        raise ValueError("data should be one-dimensional")

    n = data.size
    if n < 4:
        raise ValueError("data should have more than three entries")

    v25, v75 = np.percentile(data, [25, 75])
    dx = 2 * (v75 - v25) / (n ** (1 / 3))
    dmin, dmax = data.min(), data.max()
    Nbins = max(1, np.ceil((dmax - dmin) / dx))

    if return_bins:
        try:
            bins = dmin + dx * np.arange(Nbins + 1)
        except ValueError as e:
            if 'Maximum allowed size exceeded' in str(e):
                raise ValueError(
                    'The inter-quartile range of the data is too small: '
                    'failed to construct histogram with {} bins. '
                    'Please use another bin method, such as '
                    'bins="scott"'.format(Nbins + 1))
            else:  # Something else  # pragma: no cover
                raise
        return dx, Nbins,bins
    else:
        return dx, Nbins

    
def sturges_bin_width(data, return_bins=False):
    
    """Return the optimal histogram bin width using sturges's rule
    
    1 + log2(N) where N is the number of samples. 
    
    This is generally considered good for low N (e.g. N<200)
    
    Parameters
    ----------
    data : array-like, ndim=1
        observed (one-dimensional) data
    Returns
    -------
    width : float
        optimal bin width using Sturges rule

    """
    data = np.asarray(data)
    if data.ndim != 1:
        raise ValueError("data should be one-dimensional")

    n = data.size
    if n < 4:
        raise ValueError("data should have more than three entries")

    Nbins = np.ceil(1. + np.log(n))
    dx = (data.max() - data.min()) / Nbins
    if return_bins:
        bins = data.min() + dx * np.arange(Nbins + 1)
        return dx, Nbins,bins
    return dx,Nbins



def dist2loss(q, qI=None, qJ=None):
    """
    Convert a joint distribution model q(i,j) into a pointwise loss:

    L(i,j) = - log q(i,j)/(q(i)q(j))

    where q(i) = sum_j q(i,j) and q(j) = sum_i q(i,j)

    See: Roche, medical image registration through statistical
    inference, 2001.
    """
    qT = q.T
    if qI is None:
        qI = q.sum(0)
    if qJ is None:
        qJ = q.sum(1)
    q /= nonzero(qI)
    qT /= nonzero(qJ)
    return -np.log(nonzero(q))


class SimilarityMeasure(object):
    """
    Template class
    """
    def __init__(self, renormalize=False):
        self.renormalize = renormalize

    def loss(self, H):
        return np.zeros(H.shape)

    def npoints(self, H):
        return H.sum()

    def __call__(self, H):
        total_loss = np.sum(H * self.loss(H))
        if not self.renormalize:
            total_loss /= nonzero(self.npoints(H))
        return -total_loss

class MutualInformation(SimilarityMeasure):
    """
    Use the normalized joint histogram as a distribution model
    """
    def loss(self, H):
        return dist2loss(H / nonzero(self.npoints(H)))


class ParzenMutualInformation(SimilarityMeasure):
    """
    Use Parzen windowing to estimate the distribution model
    """
    def loss(self, H):
        if not hasattr(self, 'sigma'):
            self.sigma = SIGMA_FACTOR * np.array(H.shape)
        npts = nonzero(self.npoints(H))
        Hs = H / npts
        gaussian_filter(Hs, sigma=self.sigma, mode='constant', output=Hs)
        return dist2loss(Hs)


class DiscreteParzenMutualInformation(SimilarityMeasure):
    """
    Use Parzen windowing in the discrete case to estimate the
    distribution model
    """
    def loss(self, H):
        if not hasattr(self, 'sigma'):
            self.sigma = SIGMA_FACTOR * np.array(H.shape)
        Hs = gaussian_filter(H, sigma=self.sigma, mode='constant')
        Hs /= nonzero(Hs.sum())
        return dist2loss(Hs)


class NormalizedMutualInformation(SimilarityMeasure):
    """
    NMI = 2*(1 - H(I,J)/[H(I)+H(J)])
        = 2*MI/[H(I)+H(J)])
    """
    def __call__(self, H):
        H = H / nonzero(self.npoints(H))
        hI = H.sum(0)
        hJ = H.sum(1)
        entIJ = -np.sum(H * np.log(nonzero(H)))
        entI = -np.sum(hI * np.log(nonzero(hI)))
        entJ = -np.sum(hJ * np.log(nonzero(hJ)))
        return 2 * (1 - entIJ / nonzero(entI + entJ))




def mutual_information(arr1,arr2,norm=True,bin_rule="sturges"):
    """
    
    Computes mutual information between two images variate from a
    joint histogram.
    
    Parameters
    ----------
    
    arr1 : 1D array
    arr2 : 1D array
    
    bins:  number of bins to use 
           Default = None.  If None specificed then 
           the inital estimate is set to be int(sqrt(size/5.))
           where size is the number of points in arr1  
           
    Returns
    -------
     mi: float  the computed similariy measure
     
     
    """
    
    if bin_rule ==  None or bin_rule == "sturges":
        dx,Nbins = sturges_bin_width(arr1)
    elif bin_rule == "scott":
        dx,Nbins = scott_bin_width(arr1)
    elif bin_rule == "freedman":
        dx,Nbins = freedman_bin_width(arr1)
    else:
        raise ValueError("Unrecognised bin width rule: please use scott, sturges or freedman")

    # Convert bins counts to probability values
    hgram, x_edges, y_edges = np.histogram2d(arr1,arr2,Nbins)

    pxy = hgram/ float(np.sum(hgram))
    px = np.sum(pxy, axis=1) # marginal for x over y
    py = np.sum(pxy, axis=0) # marginal for y over x
    px_py = px[:, None] * py[None, :] # Broadcast to multiply marginals
    # Now we can do the calculation using the pxy, px_py 2D arrays
    nzs = pxy > 0 # Only non-zero pxy values contribute to the sum
    
    if norm:
        nxzx = px > 0
        nxzy = py > 0
        h_x  = -np.sum(px[nxzx]* np.log(px[nxzx]) )
        h_y  = -np.sum(py[nxzy]* np.log(py[nxzy]) )
        norm = 1.0/(max(np.amax(h_x),np.amax(h_y)))
    else:
        norm = 1.0

    i_xy=  norm*(np.sum(pxy[nzs] * np.log(pxy[nzs] / px_py[nzs])))
    
    return i_xy


def similarity_measure(image1,image2,norm=True,bin_rule=None,measure="MI"):
    """
    
    Computes mutual information between two images variate from a
    joint histogram.
    
    Parameters
    ----------
    
    arr1 : 1D array
    arr2 : 1D array
    
    bins:  number of bins to use 
           Default = None.  If None specificed then 
           the inital estimate is set to be int(sqrt(size/5.))
           where size is the number of points in arr1  
           
    Returns
    -------
     mi: float  the computed similariy measure
     
     
    """
    arr1 = image1.ravel()
    arr2 = image2.ravel()
    if bin_rule ==  None or bin_rule == "sturges":
        dx,Nbins = sturges_bin_width(arr1)
    elif bin_rule == "scott":
        dx,Nbins = scott_bin_width(arr1)
    elif bin_rule == "freedman":
        dx,Nbins = freedman_bin_width(arr1)
    elif bin_rule == 'auto':
        if len(arr1)<400:
            dx,Nbins = sturges_bin_width(arr1)
        else:
            dx,Nbins = scott_bin_width(arr1)
    else:
        raise ValueError("Unrecognised bin width rule: please use auto, scott, sturges or freedman")

    # Convert bins counts to probability values
    hgram, x_edges, y_edges = np.histogram2d(arr1,arr2,Nbins)
    if measure == "MI":
        pxy = MutualInformation(renormalize=norm)
    elif measure == "NMI":
        pxy = NormalizedMutualInformation(renormalize=norm)
    elif measure == "PMI":
        pxy = ParzenMutualInformation(renormalize=norm)
    elif measure == "DPMI":
        pxy = DiscreteParzenMutualInformation(renormalize=norm)
    else:
        pxy = NormalizedMutualInformation(renormalize=norm)
    return pxy(hgram)





