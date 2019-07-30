
import numpy as np
'''
from hyperspy.misc.math.stats.histogram_tools import scott_bin_width,\
                                                  freedman_bin_width,\
                                                   sturges_bin_width
'''
from histogram_tools import scott_bin_width,\
                         freedman_bin_width,\
                          sturges_bin_width

from scipy.ndimage import gaussian_filter
#from fastkde import fastKDE


TINY = float(np.finfo(np.double).tiny)
SIGMA_FACTOR = 0.05
# A lambda function to force positive values
nonzero = lambda x: np.maximum(x, TINY)




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


def similarity_measure(image1,image2,norm=True,bin_rule="sturges",measure="MI"):
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
    if measure == "FKDE":
        hgram,edges = fastKDE.pdf(arr1,arr2,Nbins)
        measure     =  "NMI"                
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





