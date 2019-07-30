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

