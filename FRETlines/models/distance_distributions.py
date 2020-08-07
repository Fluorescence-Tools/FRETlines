# FRETlines generator - A python library for the calcualtion of FRETlines
#    Copyright (C) 2020  Anders Barth
import numpy as np
from numba import jit


@jit(nopython=True)
def normal_distribution(x, loc=0.0, scale=1.0, norm=True):
    """Probability density function of a generalized normal distribution

    :param x:
    :param loc: mean (location)
    :param scale:
    :param norm: Boolean if true the returned array is normalized to one
    :return:
    """

    p = 1.0 / (np.sqrt(2.0 * np.pi) * scale) * np.exp(- (x - loc)**2 /
                                                      (2. * scale**2))
    if norm:
        p /= p.sum()
    return p


@jit(nopython=True)
def chi_distribution(x, loc=0.0, scale=1.0, norm=True):
    """Probability density function of a non-central chi-distribution in three
        dimensions.

    :param x:
    :param loc: mean (location)
    :param scale:
    :param norm: Boolean if true the returned array is normalized to one
    :return:
    """
    
    p = (loc / scale) * (normal_distribution(x, loc=loc, scale=scale,
                                             norm=False) +
                         (-1)*normal_distribution(x, loc=(-1)*loc, scale=scale,
                                                  norm=False))
    if norm:
        p /= p.sum()
    return p


@jit(nopython=True)
def worm_like_chain(rs, chain_length=100.0, kappa=0.5, norm=True, 
                    distance=True):
    """Calculates the radial distribution function of a worm-like-chain given
    the multiple piece-solution according to:

    The radial distribution function of worm-like chain
    Eur Phys J E, 32, 53-69 (2010)

    Parameters
    ----------
    rs: a vector at which the pdf is evaluated.
    chain_length: the total length of the chain.
    kappa: a parameter describing the stiffness (details see publication)
    norm: If this is True the sum of the returned pdf vector is normalized to
    one.
    distance: If this is False, the end-to-end vector distribution is
    calculated. If True the distribution the pdf is integrated over a sphere,
    i.e., the pdf of the end-to-end distribution function is multiplied with
    4*pi*r**2.

    Returns
    -------
    An array of the pdf

    Examples
    --------

    >>> from FRETlines import distance_distributions as dist
    >>> import numpy as np
    >>> r = np.linspace(0, 0.99, 50)
    >>> kappa = 1.0
    >>> length = 100.0
    >>> dist.worm_like_chain(r, length, kappa)

    References
    ----------

    .. [1] Becker NB, Rosa A, Everaers R, Eur Phys J E Soft Matter, 2010 May;32(1):53-69,
       The radial distribution function of worm-like chains.

    """
    if chain_length == 0.0:
        chain_length = np.max(rs)

    k = kappa
    a = 14.054
    b = 0.473
    c = 1.0 - (1.0+(0.38*k**(-0.95))**(-5.))**(-1./5.)
    pr = np.zeros_like(rs, dtype=np.float64)

    if k < 0.125:
        d = k + 1.0
    else:
        d = 1.0 - 1.0/(0.177/(k-0.111)+6.4 * np.exp(0.783 * np.log(k-0.111)))

    for i in range(len(rs)):
        r = rs[i]
        if r < chain_length:
            r /= chain_length

            pri = ((1.0 - c * r**2.0) / (1.0 - r**2.0))**(5.0 / 2.0)
            pri *= np.exp(-d * k * a * b * (1.0 + b) / (1.0 - (b*r)**2.0) * r**2.0)

            g = (((-3./4.) / k - 1./2.) * r**2. + ((-23./64.) / k + 17./16.) * r**4. + ((-7./64.) / k - 9./16.) * r**6.)
            pri *= np.exp(g / (1.0 - r**2.0))
            pri *= i0(-d*k*a*(1+b)*r/(1-(b*r)**2))
            pr[i] = pri
        else:
            break

    if norm:
        pr /= pr.sum()

    return pr


@jit(nopython=True)
def gaussian_chain(r, rmsR=100, scale=1., norm=True):
    """
    Calculates the radial distribution function of a Gaussian chain in three
    dimensions

    :param rmsR: float
        Root mean square end-to-end distance
    :param scale: float
        Unused. Only included to keep consistency with the other distance
        distributions.
    :param r: numpy-array
        values of r should be in range [0, 1) - not including 1

    ..plot:: plots/rdf-gauss.py

    Note: Third argument is ignored. It is there to achieve consistency with 
    respect to the other distance distributions which require two input
    arguments.
    """
    r2_mean = rmsR ** 2
    p = 4*np.pi*r**2/(2./3. * np.pi*r2_mean)**(3./2.) * np.exp(-3./2. * r**2 /
                                                               r2_mean)
    if norm:
        p /= p.sum()
    return p


@jit(nopython=True)
def i0(x):
    """
    Modified Bessel-function I0(x) for any real x
    (according to numerical recipes function - `bessi0`,
    Polynomal approximation Abramowitz and Stegun )

    References
    ----------

    .. [1] Abramowitz, M and Stegun, I.A. 1964, Handbook of Mathematical
       Functions, Applied Mathematics Series, Volume 55 (Washington:
       National Bureal of Standards; reprinted 1968 by Dover Publications,
       New York), Chapter 10

    :param x:
    :return:
    """

    axi = abs(x)
    if axi < 3.75:
        yi = axi / 3.75
        yi *= yi
        ayi = 1.0 + yi * (3.5156299 + yi * (
            3.0899424 + yi * (1.2067492 + yi * (
                0.2659732 + yi * (0.360768e-1 + yi *
                                  0.45813e-2)))))
    else:
        yi = 3.75 / axi
        ayi = (np.exp(axi) / np.sqrt(axi)) * \
              (0.39894228 + yi * (0.1328592e-1 + yi * (
                  0.225319e-2 + yi * (-0.157565e-2 + yi * (
                      0.916281e-2 + yi * (-0.2057706e-1 + yi * (
                          0.2635537e-1 + yi * (-0.1647633e-1 + yi *
                                               0.392377e-2))))))))
    return ayi
