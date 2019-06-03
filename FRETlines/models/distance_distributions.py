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
    y = 1.0 / (np.sqrt(2.0 * np.pi) * scale) * np.exp(- (x - loc)**2 /
                                                      (2. * scale**2))
    if norm:
        y /= y.sum()
    return y


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
def worm_like_chain(rs, chain_length=100.0, kappa=0.5, normalize=True, distance=True):
    """Calculates the radial distribution function of a worm-like-chain given the multiple piece-solution
    according to:

    The radial distribution function of worm-like chain
    Eur Phys J E, 32, 53-69 (2010)

    Parameters
    ----------
    rs: a vector at which the pdf is evaluated.
    kappa: a parameter describing the stiffness (details see publication)
    chain_length: the total length of the chain.
    normalize: If this is True the sum of the returned pdf vector is normalized to one.
    distance: If this is False, the end-to-end vector distribution is calculated. If True the distribution 
    the pdf is integrated over a sphere, i.e., the pdf of the end-to-end distribution function 
    is multiplied with 4*pi*r**2.

    Returns
    -------
    An array of the pdf

    Examples
    --------

    >>> import mfm.math.functions.rdf as rdf
    >>> import numpy as np
    >>> r = np.linspace(0, 0.99, 50)
    >>> kappa = 1.0
    >>> rdf.worm_like_chain(r, kappa)
    array([  4.36400392e-06,   4.54198260e-06,   4.95588702e-06,
             5.64882576e-06,   6.67141240e-06,   8.09427111e-06,
             1.00134432e-05,   1.25565315e-05,   1.58904681e-05,
             2.02314725e-05,   2.58578047e-05,   3.31260228e-05,
             4.24918528e-05,   5.45365051e-05,   7.00005025e-05,
             8.98266752e-05,   1.15215138e-04,   1.47693673e-04,
             1.89208054e-04,   2.42238267e-04,   3.09948546e-04,
             3.96381668e-04,   5.06711496e-04,   6.47572477e-04,
             8.27491272e-04,   1.05745452e-03,   1.35165891e-03,
             1.72850634e-03,   2.21192991e-03,   2.83316807e-03,
             3.63314697e-03,   4.66568936e-03,   6.00184475e-03,
             7.73573198e-03,   9.99239683e-03,   1.29382877e-02,
             1.67949663e-02,   2.18563930e-02,   2.85090497e-02,
             3.72510109e-02,   4.86977611e-02,   6.35415230e-02,
             8.23790455e-02,   1.05199154e-01,   1.30049143e-01,
             1.49953168e-01,   1.47519190e-01,   9.57787954e-02,
             1.45297018e-02,   1.53180248e-08])

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

    if normalize:
        pr /= pr.sum()

    return pr


@jit(nopython=True)
def gaussian_chain(r, rmsR=100, *_, norm=True):
    """
    Calculates the radial distribution function of a Gaussian chain in three
    dimensions

    :param rmsR: float
        Root mean square end-to-end distance
    :param l: float
        The segment length
    :param r: numpy-array
        values of r should be in range [0, 1) - not including 1

    ..plot:: plots/rdf-gauss.py

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
