# FRETlines generator - A python library for the calcualtion of FRETlines
#    Copyright (C) 2020  Anders Barth
from .models import distance_distributions as d
import numpy as np


def static_FRET_line(R0=50, model=d.normal_distribution, param=(50, 5),
                     variable_parameter=0, param_range=(1, 200), mode='Etau'):
    """ Perform integration over distance distribution to determine the
    species-averaged and intensity-average fluorescence lifetimes.

    :param R0: Förster radius in angstrom
    :param model: Distance distribution model, imported from
    distance_distributions submodule.
    :param param: Parameters of the distance distribution (tuple)
    :param variable_parameter: The variable parameter for the generation of the
    FRET-line, given as a number (0 or 1 for the first or second parameter of
    the distance distribution). See the definitons of the distance
    distributions for the meaning of the input parameters. For normal and chi
    distributions, the parameters denote the center distance and distribution
    widths. For the Gaussian chain model, only the first parameter is used,
    giving the root-mean-square end-to-end distance. For the worm-like chain
    model, the first parameter denotes the chain length and the second
    parameter describes the chain stiffness kappa, given as the ratio of the
    persistence length to the chain length.
    :param param_range: Range over which the variable parameter is varied
    (tuple).
    :param mode: Select output mode - 'Etau' or 'moments'.
    Etau' outputs the FRET efficiency and the intensity-averaged donor
    fluorescence lifetime. 'moments' ouputs the FRET efficiency and the
    difference between the first and second moment of the lifetime
    distribution. All values are given in units of the donor-only fluorescence
    lifetime, i.e. tauD0=1.

    :return: Dependent on 'mode' argument (see above).
    """

    xR = np.arange(0.1, 4*R0, 0.1)

    if variable_parameter is 0:
        m = lambda x: model(xR, x, param[1])
    elif variable_parameter is 1:
        m = lambda x: model(xR, param[0], x)

    tau = (1+(R0/xR) ** 6) ** (-1)

    tau1 = []  # first moment
    tau2 = []  # second moment
    for par in np.linspace(param_range[0], param_range[1], 1000):
        p = m(par)
        # integrate
        tau1.append(np.sum(p * tau))
        tau2.append(np.sum(p * (tau ** 2)))

    tau1 = np.array(tau1)
    tau2 = np.array(tau2)

    if mode is 'Etau':
        return 1-tau1, tau2/tau1
    elif mode is 'moments':
        return 1-tau1, tau1-tau2


def dynamic_FRET_line(R0=50,
                      model1=d.normal_distribution,
                      model2=d.normal_distribution,
                      param1=(40, 5),
                      param2=(60, 5),
                      mode='Etau'):
    """ Construct dynamic FRET-line between two points.

    :param R0: Förster radius in angstrom
    :param model1: Distance distribution model for the start point, imported
    from distance_distributions submodule.
    :param model2: Distance distribution model for the end point, imported from
    distance_distributions submodule.
    :param param1: Parameters of the distance distribution for the start point
    (tuple)
    :param param2: Parameters of the distance distribution for the end point
    (tuple)
    :param mode: Select output mode - 'Etau' or 'moments'.
    Etau' outputs the FRET efficiency and the intensity-averaged donor
    fluorescence lifetime. 'moments' ouputs the FRET efficiency and the
    difference between the first and second moment of the lifetime
    distribution. All values are given in units of the donor-only fluorescence
    lifetime, i.e. tauD0=1.

    :return: Dependent on 'mode' argument (see above).
    """
    xR = np.arange(0.1, 4*R0, 0.1)
    tau = (1 + (R0/xR) ** 6) ** (-1)
    # get first and second moment of lifetime distribution at pure states
    tau1_1 = np.sum(tau * model1(xR, param1[0], param1[1]))
    tau2_1 = np.sum((tau ** 2) * model1(xR, param1[0], param1[1]))
    tau1_2 = np.sum(tau * model2(xR, param2[0], param2[1]))
    tau2_2 = np.sum((tau ** 2) * model2(xR, param2[0], param2[1]))

    # calculate dynamic FRET-line
    if mode is 'Etau':
        f1 = np.linspace(0, 1, 1000)
        tau1 = tau1_1 * f1 + tau1_2 * (1-f1)
        tau2 = tau2_1 * f1 + tau2_2 * (1-f1)
        return 1 - tau1, tau2/tau1
    elif mode is 'moments':
        return np.array((1 - tau1_1, 1 - tau1_2)), np.array((tau1_1 - tau2_1,
                                                             tau1_2 - tau2_2))
