from .models import distance_distributions as d
import numpy as np


def static_FRET_line(R0=50, param=(50, 5), model=d.normal_distribution,
                     variable_parameter=1, param_range=(1, 200), mode='Etau'):
    """ Perform integration over distance distribution to determine the
        species-averaged and intensity-average fluorescence lifetimes.
    """

    xR = np.arange(0.1, 4*R0, 0.1)

    if variable_parameter is 1:
        m = lambda x: model(xR, x, param[1])
    elif variable_parameter is 2:
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
