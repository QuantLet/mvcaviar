import numpy as np
from math import isnan
from scipy.stats import norm


def generate_random_weights(num):
    # SHOULD ONLY BE APPLIED IN MAIN PROCESS!
    #
    # let p > q and p + q = 1 then P(X = 1 - \sqrt{q/p}) = p and P(X = 1 + \sqrt{p/q}) = q
    # ensures that E(X) = 1 and Var(X) = 1 and X > 0 a.s.
    # moreover, taking p = 4/5 and q = 1/5 we have X \in [1 - 1/2, 1 + 2] = [0.5, 3]
    #

    def trans(x):
        if x < .8:
            return 0.5
        else:
            return 3.0
    return np.vectorize(trans)(np.random.rand(num))


def dict_of_pars(kernel, recurrent, bias):
    return {
        'kernel': kernel,
        'recurrent': recurrent,
        'bias': bias
    }


def pars_diff(pars1, pars2):
    return {name: pars1[name] - pars2[name] for name in ['kernel', 'recurrent', 'bias']}


def pars_sum(pars1, pars2):
    return {name: pars1[name] + pars2[name] for name in ['kernel', 'recurrent', 'bias']}


def pars_left_right(pars_l, pars_r):
    pars_left = {name + '_left': pars_l[name] for name in ['kernel', 'recurrent', 'bias']}
    pars_right = {name + '_right': pars_r[name] for name in ['kernel', 'recurrent', 'bias']}
    return {**pars_left, **pars_right}


def get_left_pars(pars):
    if pars is None:
        return None
    return dict_of_pars(pars['kernel_left'], pars['recurrent_left'], pars['bias_left'])


def get_right_pars(pars):
    if pars is None:
        return None
    return dict_of_pars(pars['kernel_right'], pars['recurrent_right'], pars['bias_right'])


def clean_vec_from_nans(values):
    res = []
    for x in values:
        if not isnan(x):
            res.append(x)
    return np.array(res)


def two_moment_quantile(x, tau):
    mu, std = norm.fit(np.sqrt(np.max(x, 0)))

    return (mu + std * norm.ppf(tau)) ** 2
