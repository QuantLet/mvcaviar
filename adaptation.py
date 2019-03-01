import numpy as np
from collections import namedtuple
from scipy.stats import norm

# local
import estimator
from misc import *


TestResult = namedtuple("TestResult", ["test_val", "res_joint", "res_sep"])
EPOCHS = 10000
EPOCHS_BOOTSTRAP = 500


def cheat_estimation(train_func, is_separated, *args, **kwargs):
    tau = .1
    quantile = norm.ppf(tau)

    theta1 = dict_of_pars(np.array([[0.0, 0.2], [0.2, 0.0]]) * quantile,
                          np.array([[0.5, 0.0], [0.0, 0.5]]),
                          np.array([[0.5, 0.5]]) * quantile)
    theta2 = dict_of_pars(np.array([[0.0, 0.2], [0.2, 0.0]]) * quantile,
                          np.array([[-0.5, 0.0], [0.0, 0.5]]),
                          np.array([[0.5, 0.5]]) * quantile)
    if not is_separated:
        cheat_pars = [theta1, theta2]
    else:
        cheat_pars = [pars_left_right(theta1, theta1), pars_left_right(theta1, theta2),
                      pars_left_right(theta2, theta2)]

    ress = [train_func(*args, **kwargs, init_pars=pars) for pars in cheat_pars]
    _, idx = min([(res.loss, i) for (i, res) in enumerate(ress)])
    return ress[idx]


def _train_mvcaviar(*args, **kwargs):
    return cheat_estimation(estimator.train_mvcaviar, *args, **kwargs)


def _train_mvcaviar_w_shift(*args, **kwargs):
    return cheat_estimation(estimator.train_mvcaviar_w_shift, *args, **kwargs)


def _train_mvcaviar_separated(*args, **kwargs):
    return cheat_estimation(estimator.train_mvcaviar_separated2, *args, **kwargs)


class _CPTestWrapper:

    def __init__(self, Y, tau, breaks, message="X", prev_quantile=None):
        self.Y = Y
        self.breaks = breaks
        self.tau = tau
        self.message = message
        self.counter = 0
        (self.tmax, self.n) = np.shape(Y)
        self.prev_quantile = prev_quantile

        self.do_test()

    def do_test(self):
        self.res_joint = _train_mvcaviar(self.Y, self.tau, epochs=EPOCHS, prev_quantile=self.prev_quantile)
        self.res_breaks = [
            _train_mvcaviar_separated(self.Y, self.tau, break_point, epochs=EPOCHS,
                                               #init_pars=pars_left_right(self.res_joint.pars, self.res_joint.pars),
                                               prev_quantile=self.prev_quantile)
            for break_point in self.breaks
        ]
        self.test_val = self.res_joint.loss - min([res.loss for res in self.res_breaks])

        print("Task {}: original test done".format(self.message))

    def __call__(self, weights):
        vals = []

        for i, break_point in enumerate(self.breaks):
            shift = pars_diff(get_right_pars(self.res_breaks[i].pars), get_left_pars(self.res_breaks[i].pars))
            res_joint = estimator.train_mvcaviar_w_shift(
                self.Y, self.tau, break_point, shift,
                prev_quantile=self.res_breaks[i].prev_quantile,
                init_pars=get_left_pars(self.res_breaks[i].pars),
                sample_weights=weights, epochs=EPOCHS_BOOTSTRAP
            )
            res_break = estimator.train_mvcaviar_separated(
                self.Y, self.tau, break_point,
                prev_quantile=self.res_breaks[i].prev_quantile,
                init_pars=pars_left_right(res_joint.pars, pars_sum(res_joint.pars, shift)),
                sample_weights=weights, epochs=EPOCHS_BOOTSTRAP
            )
            vals.append(res_joint.loss - res_break.loss)

        self.counter += 1
        print("Task {}: did {} bootstrap tests".format(self.message, self.counter))
        return max(vals)


def bstrap_cp_test(Y, tau, breaks, SIM_NUM, message="X", prev_quantile=None, cheat_pars=None):
    assert(len(breaks) > 0)
    wrapper = _CPTestWrapper(Y, tau, breaks, message=message, prev_quantile=prev_quantile)

    np.random.seed(1)

    weights_list = [generate_random_weights(np.shape(Y)[0]-1) for x in range(SIM_NUM)]
    btest_vals = list(map(wrapper, weights_list))
    return wrapper.test_val, btest_vals


def select_homogeneity_interval_with_single_break(Y, tau, lengths, SIM_NUM=400):
    return
