import sys

import numpy as np
from pandas import read_csv, DataFrame
from scipy.stats import norm
import matplotlib.pyplot as plt
import seaborn as sns
from math import ceil


import estimator
from misc import *
import adaptation


################################################################################
#  
#  the complete reproduction of the simulations is done in three steps:
#
#  1) simulate the data by simulations.py
#  
#  2) produce the bootstrap values using adaptation_full; script.py helps to run independent tasks in separate processes
#
#  3) collect all the bvals, decide the interval lenght and conduct the one step ahead prediction using estimate_with_window; script_ahead.py helps to do it in parallel
#
#
#

def adaptation_full(dump=False):
    assert (len(sys.argv) == 1 or len(sys.argv) == 3)

    tau = .1
    quantile = norm.ppf(tau)

    Y = read_csv('sim_ts.csv', header=None).values.T
    sigmas = read_csv('sim_sigmas.csv', header=None).values.T

    (tmax, n) = np.shape(Y)
    step = 20
    min_t = 60

    if len(sys.argv) == 1:
        end_points = range(tmax, min_t, -step)
        filename = "bvals_{}to{}step{}.csv".format(tmax, min_t, step)
    else:
        end = (int(sys.argv[1]) // step) * step
        start = int(sys.argv[2])

        assert (start >= min_t)
        assert (end <= tmax)

        end_points = range(end, start, -step)
        filename = "bvals_{}to{}step{}.csv".format(end, start, step)

    print("doing {} .....".format(filename))

    lengths = [ceil(60 * (1.25 ** k)) for k in range(8)]

    if not dump:
        results = {}
        for end_t in end_points:
            for i in range(1, len(lengths)):
                if end_t < lengths[i]:
                    break
                message = "time{}length{}vs{}".format(end_t, lengths[i], lengths[i - 1])
                val, bvals = adaptation.bstrap_cp_test(
                    Y[(end_t - lengths[i]):end_t, :], tau, [lengths[i] - lengths[i - 1]], 30,
                    message=message, prev_quantile=sigmas[end_t - lengths[i], :] * quantile
                )
                results[message] = [val] + bvals

        df = DataFrame.from_dict(results)
        df.to_csv(filename)


def estimate_with_window():
    assert(len(sys.argv) == 4)

    end_t = int(sys.argv[1])
    start_t = int(sys.argv[2])
    length = int(sys.argv[3])

    tau= 0.1
    Y = read_csv('sim_ts.csv', header=None).values.T

    predict_ahead = []
    for t in range(end_t, start_t, -1):
        res = estimator.train_mvcaviar(Y[end_t - length: t], tau, epochs=10000, verbose=False)

        quant_last = res.y_predict[[-1], :]
        x_last = np.abs(Y[[t-1], :])

        quant_next = np.matmul(quant_last, res.pars['recurrent']) + np.matmul(x_last, res.pars['kernel']) + res.pars['bias']
        predict_ahead.append(quant_next[0])

    filename = "ahead_{}to{}length{}.csv".format(end_t, start_t, length)
    np.savetxt(filename, np.array(predict_ahead), delimiter=',')



if __name__ == "__main__":
    estimate_with_window()
