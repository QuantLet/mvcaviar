import numpy as np
from pandas import read_csv, DataFrame
import matplotlib.pyplot as plt
import subprocess
from scipy.stats import norm
from math import ceil
import sys
from misc import *


def run(operation):
    points = list(range(500, 40, -20))
    step = 20
    lengths = [ceil(60 * (1.25 ** k)) for k in range(8)]

    selected_lengths = {}
    tasks = []
    for (start, end) in zip(points[:-1], points[1:]):
        df = read_csv("simu_bvals2/bvals_{}to{}step{}.csv".format(start, end, step))
        for t in range(start, end, -step):
            print()
            print("time point {}".format(t))
            print()
            selected_lengths[t] = lengths[0]
            allpass = True
            for k in range(1, 8):
                len1 = lengths[k]
                len2 = lengths[k-1]
                if len1 > t:
                    break

                val = df["time{}length{}vs{}".format(t, len1, len2)][0]
                bvals = df["time{}length{}vs{}".format(t, len1, len2)][1:]
                #bvals_clean = clean_vec_from_nans(bvals)
                cr_val = two_moment_quantile(bvals, .95) * 0.92

                if val <= cr_val:
                    res = "PASS"
                    comp = "{} <= {}".format(val, cr_val)
                else:
                    res = "REJECT"
                    comp = "{} > {}".format(val, cr_val)
                    allpass = False

                if allpass:
                    selected_lengths[t] = len1

                #print("{} vs {}: {};  {}".format(len1, len2, res, comp))

            tasks.append(["python", "test_tf.py", str(t), str(t - 20), str(selected_lengths[t])])

    if operation == "DO":
        procs = [subprocess.Popen(task) for task in tasks]
        for i, proc in enumerate(procs):
            out, err = proc.communicate()
            if err is not None:
                print("ERROR for TASK {}".format(" ".join(tasks[i])))
                print(err)

    elif operation == "COLLECT":
        tau = .1
        quantile1 = norm.ppf(tau)
        quantile2 = norm.ppf(0.05)

        Y = read_csv('sim_ts.csv', header=None).values.T
        sigmas = read_csv('sim_sigmas.csv', header=None).values.T

        lst = []
        for (start, end) in zip(points[:-1], points[1:]):
            filename = "ahead/ahead_{}to{}length{}.csv".format(start, end, selected_lengths[start])
            lst.append(np.loadtxt(filename, dtype=np.float32, delimiter=','))

        ahead_quants = np.concatenate(lst, axis=0)

        fig, ax = plt.subplots()

        ax.plot(list(range(61, 500)), Y[61:500, 1], color='g', linewidth=.5)
        ax.plot(list(range(61, 500)), sigmas[61: 500, 1] * quantile2, color='y', linewidth=1.)
        ax.plot(list(range(61, 500)), ahead_quants[list(range(439, 0, -1)), 1] * quantile2 / quantile1, color='r', linewidth=1.)

        # Turn off tick labels
        #ax.set_yticklabels([])
        #ax.set_xticklabels([])

        plt.tick_params(direction='in', top=True, right=True)
        plt.show()


run("COLLECT")