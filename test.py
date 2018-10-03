# -*- coding: utf-8 -*-
"""
Created on Sat Mar 25 14:39:07 2017

@author: egor
"""
import sys
import numpy as np
from math import sqrt, pi, erf, cos, sin, exp

"""
TASKS:
     
    !!!) iterative optimization probably makes sense
    
"""

from optmulti import multistart_optimization
import mvcaviar
import ts_estimation_preparator

from pandas import read_csv
from bootstrap_change_point import bstrap_cp_test

#
#
#
#
#
def prepare():
    Y = np.loadtxt('crix_data/CRIX_ETH_returns_20150810_20180116.txt').T
    model = mvcaviar.mvcaviar_model(Y, 1, [.01, .01])
    b_cp = ts_estimation_preparator.ts_estimation_preparator(model)

    tmax = np.shape(Y)[1]
    edges = [x for x in reversed(range(tmax, 0, -30))]
    
    b_cp.do_preparation(edges, mvcaviar.estimate)
    b_cp.save_preparation("my_prep")
    b_cp.load_preparation("my_prep")

def test_prepare():
    Y = np.loadtxt('crix_data/CRIX_ETH_returns_20150810_20180116.txt').T
    model = mvcaviar.mvcaviar_model(Y, 1, [.01, .01])
    b_cp = ts_estimation_preparator.ts_estimation_preparator(model)
    b_cp.load_preparation("my_prep")
    print(b_cp.prep_edges)

    tmax = np.shape(Y)[1]
    interval = (tmax- 24, tmax -1)
    print(interval)
    print(b_cp.get_prepared_pars(interval))

def test_btest():
    Y = np.loadtxt('crix_data/CRIX_ETH_returns_20150810_20180116.txt').T
    model = mvcaviar.mvcaviar_model(Y, 1, [.01, .01])
    b_cp = ts_estimation_preparator.ts_estimation_preparator(model)
    b_cp.load_preparation("my_prep")

    tmax = np.shape(Y)[1]

    lens = [60, 75, 94, 118, 148, 185, 231, 289, 361, 451, 500]

    vals = []
    vals09 = []
    vals08 = []
    for i in range(len(lens) - 1):
        interval = (tmax- lens[i + 1], tmax)
        breks = list(range(tmax - lens[i], tmax - lens[i + 1], -5))

        (val, bvals) = bstrap_cp_test(model, interval, breks, 
                                  mvcaviar.estimate_simple, mvcaviar.estimate_with_break_simple, 
                                  b_cp, 48)

        vals.append(val)
        vals08.append(np.percentile(bvals, 80))
        vals09.append(np.percentile(bvals, 90))

    print("test", vals)
    print("20", vals08)
    print("10", vals09)

def test_rolling_window():
    Y = np.loadtxt('crix_data/CRIX_ETH_returns_20150810_20180116.txt').T
    model = mvcaviar.mvcaviar_model(Y, 1, [.01, .01])
    prep = ts_estimation_preparator.ts_estimation_preparator(model)
    prep.load_preparation("my_prep")
    print(prep.prep_edges)

    tmax = np.shape(Y)[1]
    length = 180
    step = 10

    intervals = [ (start, start + length) for start in reversed(range(tmax - length, 1, -step))]

    pars = ts_estimation_preparator.rolling_window(
            model, intervals, mvcaviar.estimate, prep, 
            strategy = {"prev3" : True, "global" : False, "firstmulti": True, "prep": False}
        )

    name = "rollwin_len" + str(length) + "step" + str(step) + ".txt"
    np.savetxt(name, pars)


if __name__ == '__main__':
    test_btest()

