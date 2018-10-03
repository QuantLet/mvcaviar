#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr  6 09:45:42 2017

@author: egor
"""

from multiprocessing import Pool
from multiprocessing import current_process
import numpy as np
from scipy.optimize import minimize

from num_of_processes import NUM_OF_PROCESSES


class _optimize_func_:
    def __init__(self, f, method, itnum):
        self.f = f
        self.method = method
        self.itnum = itnum
    def __call__(self, par0):
        return minimize(self.f, par0, method = self.method, 
                        options = {'maxiter': self.itnum})
    
#
#  - f MUST be picklable and callable
#  - gen generates random par0
#
def multistart_optimization(f, starts, method):
    # strategy parameters
    tols = [2000., 3., .5, .1 , .05, .02, .01]
    itnums = [100, 200, 500, 1000, 2000, 2000, 5000]
    
    min_par, min_val = 0., 0.
    if __name__ == 'optmulti':
        with Pool(min(NUM_OF_PROCESSES, len(starts))) as p:
            pars = starts
            vals = p.map(f, pars)
            succs = [False for par in pars]
            
            min_val, idx = min((val, idx) for (idx, val) in enumerate(vals))
            min_par = pars[idx]
            for step in range(len(tols)):
                #print("=======| m-start-opt: minimum value so far:", min_val)
                
                pars_in_progress = []
                for i in range(len(pars)):
                    if vals[i] <= min_val + tols[step] and succs[i] == False:
                        pars_in_progress.append(pars[i])
                
                #print("=======| m-start-opt: candidates in progress:", len(pars_in_progress))
                if len(pars_in_progress) == 0:
                    break
                
                ress = p.map(_optimize_func_(f, method, itnums[step]), pars_in_progress)
                pars = [res.x for res in ress]
                vals = [res.fun for res in ress]
                succs = [res.success for res in ress]
                
                _min_val, idx = min((val, idx) for (idx, val) in enumerate(vals))
                if _min_val < min_val:
                    min_par = pars[idx]
                    min_val = _min_val
    
    return (min_par, min_val)

def multistart_optimization_simple(f, starts, method):
    # strategy parameters
    tols = [2000., 3., .5, .1 , .05, .02, .01]
    itnums = [100, 200, 500, 1000, 2000, 2000, 5000]
    
    min_par, min_val = 0., 0.
    
    pars = starts
    vals = list(map(f, pars))
    succs = [False for par in pars]

    min_val, idx = min((val, idx) for (idx, val) in enumerate(vals))
    min_par = pars[idx]
    for step in range(len(tols)):
        #print("=======| m-start-opt-simple: minimum value so far:", min_val)

        pars_in_progress = []
        for i in range(len(pars)):
            if vals[i] <= min_val + tols[step] and succs[i] == False:
                pars_in_progress.append(pars[i])

        #print("=======| m-start-opt-simple: candidates in progress:", len(pars_in_progress))
        if len(pars_in_progress) == 0:
            break

        ress = list(map(_optimize_func_(f, method, itnums[step]), pars_in_progress))
        pars = [res.x for res in ress]
        vals = [res.fun for res in ress]
        succs = [res.success for res in ress]

        _min_val, idx = min((val, idx) for (idx, val) in enumerate(vals))
        if _min_val < min_val:
            min_par = pars[idx]
            min_val = _min_val

    return (min_par, min_val)
                