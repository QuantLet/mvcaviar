#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr  6 09:45:42 2017

@author: egor
"""
from multiprocessing import Pool
from multiprocessing import current_process
import numpy as np
from num_of_processes import NUM_OF_PROCESSES
#
#  
#

def generate_random_weights(num):
    # this function generates random mean-1 var-1 weights that take values in [0, 4]
    # SHOULD ONLY BE APPLIED IN MAIN PROCESS!
    #
    trans = lambda x : 4 * x / 3 if x <= 3/4 else 12 * x - 8
    return np.vectorize(trans)(np.random.rand(num))

class _cp_test_wrapper:
    def __init__(self, model, interval, breks, estimate, estimate_with_break, prep):
        self.model = model

        self.interval = interval
        self.breks = breks

        self.estimate = estimate
        self.estimate_with_break = estimate_with_break

        self.model.set_default_weights()

        self.par, val = estimate(model, interval, prep.get_prepared_pars(interval))        
        self.pars_l = []
        self.pars_r = []
        vals_lr = []

        self.counter = 0

        for brek in breks:
            interval_l = (interval[0], brek)
            interval_r = (brek, interval[1])

            starts_l = prep.get_prepared_pars(interval_l) + [self.par]
            if len(self.pars_l) > 0:
                starts_l.append(self.pars_l[-1])

            model.do_prepare(interval)
            par_l, val_l = estimate(model, interval_l, starts_l, prepare = False)

            starts_r = prep.get_prepared_pars(interval_r) + [self.par, par_l]
            if len(self.pars_r) > 0:
                starts_r.append(self.pars_r[-1])

            model.do_calculate(interval_l, par_l)
            par_r, val_r = estimate(model, interval_r, starts_r, prepare = False)

            vals_lr.append(val_l + val_r)
            self.pars_l.append(par_l)
            self.pars_r.append(par_r)

        self.test_val = val - min(vals_lr)
        print("Original test done for", self.interval, "; now starting bootstrap with multiprocessing...")

    def __call__(self, weights):
        self.model.set_weights(self.interval, weights)

        btest_vals = []
        for i in range(len(self.breks)):
            brek = self.breks[i]
            interval_l = (self.interval[0], brek)
            interval_r = (brek, self.interval[1])

            self.model.do_prepare(self.interval)
            bpar_l, bval_l = self.estimate(self.model, interval_l, [self.pars_l[i]], prepare = False)

            self.model.do_calculate(interval_l, bpar_l)
            bpar_r, bval_r = self.estimate(self.model, interval_r, [self.pars_r[i], bpar_l], prepare = False)

            bpar, bval = self.estimate_with_break(self.model, self.interval, brek, #self.pars_r[i] - self.pars_l[i], 
                                                  np.zeros(self.model.parlen()), [bpar_l, self.par])
            btest_vals.append(bval - bval_r - bval_l)

        self.counter += 1
        print(current_process().name + ": did", self.counter, "b_vals for", self.interval)

        return max(btest_vals)



def bstrap_cp_test(model, interval, breks, estimate, estimate_with_break, prep, SIM_NUM):
    wrapper = _cp_test_wrapper(model, interval, breks, estimate, estimate_with_break, prep)

    int_len = interval[1] - interval[0]
    weights_list = [generate_random_weights(int_len) for x in range(SIM_NUM)]
    btest_vals = []
    with Pool(NUM_OF_PROCESSES) as p:
        btest_vals = p.map(wrapper, weights_list)

    return (wrapper.test_val, btest_vals)


    
    








    
