#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr  6 09:45:42 2017

@author: egor
"""

import numpy as np
from scipy.optimize import minimize
from optmulti import multistart_optimization, multistart_optimization_simple
from multiprocessing import Pool
from math import sqrt

def q_check(x, tau):
    return (tau - float(x <= 0)) * x

def q_check_smooth(x, tau, h):
    # smoothing by kernel (1 - x^2)^2 on [-1, 1]
    # scaled by bandwitdh h
    # calculations are done via wolframalpha.com
    #
    if x >= h:
        return tau * x
    if x <= -h:
        return (tau - 1.) * x
    a = -x / h
    return tau * x - h * (a ** 2 - 1) ** 3 / 1.00667 \
           - x * ((a + 1) ** 3) * (3 * a ** 2 - 9 * a + 8) / 16.0005

    
#
# ANYTIME ONE OF THE FOLLOWING METHODS IS USED:
#  - calc_loss
#  - gen_candidate
#  - fill_quantiles
# ONE SHOULD ENSURE THAT EITHER
#  - fill_lagged_quantiles
# OR, ON SOME PRECEEDING INTERVAL,
#  - fill_quantiles
# WAS CALLED PRELIMINARY
#
class mvcaviar_model:
    def __init__(self, Y, lag, thetas, f_of_y = np.abs):
        # initialization
        #
        self.Y = Y
        self.lag = lag
        self.thetas = thetas

        # dimensions
        (self.n, self.tmax) = np.shape(Y)
        self._parlen_ = self.n * (self.lag * self.n + self.lag * self.n + 1)
        
        #fill phi
        self._fill_phi_(f_of_y)
        
        # weights
        self.weights = np.ones(self.tmax)
        
        #reserve place for conditional quantiles
        self._cond_qs_ = np.empty((self.n, self.tmax))
        
        #crutch options for avoiding bad points
        self.set_default_crutch_options()
    
    def _fill_phi_(self, f_of_y):
        # first self.lag columns are uninformative: to avoid index confusions
        self._phi_ = np.zeros((self.n * self.lag + 1, self.tmax))
        self._phi_[self.n * self.lag, self.lag:] = np.ones((self.tmax - self.lag))
        
        # the first n * lag entries of each column are arranged as follows:
        # say, for  lag = 4 for each shift we have first n entries with 
        # shift 1, then n with shift 2 and so on:
        # (   ,   ,   ,   )
        #   n   n   n   n   --- lag times n
        #
        for i in range(self.lag):
            self._phi_[(self.n * i):(self.n * i + self.n) , self.lag :] = \
                      f_of_y(self.Y[:, (self.lag - 1 - i):(self.tmax - 1 - i)])
        
    def set_default_crutch_options(self):
        # TODO:
        # this values should in fact be scaled by unconditional quantiles,
        # or the values of the time series itself
        self.QUANT_MAX = 10.
        self.MEANRISK_MAX = 5.
    
    def parlen(self):
        return self._parlen_
  
    def set_default_weights(self):
        self.weights[:] = 1.
  
    def set_weights(self, interval, weights):
        if not len(weights == interval[1] - interval[0]):
            raise ValueError("Number of weights does not match the interval.")
        self.weights[interval[0]:interval[1]] = weights

    def _parsepar_(self, par):
        # (!) TODO EXCEPTION: check size(par) == _parlen_

        # Let us explain the parameter packing:
        # first go n parameters linking 1st quantile to all n 1-lagged quantiles,
        #  then n linking 1st to 2-lagged, and so on (overall mn)
        # then mn parameters linking 2nd quantile to lagged quantiles and so on; 
        # then go nq+1 parameters linking 1st quantile to Phi, 
        # then 2nd and so on.
        #
        B = np.asmatrix(np.reshape(
              par[:(self.n * self.n * self.lag)], (self.n, self.n * self.lag)
            ))
        A = np.asmatrix(np.reshape(
              par[(self.n * self.n * self.lag):], (self.n, self.lag * self.n + 1)
            ))
        return (B, A)
  
    def _fill_lagged_quantiles_(self, interval):
        #global quantile
        (a, b) = interval
        quant0 = np.empty(self.n)
        for i in range(self.n):
            quant0[i] = np.percentile(self.Y[i, a:b], 100 * self.thetas[i])
    
        #prepare intial quantiles
        for l in range(1, self.lag + 1):
            self._cond_qs_[:, a - l] = quant0
                     
    def _fill_quantiles_(self, par, interval, with_check = False):
        # interval is a tuple!
        # (!) TODO EXCEPTION: make sure that it fits into the current data interval!
        
        # TODO: stop halfway of quantiles become too large
        (a, b) = interval

        #clean
        self._cond_qs_[:, a:b] = 0.

        (B, A) = self._parsepar_(par)

        # B and A are already matrices!!!
        for t in range(a, b):
            self._cond_qs_[:, t] += (self._phi_[:, t]).dot(np.asarray(A.T))
            
            for l in range(1, self.lag + 1):
                self._cond_qs_[:, t] += (self._cond_qs_[:, t - l]).dot(
                    np.asarray(B[:, (self.n * (l - 1)):(self.n * l)].T))
            
            if with_check and any(np.abs(self._cond_qs_[:, t]) > 
                                  np.abs(self.Y[:, t]) + self.QUANT_MAX):
                return False
        
        return True
    
    def get_quantiles(self, t):
        return self._cond_qs_[:, t]

    def _final_calc_loss_(self, interval, check_func):
        (a, b) = interval
        res = 0
        for i in range(self.n):
            res += np.inner(self.weights[a:b], 
                            np.vectorize(lambda x: check_func(x, self.thetas[i]))(
                                self.Y[i, a:b] - self._cond_qs_[i, a:b]))
        return res

    def calc_loss(self, par, interval):
        self._fill_quantiles_(par, interval)
        return self._final_calc_loss_(interval, q_check)
    
    def calc_smooth_loss(self, par, interval, h):
        self._fill_quantiles_(par, interval)
        return self._final_calc_loss_(interval, lambda x, tau : q_check_smooth(x, tau, h))
    
    def gen_candidate(self, interval):
        limit = self.MEANRISK_MAX * (interval[1] - interval[0])
        while True:
            par = np.random.rand(self._parlen_) * 4. - 2.
            if not self._fill_quantiles_(par, interval, with_check = True):
                next
            val = self._final_calc_loss_(interval, q_check)
            if val <= limit:
                return par
    
    def distance_between_pars(self, interval, par1, par2):
        self._fill_lagged_quantiles_(interval)
        
        self._fill_quantiles_(par1, interval)
        qs1 = np.array(self._cond_qs_[:, interval[0] : interval[1]]) # save a copy

        self._fill_quantiles_(par2, interval)
        qs2 = self._cond_qs_[:, interval[0] : interval[1]] # no copy required here
        
        return np.sum(np.abs(qs1 - qs2)) / (interval[1] - interval[0])
    
    def do_prepare(self, interval):
        self._fill_lagged_quantiles_(interval)
    
    def do_calculate(self, interval, par):
        self._fill_quantiles_(par, interval)
    
    def get_loss(self, interval):
        return self._final_calc_loss_(interval, q_check)
    
    def t_max(self):
        return self.tmax
    
    def t_min(self):
        return self.lag


#
#
#
class model_interval_wrapper:
    def __init__(self, model, interval):
        self.model = model
        self.interval = interval 
    def __call__(self, par):
        self.model._fill_quantiles_(par, self.interval)
        return self.model._final_calc_loss_(self.interval, q_check)


class model_interval_with_break_wrapper:
    def __init__(self, model, interval, brek, shift):
        self.model = model
        self.interval = interval
        self.interval_l = (interval[0], brek)
        self.interval_r = (brek, interval[1])
        self.shift = shift
    def __call__(self, par):
        self.model._fill_quantiles_(par, self.interval_l)
        self.model._fill_quantiles_(par + self.shift, self.interval_r)
        return self.model._final_calc_loss_(self.interval, q_check)


class model_interval_generator:
    def __init__(self, model, interval):
        self.model = model
        self.interval = interval
    def __call__(self, dummy):
        return self.model.gen_candidate(self.interval)

#
#
# 
def estimate(model, interval, start_candidates, num_gen_starts, prepare = True):
    if prepare:
        model.do_prepare(interval)
    
    starts = start_candidates
    if __name__ == 'mvcaviar':
        with Pool(16) as p:
            starts += p.map(model_interval_generator(model, interval), [0] * num_gen_starts)
    
    print(len(starts), "candidates generated successfully!")
    
    wrap_model = model_interval_wrapper(model, interval)
    return multistart_optimization(wrap_model, starts, 'Nelder-Mead')


def estimate_simple(model, interval, starts, prepare = True):
    if prepare:
        model._fill_lagged_quantiles_(interval)
    
    wrap_model = model_interval_wrapper(model, interval)
    return multistart_optimization_simple(wrap_model, starts, 'Nelder-Mead')


def estimate_with_break_simple(model, interval, brek, shift, starts, prepare = True):
    if prepare:
        model._fill_lagged_quantiles_(interval)

    wrap_model = model_interval_with_break_wrapper(model, interval, brek, shift)
    return multistart_optimization_simple(wrap_model, starts, 'Nelder-Mead')
    
    
