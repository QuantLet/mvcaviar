# -*- coding: utf-8 -*-
"""
Created on Sat Mar 25 14:39:07 2017

@author: egor
"""

from math import sqrt
import copy

import numpy as np
import seaborn
import scipy as sp
from pandas import read_csv

import threading
import logging

import mvcaviar
import time

"""
TASKS:
    
    0) pay attention that now the vector of accuracies must be scaled
       by the length of the interval
    
    1) make a connection for estimating I1 and I2 via the initial 
       quantile vector
     
    2) don't use __fill_lagged_qunatiles__() when it does same thing over and over
    
"""


logging.basicConfig(level=logging.DEBUG,
                    format='(%(threadName)-10s) %(message)s',
                    )

def my_c(par1, par2):
    return np.concatenate((par1, par2))

def split_in_half(interval):
    (a, b) = interval
    I1 = (a, a + (b - a) // 2)
    I2 = (I1[1], b)
    
    return (I1, I2)

def calc_loss_split1(par1, solver, interval):
    solver.__fill_lagged_quantiles__(interval)
    
    #split interval in half
    I1, I2 = split_in_half(interval)
    
    solver.__fill_quantiles__(par1, I1)
    return solver.__calc_loss__(I1)

def calc_loss_split2(par2, solver, interval, par1):
    #split interval in half
    I1, I2 = split_in_half(interval)
    
    #possibly unneccesary
    solver.__fill_lagged_quantiles__(interval)
    solver.__fill_quantiles__(par1, I1)
    
    solver.__fill_quantiles__(par2, I2)
    return solver.__calc_loss__(I2)

def calc_loss(par, solver, interval):
    solver.__fill_lagged_quantiles__(interval)
    solver.__fill_quantiles__(par, interval)
    
    return solver.__calc_loss__(interval)

def calc_loss_split_shift(par, solver, interval, shift):
    solver.__fill_lagged_quantiles__(interval)
    
    I1, I2 = split_in_half(interval)
    solver.__fill_quantiles__(par, I1)
    solver.__fill_quantiles__(par + shift, I2)
    
    return solver.__calc_loss__(interval)

def calc_loss_split(par1, par2, solver, interval):
    return calc_loss_split_shift(par1, solver, interval, par2 - par1)


def my_nelder_mead_wrapper(func, x, xtol, args):
    return sp.optimize.minimize(
                func, x,
                method = 'nelder-mead', options = \
                {'xtol': xtol, 'maxiter': 5000, 'disp': False},
                args = args
              )

def task_split(tasks, threads, number):
    x = tasks // threads
    r = tasks % threads
    a = number * x + min(number, r)
    b = (number + 1) * x + min(number + 1, r)
    return (a, b)    

def my_multistart_worker(handler, i, interval, xtol, regime):
    logging.debug('taking ' + str(len(handler.init_pars[i])) + ' initializations')
    handler.solvers[i].set_default_weights()
    
    counter = 0
    
    # TODO: throw exception if regime is neither 'Split' or 'Total'
    
    # run optimization for each init in the i-th list,
    # write solution into i-th results list
    for init_par in handler.init_pars[i]:
        x, val = 0, 0
        if regime == "Total": 
            res = my_nelder_mead_wrapper(
                    calc_loss, init_par, xtol, (handler.solvers[i], interval)
                  )
            logging.debug(res.message)
            
            x, val = (res.x, res.fun)
        
        if regime == "Split":
            par1, par2 = handler.solvers[i].split_par(init_par)
            res1 = my_nelder_mead_wrapper(
                    calc_loss_split1, par1, xtol,
                    (handler.solvers[i], interval)
                  )
            res2 = my_nelder_mead_wrapper(
                    calc_loss_split2, par2, xtol,
                    (handler.solvers[i], interval, res1.x)
                  )
            logging.debug(res2.message)
            
            x = my_c(res1.x, res2.x)
            val = res1.fun + res2.fun
        
        counter += 1
        logging.debug(str(counter) + '/' + str(len(handler.init_pars[i])))
        
        handler.res_pars[i].append(x)
        handler.res_vals[i].append(val)
    
    return

def my_bootstrap_worker(handler, i, interval, par12, SIM_NUM):
    logging.debug('simulating ' + str(SIM_NUM) + ' bootstrap tests')
    
    for k in range(SIM_NUM):
        handler.solvers[i].set_random_weights()
        par1, par2 = handler.solvers[i].split_par(par12)
        # starting point for par[0] minimization is par1!!!
        pars = [par1, par1, par2]
        vals = [0., 0., 0.]
        
        ITERATIONS = 1
        xtols = [1e-6, 1e-6, 1e-8, 1e-12]
        
        for it in range(ITERATIONS):
            res0 = my_nelder_mead_wrapper(calc_loss_split_shift, pars[0], 
                                           xtols[it], (handler.solvers[i], 
                                                       interval,
                                                       par2 - par1))
            res1 = my_nelder_mead_wrapper(calc_loss_split1, pars[1], 
                                           xtols[it], (handler.solvers[i], interval))
            res2 = my_nelder_mead_wrapper(calc_loss_split2, pars[2], 
                                           xtols[it], (handler.solvers[i], interval, res1.x))
            
            pars = [res0.x, res1.x, res2.x]
            vals = [res0.fun, res1.fun, res2.fun]
            logging.debug(res0.message)
        
        #additional step ensures that the statistsics is nonnegative
        pars12 = [pars[0], pars[0] + par2 - par1]
        vals12 = [.0, 0.]
        for it in range(ITERATIONS):
            res1 = my_nelder_mead_wrapper(calc_loss_split1, pars12[0],
                                         xtols[2], (handler.solvers[i], interval))
            res2 = my_nelder_mead_wrapper(calc_loss_split2, pars12[1],
                                         xtols[2], (handler.solvers[i], interval, res1.x))
            pars12, vals12 = ([res1.x, res2.x], [res1.fun, res2.fun])
        
        handler.bvals[i].append(vals[0] - min(vals[1] + vals[2], vals12[0] + vals12[1]))
    
    #return the weights to ones
    handler.solvers[i].set_default_weights()


class CaviarWithThreadsHandler:
    def __init__(self, Y, m, q, thetas, THREADS_NUM):
        self.THREADS_NUM = THREADS_NUM
        self.solvers = []
        for i in range(THREADS_NUM):
            self.solvers.append(mvcaviar.MvCaviarSolver(
                            Y, m, q, thetas
                          ))
        
        # each thread has his own list to access
        self.__clean__()
        return
        
    def __clean__(self):
        self.init_pars = [[] for i in range(self.THREADS_NUM)]
        self.res_pars = [[] for i in range(self.THREADS_NUM)]
        self.res_vals = [[] for i in range(self.THREADS_NUM)]
        
        self.bvals = [[] for i in range(self.THREADS_NUM)]
        return
    
    def multistart_optim(self, inits, interval, xtol, regime):
        self.__clean__()
        
        # split initializations
        for i in range(self.THREADS_NUM):
            (a, b) = task_split(len(inits), self.THREADS_NUM, i)
            self.init_pars[i] = inits[a:b]
            
        # start threads
        ts = []
        for i in range(self.THREADS_NUM):
            t = threading.Thread(
                    target = my_multistart_worker, name = 'Terminator' + str(i),
                    args = (self, i, interval, xtol, regime)
                )
            ts.append(t)
            t.start()
        
        for i in range(self.THREADS_NUM):
            ts[i].join()
        
        # join results from all threads
        pars = [x for lst in self.res_pars for x in lst]
        vals = [x for lst in self.res_vals for x in lst]
        
        return (pars, vals)
    
    def bstrap_test_values(self, interval, par12, SIM_NUM):
        self.__clean__()
        
        #no much need to consider case of SIMNUM not divided by THREADSNUM
        ts = []
        for i in range(self.THREADS_NUM):
            t = threading.Thread(
                    target = my_bootstrap_worker, name = 'BLRTinator' + str(i),
                    args = (self, i, interval, par12, SIM_NUM // self.THREADS_NUM)
                )
            ts.append(t)
            t.start()
        
        for i in range(self.THREADS_NUM):
            ts[i].join()
        
        bvals = [y for lst in self.bvals for y in lst]
        return bvals

class wrapped_optimizer():
    def __init__(self, H, S0):
        self.H = H
        self.S0 = S0
    
    def bootstrap_test_values(self, interval, par12, SIM_NUM):
        #if no par0, par1 and par2 it must be obtained by method estimator()
        return self.H.bstrap_test_values(interval, par12, SIM_NUM)
    
    def estimator(self, typ, interval, my_inits, regime):
        #check if regime is 'Total' or 'Split' 
        #TODO: exception for other inputs
        
        #func = calc_loss
        #if regime == "Split":
        #    func = calc_loss_split
        
        #hard options
        ITERATIONS = 6
        xtols = [1e-3, 1e-5, 1e-8, 1e-12, 1e-12, 1e-12]
        accuracies = np.array([100, 0.1, 0.05, 0.01, 0.005, 0.001]) * \
                                                   (interval[1] - interval[0])
        
        #put my initializations
        pars = []
        vals = []
        for x in my_inits:
            pars.append(x)
            vals.append(1.0)
        
        #cases
        if typ == "Trust":
            pass
        
        if typ == "Global":
            INIT_NUM = 100
            if regime == "Split":
                INIT_NUM = 100
            for i in range(INIT_NUM):
                x = self.S0.gen_candidate(interval)
                if regime == "Split":
                    x = my_c(x, self.S0.gen_candidate(interval))
                pars.append(x)
                vals.append(1.0)
        
        #if typ == "Local":
        #   pass    
        
        print("Successful initialization!")
        time.sleep(0.5)
        
        for i in range(ITERATIONS):
            min_val = min(vals)
            good_pars = []
            for j in range(len(vals)):
                if vals[j] <= min_val + accuracies[i]:
                    good_pars.append(pars[j])

            (pars, vals) = self.H.multistart_optim(good_pars, interval, xtols[i], regime)
        
        val, idx = min((val, idx) for (idx, val) in enumerate(vals))
        return (pars[idx], val)

"""
Script:
"""

true_par = np.array([0.82, -0.01, -0.12, 0.96, -0.48, -0.05, -0.15, -0.3, -0.15, -0.1])

def experement1():
    csv_data = read_csv('data.csv')
    Y = np.array(csv_data).T
    
    S0 = mvcaviar.MvCaviarSolver(Y, 1, 1, np.array([0.01, 0.01]))
    H = CaviarWithThreadsHandler(Y, 1, 1, np.array([0.01, 0.01]), THREADS_NUM = 16)
    opt = wrapped_optimizer(H, S0)
    
    a = 500
    b = 1000
    
    I = (a, b)
    
    (par0, val0) = opt.estimator("Global", I, [], "Total")
    (par12, val12) = opt.estimator("Global", I, [my_c(par0, par0)], "Split")
    
    print(val0 - val12)
    
    par1, par2 = S0.split_par(par12)
    np.savetxt("pars012_" + str(a) + "_" + str(b) + ".txt", [par0, par1, par2], fmt = '%.8f')

def experement2():
    csv_data = read_csv('data.csv')
    Y = np.array(csv_data).T
    
    S0 = mvcaviar.MvCaviarSolver(Y, 1, 1, np.array([0.01, 0.01]))
    H = CaviarWithThreadsHandler(Y, 1, 1, np.array([0.01, 0.01]), THREADS_NUM = 16)
    opt = wrapped_optimizer(H, S0)
    
    a = 500
    b = 1000
    I = (a, b)
    
    pars012 = np.loadtxt("pars012_" + str(a) + "_" + str(b) + ".txt")
    
    par0, par1, par2 = pars012[0, :], pars012[1, :], pars012[2, :]
    val0 = calc_loss(par0, S0, I)
    val12 = calc_loss_split(par1, par2, S0, I)
    
    bvals = opt.bootstrap_test_values(I, my_c(par1, par2), 160)
    bvals.append(val0 - val12)
    
    np.savetxt("__bvals_" + str(a) + "_" + str(b) + ".txt", bvals, fmt = '%.3f')
    #
    # last value in bvals.txt is original test!!!
    #
    
#def experement3():

#experement1()
experement2()


#
#######