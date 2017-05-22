#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr  6 09:45:42 2017

@author: egor
"""

import numpy as np

def check_func(x, tau):
  return (tau - float(x <= 0)) * x

class MvCaviarSolver:
  def __init__(self, Y, m, q, thetas):
    # initialization
    # take Y, m, q, what else? thetas?
    self.Y = Y
    self.m = m
    self.q = q
    self.thetas = thetas
    
    # dimensions
    (self.n, self.tmax) = np.shape(Y)
    self.shift = max(m, q)
    self.__parlen__ = self.n * (m * self.n + q * self.n + 1)
    
    # weights
    self.W = np.ones((self.n, self.tmax))
    
    # fill phi
    # first self.shift columns are uninformative: to avoid index confusions
    self.__Phi__ = np.zeros((self.n * q + 1, self.tmax))
    self.__Phi__[self.n * q, self.shift:] = np.ones((self.tmax - self.shift))
    # the first nq entries of each column are arranged as follows:
    # say, for  q = 4 for each shift we have first  n entries with 
    # shift 1, then n with shift 2 and so on:
    # (   ,   ,   ,   )
    #   n   n   n   n   --- q times n
    #
    for i in range(q):
      self.__Phi__[(self.n * i):(self.n * i + self.n) , self.shift:] = \
              abs(Y[:, (self.shift - 1 - i):(self.tmax - 1 - i)])
      
    #reserve place for conditional quantiles
    self.__condQs__ = np.empty((self.n, self.tmax))
    
    #calls counter
    self.__counter__ = 0
      
  def parlen(self):
    return self.__parlen__
  
  def set_default_weights(self):
    self.W = np.ones((self.n, self.tmax))
  
  def set_random_weights(self):
    #generate r.v. with density 3/4 on [0,1] and 1/12 on [1, 4] and 0 elsewhere
    self.W = np.vectorize(lambda x : 4 * x / 3 if x <= 3/4 else 12 * x - 8)(
                 np.random.rand(self.n, self.tmax)
             )

  def __parsepar__(self, par):
    # (!) TODO EXCEPTION: check size(par) == __parlen__
    
    # Let us explain the parameter packing:
    # first go n parameters linking 1st quantile to all n 1-lagged quantiles,
    #  then n linking 1st to 2-lagged, and so on (overall mn)
    # then mn parameters linking 2nd quantile to lagged quantiles and so on; 
    # then go nq+1 parameters linking 1st quantile to Phi, 
    # then 2nd and so on.
    #
    B = np.asmatrix(np.reshape(
          par[:(self.n * self.n * self.m)], (self.n, self.n * self.m)
        ))
    A = np.asmatrix(np.reshape(
          par[(self.n * self.n * self.m):], (self.n, self.q * self.n + 1)
        ))
    return (B, A)
  
  def __fill_lagged_quantiles__(self, interval):
    #global quantile
    (a, b) = interval
    quant0 = np.empty(self.n)
    for i in range(self.n):
      quant0[i] = np.percentile(self.Y[i, a:b], 100 * self.thetas[i])
    return
    
    #prepare intial quantiles
    for lag in range(1, self.m + 1):
      self.__condQs__[:, interval[0] - lag] = quant0
                     
  def __fill_quantiles__(self, par, interval):
    # interval is a tuple!
    # (!) TODO EXCEPTION: make sure that it fits into the current data interval!
    (a, b) = interval
    
    #clean
    for t in range(a, b):
      self.__condQs__[:, t] = np.zeros(self.n)
    
    (B, A) = self.__parsepar__(par)
    
    # B and A are already matrices!!!
    for t in range(a, b):
      for lag in range(1, self.m + 1):
        self.__condQs__[:, t] += (self.__condQs__[:, t - lag]).dot(
          np.asarray(B[:, (self.n * (lag - 1)):(self.n * lag)].T))
      self.__condQs__[:, t] += (self.__Phi__[:, t]).dot(np.asarray(A.T))
    return
  
  def get_quantile(self, i):
    return self._condQs__[:, i]

  def __calc_loss__(self, interval):
    (a, b) = interval
    res = 0
    for i in range(self.n):
      for t in range(a, b):
        res += self.W[i, t] * check_func(self.Y[i, t] - self.__condQs__[i, t], self.thetas[i])
    return res #/ (b - a)

  def calc_loss(self, par, interval):
    self.__fill_lagged_quantiles__(interval)
    self.__fill_quantiles__(par, interval)
    return self.__calc_loss__(interval)

  def gen_candidate(self, I):
    return self.gen_local_candidate(I, np.zeros(self.parlen()), 1, 1.0)
  
  def gen_local_candidate(self, I, par, spread, cut):
    while(True):
      _par = par + (np.random.rand(self.parlen()) * 2 - np.ones(10)) * spread
      if self.calc_loss(_par, I) < cut * (I[1] - I[0]) :
        return _par
  
  def split_par(self, par12):
    #TODO: make sure shape(par12) = (2 * parlen())
    par1 = par12[0 : self.__parlen__]
    par2 = par12[self.__parlen__ : 2 * self.__parlen__]
    return (par1, par2)
    
