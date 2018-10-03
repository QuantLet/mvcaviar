#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr  6 09:45:42 2017

@author: egor
"""

import numpy as np
from bisect import bisect_left, bisect_right

#
#  
#

class ts_estimation_preparator:
    def __init__(self, model):
        self.model = model
        
        self.prep_pars = []
        self.prep_edges = []
        self.is_prepared = False
    
    def do_preparation(self, edges, estimate):
        if len(edges) < 2:
            raise ValueError("Less than two edges.")
        if edges[0] <= self.model.t_min():
            raise ValueError("Bad first break point.")
        if edges[-1] > self.model.t_max():
            raise ValueError("Bad last break point.")
        for i in range(1, len(edges)):
            if edges[i - 1] >= edges[i]:
                raise ValueError("Break points unordered.")
                
        self.prep_edges = edges
        self.prep_pars = [ [0 for length in range(1, len(edges) - pos)] for pos in range(len(edges) - 1) ]
        for pos in range(len(edges) - 1):
            for length in range(1, len(edges) - pos):
                interval = (edges[pos] , edges[pos + length])
                print("interval in process:", interval)
                
                starts = []
                if length > 1:
                    starts.append(self._get_prep_par(pos, length-1))
                if pos > 0:
                    starts.append(self._get_prep_par(pos - 1, length + 1))
                
                par, val = estimate(self.model, interval, starts, 200)
                self.prep_pars[pos][length - 1] = par

        self.is_prepared = True

    def _get_prep_par(self, pos, length):
        return self.prep_pars[pos][length - 1]
    
    def save_preparation(self, name):
        if not self.is_prepared:
            raise Exception("Not prepared yet.")
        all_prep_pars = []
        for pos in range(len(self.prep_edges) - 1):
            for length in range(1, len(self.prep_edges) - pos):
                all_prep_pars.append(self._get_prep_par(pos, length))

        np.savetxt(name + ".pars", all_prep_pars, fmt = '%.5f')
        np.savetxt(name + ".edgs", self.prep_edges, fmt = '%d')
    
    def load_preparation(self, name):
        self.prep_edges = [int(x) for x in np.loadtxt(name + ".edgs")]
        
        all_prep_pars = list(np.loadtxt(name + ".pars"))
        if len(all_prep_pars) != (len(self.prep_edges) * (len(self.prep_edges) - 1) / 2):
            raise ValueError("Wrong number of prepared parameters.")

        self.prep_pars = []
        i = 0
        for pos in range(len(self.prep_edges) - 1):
            self.prep_pars.append([])
            for length in range(1, len(self.prep_edges) - pos):
                self.prep_pars[-1].append(all_prep_pars[i])
                i += 1

        self.is_prepared = True
    
    def get_prepared_pars(self, interval):
        # returns estimated pars on the preparatory intervals
        #
        if not self.is_prepared:
            raise ValueError("Bootstrap handler is not prepared.")

        poslens = []
        if interval[1] <= self.prep_edges[0]:
            poslens = [(0, 1)]
        elif interval[0] >= self.prep_edges[-1]:
            poslens = [(len(self.prep_edges) - 2, 1)]
        else:
            in_edge_l = bisect_right(self.prep_edges, interval[0])
            in_edge_r = bisect_left(self.prep_edges, interval[1]) - 1

            poslens = []
            for shift_l in [0, 1]:
                for shift_r in [0, 1]:
                    edge_l = in_edge_l - shift_l
                    edge_r = in_edge_r + shift_r
                    if edge_l < edge_r and edge_l >= 0 and edge_r < len(self.prep_edges):
                        poslens.append((edge_l, edge_r - edge_l))

        print(poslens)
        return [self._get_prep_par(pos, length) for (pos, length) in poslens]

roll_win_global_only_strategy = {
	"prev3": False, "prep": False, "global": True, "firstmulti": False
}

roll_win_standard_strategy = {
	"prev3": True, "prep": True, "global": False, "firstmulti": True
}


def rolling_window(
        model, intervals, estimate, prep = 0, 
        strategy = roll_win_standard_strategy
    ):
    for key in ["prev3", "prep", "global", "firstmulti"]:
        if not (key in strategy.keys()):
            strategy[key] = False

    if not any( [strategy[key] for key in ["prep", "global", "firstmulti"]] ):
        raise ValueError("Insufficient strategy.")

    if strategy["prep"] == True and prep == 0:
        raise ValueError("Preparator required for this strategy.")

    if strategy["prep"] and not prep.is_prepared:
        raise ValueError("Preparator is not prepared.")

    
    global_par = 0
    if strategy["global"]:
        global_interval = ( min(x for (x,y) in intervals), max(y for (x, y) in intervals) )
        global_par, val = estimate(model, global_interval, [], 200)

    pars = []
    for interval in intervals:
        print(interval)
        starts = []
        starts_to_gen = 0

        if strategy["prep"]:
            starts += prep.get_prepared_pars(interval)

        if strategy["global"]:
        	starts += [global_par]

        
        if strategy["firstmulti"] and len(pars) == 0:
            starts_to_gen = 200

        if strategy["prev3"] and len(pars) > 0:
            a = max(0, len(pars) - 3)
            starts += pars[a :]

        par, val = estimate(model, interval, starts, starts_to_gen)
        pars.append(par)

    return pars







    
