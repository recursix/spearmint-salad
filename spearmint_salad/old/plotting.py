# -*- coding: utf-8 -*-
'''
Created on Oct 9, 2013

@author: alexandre
'''


import matplotlib.pyplot as pp
import numpy as np


class Plotter1d:
    
    def __init__(self, optimizer, func=None):
        
        self.optimizer = optimizer
        var_grid = optimizer.variable_grid()[:, 0]

        self.sort_idx = np.argsort(var_grid)
        self.var_grid = var_grid[self.sort_idx]
        if func is not None:
            self.y = func(self.var_grid)        
        else:
            self.y = None
        
    def plot(self):
        
        
        if self.y is not None:
            pp.subplot(2, 1, 1)    
            pp.plot(self.var_grid, self.y)
        
        if hasattr(self.optimizer.chooser, 'ei'):
            
            pp.subplot(2, 1, 1)
            
            func_m = self.optimizer.chooser.func_m[self.sort_idx]
            func_s = np.sqrt(self.optimizer.chooser.func_v[self.sort_idx])
            pp.plot(self.var_grid, func_m)
            pp.plot(self.var_grid, func_m + func_s)
            pp.plot(self.var_grid, func_m - func_s)

            
            pp.subplot(2, 1, 2)
            
            ei = self.optimizer.chooser.ei[self.sort_idx]
            pp.plot(self.var_grid, ei)
            
            best_idx = np.nanargmax(ei)
            pp.plot(self.var_grid[best_idx], ei[best_idx], '.', markersize=10)
            
            
        pp.show()