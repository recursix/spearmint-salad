# -*- coding: utf-8 -*-
'''
Created on Oct 9, 2013

@author: alexandre
'''

import numpy as np

class BaseChooser:
    
    def _save_ei(self, grid, candidates, ei, func_m, func_v):
        self.ei = np.zeros( grid.shape[0] ) + np.nan
        self.func_m = np.zeros( grid.shape[0] ) + np.nan
        self.func_v = np.zeros( grid.shape[0] ) + np.nan
        
        self.func_m[ candidates] = func_m
        self.func_v[ candidates] = func_v
        self.ei[candidates]  = ei