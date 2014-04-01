# -*- coding: utf-8 -*-
'''
Created on Sep 19, 2013

@author: alexandre
'''

import numpy        as np
from sobol_lib import i4_sobol_generate

CANDIDATE_STATE = 0
SUBMITTED_STATE = 1
RUNNING_STATE   = 2
COMPLETE_STATE  = 3
BROKEN_STATE    = -1


class ExperimentGrid:


    def __init__(self, n_variable, grid_size=None, shuffle=True ):


        
        self.grid = i4_sobol_generate(n_variable,grid_size,1).T
        
        if shuffle:
            idx = np.arange(grid_size)
            np.random.shuffle(idx)
            self.grid = self.grid[idx,:]
        
        
        self.status = np.zeros(grid_size, dtype=int) + CANDIDATE_STATE
        self.values = np.zeros(grid_size) + np.nan
        self.durations = np.zeros(grid_size) + np.nan


    def get_candidates(self):
        return np.nonzero(self.status == CANDIDATE_STATE)[0]

    def get_pending(self):
        return np.nonzero((self.status == SUBMITTED_STATE) | (self.status == RUNNING_STATE))[0]

    def get_complete(self):
        return np.nonzero(self.status == COMPLETE_STATE)[0]

    def get_broken(self):
        return np.nonzero(self.status == BROKEN_STATE)[0]

    def get_params(self, index):
        return self.grid[index,:]

    def get_best(self):
        best_idx = np.nanargmin( self.values ) # will ignore nan values
        if np.isnan(best_idx): # if the list is empty (or all nans)
            return np.nan, -1
        else:                   
            return self.values[best_idx], best_idx 
        

    def add_to_grid(self, candidate):
        self.grid   = np.vstack((self.grid, candidate))
        self.status = np.append(self.status, CANDIDATE_STATE )
        self.values = np.append(self.values, np.nan )
        self.durations = np.append(self.durations, np.nan)

        return self.grid.shape[0]-1

    def set_candidate(self, idx):
        self.status[idx] = CANDIDATE_STATE

    def set_submitted(self, idx):
        self.status[idx] = SUBMITTED_STATE

    def set_running(self, idx):
        self.status[idx] = RUNNING_STATE

    def set_complete(self, idx, value, duration):
        self.status[idx] = COMPLETE_STATE
        self.values[idx] = value
        self.durations[idx] = duration

    def set_broken(self, idx):
        self.status[idx] = BROKEN_STATE


    def get_info(self):
        candidates = self.get_candidates()
        pending    = self.get_pending()
        complete   = self.get_complete()
        
        return self.grid, self.values, self.durations, candidates, pending, complete

class VariableMap:
    
    def __init__(self, name, min_val, max_val, transform=None, is_int=False):
        """
        Range is within [min_val, max_val).        
        """
        
        self.name = name
        self.min_val = min_val
        self.max_val = max_val
        self.is_int = is_int
        self.transform = transform # may be useful for working in logspace
        
    def map(self, unit_value):
       
        if unit_value >= 1: # makes sure it stays within [0,1)
            unit_value = 1. - np.finfo(1.).eps
        
        span = self.max_val - self.min_val
                    
        val = self.min_val + unit_value*span

        if self.transform is not None:
            val = self.transform(val) 

        if self.is_int:
            val =  int(np.floor(val))
            
        return val

    def map_vec(self, unit_value):
        """
        unit_value can be a vector to efficiently convert several values simultaneously
        """
        
        unit_value = np.asarray(unit_value)
        
        # makes sure it stays within [0,1)
        unit_value[unit_value >=1.]  = 1. - np.finfo(1.).eps
               
        span = self.max_val - self.min_val
        val = self.min_val + unit_value*span

        if self.transform is not None:
            val = self.transform(val) 

        if self.is_int:
            val =  np.floor(val).astype(np.int)
            
        return val


class Optimizer:
    
    def __init__(self, variable_mapper_list, chooser, grid_size=1000 ):
        
        # extract into list to have a constant order across the different calls
        self.variable_mapper_list = variable_mapper_list
        
        self.chooser = chooser
#        self.chooser.keep_ei = True
        self.grid = ExperimentGrid(len(variable_mapper_list), grid_size ) 
        
    
    def next(self):
        
        """
        Ask the chooser which point to explore according to the current information.
        It will also take into account the pending points.
        """
        
        idx = self.chooser.next( *self.grid.get_info() )
        
        
        self.grid.set_submitted(idx)
        
        u = self.grid.get_params(idx)
        x = self._map_variables(u)
        return x, idx
        
    
    def variable_grid(self ):
        grid = []
        for i, var_mapper in enumerate(self.variable_mapper_list):
            grid.append( var_mapper.map_vec( self.grid.grid[:,i] ) )
            
        return np.array(grid).T 
    
    def _map_variables(self,u):
        """
        The chooser works in the unit hypercube and this helper function
        maps the values back to the original space for each variables. 
        """
        
        var_dict = {}
        i = 0
        for var_mapper in self.variable_mapper_list:
            var_dict[var_mapper.name] = var_mapper.map( u[i] )
            i+=1
            
        return var_dict
            
    
    def set_observation(self, x, idx, y ):
        """
        update the grid and save state
        """

        # do some code to update the state of the grid
        self.grid.set_complete( idx, y, 0. )
        
        
