# -*- coding: utf-8 -*-
'''
Created on Sep 19, 2013

@author: alexandre
'''

class Variable:
    """
    Defines the information contained in a variable 
    """
    
class Optimizer:
    
    def __init__(self, variable_list, chooser ):
        self.variable_list = variable_list
        self.chooser = chooser
        self._create_grid()
        
    def _create_grid(self):
        """
        create some useful data structure from the variable_list
        """
    
    
    def next(self):
        
        """
        Ask the chooser which point to explore according to the current information
        It will also take into account the pending points.
        """
        
        idx = self.chooser.next( *self.grid.get_info() )
        x = self.grid.set_pending(idx)
        
        return x
        
    
    def set_observation(self, x, y ):
        """
        update the grid and save state
        """

        # do some code to update the state of the grid
        self.grid.update( x, y )
        
        


    
    

    