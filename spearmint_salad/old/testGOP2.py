# -*- coding: utf-8 -*-
'''
Created on Oct 7, 2013

@author: alexandre
'''

from grid import Optimizer, VariableMap
from spearmint.chooser.GPEIChooser import GPEIChooser
from spearmint.chooser.GPEIOptChooser import GPEIOptChooser
from spearmint.chooser.RandomForestEIChooser import RandomForestEIChooser
import os
import shutil
import numpy as np
from plotting import Plotter1d


def quadratic(x):
    x = np.array(x)
    return (x - 3) ** 2

def sinc(x):
    return -2* np.sinc(0.5*x )

myFunc = quadratic

exp_dir = '/tmp/GPEI'
if os.path.exists(exp_dir):
    shutil.rmtree(exp_dir)
os.makedirs(exp_dir)



var_list = [VariableMap('x', -10, 10)]

chooser = GPEIChooser(exp_dir, mcmc_iters=10, keep_ei=True)
#chooser = RandomForestEIChooser(100, min_split=1 )
#chooser = GPEIOptChooser(exp_dir, mcmc_iters=0, keep_ei=True)
optimizer = Optimizer(var_list, chooser, grid_size=10000)

plotter = Plotter1d(optimizer, myFunc)


i = 0
while True:
    i += 1
    x, idx = optimizer.next()
    
    y = myFunc(**x)
    print '%3d: idx = %d, x=%.3f, y=%.3f' % (i, idx, x['x'], y)
    
    if i>=3:
        plotter.plot()
        

    optimizer.set_observation(x, idx, y)
    
    
