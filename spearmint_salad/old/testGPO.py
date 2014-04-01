# -*- coding: utf-8 -*-
'''
Created on Sep 18, 2013

@author: alexandre
'''



from spearmint.chooser.GPEIChooser import GPEIChooser
from spearmint.ExperimentGrid import ExperimentGrid
from spearmint_pb2 import Experiment
from graalUtil.num import uHist
from matplotlib import pyplot as pp
import numpy as np

import os
import shutil

exp_dir = '/tmp/2'
if os.path.exists(exp_dir):
    shutil.rmtree(exp_dir)
os.makedirs(exp_dir)

class V: pass

v = V()
v.name = 'x'
v.size = 1
v.type = Experiment.ParameterSpec.FLOAT
v.min = -20.
v.max = 20.


expt_grid = ExperimentGrid(exp_dir,[v], 105,11 )


def myFunc(x):
    x = np.array(x)
    return (x-3)**2

chooser = GPEIChooser(exp_dir)

xL = []

for i in range(10):

    grid, values, durations = expt_grid.get_grid()
    candidates = expt_grid.get_candidates()
    pending    = expt_grid.get_pending()
    complete   = expt_grid.get_complete()
    
    idx = chooser.next( grid, values, durations, candidates, pending, complete)
    
    print idx
    
    params = expt_grid.get_params(idx)
    
    x = params[0].dbl_val
    xL.append(x)
    y = myFunc(x)
    
    print x, y
    
    expt_grid.set_complete(idx, y, 1.)

print 'ending'


grid, values, durations = expt_grid.get_grid()

x = np.linspace(-10,10, 1000 )
pp.plot( x, myFunc(x) )
pp.plot( np.array(xL), myFunc(xL), '.' )
pp.show()

