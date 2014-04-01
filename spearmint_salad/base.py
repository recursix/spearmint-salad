# -*- coding: utf-8 -*-
'''
Created on Oct 11, 2013

@author: alexandre
'''

import numpy as np
import sys
from traceback import format_exc
from sobol_lib import i4_sobol_generate
import time
from hp import HpConfiguration
import os

def argmin_with_ties( eLoss_kN ):
    min_eLoss_k1 = np.min(eLoss_kN, 1).reshape(-1, 1) # the min value for the k different bootstrap
    amin_kN = eLoss_kN == min_eLoss_k1 # for each bootstrap, flag each hp having the minimal eLoss (may be not unique)
    count_k1 = amin_kN.sum(1).reshape(-1, 1).astype(np.float) # for each bootstrap, count how many were having the min (usually 1)

    return amin_kN / count_k1 # normalize the flag matrix so that sums to one for each bootstrap

def aB_prob(w_km, loss_mN):
    """
    Agnostic Bayes probability distribution
    """
    
    eLoss_kN = np.dot(w_km, loss_mN) # computational bottleneck O(kmN)   

    amin_kN = argmin_with_ties(eLoss_kN) # takes into account the possibility of having several mins
    prob_N = np.mean(amin_kN, 0) # sum across the bootstrap
    
    np.testing.assert_almost_equal(prob_N.sum(), 1,) 

    return prob_N

import random
def rand_arg_min(val_n):
    val_n = np.asarray(val_n)
    min_val = np.min(val_n)
    idxL = np.where(min_val == val_n)[0]
    return random.choice(idxL)

def arg_min_prob( loss_mN ):
    prob_N = np.zeros( loss_mN.shape[1] )
    best_idx = rand_arg_min( np.mean(loss_mN,0) )
    prob_N[best_idx] = 1.
    return prob_N

class NotAPool:
    
    _processes = 1
    
    def apply_async(self, func, args=(), kwds={}, callback=None):
        result = func(*args, **kwds)
        if callback is not None:
            callback(result)
        
        return ApplyResult(result)
    
    def close(self): pass
    
class ApplyResult:
    
    def __init__(self, value):
        self._value = value
        
    def get(self):
        return self._value

class Profiler:
    """Basic profiler"""
    def __enter__(self):
        self.t0 = time.time()
        self.cpu_t0 = time.clock()
        return self

    def __exit__(self,*args):
        self.eval_info = {
            'time':time.time() - self.t0,
            'cpu_time':time.clock() - self.cpu_t0,
            }
        


class Evaluator:
    
    def __init__(self, ds_partition):
        self.ds_partition= ds_partition
        
    def __call__(self, hp_id, hp_config ):
        learner = hp_config.instantiate()
        with Profiler() as profiler:
            eval_info = eval_learner(learner, self.ds_partition)
            
        eval_info.update( profiler.eval_info )
        eval_info["hp_id"]= hp_id
        return eval_info
    
    def get_y_dict(self):
        y_dict = {}
        for name, ds_loader in self.ds_partition.items():
            if name == 'trn': continue
            y_dict[name] = ds_loader.get_dataset().y
        return y_dict
    
    @property
    def ds_name(self):
        return self.ds_partition.name

def eval_learner(learner, ds_partition):
    eval_info = {}
    
    # learn the estimator
    try:
        trn_ds = ds_partition['trn'].get_dataset()

        if hasattr(learner, 'set_valid_info'):
            val_ds = ds_partition['val'].get_dataset()
            learner.set_valid_info(val_ds.x, val_ds.y )
        if hasattr(learner, 'learn' ): 
            estimator = learner.learn(trn_ds )
        elif hasattr(learner, 'fit' ):
            estimator = learner.fit( trn_ds.x, trn_ds.y )
        else:
            raise Exception("Learner doesn't have any of the required member function learn, train, fit.")        
        
        eval_info['learn_status'] = 0
    except:
        estimator = None
        eval_info['learn_status'] = 1
        sys.stderr.write( format_exc() )

    # evaluate the estimator on the partition dict
    if estimator is not None:
        for partition_name, dsLoader in ds_partition.iteritems():
            if partition_name == 'trn' : continue
            
            ds = dsLoader.get_dataset()
            predict_info = {'m':len(ds.y)}

            try:
                if hasattr(ds, 'x' ):
                    x = ds.x
                elif hasattr(ds,'X'):
                    x = ds.X
                else:
                    raise Exception("dataset doesn't have attribute x or X")
                
                y = estimator.predict( x )
                predict_info['y'] = y
                
                predict_info['predict_status'] = 0
            except:
                predict_info['predict_status'] = 1
                sys.stderr.write( format_exc() )
                        

            eval_info[partition_name] = predict_info
    return eval_info

    

#async_eval_learner = async.AsyncFunc(eval_learner)
import tempfile

def get_tmp_dir():
    return tempfile.mkdtemp('spearmint_salad')


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

def build_bootstrap_matrix(k_bootstrap, m, essr, rng=None):
    bootstrap_matrix_km = np.zeros((k_bootstrap, m))
    for i in range(k_bootstrap):
        bootstrap_matrix_km[i, :] = bootstrap_weight(m, essr, rng) 
    
    return bootstrap_matrix_km

def bootstrap_weight(n, essr=1.0, rng=None):
    if rng is None: rng = np.random.RandomState()
    mSample = max(1,int(round(n*essr)))
    count_n =  np.bincount(rng.randint(0, n, mSample), minlength=n)
    assert count_n.sum() == mSample
    prob_n = count_n.astype(np.float) / np.sum(count_n)
    return prob_n



class Grid(object):
    
    def add_point(self, point):
        assert len(point) == self.n_dims # sanity check
        self.grid = np.vstack( (self.grid, point) )
        hp_id = self.grid.shape[0]-1
        np.testing.assert_equal( self.grid[hp_id,:], point )  # sanity check
        return hp_id

    def get_point(self, hp_id):
        return self.grid[hp_id,:]

    @property
    def size(self):
        return self.grid.shape[0]

    def _shuffle(self):
        if self.shuffle:
            idx = np.arange(self.size)
            np.random.shuffle(idx)
            self.grid = self.grid[idx,:]

class SobolGrid(Grid):
    
    def __init__(self, grid_size=10000, shuffle=True):
        self._grid_size = grid_size
        self.shuffle = shuffle

    def __str__(self):
        return "grid_size: %d, shuffle: %s"%(self._grid_size, self.shuffle)
        
    def init_grid(self,n_dims):
        self.n_dims = n_dims
        self.grid = i4_sobol_generate( self.n_dims, self._grid_size,1).T
        self._shuffle()

import itertools, random
class EvenGrid(Grid):
    
    def __init__(self, dim_size=10, shuffle=True ):
        self.dim_size = dim_size
        self.shuffle = True
        
    def init_grid(self,n_dims):
        self.n_dims = n_dims
        x =np.linspace(0,1,self.dim_size)
        self.grid = np.array( list(itertools.product( x, repeat=n_dims )) )
        self._shuffle()
        
class RandomChooser(object):

    def next(self, grid, values, durations,
            candidates, pending, complete):
        
        return random.choice( candidates )

def BasicGridOptimizer(hp_space, dim_size=10):
    return Optimizer( hp_space, RandomChooser , grid=EvenGrid(dim_size,shuffle=False) )


class GPEIChooseWrapper:
    
    def __init__(self, *args, **kwargs ):
        self.args = args
        self.kwargs = kwargs
    
    def __call__(self):
        
        if self.kwargs.get('mcmc_iters', 10) == 0:
            from spearmint.chooser.GPEIChooser import GPEIChooser 
            if 'burnin' in self.kwargs:
                del self.kwargs['burnin']
            return GPEIChooser(get_tmp_dir(),*self.args, **self.kwargs) 
        
        from spearmint.chooser.GPEIOptChooser import GPEIOptChooser
        return GPEIOptChooser(get_tmp_dir(),*self.args, **self.kwargs) 
        

def GPEIOptimizer(hp_space, grid_size=10000, mcmc_iters=0, **kwargs):
    return Optimizer( hp_space, GPEIChooseWrapper(mcmc_iters=mcmc_iters, **kwargs ), grid = SobolGrid(grid_size ))
    
def RandomSobol(hp_space, grid_size=10000 ):
    return Optimizer( hp_space, RandomChooser, grid = SobolGrid(grid_size ))

class Optimizer:
    
    def __init__(self, hp_space, Chooser = None, grid = None ):
        
        self.hp_space = hp_space
        
        if grid is None: grid = SobolGrid( )
        self.grid = grid 

        self.Chooser = Chooser
        self.chooser = None
    
    def __str__(self):
        return '  Chooser : %s\n  grid: %s\n  hp_space: %s'%(self.Chooser, self.grid, self.hp_space )
    
    def _init(self):
        self.chooser = self.Chooser()

        self.grid.init_grid(len(self.hp_space))
    
    def next(self, result_list, pending):
        if self.chooser is None:
            self._init()
        
#         
#         if self.chooser.mcmc_iters > 0:
#             print 'resetting state'
#             self.chooser.D = -1  # triggers a reinit
#             if os.path.exists(self.chooser.state_pkl):
#                 print 'removing choosser.pkl'
#                 os.remove(self.chooser.state_pkl)
#             self.chooser.needs_burnin = True
        
        
        hp_id = self.chooser.next( *self._build_grid_info(result_list, pending ))
        if isinstance(hp_id, tuple):
            _candidate_id, point = hp_id
            hp_id = self.grid.add_point(point)
           
        return HpConfiguration( self.hp_space, self.grid.get_point(hp_id) ), hp_id
    

    
    def _build_grid_info(self, result_list, pending ):
        
        pending =np.array(pending, dtype=np.int)
        grid_sz = self.grid.size
        
        values     = np.zeros(grid_sz) + np.nan
        durations  = np.zeros(grid_sz) + np.nan
        complete = []
        
        for hp_id, value, duration in result_list:
            complete.append( hp_id )
            values[hp_id] = value
            durations[hp_id] = duration
            
        candidates = set(range(grid_sz))
        candidates.difference_update(pending)
        candidates.difference_update(complete)
        
        return self.grid.grid, values, durations, np.array(list(candidates)), pending, np.array(complete)

        

