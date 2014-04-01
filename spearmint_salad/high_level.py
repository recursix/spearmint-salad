# -*- coding: utf-8 -*-
'''
Created on Mar 31, 2014

@author: alex
'''

from spearmint_salad.dataset import PickledDatasetLoader,  DatasetPartition, TrnTstSplitter
from spearmint_salad import salad,  base, experiments_folder
from os import path

def make_salad( hp_space, metric, dataset_path, salad_size = 25, max_iter = 100, partition=(0.6, 0.2, 0.2) ):
    
    ds_loader = PickledDatasetLoader(dataset_path) 
    ds_partition = DatasetPartition(ds_loader.shape.name, {'trn':ds_loader})
    ds_partition = TrnTstSplitter(partition[0], split_name='tst')(ds_partition)
    split_ratio = partition[1] / (partition[1] + partition[2])
    ds_partition = TrnTstSplitter(split_ratio, split_name='val')(ds_partition)
    
    
    trace_name = "trace_%s_on_%s_with_%d_voters.pkl"%(hp_space.name, ds_partition.name, salad_size)
    trace_path = path.expandvars(path.join( experiments_folder, 'tests', trace_name ))

    salad.SpearmintSalad(
        evaluator=base.Evaluator(ds_partition),
        metric=metric,
        optimizer = base.GPEIOptimizer(hp_space, grid_size=10000, mcmc_iters = 0, burnin=50),
        salad_size=salad_size,
        n_iter=max_iter,
        trace_path = trace_path,
        Pool = None,
        )()
        

        
        