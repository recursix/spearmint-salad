# -*- coding: utf-8 -*-
'''
Created on Mar 31, 2014

@author: alex
'''

from spearmint_salad import salad,  base, experiments_folder
from spearmint_salad.analyze.analyze_trace import get_collection_structure
from pkl_trace import  TraceDBFile, get_column_dict
import numpy as np
from os import path
from spearmint_salad.dataset import DatasetPartition, DatasetView, SplitMask

def make_salad( hp_space, metric, dataset_partition, salad_size = 25, max_iter = 100 ):
    
    
    
    trace_name = "trace_%s_on_%s_with_%d_voters.pkl"%(hp_space.name, dataset_partition.name, salad_size)
    trace_path = path.expandvars(path.join( experiments_folder, 'tests', trace_name ))


    salad.SpearmintSalad(
        evaluator=base.Evaluator(dataset_partition),
        metric=metric,
        optimizer = base.GPEIOptimizer(hp_space, grid_size=10000, mcmc_iters = 0, burnin=50),
        salad_size=salad_size,
        n_iter=max_iter,
        trace_path = trace_path,
        Pool = None,
        )()
        
    return TraceDBFile(trace_path)


def make_partition( dataset_loader, trn_ratio=0.6, val_ratio=0.2, dataset_name=None):
    """
    dataset_loader : a callable that returns an object with the data and target attributes
    trn_ratio : the ratio of the samples used for training
    val_ratio : the ratio of the samples used for validation
    tst_ratio : 1 - trn_ratio - val_ratio 
    """
    
    
    if dataset_name is None:
        try:
            dataset_name = dataset_loader.__name__
        except AttributeError:
            try:
                dataset_name = dataset_loader.__class__.__name__
            except AttributeError:
                pass
                
    
    return DatasetPartition( dataset_name,
        trn = DatasetView( dataset_loader, SplitMask(0., trn_ratio) ),
        val = DatasetView( dataset_loader, SplitMask(trn_ratio, trn_ratio+val_ratio) ),
        tst = DatasetView( dataset_loader, SplitMask(trn_ratio+val_ratio, 1.) ),
    )
        
    
def get_final_risk(trace):
    col_dict = get_column_dict( trace.db.analyze, 'i', 'salad_risk.tst', 'salad_risk.val', 'argmin_risk.tst', 'argmin_risk.val' ) 
    idx = np.argmax(col_dict.pop('i'))
    risk_dict = {}
    for key, col in col_dict.iteritems():
        risk_dict[key]= col[idx]
        
    return risk_dict


def format_final_risk(trace):
    risk_dict = get_final_risk(trace)
    
    line_list = [
        'Final Risk',
        '----------',
        ]
    for key in sorted(risk_dict.keys()):
        line_list.append( '%20s : %.3g'%(key, risk_dict[key]) )
    return '\n'.join(line_list)

        
def format_trace_structure(trace):
    line_list = [
        'Trace Structure',
        '---------------',
        '']
    for collection_name in trace.db.get_collection_list():
        line_list.append( 'COLLECTION: %s'%collection_name )
        collection = getattr( trace.db, collection_name )
        collection_structure  = get_collection_structure( collection )

        
        for key in sorted( collection_structure.keys() ):
            if key in ['_id','proc_id']: continue
            count, type_set = collection_structure[key]
            types = ', '.join( [ t.__name__ for t in type_set ]  )
            line_list.append( '%20s : %d occurences of type %s'%(key, count,types ) )
        line_list.append('')
    return '\n'.join(line_list)
        
    
    
def get_final_predictions(trace):
    prediction_dict = {}
    col_dict = get_column_dict( trace.db.predict, 'i', 'salad_predict.tst', 'salad_predict.val', 'argmin_predict.val', 'argmin_predict.tst' ) 
    idx = np.argmax(col_dict.pop('i'))
    for key, col in col_dict.iteritems():
        prediction_dict[key]= col[idx]
        
    return prediction_dict