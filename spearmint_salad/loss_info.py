# -*- coding: utf-8 -*-
'''
Created on Dec 10, 2013

@author: alexandre
'''

import numpy as np
from spearmint_salad import pkl_trace, base, greedy
from os import path


def load_eval_info(trace_path):
    """
    Extract from the trace the predictions on the test and the validation sets for
    every models. 
    """
    trace = pkl_trace.TraceDBFile(trace_path)
    y_tst_N, y_val_N = pkl_trace.get_column_list(trace.db.eval_info, 'tst.y', 'val.y')
    y_tst_m, y_val_m = pkl_trace.get_column_list( trace.db.y, 'tst', 'val' )
    metric, ds_name, hp_space = pkl_trace.get_column_list( trace.db.main, '__dict__.metric', 'ds_name', 'hp_space' )
    
    tst_eval_info = EvalInfo( y_tst_m[0], y_tst_N, metric[0], ds_name[0], hp_space[0].name )
    val_eval_info = EvalInfo( y_val_m[0], y_val_N, metric[0], ds_name[0], hp_space[0].name )
    return tst_eval_info, val_eval_info



def merge_eval_info( eval_info_list ):
    """
    Merge eval info from different sources to be able to merge different
    hyperparameters space.
    """
    
    eval_info =eval_info_list[0]
    y_target_m = eval_info.y_target_m
    metric     = eval_info.metric
    ds_name    = eval_info.ds_name
        
    
    y_estimate_N = []
    hp_space_list = []
    for eval_info in eval_info_list:
        np.testing.assert_almost_equal(y_target_m, eval_info.y_target_m)
        assert eval_info.metric.__class__ == metric.__class__
        assert ds_name == eval_info.ds_name
        y_estimate_N +=  eval_info.y_estimate_N
        hp_space_list.append( eval_info.hp_space)
    
    return EvalInfo( y_target_m, y_estimate_N, metric, ds_name, '-'.join(hp_space_list) )
        
class EvalInfo:
    
    def __init__(self, y_target_m, y_estimate_N, metric, ds_name=None, hp_space=None):
#         assert y_target_m.shape[0] == y_estimate_mN.shape[0]
        assert y_target_m.ndim == 1
#         assert y_estimate_mN.ndim == 2
        
#        print "m=%d, N=%d"%( y_estimate_mN.shape )
        
        self.y_target_m = y_target_m
        self.y_estimate_N = y_estimate_N
        
        self.metric = metric
        self.ds_name = ds_name
        self.hp_space = hp_space

    def __str__(self):
        return 'LossInfo( metric=%s)'%( str(self.metric) )
    
    def get_loss(self):
        loss_Nm = [ self.metric.loss(self.y_target_m, y_estimate) 
                   for y_estimate in self.y_estimate_N ]
        loss_mN = np.array(loss_Nm).T
#         print loss_mN.shape, self.y_target_m.shape, len(self.y_estimate_N)
        assert loss_mN.shape ==  ( len( self.y_target_m ), len(self.y_estimate_N) )
        return loss_mN

    def subset(self, idxL):
        self.y_estimate_N = self.y_estimate_N[idxL]

    def test_distr(self, prob_N):
        
        assert prob_N.shape[0] == len( self.y_estimate_N)
        if not (prob_N >= 0).all():
            print 'WARNING : negative probs.'
        
        np.testing.assert_almost_equal(prob_N.sum(0),1 )
        
        y_estimate_m = self.metric.model_averaging(self.y_estimate_N, prob_N)
#        print unicodeHist( y_estimate_m, 'y_estimate_m' )
        return self.metric.loss(self.y_target_m, y_estimate_m)


import os, fnmatch


def find_files(pattern='*', *directory_list ):
    for directory in directory_list:
        directory = path.expandvars(directory)
        for root, _dirs, files in os.walk(directory):
            for basename in files:
                if fnmatch.fnmatch(basename, pattern):
                    yield os.path.join(root, basename)

def load_eval_info_dict( trace_path_list ):
    eval_info_dict = {}
   
    for trace_path in trace_path_list:
        tst_eval_info, val_eval_info = load_eval_info(trace_path)
        ds_name = tst_eval_info.ds_name
        assert ds_name == val_eval_info.ds_name
        
        eval_info_dict[ds_name] = (tst_eval_info, val_eval_info)
            
#        eval_info_dict[ds_name].append( (tst_eval_info, val_eval_info) )
#        
#
#    for ds_name in eval_info_dict.iterkeys():
#
#        tst_info, val_info = zip(*eval_info_dict[ds_name])
#        
#        eval_info_dict[ds_name] = (
#            merge_eval_info(tst_info), 
#            merge_eval_info(val_info) 
#            )
#    
    
    return eval_info_dict
    
def merge_info_dict( info_dict_list ):
    
    ds_name_set = set()
    for info_dict in info_dict_list:
        ds_name_set.update( info_dict.keys() )
    
    
    eval_info_dict = {}
    for ds_name in ds_name_set:
        tst_info, val_info = zip( * [ info_dict[ds_name] 
            for info_dict in info_dict_list if ds_name in info_dict ] )
        
        eval_info_dict[ds_name] = (
            merge_eval_info(tst_info), 
            merge_eval_info(val_info) 
            )   
    return eval_info_dict


class Method(object): 
    @property
    def name(self):
        return self.__class__.__name__
    
class AB_bootstrap(Method):
    
    def __init__(self,  k_bootstrap=100, essr=0.5 ):
        self.k_bootstrap = k_bootstrap
        self.essr = essr
    
    @property
    def name(self):
        return 'AB'

    def get_loss_info(self, tst_eval_info, val_eval_info ):

        loss_mN = val_eval_info.get_loss()
        w_km = base.build_bootstrap_matrix(self.k_bootstrap, loss_mN.shape[0], self.essr)
        
        prob_N = base.aB_prob(w_km, loss_mN)
        
        loss_m = tst_eval_info.test_distr(prob_N)
        
        
        return loss_m

class Greedy(Method):
    
    def __init__(self, ensemble_size= 10,  k_bootstrap=100, essr=0.5):
        self.ensemble_size = ensemble_size
        self.k_bootstrap = k_bootstrap
        self.essr = essr
        
    @property
    def name(self):
        return 'greedy-%.1f'%self.essr
        
    def get_loss_info(self, tst_eval_info, val_eval_info ):
        
        loss_mN = val_eval_info.get_loss()
        m,N = loss_mN.shape

        w_km = base.build_bootstrap_matrix(self.k_bootstrap, m, self.essr)
        ensemble_d, prob_d = greedy.greedy_ensemble(loss_mN, w_km, self.ensemble_size)
        prob_N = np.zeros( N )
        prob_N[ensemble_d] = prob_d

        loss_m = tst_eval_info.test_distr(prob_N)
        return loss_m

        
        
class ArgMin(Method):
    def get_loss_info(self, tst_eval_info, val_eval_info ):
        loss_mN = val_eval_info.get_loss()
        prob_N =  base.arg_min_prob(loss_mN)
        return tst_eval_info.test_distr(prob_N)

if __name__ == "__main__":
    exp_path_list = [
        "$EXPERIMENTS_FOLDER/Forest_on_truncated_torgo_10Dec2013-19.56.44",
        "$EXPERIMENTS_FOLDER/SVR_on_truncated_torgo_10Dec2013-20.44.50",
        ]
    
#    trace_path_list = [
#        "/tmp/trace_10Dec2013-18_07_28_t_gcPg.pkld",
#        "/tmp/trace_10Dec2013-18_15_36_CkwwcP.pkld",
#        ]
    

    trace_path_list = list( find_files( 'trace_*.pkl', *exp_path_list ) )
    eval_info_dict = load_eval_info_dict( trace_path_list )
    method = AB_bootstrap(  )

    for ds_name, (tst_info, val_info) in eval_info_dict.iteritems():
        loss_m  =  method.get_loss_info(tst_info, val_info)
        print ds_name, method.hp_space, np.mean(loss_m)
        
#    
#    loss_m, ds_name = method.get_loss_info()
#    print method.name, ds_name, method.hp_space, np.mean(loss_m)
#    
    
    
    
    
    