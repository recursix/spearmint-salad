# -*- coding: utf-8 -*-
'''
Created on Oct 10, 2013

@author: alexandre
'''


import time
import base
import numpy as np
import random
from pkl_trace import Trace
from threading import RLock 
import os
from greedy import greedy_ensemble

verbose = 2
from base import aB_prob


def make_pool(Pool):
    if Pool is None:  
        return base.NotAPool()
    elif Pool == 'mp':
        
        import sys
        sys.stdout = sys.stderr
        
        import multiprocessing as mp
        
        
        ppn = os.environ.get('JD_PPN')
        if ppn is not None: ppn = int(ppn)
        return mp.Pool(ppn)
        
        
    elif Pool == 'mpi':
        from pyjobber.mpi import mpi_pool
        return mpi_pool.Pool()
    else:        
        return Pool()
        



class SpearmintSalad:
    
    def __init__(self, evaluator, metric, optimizer, salad_size=30, 
            essr=0.5, n_iter=100, Pool=None, trace_path=None, seed=None):
        
        self.evaluator = evaluator
        self.metric = metric
        self.optimizer = optimizer
        self.salad_size = salad_size
        self.essr = essr
        self.n_iter = n_iter 
        self.Pool = Pool
        self.trace_path = trace_path
        self.seed = seed
      
    def _init(self):
        if os.path.exists(self.trace_path):
            os.remove(self.trace_path)
            
        trace = Trace(self.trace_path, trace_level=2)
        trace.write( 1, 'main', {
            'hp_space': self.optimizer.hp_space,
            '__dict__': self.__dict__,
            'ds_name':self.evaluator.ds_name,
             } )
        
        self.trace = trace
        self.pool = make_pool(self.Pool)
        
        self._last_print = 0.
        self.last_eval_size = 0
        self.result_dict = {}
        self.hp_id_seq = []
        self.hp_id_pending = {}
        self.lock = RLock()
        
        self.y_dict = self.evaluator.get_y_dict()
        self.y_val_m = self.y_dict['val']
        self.trace.write(1,'y',self.y_dict) # trace this info for further analysis
        
        self.m = len(self.y_val_m)
        
        if self.salad_size == 1:
            self.bootstrap_matrix_km = np.ones( (1,self.m) )/ self.m
            
        else:
            rng = np.random.RandomState(self.seed) # provides the opportunity to have the same bootstrap matrix across different experiments.
            self.bootstrap_matrix_km = base.build_bootstrap_matrix(self.salad_size,self.m,self.essr, rng)

            
    def _submit_hp( self, hp_id, hp_config ):
        
        with self.lock:
            self.hp_id_pending[ hp_id ] = None
            
        submit_info = self.pool.apply_async( 
            self.evaluator, (hp_id, hp_config), callback = self._result_callback )
        
        with self.lock:
            if self.hp_id_pending.has_key(hp_id): # pool is of type NotAPool, hp_id is already deleted from pending
                self.hp_id_pending[ hp_id ] = submit_info

    
    def _result_callback( self, eval_info ):
        hp_id = eval_info['hp_id']
        with self.lock:
            self.hp_id_seq.append(hp_id)
            del self.hp_id_pending[hp_id]
            self.result_dict[hp_id] = eval_info
    
    def trace_result(self, hp_id):
        
        eval_info = self.result_dict[hp_id]
        
        # add little extras
        for key in self.y_dict.keys():
            eval_info[key]['risk'] = np.mean(self._eval(eval_info[key]['y'], key)) 

        self.trace.write(2,'eval_info',eval_info )

    
    def _print_info(self):
        if verbose >= 1:
            print 'salad size : %d' % (self.salad_size)
            print 'essr : %.3g' % self.essr 
            
#            for name, y in self.y_dict.items():
#                print uHist(y, 'y_%s'%name)
            
            print 'k,m', self.bootstrap_matrix_km.shape
    
    @property
    def pool_size(self):
        return self.pool._processes
    
    @property
    def pending_count(self):
        return len(self.hp_id_pending)
    
    @property
    def pending_list(self):
        return self.hp_id_pending.keys()
    
    def __call__(self):
        self._init()
        self._print_info()
        
        self.i = 0
        
        
        while self.last_eval_size < self.n_iter: # wait until all results are analysed
             
            if self.i < self.n_iter: # if there are still some evaluations to be launched
                if self.pending_count < self.pool_size: # if there is some worker available
                    
#                    if verbose >= 1:
#                        print 'Pending count : %d/%d. Searching for another configuration.'%( self.pending_count, self.pool_size )

                    
                    with base.Profiler() as choose_profiler:
                        result_list = self.build_values(self.i % self.salad_size)
                        hp_config, hp_id = self.optimizer.next(result_list, self.pending_list )
                    
                    if verbose >=1 :
                        print 'trying %d : %s'%(hp_id, str(hp_config.var_dict()))
    
                    self.trace.write( 1, 'candidates', { 
                        'i': self.i,
                        'hp_id':int(hp_id), 
                        'unit_value':hp_config.unit_value_list,
                        'choose': choose_profiler.eval_info,
                        } )
                    
                    self._submit_hp(hp_id, hp_config)
#                    self.result_dict[ hp_id ] = async_eval_learner(hp_config, self.ds_partition)
                    self.i += 1
                
            if len(self.hp_id_seq) > self.last_eval_size: # if we have some analysis work to do
                
                with self.lock: # hp_id_seq may be modified by the completed callback from another thread
                    hp_id_N = self.hp_id_seq[:self.last_eval_size+1] # copying the sequence of completed hp_id to work with   
                
                self.analyse_results(hp_id_N)
                
                self.last_eval_size += 1
  
            
            time.sleep(0.001)


    def build_values(self, bootstrap_idx):
        hp_id_N = self.result_dict.keys()
        duration_N = self._get_field(hp_id_N, 'time')
        
        w_m = self.bootstrap_matrix_km[bootstrap_idx, :]
        eLoss_N = np.dot(w_m, self._get_loss_matrix(hp_id_N)) 
        
        assert len(eLoss_N) == len(hp_id_N)
        
#        if verbose >= 2:
#            print uHist(eLoss_N, 'eLoss_N[%d]' % bootstrap_idx)
#   
        return zip(hp_id_N, eLoss_N, duration_N)

    
    def _get_field(self, hp_id_N, field_name):
        return [ self.result_dict[hp_id][field_name] for hp_id in hp_id_N ]          
            
    
    def _get_loss_matrix(self, hp_id_N, optim_loss = True):
        
        """
        get the loss values for each evaluated predictor
        on each validation sample
        """
        if optim_loss: loss_func = self.metric.optim_loss
        else:          loss_func = self.metric.loss

        loss_mN = np.zeros((self.m, len(hp_id_N)))
        for i, hp_id in enumerate(hp_id_N):
            y_val_predict_m = self.result_dict[hp_id]['val']['y']
            loss_mN[:, i] = loss_func(self.y_val_m, y_val_predict_m)
      
        return loss_mN
    
    def _get_prob(self, hp_id_N, bootstrap_matrix_km = None):
        
        if bootstrap_matrix_km is None:
            bootstrap_matrix_km = self.bootstrap_matrix_km
        
        loss_mN = self._get_loss_matrix(hp_id_N)        

        return aB_prob(bootstrap_matrix_km, loss_mN)


    def _get_y_matrix(self, hp_id_N, partition='tst'):
        y_Nn = np.zeros((len(hp_id_N), len(self.y_dict[partition])))
        for i, hp_id in enumerate(hp_id_N):          
            y_Nn[i, :] = self.result_dict[hp_id][partition]['y']
        return y_Nn

    def _get_y_list(self, hp_id_N, partition='tst'):
        return [self.result_dict[hp_id][partition]['y'] for hp_id in hp_id_N ]           

    def _build_and_eval_salad(self, hp_id_N, bootstrap_matrix_km = None):
        prob_N = self._get_prob(hp_id_N,bootstrap_matrix_km)
        lossD, predictD = self._eval_salad( hp_id_N, prob_N )
        return lossD, predictD, prob_N
    
    
    
    def _eval_salad(self, hp_id_N, prob_N):
        
        lossD = {}
        predictD = {}
        for partition in ['val', 'tst']:
            y_predict = self.metric.model_averaging(self._get_y_list(hp_id_N, partition), prob_N)
            lossD[partition] = self._eval(y_predict, partition)
            predictD[partition] = y_predict
            
        return lossD, predictD
    
    def _eval(self, y_predict, partition='tst'):
        return self.metric.loss(self.y_dict[partition], y_predict)
    
    
    def _get_argmin_all(self, hp_id_N):
        """
        When using the zero-one loss, there is often several hp_configuration 
        with the same expected validation loss
        """
        
        loss_mN = self._get_loss_matrix(hp_id_N, optim_loss=False)
        assert loss_mN.shape == (self.y_val_m.shape[0], len(hp_id_N))
         
        eLoss_N = np.mean(loss_mN, 0)
        
#        if verbose >= 2:
#            print uHist(eLoss_N, 'eLoss_N' )
        
        return  hp_id_N[eLoss_N == np.min(eLoss_N)]
        

    def _eval_argmin(self,hp_id_N):
        
        argmin_list = self._get_argmin_all(hp_id_N)
        chosen_hp_id = int(random.choice( argmin_list ))
        
        self.trace.write( 2,'argmin', {
            'argmin_list': [int(hp_id) for hp_id in argmin_list ], 
            'chosen_hp_id':chosen_hp_id,
            'i':len(hp_id_N)-1 } )
        
        
        lossD = {}
        predictD = {}
        for partition in ['val', 'tst']:
            y_predict = self.result_dict[chosen_hp_id][partition]['y']
            lossD[partition] = self._eval(y_predict, partition)
            predictD[partition] = y_predict
        return lossD, predictD
    
    
    def _analyze_greedy(self,hp_id_N, ensemble_size=10):
        if not hasattr(self, 'greedy_w_km'):
            self.greedy_w_km = base.build_bootstrap_matrix(100, len(self.y_val_m), self.essr)

        loss_mN = self._get_loss_matrix(hp_id_N)
        ensemble_d, prob_d = greedy_ensemble(loss_mN, self.greedy_w_km, ensemble_size)
        hp_id_d = hp_id_N[ensemble_d]
        return self._eval_salad( hp_id_d, prob_d )
        
    
    def analyse_results(self, hp_id_N):
        
        t_ref =time.clock()
        
        hp_id_N = np.asarray(hp_id_N)
        if verbose >=1 :
            print 'analyzing results %d'%len(hp_id_N)
        
        self.trace_result(hp_id_N[-1])
        argmin_lossD, argmin_predictD = self._eval_argmin(hp_id_N)
        salad_lossD, salad_predictD, prob_N  = self._build_and_eval_salad(hp_id_N)
        
        for essr in [ 0.2, 0.5, 1.]: # try for different essr
            bootstrap_matrix_km = base.build_bootstrap_matrix(2*self.salad_size,len(self.y_val_m),essr)
            salad_lossD['%d%%'%(round(essr*100))] = self._build_and_eval_salad(hp_id_N,bootstrap_matrix_km)[0]

#         for ensemble_size in [5,10,20]:
#             salad_lossD['greedy-%02d'%ensemble_size] = self._analyze_greedy(hp_id_N,ensemble_size)
#         
        
        # sparsify for storage
        prob_list = [ ( int(hp_id), float(prob)) for hp_id, prob in zip( hp_id_N, prob_N ) if prob > 0]
        self.trace.write(2,'predict',dict(
            i = len(hp_id_N)-1,
            salad_predict = salad_predictD,
            argmin_predict = argmin_predictD,
            prob = prob_list,
            ))
        
        
        self.trace.write(2,'loss',dict(
            i = len(hp_id_N)-1,
            hp_id = int(hp_id_N[-1]),
            argmin_loss = argmin_lossD,
            salad_loss = salad_lossD,
            ))
        
        
        
        self.trace.write(1,'analyze', dict(
            i = len(hp_id_N)-1,
            hp_id = int(hp_id_N[-1]),
            argmin_risk = _compute_risk( argmin_lossD ),
            salad_risk = _compute_risk( salad_lossD ),
            analyse_time = time.clock() - t_ref,
            ))

def _compute_risk( lossD ):
    riskD = {}
    for key, val in lossD.iteritems():
        if isinstance( val, dict ):
            riskD[key] = _compute_risk( val )
        else:
            riskD[key] = float(np.mean(val))

    return riskD
            
    
        


    
    
