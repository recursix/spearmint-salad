# -*- coding: utf-8 -*-
'''
Created on Nov 19, 2013

@author: alexandre
'''

#import sys
#print 'theano_loaded : ', sys.modules.has_key('theano')
#
#import os
#pid = os.getpid()
#slot = pid%16
##print 'pid:%d, slot%d'%(pid,slot)
#os.environ['THEANO_FLAGS'] = "base_compiledir=/tmp/theano_%d" % (slot)



from spearmint_salad import salad, metric, sklearn_models, base
#from spearmint_salad import pylearn2_models

from dataset import FolderContext, TrnTstSplitter
from os import path
import traceback


def salad_on_context( hp_space, context_name, metric, Pool='mp', trace_folder='.', grid=False, **kwargs ):
    print 'before making pool'
    pool = salad.make_pool(Pool)
    print 'after making pool'
    
    
    context = FolderContext( path.join( '$CONTEXT_FOLDER', context_name ) )

    if grid:
        print 'Using RandomSobol'
        optimizer=base.RandomSobol( hp_space, grid_size=kwargs['n_iter'] )
    else:
        print 'Using GPEIOptimizer'
        optimizer=base.GPEIOptimizer( hp_space )


    result_list = []
    for ds_partition in context: 
        
        ds_partition = TrnTstSplitter(0.7,split_name='tst')(ds_partition)
        ds_partition = TrnTstSplitter(0.6,split_name='val')(ds_partition)

        trace_name = 'trace_%s_on_%s.pkl'%(hp_space.name, ds_partition.name)
        ss = salad.SpearmintSalad(
            evaluator = base.Evaluator(ds_partition),
            metric = metric,
            optimizer = optimizer,
            trace_path = path.join(trace_folder, trace_name),
            seed=None, 
            **kwargs )
        
        result_list.append( pool.apply_async(ss) )
    
    print 'submitted, waiting for results'
    for result in result_list:
        try: 
            result.get() # will raise the error if any
        except:
            traceback.print_exc()
            

    print pool.close()


def launch():
    classification = True
    
    if classification :
        context_name = 'classification'
        metric_ = metric.ZeroOneLoss()
        hp_space = sklearn_models.svc_space()
#        from spearmint_salad import pylearn2_models
#        hp_space = pylearn2_models.maxout_classifier_fixed_units

    else:
        context_name = 'debug_regression'
        metric_ = metric.SquareDiffLoss()   
        hp_space = sklearn_models.gbr_space_3D()
    
    
    salad_on_context(hp_space, context_name, metric_, n_iter = 20,Pool='mp')

if __name__ == "__main__":
    launch()

    
    