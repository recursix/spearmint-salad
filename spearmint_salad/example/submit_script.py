# -*- coding: utf-8 -*-
'''
Created on Mar 31, 2014

@author: alex
'''

from spearmint_salad.high_level import make_salad, format_trace_structure, format_final_risk, get_final_predictions, make_partition


from spearmint_salad import  hp
from sklearn.svm import SVC
hp_space = hp.Obj(SVC)(
    C = hp.Float( 0.01, 1000, hp.log_scale ),
    gamma = hp.Float( 10**-5, 1000, hp.log_scale ),
)

from sklearn.datasets.base import load_digits
dataset_partition = make_partition(load_digits, trn_ratio=0.6, val_ratio=0.2)

from spearmint_salad import  metric
metric = metric.ZeroOneLoss()

trace = make_salad( hp_space, metric, dataset_partition, max_iter = 100)

print format_trace_structure(trace)
print
print format_final_risk(trace)
print

prediction_dict = get_final_predictions(trace)
print 'predictions available for %s.'%(', '.join(prediction_dict.keys()))
