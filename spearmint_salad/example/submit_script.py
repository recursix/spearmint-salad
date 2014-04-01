# -*- coding: utf-8 -*-
'''
Created on Mar 31, 2014

@author: alex
'''

from spearmint_salad import  metric, hp
from spearmint_salad.high_level import make_salad
from os import path


from sklearn.svm import SVC
hp_space = hp.Obj(SVC)(
    C = hp.Float( 0.01, 1000, hp.log_scale ),
    gamma = hp.Float( 10**-5, 1000, hp.log_scale ),
)


dataset_path = path.join('$CONTEXT_FOLDER', 'many_binary' , 'vote')

metric = metric.ZeroOneLoss()

make_salad( hp_space, metric, dataset_path )

