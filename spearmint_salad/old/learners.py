# -*- coding: utf-8 -*-
'''
Created on Oct 10, 2013

@author: alexandre
'''

from sklearn.ensemble import RandomForestClassifier, GradientBoostingRegressor
from sklearn import svm
from base import VariableMap
import numpy as np


def _convert_max_features(param_dict, n_feat):
    max_features = param_dict.get( 'max_features', 'auto' )
    if isinstance(max_features,(float, np.float)):
        max_features = max(1, int(np.round(n_feat*max_features)) )
        param_dict['max_features'] = max_features

class BaseLearner:
    
    def __init__(self, **param_dict):
        self.param_dict = param_dict 


def inv_log10(x):
    return 10**x
    
    
    
class SVC(BaseLearner):
    
    def learn(self, ds ):
        self.param_dict['max_iter']  =1e7
        
        return svm.SVC( **self.param_dict ).fit(ds.x, ds.y)


svc_grid = [
    VariableMap('C', -2, 5, inv_log10 ),
    VariableMap('gamma', -8, 3, inv_log10 )
    ]



class RF(BaseLearner):
    def learn(self, ds ):
        _convert_max_features(self.param_dict, ds.x.shape[1])
#        print 'max_features : ', self.param_dict['max_features']

        if not 'n_estimators' in self.param_dict:
            self.param_dict['n_estimators'] = 100 # our default value
        return RandomForestClassifier(**self.param_dict).fit(ds.x, ds.y)

rf_grid = [
    VariableMap('max_depth', 1, 20, is_int=True ),
    VariableMap('max_features', 0, 1 )
    ]



class SVR(BaseLearner):
    
    def learn(self, ds ):
        return svm.SVR( **self.param_dict ).fit(ds.x, ds.y)

svr_grid = [
    VariableMap('C', -2, 3, inv_log10 ),
    VariableMap('gamma', -8, 3, inv_log10 )
]


class GBR(BaseLearner):
    def learn(self, ds ):
        if not 'n_estimators' in self.param_dict:
            self.param_dict['n_estimators'] = 100 # our default value
        return GradientBoostingRegressor(**self.param_dict).fit(ds.x, ds.y)

from hp import Variable, HpObj, pow10

gbr_hp = HpObj(GradientBoostingRegressor)(
    max_depth = Variable(1, 15, is_int=True ),
    learning_rate = Variable( -2, 0, inv_log10 ),
    n_estimators = 100,
    )


gbr_grid = [
    VariableMap('max_depth', 1, 15, is_int=True ),
    VariableMap('learning_rate', -2, 0, inv_log10 )
    ]

class SkWrap:
    
    def __init__(self, learner):
        self.learner = learner
    
    def __call__(self, *argL, **argD):
        self.argL = argL
        self.argD = argD
    
    def train(self, ds ):
        return self.learner( *self.argL, **self.argD ).fit(ds.x, ds.y )

gbr_hp_space = HpObj(GradientBoostingRegressor)(
    max_depth = Variable(1, 15, is_int=True ),
    learning_rate = Variable( -2, 0, pow10 ),
    n_estimators = 100,
)
