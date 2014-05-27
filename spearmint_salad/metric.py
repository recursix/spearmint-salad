# -*- coding: utf-8 -*-
'''
Created on Mar 4, 2013

@author: alexandre
'''

import numpy as np



def square_diff_loss( target_md, predict_md):
    target_md, predict_md = _check_y_shape( target_md, predict_md )
    return ((target_md - predict_md)**2).sum(1)
    
def abs_diff_loss(target_md, predict_md):
    target_md, predict_md = _check_y_shape( target_md, predict_md )
    return np.abs(target_md- predict_md).sum(1)

def abs_impalance_loss( target_md, predict_md):
    target_md, predict_md = _check_y_shape( target_md, predict_md )
    loss = target_md - predict_md
    loss[loss < 0 ] *= -10
    return loss.sum(1)

def zero_one_loss( yTarget_md, y_md):
    yTarget_md, y_md = _check_y_shape( yTarget_md, y_md )
    loss_m = (y_md != yTarget_md).all(1).astype(np.int)
#     print '01 loss.shape:', yTarget_md.shape, y_md.shape, loss_m.shape
    return loss_m


def zero_one_loss_prob( yTarget_m, p_md):
#     print p_md.shape
    np.testing.assert_almost_equal( p_md.sum(1), 1 )
    y_m = np.argmax(p_md,1)
    yTarget_m1, y_m1 = _check_y_shape( yTarget_m, y_m )
    loss_m = (y_m1 != yTarget_m1).all(1).astype(np.int)
#     print '01 loss.shape:', yTarget_md.shape, y_md.shape, loss_m.shape
    return loss_m

def _check_y_shape( y_target_md, y_md ):
    if y_target_md.ndim == 1:
        y_target_md = y_target_md.reshape(-1,1)
    if y_md.ndim == 1:
        y_md = y_md.reshape(-1,1)
    assert y_target_md.shape == y_md.shape, "%s,%s"%(y_target_md.shape, y_md.shape)
    return y_target_md, y_md

def reshape_weight( y_dm, w_dm):
    d,m = y_dm.shape
    
    if w_dm is None:
        w_dm = np.ones((d,1))/float(d) # uniform distribution
    if np.ndim(w_dm) == 1:
        assert len(w_dm) == d
        w_dm = w_dm.reshape(d, 1)
    
    
    np.testing.assert_array_equal(w_dm >=0, True, err_msg='The weights must be  positive values.' )
    
    n = w_dm.shape[1]
    assert n==1 or n==m
    return w_dm

def weighted_argmax( y_dm1, w_d ):
    y_dm1 = np.asarray(y_dm1)
    
    if y_dm1.ndim == 3:
        assert y_dm1.shape[2] == 1, 'this model averaging method is meant to work only with class labels'
        assert y_dm1.shape[0] == w_d.shape[0]
        y_dm = y_dm1.reshape(y_dm1.shape[0],-1)
    else:
        assert y_dm1.ndim == 2
        y_dm = y_dm1
    
    m = y_dm.shape[1]
    ySet_c = np.unique(y_dm) # c is the number of classes
    
    cumWeight_cm = np.zeros((len(ySet_c), m))
    
    for i, y in enumerate(ySet_c):
        cumWeight_cm[i, :] = ((y_dm == y) * w_d.reshape(-1,1)).sum(0)
    
    idx_m = np.argmax(cumWeight_cm, axis=0)
    y_m = ySet_c[idx_m]
    
    return y_m


def merge_prob( y_dmc, w_d ):
    # d: the number of predictors
    # m: the number of samples in the batch
    # c: the number of classes 
    y_dmc = np.asarray(y_dmc)
    
#     print 'y_dmc.shape:', y_dmc.shape
    assert y_dmc.shape[0] == w_d.shape[0]
    np.testing.assert_almost_equal(w_d.sum(), 1 )
    np.testing.assert_almost_equal(y_dmc.sum(2), 1 )
#     print y_dmc.shape, w_d.shape
    
    p_mc = (w_d.reshape(-1,1,1) * y_dmc ).sum(0)
    np.testing.assert_almost_equal(p_mc.sum(1), 1 )
    return p_mc
#     return np.argmax( p_mc, 1 ) 
    
    
    

def weighted_median_( x_n, w_n=None, quantile=0.5):
    
    if w_n is None and quantile == 0.5: # use the existing median algorithm
        return np.median(x_n)
    
    idx_n = np.argsort( x_n )
    wSorted_n = w_n[idx_n]
    xSorted_n = x_n[idx_n]
    cumW_n = np.cumsum(wSorted_n)/np.sum(w_n)
    
    medianIdx = np.searchsorted(cumW_n, quantile)
    
    if medianIdx == 0: # the case where the smallest answer has more than 50% of the weight
        return xSorted_n[0]
    
    r = (quantile - cumW_n[medianIdx-1]) / ( cumW_n[medianIdx] - cumW_n[medianIdx-1] )
    assert r >=0 
    assert r <= 1
    
    x = xSorted_n[medianIdx-1]*(1-r) + xSorted_n[medianIdx] * r
    return x 
    

def weighted_median( y_dm, w_dm=None ):
    w_dm = reshape_weight( y_dm, w_dm )

    m = y_dm.shape[1]
    n = w_dm.shape[1]
    yL = [ weighted_median_( y_dm[:,i], w_dm[:,i%n] ) for i in range(m) ]
    return np.array(yL)


def weighted_average( y_dm, w_d ):
    y_dm = np.asarray(y_dm)
    w_d1 = w_d.reshape(-1,1)
    assert w_d1.shape[0] == y_dm.shape[0]
    return np.sum( y_dm * w_d1, 0 )



def extract_prob( prob_mc, idx_m ):
    idx_m = idx_m.astype(np.int) 
    range_m = np.arange(len(idx_m)) 
    return prob_mc[range_m, idx_m ]

def nll( y_target_m, y_prob_mc ):
    np.testing.assert_almost_equal( y_prob_mc.sum(1), 1) # otherwise it's cheating 
    
    p_m = extract_prob( y_prob_mc, y_target_m )
    return -np.log(p_m)

class Metric:
    
    @property
    def name(self):
        return self.__class__.__name__
    
    def optim_loss(self,*args, **kwargs):
        return self.loss( *args, **kwargs)

class ZeroOneLoss(Metric):    
    short_name = '01'
    loss = staticmethod( zero_one_loss )
    optim_loss = staticmethod( zero_one_loss )
    model_averaging = staticmethod( weighted_argmax )
    
class SquareDiffLoss(Metric):
    short_name = 'sq'
    loss = staticmethod( square_diff_loss )
    optim_loss = staticmethod( square_diff_loss )
    model_averaging = staticmethod( weighted_average )
    
class AbsDiffLoss(Metric):
    short_name = 'abs'
    loss = staticmethod( abs_diff_loss )
    optim_loss = staticmethod( abs_diff_loss )
    model_averaging = staticmethod( weighted_average ) # not the median ! 

class AbsImbalanceLoss(Metric):
    short_name = 'absi'
    loss = staticmethod( abs_impalance_loss )
    optim_loss = staticmethod( abs_impalance_loss )
    model_averaging = staticmethod( weighted_average ) # not the median ! 

class ZeroOneNLL(Metric):
    short_name = '01-nll'
    loss = staticmethod( zero_one_loss_prob )
    optim_loss = staticmethod( nll )
    model_averaging = staticmethod( merge_prob )
    

class NLL(Metric):
    short_name = 'nll'
    loss = staticmethod( nll )
    optim_loss = staticmethod( nll )
    model_averaging = staticmethod( merge_prob )
    

if __name__ == "__main__":
    metric =  ZeroOneNLL()
    print metric.name, metric.short_name
    