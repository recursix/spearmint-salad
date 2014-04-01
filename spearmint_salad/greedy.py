# -*- coding: utf-8 -*-
'''
Created on Nov 25, 2013

@author: alexandre
'''
import numpy as np
from spearmint_salad.base import argmin_with_ties, rand_arg_min

def agnostic_bayes_prob( eLoss_kN ):
    return np.mean( argmin_with_ties(eLoss_kN), 0 ) 

def greedy_prob( eLoss_kN, ensemble_d ):
    """
    Computes the probability mass of the ith model if it was
    added to the current ensemble (for i in range(N)).
    """
    
    N = eLoss_kN.shape[1]

    prob_N = np.zeros(N)
    for i in range(N):
       
        new_ensemble_D = np.append(ensemble_d, i ) # add model i to the ensemble
        prob_D = agnostic_bayes_prob(eLoss_kN[:,new_ensemble_D] ) 
        prob_N[i] = prob_D[-1] # collect the probability of the recently added model
        
    return prob_N
    
    
def greedy_ensemble( loss_mN, w_km, ensemble_size=10 ):
    """
    Progressively build an ensemble, by greedily selecting the 
    next model that would have the highest probability if added
    to the current ensemble.
    """
    
    eLoss_kN = np.dot(w_km, loss_mN)
    
    ensemble_d = [np.argmin( np.mean(loss_mN, 0) )] # start the ensemble with the best model
    
    for _i in range(ensemble_size-1):
        greedy_prob_N = greedy_prob( eLoss_kN, ensemble_d )
        greedy_prob_N[ensemble_d] = 0 # prevents from selecting twice the same
        ensemble_d.append( rand_arg_min( -greedy_prob_N ) )
        
    prob_d = agnostic_bayes_prob(  eLoss_kN[:,ensemble_d] )
    return ensemble_d, prob_d
    