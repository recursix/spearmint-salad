#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''
Created on Oct 23, 2013

@author: alexandre
'''


from os import path
from jobDispatcher.dispatch import newExperiment
from jobDispatcher import dispatcher

job_conf={
    "nNode"     : 1,
    "walltime"  : 1*60*60,
#    "queue" : 'courte',
    }    

experiment = newExperiment( 'maxout' )

job = experiment.newJob("main")

job.setConf( mpi=True, **job_conf )
job.writeCmd('train.py /RQusagers/recursix/_exp/mnist.yaml')


print "cd '%s'"%experiment.folder

d = dispatcher.getHostDispatcher()
d.submitExperiment(experiment)
