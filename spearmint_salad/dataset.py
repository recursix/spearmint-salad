# -*- coding: utf-8 -*-
'''
Created on Nov 24, 2013

@author: alexandre
'''

import numpy as np
from scipy import sparse

# for unpickling  deprecated classes
class PickledDatasetLoader(object): pass 
class Dataset(object): pass


class Bunch(dict): # this should just be part of the standard library
    """
    Makes dictionnary behave like objects i.e., d['x'] is equivalent to d.x
    This should just be part of standard python library ...
    """
    def __init__(self, **kwargs):
        dict.__init__(self, kwargs)
        self.__dict__ = self

def subList(x, idx):
    """
    for compatibility with sparse matrices and lists
    """
    if sparse.isspmatrix(x):
        return x[idx, :]
    elif isinstance(x, np.ndarray):
        return x[idx]
    else:
        return [x[i] for i in idx]


class Slice:
    """
    for compatibility with sparse matrices
    """
    def __init__(self, x):
        self.x = x

    def __getitem__(self, slc):
        if sparse.isspmatrix(self.x):
            return self.x[slc.start: slc.stop: slc.step, :]
        else:
            return self.x[slc.start: slc.stop: slc.step]


def n_samples(x):
    """
    Equivalent of len(x) but compatible with sparse matrix
    """
    return np.shape(x)[0]

class SplitMask:
    
    def __init__(self, start =0., stop=1.):
        self.start = start
        self.stop = stop

    def __call__(self, x):
        m = n_samples(x)
        
        if isinstance( self.start, float):
            start = int(round(m*self.start))
        else:
            start = self.start
            
        if isinstance( self.start, float):
            stop  = int(round(m*self.stop ))
        else:
            stop  = self.stop

        return Slice(x)[start:stop]




class MaskList:
    def __init__(self, *mask_list):
        self.mask_list = []
        self.addMask(*mask_list)

    def addMask(self, *mask_list):
        for mask in mask_list:
            if mask is not None:
                self.mask_list.append(mask)

    def __call__(self, x):
        idx = np.arange(np.shape(x)[0])
        for mask in self.mask_list:
            idx = mask(idx)
        return subList(x, idx)




class DatasetView(object):
    def __init__(self, loader, mask=None, name = '<unknown_dataset>'):
        assert callable(loader)
        self.mask_list = MaskList(mask)
        self.name = name
        self.load_dataset = loader

    def get_dataset(self):
        dataset = self.load_dataset()
        self.apply_mask(dataset)
        return dataset

    def add_mask(self, mask):
        self.mask_list.addMask(mask)

    def apply_mask(self, dataset):
        if hasattr(dataset,'mask'):
            dataset.mask(self.mask_list)
        else:
            dataset.data = self.mask_list( dataset.data )
            dataset.target = self.mask_list( dataset.target )
            
    def clone(self, mask=None):
        ds_view = DatasetView(self.load_dataset, self.mask_list, self.name )
        
        if mask is not None:
            ds_view.add_mask(mask)

        return ds_view






class DatasetPartition(dict):
    """
    This is simply a dictionary of dataset loader with a name.
    """
        
    def __init__(self, name, trn, val, **partition ):
        dict.__init__(self,trn=trn, val=val,**partition)
        self.name = name
    
    def clone(self):
        """
        shallow copy of the partition
        """
        return DatasetPartition(self.name,**self)

     
class XvSplitter:
     
    def __init__(self,  k_fold=5):
        self.k_fold = k_fold
     
    def __call__(self, ds_partition):
        return k_fold_split(ds_partition, self.k_fold)
 
 
def k_fold_split(ds_partition, k_fold=5, src='trn' ):
     
    for i in range(k_fold):
        fold_partition = ds_partition.clone()
        
        ds_loader = ds_partition[src]
                
        fold_partition["%s_cv_%d"%(src,i)] = ds_loader.clone(kFoldMaskTrn(i,k_fold))
        fold_partition['val_cv_%d'%i] = ds_loader.clone(kFoldMaskVal(i,k_fold))

        yield fold_partition

 
     
class kFoldMaskVal:
    def __init__(self, i, k):
        self.i = i
        self.k = k
 
    def __call__(self, x):
        return Slice(x)[self.i::self.k]
 
 
 
class kFoldMaskTrn:
    def __init__(self, i, k):
        self.i = i
        self.k = k
 
    def __call__(self, x):
        idx = range(len(x))
        del idx[self.i::self.k]
        return subList(x, idx)

                