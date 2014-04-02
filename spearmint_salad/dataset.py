# -*- coding: utf-8 -*-
'''
Created on Nov 24, 2013

@author: alexandre
'''
from sklearn.datasets.base import Bunch
from os import path
from util import expandVarsRec, readFile, readPklz
import os
import numpy as np
from scipy import sparse
import copy

class Dataset:
    def __init__(self, x, y, idx=None, shape=None, name=None, path=None):
        self.x = x
        self.y = y
        if idx is not None:
            self.idx = idx
        else:
            self.idx = np.arange(self.n_samples)

        self.name = name
        
#        if shape is None:
#            shape = DatasetShape(
#                infer_space(self.x), 
#                infer_space(self.y), 
#                self.n_samples, 
#                self.name)
                
        self.shape = shape
        self.path = path

    @property
    def n_samples(self):
        return np.shape(self.x)[0]

    def mask(self, mask=None):
        if mask is None:
            return

        self.x = mask(self.x)
        self.y = mask(self.y)

        if self.idx is not None:
            self.idx = mask(self.idx)

    def __str__(self):
        return '<%s> %s m=%d, xSpace:%s, ySpace:%s' % (
            self.__class__.__name__, self.name, self.n_samples, self.shape.x_space, self.shape.y_space)
        

class DatasetShape:
    def __init__(self, x_space=None, y_space=None, n_samples=None, name=None):
        self.x_space = x_space
        self.y_space = y_space
        self.n_samples = n_samples
        self.n_samples_before_mask = n_samples
        self.name = name

    def mask(self, mask=None):
        if mask is None:
            return
        self.n_samples = mask.count(self.n_samples)

    def __str__(self):
        return '<%s> %s m=%d, xSpace:%s, ySpace:%s' % (
            self.__class__.__name__, self.name, self.n_samples, self.x_space, self.y_space)


def subList(x, idx):
    if sparse.isspmatrix(x):
        return x[idx, :]
    elif isinstance(x, np.ndarray):
        return x[idx]
    else:
        return [x[i] for i in idx]


class Slice:
    def __init__(self, x):
        self.x = x

    def __getitem__(self, slc):
        if sparse.isspmatrix(self.x):
            return self.x[slc.start: slc.stop: slc.step, :]
        else:
            return self.x[slc.start: slc.stop: slc.step]


class SplitMask:
    def __init__(self, start, stop):
        self.start = start
        self.stop = stop

    def __call__(self, x):
        return Slice(x)[self.start:self.stop]

    def count(self, m):
        return self.stop - self.start


class kFoldMaskVal:
    def __init__(self, i, k):
        self.i = i
        self.k = k

    def __call__(self, x):
        return Slice(x)[self.i::self.k]

    def count(self, m):
        return len(xrange(self.i, m, self.k))


class kFoldMaskTrn:
    def __init__(self, i, k):
        self.i = i
        self.k = k

    def __call__(self, x):
        idx = range(len(x))
        del idx[self.i::self.k]
        return subList(x, idx)

    def count(self, m):
        return m-len(xrange(self.i, m, self.k))



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

    def count(self, m):
        for mask in self.mask_list:
            m = mask.count(m)
        return m


class DatasetLoader(object):
    def __init__(self, path, mask=None):
        self.path = path
        self.mask_list = MaskList(mask)
        self.shape = self.get_shape()

    def get_dataset(self):
        return NotImplementedError('get_dataset is not implemented')

    def get_shape(self):
        dataset = self.get_dataset()
        return DatasetShape( n_samples = dataset.n_samples, name = dataset.name)
        

    def add_mask(self, mask):
        self.mask_list.addMask(mask)
        if self.shape is not None:
            self.shape.mask(mask)

    def get_name(self):
        return os.path.basename(self.path).split('.')[0]

    def clone(self, mask):
        dsLoader = copy.deepcopy(self)
        if mask is not None:
            dsLoader.add_mask(mask)

        return dsLoader

#    def partition_dataset(self, partitioner=SplitPartitioner(0.5)):
#        return partitioner(self)


class PickledDatasetLoader(DatasetLoader):
    def __init__(self, dataset_path, mask=None):
        super(PickledDatasetLoader, self).__init__(dataset_path, mask)

    def get_dataset(self):
        dsPath = os.path.expandvars(os.path.join(self.path, "ds.pklz"))
        ds = readPklz(dsPath)
        ds.path = os.path.expandvars(self.path)

        if getattr(ds, 'idx', None) is None:
            ds.idx = np.arange(ds.n_samples)

        ds.mask(self.mask_list)
        return ds

    def get_shape(self):
        dsShapePath = os.path.expandvars(os.path.join(self.path, 'dsShape.pklz'))
        if os.path.exists(dsShapePath):
            shape = readPklz(dsShapePath)
            shape.mask(self.mask_list)
            return shape
        else:
            return None

    def get_name(self):
        return self.shape.name

    def __repr__(self):
        return "dsLoader for %s(%s) (m=%d, x=%s, y=%s, mask=%s)" % (
            self.get_name(), self.path, self.shape.n_samples, str(self.shape.x_space), str(self.shape.y_space),
            str(self.mask_list))

class SklearnDatasetLoader(DatasetLoader):
    
    def __init__(self, ds_name):
        self.ds_name = ds_name
    
    def get_dataset(self):
        pass
    

class DatasetPartition(dict):
    """
    This is simply a dictionary of dataset loader with a name.
    """
        
    def __init__(self, name, partition={} ):
        dict.__init__(self,**partition)
        self.name = name
    
    def clone(self):
        """
        shallow copy of the partition
        """
        return DatasetPartition(self.name,self)

    
class XvSplitter:
    
    def __init__(self,  k_fold=5):
        self.k_fold = k_fold
    
    def __call__(self, ds_partition):
        return k_fold_split(ds_partition, self.k_fold)


def k_fold_split(ds_partition, k_fold=5, src='trn', dst='val' ):
    
    for i in range(k_fold):
        fold_partition = ds_partition.clone()
        
        for key, ds_loader in ds_partition.items():
            if key == src:
                fold_partition[src] = ds_loader.clone(kFoldMaskTrn(i,k_fold))
                fold_partition[dst] = ds_loader.clone(kFoldMaskVal(i,k_fold))

        yield fold_partition

class TrnTstSplitter:
    
    def __init__(self, trn_ratio=0.5, min_trn = None, max_trn =None, split_name='tst' ):
        
        assert trn_ratio <= 1 and trn_ratio >= 0
        if max_trn is not None and min_trn is not None:
            assert max_trn >= min_trn

        self.trn_ratio = trn_ratio
        self.min_trn = min_trn
        self.max_trn = max_trn
        self.split_name = split_name
    
    def get_split_size(self, m):

        m_trn = int(round(m*self.trn_ratio))
        
        if self.min_trn is not None and m_trn < self.min_trn: 
            m_trn = self.min_trn
        if self.max_trn is not None and m_trn > self.max_trn:
            m_trn = self.max_trn

        return m_trn
    
    def __call__(self, ds_partition):
        
        trn_ds_loader = ds_partition['trn']
        m = trn_ds_loader.shape.n_samples
        m_trn = self.get_split_size(m)
        new_ds_partition = ds_partition.clone()
        
        new_ds_partition['trn'] = trn_ds_loader.clone(SplitMask(0,m_trn))
        new_ds_partition[self.split_name] = trn_ds_loader.clone(SplitMask(m_trn,m))
        return new_ds_partition


class Context(object):

    def _get_name(self):
        raise NotImplementedError()
    
    name = property(_get_name)

    def _get_description(self):
        raise NotImplementedError()
    description = property(_get_description)

    def __iter__(self):
        raise NotImplementedError()

    def __str__(self):
        return '<%s> %s' % (self.__class__.__name__, self.name)



class FolderContext(Context):
    
    def __init__(self, folder, shuffle_idx=None):
        self._folder = folder
        self.shuffle_idx = shuffle_idx

    def _get_folder(self):
        return expandVarsRec(self._folder)

    folder = property(_get_folder)
    
    def _get_name(self):
        name = readFile(self.folder, 'name')
        if name is None:
            name = path.basename(self.folder)
        else:
            name = name.strip()
        s = name
        if self.shuffle_idx is not None:
            s += '-shuffle%02d' % (self.shuffle_idx)
        return s

    name = property(_get_name)
    
    def _get_description(self):
        return readFile(self.folder, 'description')

    def __iter__(self):
        folder = self.folder
        for name in os.listdir(folder):
            if path.isdir(path.join(folder, name)):
                ds_folder = path.join(self._folder, name)
                
                ds_loader = PickledDatasetLoader(ds_folder)
                if self.shuffle_idx is not None:
                    ds_loader.add_mask(PklShuffleMask(ds_folder, self.shuffle_idx))
                
                yield DatasetPartition(ds_loader.shape.name, {'trn':ds_loader}) 
                
                
                