# -*- coding: utf-8 -*-
'''
Created on Mar 11, 2013

@author: alexandre
'''

import numpy as np

# used for pickling and unpickling lambda functions
import marshal 
import types

class HpConfiguration:
    
    def __init__(self, hp_space, unit_value_list):
        self.hp_space = hp_space
        self.unit_value_list = unit_value_list
        
    def instantiate(self):
        return self.hp_space._map( self.unit_value_list )[0]
    
    def var_list(self):
        return self.hp_space.var_list(self.unit_value_list)

    def var_dict(self):
        return self.hp_space.var_dict(self.unit_value_list)


class HpSpace(object): 
    
    def get_hp_keys(self):
        val_list_rec = self._map( [0.5]* len(self) )[1]
        keys = [ key for key, _val in flatten_var_list(val_list_rec)]
        return keys

    def var_list(self, point ):
        return flatten_var_list( self._map(point)[1])

    def var_dict(self, point):
        return dict(flatten_var_list(self._map(point)[1]))  

class log_scale:
    @staticmethod
    def f( x):  return np.log(x)
    
    @staticmethod
    def inv(x): return np.exp(x)

class identity:
    @staticmethod
    def f( x):  return x
    
    @staticmethod
    def inv(x): return x

    

class Variable(HpSpace):
    
    def __init__(self,  min_val, max_val, scale=identity, is_int=False, format="%s"):
        """
        Range is within [min_val, max_val).        
        """
        
        self.min_val = scale.f( min_val )
        self.max_val = scale.f(max_val )
        self.is_int = is_int
        self.scale = scale 
        self.format = format
    
    
    def _map(self, unit_value_list):
       
        assert len(unit_value_list) == 1
        unit_value = unit_value_list[0]
       
        if unit_value >= 1: # makes sure it stays within [0,1)
            unit_value = 1. - np.finfo(1.).eps
        
        span = self.max_val - self.min_val
                    
        val = self.min_val + unit_value*span

        val = self.scale.inv(val) 

        if self.is_int:
            val =  int(np.floor(val))
            
        return val, val

#    def _map_str(self,unit_value_list):
#        return self.format%self._map(unit_value_list)
        
    def __len__(self): return 1

class Int(Variable):
    def __init__(self,  min_val, max_val, scale=identity, format="%d" ):
        super(Int,self).__init__(
            min_val, max_val, scale, is_int=True, format=format )

class Float(Variable):
    def __init__(self,  min_val, max_val, scale=identity, format="%.3g" ):
        super(Float,self).__init__(
            min_val, max_val, scale, is_int=False, format=format )


class Enum(Variable):
    def __init__(self, *value_list ):
        self.value_list = value_list
        super(Enum,self).__init__(0, len(value_list), is_int=True )

    def _map(self, unit_value_list):
        val = super(Enum,self)._map(unit_value_list)[0]
        return self.value_list[val], self.value_list[val]

class Void: pass
void = Void()

class HpStruct(HpSpace):
    
    def __init__(self, *argL, **argD ):
        self._call(*argL, **argD)
    
    def _call(self, *argL, **argD ):
        
        self.argL = zip( range(len(argL)), argL )
        self.argL += argD.items()    
            
        self.mapD = {} # variable params

        self._len = 0
        
        for i, key_arg in enumerate(self.argL):
            key, arg = key_arg
            if isinstance( arg, HpSpace ):

                self._len += len(arg)
                self.mapD[key] = arg
                self.argL[i] = (key,void) # temporary unset this variable

        return self

    def __len__(self): return self._len

    def _map(self, unit_value_list):
        assert len(unit_value_list) == len(self)
        
        argL = []
        argD = {}
        i = 0
        
        var_list = []
        
        for key, arg in self.argL:
            if isinstance(arg, Void ):
                hp_space = self.mapD[key]
                n = len(hp_space)
                value, sub_var_list = hp_space._map( unit_value_list[i:i+n] )
                i += n
                var_list.append( (key, sub_var_list) )
            else:
                value = arg
            
            if isinstance(key, int): 
                argL.append(value)
            else:
                argD[key] = value
        
        return self._instantiate( *argL, **argD ), var_list

class Obj( HpStruct ):
    
    def __init__(self, constructor, name=None):
        self._instantiate = constructor
        if name is None:
            self.name = constructor.__name__
        else:
            self.name = name
        
    def __call__(self, *argL, **argD ):
        return self._call(*argL, **argD)

class Dict( HpStruct ):
    
    
    def _instantiate(self, *argL, **argD ):
        assert len(argL) == 0
        return argD



class Tuple( HpStruct ):
    
    def _instantiate(self, *argL, **argD ):
        assert len(argD) == 0
        return argL

class List( HpStruct ):
    
    def _instantiate(self, *argL, **argD ):
        assert len(argD) == 0
        return list(argL)

def flatten_var_list_rec(var_list):
    var_list_flat = []
    for key,val in var_list:
        if isinstance( val, list ):
            for (path, val_) in flatten_var_list_rec(val):
                var_list_flat.append( ((key,) + path, val_) )
        else:
            var_list_flat.append(  ( (key,), val )  )
    return var_list_flat

def flatten_var_list(var_list):
    var_list_flat = []
    for path, val in flatten_var_list_rec(var_list):
        key_list = ['hp']
        for key in path:
            if isinstance(key, int):
                key_list.append('[%d]'%key)
            else:
                key_list.append('.%s'%key)
        
        var_list_flat.append( ( ''.join(key_list) , val ) )

    return var_list_flat

def wrap_lambda( f ):
    if getattr(f, '__name__',None ) == '<lambda>':
        return LambdaFunc(f)
    else :
        return f


class LambdaFunc:
    
    def __init__(self, f ):
        self.f = f
    
    def __getstate__(self):
        odict = self.__dict__.copy()
        odict['f'] = marshal.dumps( self.f.func_code )
        return odict
    
    def __setstate__(self, d):
        self.__dict__.update(d)  
        self.f = types.FunctionType(marshal.loads(self.f), globals()) 
           
    def __call__(self, *argL, **argD):
        return self.f( *argL, **argD )

def obj_example():
    
    class Kernel:
        
        def __init__(self, g):
            self.g = g
    
        def __str__(self):
            return '{g=%.3g}'%self.g
    
    class Learner:
        
        def __init__(self, a, b =5, k=None ):
            self.a = a
            self.b = b
            self.k = k
        
        def __str__(self):
            return 'a=%d, b=%d, k=%s'%(self.a, self.b, str(self.k) )
    
    kernel_space = Obj( Kernel )( 
        g = Variable( -3, 3, lambda x:10**x ) # corresponds to a logspace from 10**-3 to 10**3
    )
    
    hp_space = Obj( Learner )(
        a = Variable( 1,10, is_int = True ),
        k = kernel_space,
        b = 8, # it is possible to set a constant
    )
    
    print 
    for unit_value_list in np.random.rand(10,len(hp_space)):
        hp_configuration = HpConfiguration(hp_space, unit_value_list)
        
        # we can instantiate the appropriate object
        print 'learner : ', hp_configuration.instantiate()
        
        # we can also obtain a dictionnary of the variables (does not include constants)
        print hp_configuration.var_dict()
        
        # var_list will return a list aligned with unit_value_list
        key_list, val_list =  hp_configuration.var_list()
        for unit_value, key, val in zip(unit_value_list, key_list, val_list ):
            print '%6s = %s (unit value = %.3f)'%(key, str(val), unit_value)
            

        print


def sample_models(hp_space, n=10):
    for unit_value_list in np.random.rand(n,len(hp_space)):
        hp_configuration = HpConfiguration(hp_space, unit_value_list)
        
        # we can instantiate the appropriate object
        print 'learner : ', hp_configuration.instantiate()
        
        # we can also obtain a dictionnary of the variables (does not include constants)
        print hp_configuration.var_dict()
        
        # var_list will return a list aligned with unit_value_list
        key_list, val_list =  hp_configuration.var_list()
        for unit_value, key, val in zip(unit_value_list, key_list, val_list ):
            print '%6s = %s (unit value = %.3f)'%(key, str(val), unit_value)
            

def dict_example():
    
    hp_space = Dict(
        a = Int( 1,10 ),
        C = List( Float( -1, 1,  ), 0.5, Float( 10**-3, 10**3, log_scale ) ),
        e = Enum('left', 'right' ),
    )
    
    from cPickle import dumps, loads
    pickle = dumps( hp_space, 2 ) # just makes sure we can pickle the lambda function
    print 'pickle len : %d'%len(pickle)
    hp_space = loads(pickle)

    for unit_value_list in np.random.rand(10,len(hp_space)):
        hp_conf = HpConfiguration(hp_space, unit_value_list)
        print 'as instance', hp_conf.instantiate()
        print 'as var dict', hp_conf.var_dict()
        print

def dict2_example():
    
    hp_space = Dict(**{
        'a*':Int( 1,10 ),
#         'C':List( Float( -1, 1,  ), 0.5, Float( 10**-3, 10**3, log_scale ) ),
        'e':Enum('left', 'right' ),
    })
    
    from cPickle import dumps, loads
    pickle = dumps( hp_space, 2 ) # just makes sure we can pickle the lambda function
    print 'pickle len : %d'%len(pickle)
    hp_space = loads(pickle)

    for unit_value_list in np.random.rand(10,len(hp_space)):
        hp_conf = HpConfiguration(hp_space, unit_value_list)
        print 'as instance', hp_conf.instantiate()
        print 'as var dict', hp_conf.var_dict()
        print

if __name__ == "__main__":
    obj_example()
#     dict2_example()
    