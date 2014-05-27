# -*- coding: utf-8 -*-
'''
Created on Oct 16, 2013

@author: alexandre
'''


import cPickle
import time as t
import os
import socket
import traceback
import getpass
import fcntl
import tempfile


def get_sys_info():
    info = {
        'main': traceback.extract_stack()[0][0],
        'user' : getpass.getuser(),
        }
    return info



default_trace_path = None 

def get_default_trace_path():
    global default_trace_path

    if default_trace_path is None:
        prefix = t.strftime('trace_%d%b%Y-%H_%M_%S_')
        with tempfile.NamedTemporaryFile('r', suffix='.pkld', prefix=prefix, delete=False) as fd:
            default_trace_path = fd.name
        
    return default_trace_path
    
    
class Trace:
    
    
    def __init__(self, trace_path=None, trace_level=10):
        
        if trace_path is None:
            trace_path = get_default_trace_path()
            print 'trace_path:', trace_path
         
        self.trace_level = trace_level
        self.trace_path = trace_path
        self.pid = os.getpid()
        self.hostname = socket.gethostname()
        self.write(-1,'process',get_sys_info())
    
    def write(self, trace_level, collection, info  ):
        if trace_level > self.trace_level: return
        
        data = cPickle.dumps( (t.time(),self.pid,self.hostname,info, collection), cPickle.HIGHEST_PROTOCOL)
        with open( self.trace_path, 'a' ) as fh:
            fcntl.flock(fh, fcntl.LOCK_EX ) # lock the file
            fh.write(data)
            fcntl.flock(fh, fcntl.LOCK_UN ) # release the lock

    def __call__(self, info ):
        self.write(0, 'main', info)


#def _parse_path( col_name ):
#    return col_name.split('.')


#def _extract_info(doc, path_list ):
#    val_list = []
#    for path in path_list:
#        sub_doc = doc
#        for key in path: 
#            sub_doc = sub_doc[key]
#        val_list.append(sub_doc)
#    return val_list


class Collection:
    
    def __init__(self):
        self._l = []
        self._id = 0
    
    @property
    def docs(self):
        return self._l
    
    def insert(self, doc):
        
        try:
            _id = doc['_id']
        except KeyError:
        
            _id = self._id
            doc['_id'] = _id
            self._id +=1
            
            
        self._l.append( doc )
        return (_id,)

    def find(self, query={}):
        for doc in self._l:
            match = True
            
            for col_name, pattern in query.items():
                assert pattern['$exists'] == True 
                try:
                    _val = get_recursif(doc, col_name)
                except (KeyError, IndexError):
                    match = False
                    break
                
            if match:
                yield doc
            


class DB:
    
    def __init__(self):
        self._d= {}
    
    def get_collection_list(self):
        return self._d.keys()
    
    def __getattr__(self, key):
        return self.__getitem__(key)
    
    def __getitem__(self, key ):
        if not self._d.has_key(key):
            self._d[key] = Collection()
        return self._d[key]

def get_column_dict( collection, *column_names):
    
    column_dict = { col_name: [] for col_name in column_names}
    query = { col_name: {'$exists':True} for col_name in column_names}
    
    for doc in collection.find(query):
        for col_name in column_names:
            column_dict[col_name].append( get_recursif(doc,col_name) )

    return column_dict

def get_column_list( collection, *col_names):
    col_dict = get_column_dict(collection, *col_names)
    return [ col_dict[col_name] for col_name in col_names ]


class TraceDB(object):
    pass

from traceback import print_exc

class TraceDBFile(TraceDB):
    
    def __init__(self, trace_path=None):
        if trace_path is None:
            trace_path = get_latest_trace()
            
        self.trace_path = trace_path
        
        self.db = DB()
        self.seek = 0
        self.id = 0
        self.verbose = 2
        self.process_dict = {}
        self.update_trace()
        
    def update_trace(self):
        with open( self.trace_path, 'r' ) as fh:
            if self.seek != 0:
                if self.verbose >= 1:
                    print 'setting seek to %d'%self.seek
                fh.seek(self.seek)
            
            while True:
                try:
                    _timestamp, pid, hostname, info, collection = cPickle.load( fh )
                    try:
                        info['_id'] = self.id
                    except TypeError:
                        print info
                        raise 
                    self.id += 1
                    
                    if collection == 'process' :
                        info['pid'] = pid
                        info['hostname'] = hostname
                        self.process_dict[(pid,hostname)] = self.db[collection].insert(info)[0]
                    else:
                        info['proc_id'] = self.process_dict.get( (pid,hostname) )
                        self.db[collection].insert(info)
 
                                
                except EOFError:
                    break
#                 except AttributeError:
#                     print_exc()
                    
            self.seek = fh.tell()

#        print 'next seek:', self.seek
    
#    
#class TraceLoader:
#    
#    def __init__(self, trace_path=None, db=None ):
#        
#        if trace_path is None:
#            trace_path = get_latest_trace()
#        
#        self.trace_path = trace_path
#        self.seek = 0
#        self.id = 0
#        
#        if db is None:
#            db_name = path.splitext(path.basename(trace_path))[0]
#            db = get_db(db_name)
#        
#        self.db = db
#        self.process_dict = {}
#
#    def update_trace(self):
#        
#        with open( self.trace_path, 'r' ) as fh:
#            if self.seek != 0:
#                print 'setting seek to %d'%self.seek
#                fh.seek(self.seek)
#            
#            while True:
#                try:
#                    _timestamp, pid, hostname, info, collection = cPickle.load( fh )
#                    
#                    info['_id'] = self.id
#                    self.id += 1
#                    
#                    try:
#                        if collection == 'process' :
#                            info['pid'] = pid
#                            info['hostname'] = hostname
#                            self.process_dict[(pid,hostname)] = self.db[collection].insert(info)[0]
#                        else:
#                            info['proc_id'] = self.process_dict.get( (pid,hostname) )
#                            
#                            
#                            try:
#                                self.db[collection].insert(info)
#                            except bson.errors.InvalidDocument :
#                                
##                                stderr.write("can't document convert to bson.\n")
#                                traceback.print_exc()
#                                stderr.write( 'collection : %s\n'%collection )
#                                for key,val in info.items():
#                                    stderr.write('%s ( %s ): %s\n'%( str(key), str(type(val)), str(val) ) )
#                                
#                                stderr.write('\n')
#                                
#                    except pymongo.errors.DuplicateKeyError:
##                        print 'already in db'
#                        pass
#                except EOFError:
#                    break
#            self.seek = fh.tell()
#            print 'next seek:', self.seek

def get_latest_trace(dir='/tmp'):
    
    latest_trace = None
    best_mtime = 0
    
    for name in os.listdir(dir):
        if name.startswith('trace_'):
            trace_path = os.path.join( dir, name )
            mtime = os.path.getmtime(trace_path)
            if mtime > best_mtime:
                latest_trace = trace_path
                best_mtime = mtime

    return latest_trace

#def get_db(db_name='trace_db'):
#    client = pymongo.MongoClient('localhost', 27017)
#    return client[db_name]
    


def get_recursif(struct, path ):
    for name in path.split('.'):
        try:
            struct = struct[name]
        except (TypeError, ValueError):
            struct = struct[int(name)]
            
    return struct
    

def get_recursif_(struct, path ):
    for key in path:
        struct = struct[key]
    return struct


def plot_stats(collection, axes, x_key, *y_key_list):
    
    col_dict = get_column_dict( collection, * ((x_key, ) + y_key_list)  )
    if x_key is not None:
        x = col_dict.get(x_key)
        if x is None: print 'unknown key %s for x axis'%(x_key) 

    for y_key in y_key_list:

        y = col_dict.get(y_key)
        if y is None: print 'unknown key %s for y axis'%(y_key) 
        
        if x_key is None:
            axes.plot(y,'.-',label=y_key)
        else:
            axes.plot(x,y,'.-',label=y_key)
        
    axes.set_xlabel(x_key)
    axes.legend(loc='best')
    


