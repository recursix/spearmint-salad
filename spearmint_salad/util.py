# -*- coding: utf-8 -*-
'''
Created on Nov 24, 2013

@author: alexandre
'''

from os import path

import cPickle
import fcntl
import zlib
from scipy import signal


def expandVarsRec( vPath ):
    """
    like path.expandvars but will be called recursively while vPath changes.
    This is useful when a variable contains a variable;
    Ex.: vPath = $HOME
    $HOME=/home/$USER
    $USER=nigel
    then path.expandvars( vPath ) will return /home/$USER
    while expandVarsRec( vPath ) will return /home/nigel 
    """
    maxRec = 10
    while True:
        nPath = path.expandvars( vPath )
        if nPath == vPath:
            return nPath
        else:
            vPath = nPath
        maxRec -=1
        if maxRec <=0 : break # some security to avoid infinite recursion

def readFile( *args ):
    """
    A wrapper to simplify file reading
    fileName = path.join(*args)
    If fileName doesn't exist None is returned.
    """
    filePath = path.join( *args )
    if not path.exists( filePath ):
        return None
    with open( filePath, 'r' ) as fd:
        return fd.read()
    
    
    
    
def writeFile( str_, *args ):
    """
    A wrapper to simplify file writing
    fileName = path.join(*args)
    """
    filePath = path.join( *args )
    with open( filePath, 'w' ) as fd:
        fd.write(str_)
    
    

def writePkl( obj, *args, **kwArgs ):
    """
    Serialize an object to file.
    the fileName is path.join(*args)
    """
    pklPath = path.join( *args )
    with open( pklPath, 'w' ) as fd:
        try: 
            if kwArgs['lock']: fcntl.flock(fd, fcntl.LOCK_EX+ fcntl.LOCK_NB )
        except KeyError: pass
        if 'compress' in kwArgs and kwArgs['compress']:
            fd.write( zlib.compress( cPickle.dumps(obj,cPickle.HIGHEST_PROTOCOL) ) )
        else:
            cPickle.dump( obj, fd, cPickle.HIGHEST_PROTOCOL )


def writePklz( obj, *args ):
    writeFile( zlib.compress( cPickle.dumps(obj,cPickle.HIGHEST_PROTOCOL) ), *args )

def readPklz( *args ):
    with open( path.join(*args), 'r' ) as fd:
        str_ = fd.read() 
    return cPickle.loads(zlib.decompress( str_ ))
        
def readPkl( *args, **kwArgs ):
    """
    Unserialize an object from file.
    the fileName is path.join(*args)
    """
    try:
        with open( path.join( *args ), 'r') as fd: return cPickle.load(fd)
    except IOError: 
        if kwArgs.has_key('defaultVal'): return kwArgs['defaultVal']
        else: raise
 
 
def gaussConv(std=3., *xL):
    f = signal.gaussian(int(4*std), std)
    f /= f.sum()
    return [ signal.convolve(x, f, 'valid') for x in xL ]
