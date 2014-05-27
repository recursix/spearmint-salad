# -*- coding: utf-8 -*-
'''
Created on Nov 4, 2013

@author: alexandre
'''
import numpy as np
from mayavi import mlab
from traits.api import Range, on_trait_change, Bool, Button, Str, Float, Int
from traits.api import HasTraits, Instance
from traitsui.api import View, Item, Group, HFlow, HGroup, VGroup, VGrid, CustomEditor, spring
from mayavi.core.ui.api import SceneEditor, MlabSceneModel, MayaviScene
from tvtk.pyface.api import Scene
import matplotlib.pyplot as pp
from spearmint_salad.pkl_trace import get_column_list
from scipy.stats import  binom

#from mlEval.freqTest import sgnTestPValueH0
#def get_column_mat( collection_list, key_list ):



def sgn_test_p_value_( wins, count ):
    p_floor = binom.cdf( np.floor(count-wins), count, 0.5 )
    p_ceil  = binom.cdf( np.ceil(count-wins), count, 0.5 )
    return (p_floor + p_ceil) / 2.

def sgn_test_p_value( wins, count ):
    return binom.cdf( count-wins, count, 0.5 )

def sgn_test_threshold( count, p_value=0.05  ):
    return (count - binom.ppf(p_value, count,0.5) + 1)/count

def sign_test_over_time(trace_list, key_A='salad_risk.tst', key_B='argmin_risk.tst' ):
    
    wins  = np.zeros(1)
    lose = np.zeros(1)
    
    for trace in trace_list:
        a, b = get_column_list( trace.db.analyze, key_A, key_B )
        a = np.array(a)
        b = np.array(b)

        n = max( len(a), len(wins) )
        if n > len(wins):
            print 'resizing', wins.shape
            wins.resize(n)
            lose.resize(n)
        
        elif n > len(a):
            a.resize(n)
            b.resize(n)
        
        print wins.shape, lose.shape, a.shape, b.shape
        
#         mask = a!=b
        wins[a<b] += 1.
        lose[a>b] += 1
#         wins[a==b] += 0.5

#         count[:n] += 1

    return wins, wins+lose




def plot_sign_test_(trace_list, key_pair_list = None, axes=None):
    if axes is None:
        axes = pp.gca()

    if key_pair_list is None:
        key_pair_list = [ ('salad_risk.tst', 'argmin_risk.tst' ) ]
    
    for key_A, key_B in key_pair_list:
        wins, count = sign_test_over_time(trace_list, key_A, key_B )
        axes.plot(wins/count, label='%s vs %s'%(key_A, key_B))
    
    for p_value in [0.05, 0.1]:
        threshold = sgn_test_threshold(count,p_value)
        axes.plot( threshold, label='sgn-test @ %.2f'%p_value)
    axes.legend(loc='best')
    axes.set_xlabel('iteration')

def plot_sign_test(trace_list, key_pair_list = None, axes=None):
    if axes is None:
        axes = pp.gca()

    if key_pair_list is None:
        key_pair_list = [ ('salad_risk.tst', 'argmin_risk.tst' ) ]
    
    for key_A, key_B in key_pair_list:
        wins, count = sign_test_over_time(trace_list, key_A, key_B )
        p_value = sgn_test_p_value( wins, count )
        axes.plot(p_value, label='%s vs %s'%(key_A, key_B))
    
    
    axes.axhline( y =  0.05, color = 'g', label='highly significant' )
    axes.axhline( y =  0.1,  color = 'y', label='significant' )
    
    axes.legend(loc='best')
    axes.set_xlabel('iteration')
    axes.set_ylabel('p-value')


class Hp(HasTraits):
    
    C  = Float()
    gamma = Float()
    epsilon = Float()
    nd = Int()
    
    view = View('C','gamma','epsilon', 'nd')

def get_shape(n):
    n_col = np.ceil(np.sqrt(n))
    n_row = np.ceil( n/ n_col )
    return int(n_row), int(n_col)

class ManyPlots(HasTraits):
    
    a = Range(5,20)
    _pad = Str()
    _sync = Button()
    
    hp = Instance(Hp,())
    def __init__(self):
        super(ManyPlots,self).__init__()
        print 'init'
        self.sceneL = []
        self.itemL = []
        for i in range(4):
            scene_name = 'scene_%d'%(i+1)
            print scene_name
            self.add_trait(scene_name,Instance(MlabSceneModel, ()))
            scene = getattr( self,scene_name )
            scene.on_trait_change(self._add_points, 'activated' )
#            scene.on_trait_change(self.handler)
            item = Item(scene_name, editor=SceneEditor(scene_class=Scene),
                        height=50, width=100, 
                        show_label=False)
            self.itemL.append( item )
            self.sceneL.append( scene )
            
#        for i in range(2):
#            self.itemL.append(Item('_pad',show_label=False, height=50, width=100, resizable=True, style = 'readonly' ) )
        self.itemL.append(spring)
        self.itemL.append( Item('hp', show_label=False, style='custom' ) )

    def grid_arrange(self):
        return View( Group(*self.itemL, columns = 3),resizable=True, style='custom' )
    
    
    
    def group_of_group(self):
        grL= []
        i =0
        n= 3
        for j in range(n):
            grL.append(HGroup(*self.itemL[i:i+n],show_border=True) )
            i+=n
        grL.append( VGroup('_sync') )
        return View(
            Group( *grL),
            resizable=True)


    def start(self):
#        view = self.group_of_group()
        view = self.grid_arrange()
        self.configure_traits(view= view)


    @on_trait_change('_sync')
    def sync(self):
        print 'sync'
        for scene in self.sceneL:
            mlab.sync_camera(self.sceneL[0],scene)
    
    def handler(self, obj, name, old, new):
        print name, new, old

        
    def _add_points(self, scene, name=None, new=None):

        scene.disable_render = True

        mlab.clf(figure=scene)
        x,y,z = np.random.rand(3,50)
        mlab.points3d(x,y,z, figure=scene.mayavi_scene )
        mlab.axes()
        
        scene.disable_render = False

#    @on_trait_change('scene_1.activated,a')
#    def init_view(self):
#
#        self._add_points(self.scene_1)
#        self._add_points(self.scene_2)

import os
from fnmatch import fnmatch

def load_trace_rec(folder, filter_='trace.pkl'):
    trace_path_list= []
    for root, dirs, files in os.walk(folder):
        for file_ in files:
            if fnmatch( file_, filter_ ):
                trace_path_list.append( os.path.join( root, file_ ) ) 
    return trace_path_list



def get_last_exp(exp_folder='$EXPERIMENTS_FOLDER', filter_='*'):
    latest_exp= None
    best_mtime = 0
    
    exp_folder = os.path.expandvars(exp_folder)
    for name in os.listdir(exp_folder):
        if fnmatch(name, filter_):
            exp_path = os.path.join( exp_folder, name )
            mtime = os.path.getmtime(exp_path)
            if mtime > best_mtime:
                latest_exp = exp_path
                best_mtime = mtime
                
    return latest_exp

if __name__ == "__main__":
    #print '\n'.join( load_trace_rec( get_last_exp() ) )
    ManyPlots().start()
