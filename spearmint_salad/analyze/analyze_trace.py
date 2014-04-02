# -*- coding: utf-8 -*-
'''
Created on Oct 17, 2013

@author: alexandre
'''

import matplotlib
# We want matplotlib to use a wxPython backend
matplotlib.use('WXAgg', warn=False)

from spearmint_salad import pkl_trace
from spearmint_salad.pkl_trace import get_column_dict, get_column_list, plot_stats
from spearmint_salad.hp import HpConfiguration
import matplotlib.pyplot as pp
import scipy.stats as sps
from matplotlib.colors import NoNorm
import numpy as np
from graalUtil.num import uHist
from spearmint_salad.spearmint.chooser.gp import GP
import scipy.linalg   as spla
from plot_nd import Plot3d_with_params
from os import path


cmap = 'YlOrRd'


class MyGP:
    
    def __init__(self, *argL, **argD ):

        self.gp = GP(*argL, **argD)
    
    
    def __str__(self):
        
        infoD = {}
        for field in ['mean', 'amp2', 'noise', 'ls']:
            infoD[ field ] = getattr(self.gp, field) 
        
        return '\n'.join([ "%s : %s"%(key, str(val)) for key, val in infoD.items() ]) 
    
    def fit(self, x, y):
        
        m,d = x.shape
        assert y.shape == (m,), "m = %d, y.shape = %s"%(m, str(y.shape))
        
        self.x = x
        self.y = y
        
        self.gp.real_init(d,y)
        
        self.gp.optimize_hypers(x,y)

        # The primary covariances for prediction.
        K   = self.gp.cov(x)
        assert K.shape == (m,m)

        # Compute the required Cholesky.
        obsv_cov  = K + self.gp.noise*np.eye(x.shape[0])
        obsv_chol = spla.cholesky( obsv_cov, lower=True )

        # Solve the linear systems.
        self.alpha  = spla.cho_solve((obsv_chol, True), y - self.gp.mean)
        self.obsv_chol = obsv_chol
        return self
        
    def predict(self,x):
        cand_cross = self.gp.cov(self.x, x) # O(mdn)
        assert cand_cross.shape == (self.x.shape[0], x.shape[0])
        y_mean = np.dot(cand_cross.T, self.alpha) + self.gp.mean # O(mn)
        return y_mean

    def predict_proba(self,x):
        

        cand_cross = self.gp.cov(self.x, x) # O(mdn) 
        beta   = spla.solve_triangular(self.obsv_chol, cand_cross, lower=True) # O(mmn)
        assert cand_cross.shape == (self.x.shape[0], x.shape[0])
        assert beta.shape == (self.x.shape[0],x.shape[0])

        # Predict the marginal means and variances at candidates.
        y_mean = np.dot(cand_cross.T, self.alpha) + self.gp.mean # O(mn)
        y_var = self.gp.amp2*(1+1e-6) - np.sum(beta**2, axis=0) # O(mn)

        return y_mean, y_var

    def ei(self,x):
        best = np.min(self.x)
        y_mean, y_var = self.predict_proba(x)
        
        y_std = np.sqrt(y_var)
        u      = (best - y_mean) / y_std
        ncdf   = sps.norm.cdf(u)
        npdf   = sps.norm.pdf(u)
        ei     = y_std*( u*ncdf + npdf)
        
        return ei, y_mean, y_var



class Uniformize:
    
    def __init__(self, x, sz = 1000 ):
        x = np.asarray(x)
        x_  = x+ np.std(x)*0.01*np.random.randn(*x.shape)
        hist, bins = np.histogram(x_, sz )
        hist = hist.astype(np.float)/ hist.sum()
        self.cdf = np.insert( np.cumsum(hist), 0, 0 )
        self.bins = bins[:-1] + np.diff(bins)*0.5
        
    
    def __call__(self, x ):
        return self.cdf[ np.searchsorted(self.bins, x) ] 
        




def plot_selected_hp_trace(trace):
    prob_dict, = get_column_list(trace.db.salad,  'prob' )
    hp_id_list,  = get_column_list(trace.db.eval_info, 'hp_id' )
    
    
    hp_id_map = dict( zip( hp_id_list, range(len(hp_id_list))))
    
    prob_mat = np.zeros( (len(hp_id_list),len(prob_dict)) )
    
    for j, prob_list in enumerate( prob_dict):
        for hp_id, p in prob_list:
            i = hp_id_map[hp_id]
            prob_mat[ i, j ] = p
    
    pp.imshow(prob_mat, origin='lower', aspect='auto', cmap='binary', interpolation='nearest' )
    
    
    col_list = get_column_list(trace.db.argmin, 'i', 'argmin_list', 'chosen_hp_id' )
    
    for i, argmin_list, chosen_hp_id in zip(*col_list):
        idxL = np.array([ hp_id_map[hp_id] for hp_id in argmin_list ])
        
        pp.scatter( [i]*len(idxL), idxL, 2 , color='blue',alpha=0.2  ) # plot all candidates
        pp.scatter( i, hp_id_map[chosen_hp_id], 10,  facecolors='none', edgecolors='r' ) # plot the chosen one
    
    pp.xlabel('iteration')
    pp.ylabel('candidate')
    pp.title('The chosen candidate for each iteration' )

def plot_selected_hp(trace, hp_name_x, hp_name_y):
    
    
    prob_dict, = get_column_list(trace.db.salad, 'prob' )
    
    hp_id_list, hp_dict_list = get_column_list(trace.db.eval_info, 'hp_id', 'hp_' )
    hp_map = { hp_id : hp_dict for hp_id, hp_dict in  zip( hp_id_list, hp_dict_list ) }
    
    point_list= []
    for hp_id, prob in prob_dict[-1]:
        hp_dict = hp_map[hp_id]
        x = hp_dict[hp_name_x ]
        y = hp_dict[hp_name_y ]
        point_list.append((x,y,prob))
    
    x,y,prob = np.array(point_list).T


    pp.scatter(x,y, s=prob/min(prob)*20,c='b', marker='+',linewidths=0.5)
    
    i, amin_list, chosen = get_column_list(trace.db.argmin, 'i', 'argmin_list', 'chosen_hp_id' )
    
    idx = max(i)
    amin_list = amin_list[idx]
    hp = hp_map[chosen[idx]]
    
    pp.scatter(hp[hp_name_x],hp[hp_name_y],s=200,  marker='+',c='g',linewidths=1)


def remap(id_map, ids, vals ):
    val_ = np.zeros(len(vals))
    for id_, val in zip(ids,vals):
        val_[ id_map[id_] ] = val
        
    return val_

def plot_time(trace, axes=None):
    if axes is None:
        axes = pp.gca()
        
    hp_id, choose_time = get_column_list(trace.db.candidates,  'hp_id', 'choose.time' )
    hp_id_analyse, analyse_time = get_column_list(trace.db.analyze,  'hp_id', 'analyse_time' )
    hp_id_duration, duration = get_column_list(trace.db.eval_info, 'hp_id', 'time' )
    
    max_len = min( (len(hp_id), len(hp_id_analyse), len(hp_id_duration) ) )
    choose_time = choose_time[:max_len]
    analyse_time = analyse_time[:max_len]
    duration = duration[:max_len]

    hp_id_map = dict( zip( hp_id, range( len(hp_id)) ) )
    
    analyse_time = remap( hp_id_map, hp_id_analyse, analyse_time )
    duration = remap( hp_id_map, hp_id_duration, duration )

    
#    print uHist(choose_time, 'choose_time' )
#    print uHist(analyse_time, 'analyse_time' )
#    print uHist(duration, 'duration' )
    
    i = np.arange( max_len )
    
    axes.bar(i,analyse_time, color='r', label = 'analyse time' )
    axes.bar(i,choose_time, bottom = analyse_time, color='b', label='choose time' )
    axes.bar(i,duration, bottom = analyse_time+choose_time, color='g', label='learn time' )
    axes.set_xlabel('iteration')
    axes.set_ylabel('time (s)' )
    axes.legend(loc='best')
    axes.set_title('time per iteration for different components')
    

from graalUtil.num import gaussConv
def plot_curve(collection, x_key, *y_key_list):
    
    col_dict = get_column_dict( collection, * ((x_key, ) + y_key_list)  )
    
    x = col_dict.get(x_key)
    x_, = gaussConv(1,x )
    
    color_cycle = pp.gca()._get_lines.color_cycle

    for y_key in y_key_list:

        color= color_cycle.next()
        
        y = col_dict.get(y_key)
        y_, = gaussConv(1,y )
        pp.plot(x,y,'.', color=color,markersize=2)
        
        pp.plot(x_,y_,'-', label=y_key, color=color)

        
    pp.xlabel(x_key)
    pp.legend(loc='best')




def make_map(id):
    return dict( zip( id, range(len(id))))

class HpInfo:
    
    def __init__(self, trace):
        self.hp_space = get_column_list(trace.db.main, 'hp_space' )[0][0]
        self.trace = trace
        self.hp_id_list, unit_list = get_column_list( trace.db.candidates, 'hp_id', 'unit_value')

        self.hp_id_map = make_map(self.hp_id_list)
    
        self.unit_grid = np.array(unit_list)
        
        row_list = []
        for unit_value in self.unit_grid:
            hp_conf = HpConfiguration(self.hp_space, unit_value)
            hp_keys, val = zip( *hp_conf.var_list() )
            row_list.append(val)
        self.col_list = [ np.array(col) for col in zip(*row_list)]
        
        self.hp_keys = hp_keys
        self.hp_key_map = make_map(hp_keys)
        
        for hp_key, col in zip(hp_keys, self.col_list):
            try:
                print uHist(col, hp_key)
            except: pass 
        
    def map_hp_id_list(self, hp_id_list):
        return np.array([self.hp_id_map[hp_id] for hp_id in hp_id_list])
    
    
    def get_col_unit(self, hp_key ):
        idx = self.hp_key_map[hp_key]
        return self.unit_grid[:,idx]


class TimeLine:
    pass

class Plot2d:
    
    def __init__(self, compute_std=False ):
        self.compute_std = compute_std

    def set_stats(self, y_val ):
        self.monotonic_func = Uniformize(y_val)


    def plot(self, x1,x2, y, w, y_mean, y_std):
        
        pp.scatter(x1, x2, c=self.monotonic_func(y), 
            cmap=cmap, s=40*w, linewidths=0.5, norm=NoNorm(), edgecolor='black')
    
        y_sample = np.random.randn( *y_mean.shape) * y_std + y_mean 
        pp.imshow( self.monotonic_func(y_sample) ,origin='lower', 
            extent=(0,1,0,1), aspect='auto',  cmap=cmap, norm=NoNorm())


def clean_hp_name( hp_name ):
    if hp_name.startswith('hp.'):
        hp_name = hp_name[3:]
        
    for char in ['.','[' ]:
        hp_name= hp_name.replace(char,'_')
    hp_name= hp_name.replace(']','')
    
    return hp_name

def extract_metric_set(*collection_list):
    metric_set = set()
    for collection in collection_list:
        for doc in collection.find():
            for metric_path in zip( * flatten_rec_dict(doc) )[0]:
                metric = '.'.join(metric_path)
                metric_set.add(metric)  
    
    return metric_set

def flatten_rec_dict(rec_dict):
    keyL = []
    for key,val in rec_dict.iteritems():
        if isinstance( val, dict ):
            for (path, val_) in flatten_rec_dict(val):
                keyL.append( ((key,) + path, val_) )
        else:
            keyL.append(  ( (key,), val )  )
    return keyL

def flatten_doc( doc ):
    flat_doc = {}
    for key_path, val in flatten_rec_dict(doc):
        flat_doc[ '.'.join( key_path ) ] = val
         
    return flat_doc

def get_collection_structure(collection):
    info_dict = {}
    for doc in collection.find():
        flat_doc = flatten_doc( doc)
        for key, val in flat_doc.iteritems():
            count, type_set = info_dict.get( key, (0,set()) )
            type_set.add( type(val) )
            info_dict[key] = count+1, type_set
    return info_dict

def plot_eval_info( plot, hp_info, y_key, perm = None ):
    
    
    col_dict = get_column_dict( hp_info.trace.db.eval_info, 'hp_id', y_key )
    
    idx = hp_info.map_hp_id_list(col_dict['hp_id'])
    
    if len(idx) == 0:
        print 'no results yet'
        return
    
    gp = MyGP(mcmc_iters=0, noiseless=False)
    
    y = np.array(col_dict[y_key])
    
    print '%s.shape:'%y_key, y.shape
    
    X = hp_info.unit_grid[idx,:]
    
    hp_keys = hp_info.hp_keys
    print hp_keys
    if perm is not None:
        X = X[:,perm]
        hp_keys = [hp_keys[i] for i in perm ]
    
    hp_keys = [ clean_hp_name(hp_key) for hp_key in hp_keys ]
    print hp_keys
    plot.set_info(X, y, hp_keys, y_key, hp_info.hp_space.var_list, gp)


if __name__ == "__main__":
    pp.ion()
    
    trace_folder = path.expandvars('$HOME/trace')
#    trace_name = "trace_28Oct2013-10_50_00_HTCd2l.pkld"   # maxout on mnist
#    trace_name = "trace_30Oct2013-14_28_29_Tqp86d.pkld" # SVR on AirplaneCompanies
    trace_name = "trace_28Oct2013-20_02_27_hUgl_O.pkld" # SVR on CaliforniaHousing
    trace_path = path.join( trace_folder, trace_name )
    
#    trace_path = None
    trace= pkl_trace.TraceDBFile(trace_path)
#    trace.update_trace()
    

    hp_info = HpInfo(trace)

    fig = pp.figure()
#    axes = fig.add_subplot(111)
    plot_time(trace)
    
#    pp.figure()
#    plot_selected_hp(trace, *hpL)
    
#    pp.figure()
#    plot_grid(trace, 'hp_.%s'%hpL[0], 'hp_.%s'%hpL[1], 'tst.risk', 'val.risk')
    
    
    
    fig = pp.figure()
    axes = fig.add_subplot(111)
    plot_stats(trace.db.risk, axes, 'i', 'salad_risk.tst', 'argmin_risk.tst','salad_risk.val', 'argmin_risk.val')
    
    pp.figure()
    plot_curve(trace.db.risk, 'i', 
        'salad_risk.tst', 'argmin_risk.tst', 
        'salad_risk.5%.tst', 'salad_risk.10%.tst',
        'salad_risk.50%.tst','salad_risk.100%.tst' )
##    

    thread = plot_eval_info(hp_info, 'val.risk').configure_traits()

#    pp.show()

   
#    thread.join()
