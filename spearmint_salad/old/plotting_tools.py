# -*- coding: utf-8 -*-
'''
Created on Oct 30, 2013

@author: alexandre
'''


#
#def plot_eval_info(hp_info, y_key_list, plotter=None, hp_keys=None, grid_shape=(100,100,5,5)):
#    
#    if plotter is None: plotter = Plot2d()
#    
#    if hp_keys is None:
#        hp_keys_ = list(hp_info.hp_keys)
#        hp_keys = hp_keys_ + [None]*(4-len(hp_keys_))
#    
#    else :
#        hp_keys_ = [ hp_key for hp_key in hp_keys if hp_key is not None ]        
#    
#    
#    col_dict = get_column_dict( hp_info.trace.db.eval_info, 'hp_id', *y_key_list )
#    y_val_cat = np.hstack( [ col_dict[y_key] for y_key in y_key_list ] )
#    plotter.set_stats( y_val_cat )
#    
#    idx = hp_info.map_hp_id_list(col_dict['hp_id'])
#    
#    ndims = len(hp_keys_)
#    if ndims > 4 or ndims < 2: 
#        raise Exception( 'Can only plot from 2D to 4D.' )
#    
#    grid_shape = [ 1 if hp_key is None else size 
#        for size, hp_key in zip( grid_shape, hp_keys ) ]
#    permute = np.array( [ hp_info.hp_key_map[ hp_key ] for hp_key in hp_keys_ ] )
#    
##    print hp_keys
##    print hp_keys_
##    print hp_info.hp_keys
##    print 'permute', permute
##    print 'grid_shape', grid_shape
#    
#    
#    for y_key in y_key_list:
#        
#        pp.figure()
#        gp = MyGP(mcmc_iters=0, noiseless=False)
#        
#        y = np.array(col_dict[y_key])
#        X = hp_info.unit_grid[idx,:][:,permute]
#        plot_nd = PlotNd(X, y, hp_keys, None, gp, plotter)
#
#        plot_number = 1
#        for x2 in np.linspace(0,1,grid_shape[2]):
#            for x3 in np.linspace( 0,1, grid_shape[3] ):
#                
#                pp.subplot( grid_shape[3], grid_shape[2], plot_number )
#                template = np.array( [np.nan, np.nan, x2, x3] )[:len(hp_keys_)]
#                plot_nd.plot(template)
#                plot_number += 1
#        
#                pp.xlabel(hp_keys[0].replace('hp.','',1))
#                pp.ylabel(hp_keys[1].replace('hp.','',1))
#                title = []
#                for x, hp_key in zip( (x2,x3), hp_keys[2:] ):
#                    if hp_key is not None:
#                        title.append( '%s=%.3f'%(hp_key.replace('hp.','',1), x) )
#                pp.title('\n'.join(title) )
#        
#            
#class DataGrid:
#    
#    def __init__(self, hp_grid, y_val, y_idx, learner, grid_conf ):
#        
#        self.hp_grid = hp_grid
#        self.y_val = y_val 
#        self.y_idx = y_idx
#        self.grid_conf = grid_conf
#        self.estimator = learner.fit( hp_grid.unit_grid[y_idx,:], y_val ) 
#    
#    
#    def plot(self):
#        
#        n_dims = sum( [hp_key is not None for hp_key, _size in self.grid_conf])
#        assert len(self.hp_grid.hp_keys) == n_dims
#    
#    



#class RbfFilter:
#    def __init__(self, X, val_mapper, std=0.15):
#        self.X = X
#        self.std = std
#        
#    def __call__(self, template ):
#        mask = np.logical_not( np.isnan(template) )
#        template = template[mask]
#        if len(template) == 0:
#            return np.ones(self.X.shape[0])
#        X = self.X[:,mask]
#        sq_dist = (X - template.reshape(1,-1))**2
#        return np.exp( -0.5* sq_dist / self.std**2 )
#
#def predict_grid(template, estimator, dims, grid_shape=(100,100), include_std =False):
#
#    
#    x1_grid, x2_grid = np.mgrid[0:1:grid_shape[0]*1j,0:1:grid_shape[1]*1j]
#    X = np.tile(template.reshape(-1,1), np.prod(grid_shape)).T
#    X[:,dims[0]] = x1_grid.flat
#    X[:,dims[1]] = x2_grid.flat
#
#    if include_std:
#        y_mean, y_var = estimator.predict_proba( X )
#    else:
#        y_mean = estimator.predict( X )
#        y_var = np.zeros(y_mean.shape)
#        
#    y_mean.shape = grid_shape
#    y_var.shape = grid_shape
#    y_std = np.sqrt(y_var)
#
#    return y_mean, y_std
#
#class PlotNd:
#    
#    def __init__(self, X, y, dim_names, val_mapper, learner, plotter=None):
#        self.estimator = learner.fit(X,y)
#        self.X = X
#        self.y = y
#        self.dim_names = dim_names
#        self.val_mapper = val_mapper
#        self.plotter = plotter if plotter is not None else Plot2d()
#        self.filter = RbfFilter(X, val_mapper)
#        
#    def plot( self, template ):
#        assert np.isnan(template).sum() == 2 
#        dims = np.arange(template.size)[ np.isnan(template) ]
#        print template, dims
#        
#        w = self.filter(template)
#        y_mean, y_std = predict_grid( template, self.estimator,dims, include_std= self.plotter.compute_std )
#        self.plotter.plot(self.X[:,dims[0]], self.X[:,dims[1]], self.y, w, y_mean, y_std )
#  







def plot_with_fit_2d(x,y, z, estimator, grid_shape=(100,100), monotonic_func=None, include_noise=True, template=None ):
    
    monotonic_func  = lambda x : x if monotonic_func is None else monotonic_func # idendity
    

    
    pp.scatter(x, y, c= monotonic_func(z) ,cmap=cmap, s=40, linewidths=0.5, norm=NoNorm(), edgecolor='black')
#    pp.scatter(x, y, c= monotonic_func(z) ,cmap='Blues', s=30, linewidths=0.2)

    x_grid, y_grid = np.meshgrid(np.linspace(0,1,grid_shape[0]), np.linspace(0,1,grid_shape[1]))
    
    data_x = np.vstack( (x_grid.flat, y_grid.flat) ).T 
    if template is not None:
        data_x = template(data_x)
    if include_noise:
        z_mean, z_var = estimator.predict_proba(data_x)
    else:
        z_mean = estimator.predict( data_x )
        z_var = np.zeros(z_mean.shape)
        
    z_mean.shape = grid_shape
    z_var.shape = grid_shape
    
    z_sample = np.random.randn( *z_mean.shape) * np.sqrt(z_var) + z_mean 
    pp.imshow( monotonic_func(z_sample) ,origin='lower', extent=(0,1,0,1), aspect='auto',  cmap=cmap, norm=NoNorm())

from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm

def plot_with_fit_3d( x,y, z, learner, grid_shape=(20,20) ):
    
    
    # predict grid
    learner.fit( np.vstack((x,y)).T, z )
    print 'learner state : '
    print learner
    print 
    
    ax = pp.gcf().gca(projection='3d')
    ax.scatter(x, y, z )
#    pp.scatter(x, y, c= monotonic_func(z) ,cmap='Blues', s=30, linewidths=0.2)

    x_grid, y_grid = np.meshgrid(np.linspace(0,1,grid_shape[0]), np.linspace(0,1,grid_shape[1]))
    z_mean, z_var = learner.predict_proba( np.vstack( (x_grid.flat, y_grid.flat) ).T ) 
    z_mean.shape = grid_shape
    z_var.shape = grid_shape
    z_std = np.sqrt(z_var)
    
    ax.plot_wireframe( x_grid, y_grid, z_mean+z_std, alpha=0.2 )
    ax.plot_wireframe( x_grid, y_grid, z_mean-z_std, color='green', alpha=0.2 )
#    ax.plot_surface( x_grid, y_grid, z_mean, rstride=1, cstride=1, 
#        linewidth=0, antialiased=True, alpha=0.5 )
    
#    for i in range(grid_shape[0]):
#        for j in range(grid_shape[1]):
#            x_ = x_grid[i,j]
#            y_ = y_grid[i,j]
#            z_ = z_mean[i,j]
#            std_ = z_std[i,j]
#            ax.plot([x_,x_], [y_,y_], [z_+std_, z_-std_], marker="_", color='blue')
    


def plot_grid( trace, x_key, y_key, *z_key_list ):
    
    col_dict = get_column_dict( trace.db.eval_info, * ((x_key, y_key) + z_key_list)  )
    x_val = np.array(col_dict[x_key])
    y_val = np.array(col_dict[y_key])


    z_val_cat = []
    for i,z_key in enumerate(z_key_list):
        z_val_cat += col_dict[z_key] 
    monotonic_func = Uniformize(np.array(z_val_cat))
    
    
    for i,z_key in enumerate(z_key_list):
        
        z_val = np.array(col_dict[z_key])
        print uHist( z_val, z_key )

        gp = MyGP(mcmc_iters=0, noiseless=False)
        
        pp.subplot(2,len(z_key_list),2*i+1,  projection='3d')
        plot_with_fit_3d( x_val, y_val, z_val, gp )
        pp.title(z_key)
        pp.xlabel(x_key)
        pp.ylabel(y_key)
        
        pp.subplot(2,len(z_key_list),2*i+2)
        plot_with_fit_2d(x_val, y_val, z_val, gp, monotonic_func=monotonic_func )
        pp.title(z_key)
        pp.xlabel(x_key)
        pp.ylabel(y_key)
        
