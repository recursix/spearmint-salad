from mayavi import mlab
from traits.api import Range, on_trait_change
from traits.api import HasTraits, Instance
from traitsui.api import View, Item, Group
from mayavi.core.ui.api import SceneEditor, MlabSceneModel, MayaviScene
import numpy as np
from threading import Thread

class RbfFilter:
    def __init__(self, X, val_mapper, std=0.05):
        self.X = X
        self.std = std
        
    def __call__(self, val ):
        if len(val) == 0:
            return np.ones(self.X.shape[0])
        
        sq_dist = (self.X - val.reshape(1,-1))**2
        sq_dist.reshape(-1)
        return np.exp( -0.5* sq_dist / self.std**2 ) + 0.1


def predict_grid(constants, estimator,  grid_shape=(50,50), include_std =False):
    constants = np.asarray(constants)

    
    x1_grid, x2_grid = np.mgrid[0:1:grid_shape[0]*1j,0:1:grid_shape[1]*1j]
    
    X = np.hstack((
        x1_grid.reshape(-1,1), 
        x2_grid.reshape(-1,1),
        np.tile(constants.reshape(-1,1), np.prod(grid_shape)).T
        ))
    
    
    if include_std:
        y_mean, y_var = estimator.predict_proba( X )
    else:
        y_mean = estimator.predict( X )
        y_var = np.zeros(y_mean.shape)
        
    y_mean.shape = grid_shape
    y_var.shape = grid_shape
    y_std = np.sqrt(y_var)

    return x1_grid, x2_grid, y_mean, y_std


def predict_grid_3d(constants, estimator,  grid_shape=(50,50,50), include_std =False):
    constants = np.asarray(constants)

    
    
    x1_grid, x2_grid, x3_grid = np.mgrid[
        0:1:grid_shape[0]*1j,
        0:1:grid_shape[1]*1j,
        0:1:grid_shape[2]*1j ]
    
    X = np.hstack((
        x1_grid.reshape(-1,1), 
        x2_grid.reshape(-1,1),
        x3_grid.reshape(-1,1),
        np.tile(constants.reshape(-1,1), np.prod(grid_shape)).T
        ))
    
    
    if include_std:
        y_mean, y_var = estimator.predict_proba( X )
        y_var.shape = grid_shape
    else:
        y_mean = estimator.predict( X )
        y_var = 0.
        
    y_mean.shape = grid_shape
    y_std = np.sqrt(y_var)

    return x1_grid, x2_grid, x3_grid, y_mean, y_std



from graalUtil.num import uHist

def Plot2d_with_params( X, y, dim_names, val_mapper, learner, title, clip = 0.96, include_std=False):
    
    estimator = learner.fit(X,y)
    filter_ = RbfFilter(X[:,2:],None)
    x1_name, x2_name = dim_names[:2]
    param_names = list( dim_names[2:] )
    
    clip_value= np.percentile( y, clip*100 )
    z_scale = 1./clip_value
    print 'clip_value', clip_value
    print uHist(y,'y')
    print X.shape, y.shape
    print x1_name, x2_name, param_names
    
    class _Plot2d_with_params(HasTraits):
        
        def __init__(self):
            HasTraits.__init__(self)
            for name in param_names:
                self.add_trait(name, Range(0., 1., 0.5 ))
    
        scene = Instance(MlabSceneModel, ())
        
        def get_data(self):
            val_list = [ getattr(self,param_name) for param_name in param_names ]
            x1 = X[:,0]
            x2 = X[:,1]
            
            w = filter_(np.array(val_list)).reshape(-1)
            
            x1_grid, x2_grid, y_mean, y_std =predict_grid(val_list, estimator, include_std=include_std)

            y_ = y.copy()
            y_[ y_>clip_value ] =clip_value
            y_std[ y_mean >= clip_value ] = 0
            y_mean[ y_mean >= clip_value ] = clip_value
            
            y_mean *= z_scale
            y_ *= z_scale
            y_std *= z_scale
            
#            print uHist(y_std,'y_std')
            
            return x1,x2,y_,w, x1_grid, x2_grid, y_mean, y_std
        
        def pick(self, picker):
            pos = picker.mapper_position
            self.click.mlab_source.reset(x=pos[0],y=pos[1],z=pos[2])

        
        @on_trait_change('scene.activated')
        def init_view(self):
            x1,x2,y,w,x1_grid, x2_grid, y_mean, y_std = self.get_data()
            
            self.scene.disable_render = True
            
            
            self.points = self.scene.mlab.points3d(x1,x2,y,w, 
                scale_factor=.02, opacity=1, color=(0.5,0.5,0.9 ))
            self.points.glyph.glyph.clamping = False

            self.surf  = self.scene.mlab.surf(x1_grid, x2_grid, y_mean,      opacity=0.8 )
            
            if include_std:
                self.surf2 = self.scene.mlab.surf(x1_grid, x2_grid, y_mean-y_std,opacity=0.3)
                self.surf3 = self.scene.mlab.surf(x1_grid, x2_grid, y_mean+y_std,opacity=0.3)

            self.scene.mlab.axes(xlabel=x1_name, ylabel=x2_name, zlabel='risk',ranges=(0,1,0,1,0,1./z_scale))
            
            self.click = self.scene.mlab.points3d(0.5,0.5,0.5,scale_factor=.02,  color=(0.9,0.5,0.5 ))
            self.click.glyph.glyph.clamping = False
            fig = mlab.gcf()
            fig.on_mouse_pick(self.pick)
            
            self.scene.mlab.title(title, size=0.3,height=0.95)

            self.scene.disable_render = False

            
        @on_trait_change(','.join(param_names))
        def update_plot(self):
            self.scene.disable_render = True
            
            x1,x2,y,w,x1_grid, x2_grid, y_mean,y_std = self.get_data()

            self.surf.mlab_source.reset( scalars=y_mean)
            if include_std:
                self.surf2.mlab_source.reset(scalars=y_mean-y_std)
                self.surf3.mlab_source.reset(scalars=y_mean+y_std)

            self.points.mlab_source.reset(x=x1,y=x2,z=y,scalars=w)
            
            self.scene.disable_render = False
    
        # The layout of the dialog created
        view = View(
            Item('scene', editor=SceneEditor(scene_class=MayaviScene),
                height=250, width=300, show_label=False),
            Group( ['_'] + param_names ),
            resizable=True)

    return _Plot2d_with_params()


#class ValMapper:
#    
#    def __init__(self, hp_space):
#        self.hp_space = hp_space
##        self.key_list = hp_space.get_hp_keys()
#    
#    def __call__(self, point ):
#        return hp.HpConfiguration(self.hp_space, point).var_list()



class Plot3d_with_params(HasTraits):
    
    scene = Instance(MlabSceneModel, ())

    
    def set_info(self, X, y, dim_names, y_name, val_mapper, learner, title=None, include_std=False):
            
        assert X.shape[1] == len(dim_names)
        
        self.original_dim = len(dim_names)
        if len(dim_names) == 2:
            dim_names.append( 'None' )
            X = np.hstack( (X, np.zeros( (X.shape[0],1) ) ) )

        

        self.learner = learner
        self.X = X
        self.set_y(y, y_name )
        self.val_mapper = val_mapper
        self.filter_ = RbfFilter(X[:,3:],None)
        self.x1_name, self.x2_name, self.x3_name = dim_names[:3]
        self.param_names = list( dim_names[3:] )
        self.title = title
        self.include_std = include_std
    
        
#        print uHist(y,'y')
#        print X.shape, y.shape
#        print self.x1_name, self.x2_name, self.x3_name, self.param_names
    
        self.grid_cache = {}
#        print hasattr(self, 'grid_cache')

        self.init_view()
    
    def set_y(self,y,y_name):
        self.y_name = y_name
        self.y = y
        self.estimator = self.learner.fit(self.X,y)
        
    
    def compute_data(self):
        val_list = [ getattr(self,param_name) for param_name in self.param_names ]
        self.x1 = self.X[:,0]
        self.x2 = self.X[:,1]
                    
        self.x3 = self.X[:,2]
        
        self.w = self.filter_(np.array(val_list)).reshape(-1)
        val_list = tuple([ getattr(self,param_name) for param_name in self.param_names ])
        
        print 'has grid_cache:', hasattr(self, 'grid_cache')

        if val_list not in self.grid_cache:
            print 'computing grid'
            self.grid_cache[val_list] = predict_grid_3d(val_list, self.estimator, (50,50,50), False)
            
        self.x1_grid, self.x2_grid, self.x3_grid, self.y_mean, self.y_std = self.grid_cache[val_list]

    
    def _add_selection(self):
#            self.selection= mlab.points3d(0.5,0.5,0.5, 0.05, opacity = 0.5 )
#            self.selection.glyph.glyph.clamping = False

        self.selection = mlab.outline(line_width=3)
        self.selection.outline_mode = 'cornered'
        self.selection.bounds = (0.04, 0.06)*3

        self.glyph_points = self.points.glyph.glyph_source.glyph_source.output.points.to_array()
        
        picker = mlab.gcf().on_mouse_pick(self.picker_callback)
#
#            # Decrease the tolerance, so that we can more easily select a precise
#            # point.
        picker.tolerance = 0.01
        
    def picker_callback(self,picker):
        """ Picker callback: this get called when on pick events.
        """
        if picker.actor in self.points.actor.actors:
            # Find which data point corresponds to the point picked:
            # we have to account for the fact that each data point is
            # represented by a glyph with several points
            point_id = picker.point_id/self.glyph_points.shape[0]
            print 'id', picker.point_id, self.glyph_points.shape
            # If the no points have been selected, we have '-1'
            if point_id != -1:
                # Retrieve the coordinnates coorresponding to that data point
                x, y, z = self.x1[point_id], self.x2[point_id], self.x3[point_id]
                print point_id, x,y,z
                # Move the outline to the data point.
                w = 0.01
#                    self.selection.mlab_source.reset( x = x, y = y, z = z )
                self.selection.bounds = (x-w, x+w, y-w, y+w, z-w, z+w)
                
    def _add_points(self):
        if not hasattr(self,'X'):
            return
        
#            
#            self.points = mlab.points3d(
#                self.x1,self.x2,self.x3, self.y, mode='sphere', scale_factor=.01)
        
        self.points = mlab.quiver3d(
            self.x1,self.x2,self.x3, self.w, self.w, self.w, 
            scalars=self.y, mode='sphere', scale_factor=.01)
        
        self.points.glyph.color_mode = 'color_by_scalar'
        self.points.glyph.glyph_source.glyph_source.center = [0, 0, 0]
    
        lut = self.points.module_manager.scalar_lut_manager
        lut.reverse_lut = True
        lut.show_legend = True
        lut.data_name = self.y_name
        lut.scalar_bar_representation.position  = [ 0.01, 0.05]
        lut.scalar_bar_representation.position2 = [ 0.05, 0.3]
    
        self.lut = lut
    
    
    
    def _point_to_str(self, point):
        point_ = point[ :self.original_dim ]# bring back to the right dimensionality 
        hp_str_list = []
        for name, val in self.val_mapper(point_):
            if isinstance(val,float): val_str = "%.3g"%(val)
            else:                     val_str = str(val)
            hp_str_list.append( '%s: %s'%(name,val_str) )
        
        y_mean, y_std = self.estimator.predict_proba( np.array([point]).reshape(1,-1) )
        hp_str_list.append( '%s: %.3g'%(self.y_name, y_mean ) )
        hp_str_list.append( ' +-: %.3g'%( y_std ) )
        return '\n'.join(hp_str_list)
    
    def _plane_cb(self,obj, evt):
#            print 'plane_orientation: ', getattr(obj,'plane_orientation', None)


        point = np.asarray(obj.GetCurrentCursorPosition())
        
        point[obj.GetPlaneOrientation()] = obj.GetSlicePosition()

        self.scene.disable_render = True
        self.hp_val_text.text = self._point_to_str(point)
        self.hp_val_text.actor.visibility = True
        self.scene.disable_render = False

    def _plane_cb_end(self,*argL):
        self.hp_val_text.actor.visibility = False


    
    def _make_cut_plane(self, plane_orientation='x_axes'):
        plane = mlab.pipeline.image_plane_widget(
            self.field, plane_orientation=plane_orientation )
        
        plane.ipw.add_observer('InteractionEvent', self._plane_cb)
        plane.ipw.add_observer('EndInteractionEvent', self._plane_cb_end)

        plane.ipw.right_button_action = 0
        plane.ipw.use_continuous_cursor = True
        plane.ipw.margin_size_x = 0
        plane.ipw.margin_size_y = 0
        
        plane.module_manager.scalar_lut_manager.lut.table  = self.lut.lut.table
        return plane
    
    def _add_field(self ):
        self.field = mlab.pipeline.scalar_field(
            self.x1_grid, self.x2_grid, self.x3_grid, self.y_mean)
#            mlab.pipeline.volume(field,vmin=0, vmax=0.8)
        
        self.x_plane = self._make_cut_plane('x_axes')
        self.y_plane = self._make_cut_plane('y_axes')
        
        contours = [np.percentile(self.y, p) for p in [10,20] ]
        mlab.pipeline.iso_surface(self.field, contours=contours, opacity=0.2)


    def show_in_thread(self):
        thread= Thread( target=self.configure_traits)
        thread.start()
        return thread

            
#     @on_trait_change('scene.activated')
    def init_view(self):
        
        mlab.clf()
        self.compute_data()
        
        self.scene.disable_render = True

        
        self._add_points()
#            self._add_selection()
        self._add_field()
        
        
        self.hp_val_text = self.scene.mlab.text(0.01,0.98,'hp values',width=0.12)
        self.hp_val_text.actor.visibility = False
        self.hp_val_text.actor.text_scale_mode = 'viewport'
        self.hp_val_text.property.vertical_justification = 'top'
        
        axes = self.scene.mlab.axes(xlabel=self.x1_name, ylabel=self.x2_name, zlabel=self.x3_name )
        axes.label_text_property.opacity = 0
        axes.axes.corner_offset = 0.1
#            axes2 = self.scene.mlab.axes(xlabel=x1_name, ylabel=x2_name, zlabel=x3_name )
#            axes.axes.font_factor = 0.8
#            self.points.glyph.glyph.clamping = False

        

        if self.title is not None:
            self.scene.mlab.title(self.title, size=0.3,height=0.95)

        self.scene.disable_render = False

        

    def default_traits_view(self):
        return View(
            Item('scene', editor=SceneEditor(scene_class=MayaviScene), show_label=False),
#             Group( ['_'] + self.param_names ),
            resizable=True)






#from os import path
#if __name__ == "__main__":
#    trace_folder = path.expandvars('$HOME/trace')
##    trace_name = "trace_28Oct2013-10_50_00_HTCd2l.pkld"   # maxout on mnist
##    trace_name = "trace_30Oct2013-14_28_29_Tqp86d.pkld" # SVR on AirplaneCompanies
#    trace_name = "trace_28Oct2013-20_02_27_hUgl_O.pkld" # SVR on CaliforniaHousing
#    trace_path = path.join( trace_folder, trace_name )
#    
#    trace_path = None
#    trace= pkl_trace.TraceDBFile(trace_path)
#    trace.update_trace()
#    
#    hp_info = HpInfo(trace)
#    plot = plot_eval_info(hp_info, 'val.risk', perm = [0,1,2])
#    plot.configure_traits()
#    