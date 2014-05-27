from mayavi import mlab
from traits.api import Range, on_trait_change
from traits.api import HasTraits, Instance, List, Enum, Str, Bool
from traitsui.api import View, Item, Group, HGroup, Spring, CheckListEditor, Label
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



class Plot3d_with_params(HasTraits):
    
    scene = Instance(MlabSceneModel, ())
    y_keys = List(Str)
    y_selected = Enum(values="y_keys",
        desc = 'which data is used to assign to color of the points.',
        label = 'y')
    
    cuts = List(  ['z'],
        desc = """which cut plane to show. 
The cut planes reveal the predictions of the Gaussian process for unexplored points. \
Move these cut planes with the mouse to explore different regions and right click to obtain the \
values of the hyperparameter configuration for a given point.""",
        editor = CheckListEditor(
            values = [ 'x', 'y', 'z'  ],
            cols   = 3 ) )
    
#     lut_scale = Enum(  "log10", "linear",
#         desc="the scale of the colormap.")
    
    lut_mode = Enum('autumn', 'blue-red', 'bone', desc="the colormap to use.")
    
    lut_range = Range(0.,1., value = 1., desc="the ratio of points to use for the range of the color map.")
#     
#     reverse_lut = Bool(True, )
    
    reverse_lut = List( ['Reverse colormap'],
         desc="if the colormap is reversed.",
         editor = CheckListEditor( values = [ 'Reverse colormap' ]) )
    
    level_surface = Range(0.,1.,value=0.,
        desc="""a surface of equal values over the prediction of the Gaussian process. The selected value indicates the ratio of points included inside the surface.""")
    
    
    min_is_best = List( ['min is best'],
         editor = CheckListEditor( values = [ 'min is best' ]) )
    
    def default_traits_view(self):
        return View(
            HGroup(
                Group( 
                    Label('Data' ),         
                    Item('y_selected',show_label=False),              
                    Item('_'), Label('Cut planes'),    
                    Item('cuts', style='custom', show_label=False ) , 
                    Item('_'), Label('ISO Surfaces'),  
                    Item('level_surface', show_label=False),          
                    Item('_'), Label('Colormap'), 
#                     Item('lut_scale', show_label=False),
                    Item('lut_mode', show_label=False),
                    Item('reverse_lut', style='custom', show_label=False),
                    Item('min_is_best', style='custom', show_label=False),
                    Label('zoom color'),
                    Item('lut_range', show_label=False),
                    Item('_'),
                    Spring() ),
                Item('scene', editor=SceneEditor(scene_class=MayaviScene), show_label=False),
            ),
#             Group( ['_'] + self.param_names ),
            resizable=True)
    
    def set_info(self, X, y_dict, selected_y_key, dim_names, val_mapper, learner, title=None, include_std=False):
            
        assert X.shape[1] == len(dim_names)
        
        self.original_dim = len(dim_names)
        if len(dim_names) == 2:
            dim_names.append( 'None' )
            X = np.hstack( (X, np.zeros( (X.shape[0],1) ) ) )

        mlab.clf()

        self.learner = learner
        self.X = X
        self.y_dict =  y_dict
        self.y_keys = y_dict.keys()
        
        self.val_mapper = val_mapper
        self.filter_ = RbfFilter(X[:,3:],None)
        self.x1_name, self.x2_name, self.x3_name = dim_names[:3]
        self.param_names = list( dim_names[3:] )
        self.title = title
        self.include_std = include_std

        self.grid_cache = {}
        self.lut_manager_list = []

        self.initialized= False
        if self.y_selected == selected_y_key:
            self.set_y()
        else:
            self.y_selected = selected_y_key
        
        self.initialized= True


    @on_trait_change('lut_range')
    def set_lut_range(self):
        if len(self.min_is_best) > 0:
            percentile = np.percentile(self.y, self.lut_range*100)
            data_range = np.array([np.min(self.y), percentile])
        else:
            percentile = np.percentile(self.y, (1-self.lut_range)*100)
            data_range = np.array([percentile, np.max(self.y)])
        
        
        for lut_manager in self.lut_manager_list:
            lut_manager.data_range = data_range
        

    @on_trait_change('reverse_lut')
    def _reverse_lut(self):
        for lut_manager in self.lut_manager_list:
            lut_manager.reverse_lut =  len(self.reverse_lut) > 0

    @on_trait_change('lut_mode')
    def set_lut_mode(self):
        for lut_manager in self.lut_manager_list:
            lut_manager.lut_mode = self.lut_mode
    
#     @on_trait_change('lut_scale')
#     def set_lut_scale(self):
#         for lut_manager in self.lut_manager_list:
#             lut_manager.lut.scale = self.lut_scale
#             
#         self.scene.mlab.draw()

    
    def update_lut(self):
#         self.scene.disable_render = True
        self.set_lut_range()
        self._reverse_lut()
        self.set_lut_mode()
#         self.set_lut_scale()
#         self.scene.disable_render = False

        
    @on_trait_change('y_selected')
    def set_y(self):
        print 'set y', 'initialize = %s'%self.initialized
        self.y = self.y_dict[self.y_selected]
        self.scene.disable_render = True
        

        self.init_view()
            
                    
        self.estimator = self.learner.fit(self.X,self.y)

        self._add_field((50,50,50))


        self.update_lut()
        self._set_contour()
        self.show_cuts()
        
        self.scene.disable_render = False


    def compute_field(self, shape=(10,10,10)):
        
        val_list = self._get_val_list()
        key = (val_list, self.y_selected, shape)
         
        print 'has grid_cache:', hasattr(self, 'grid_cache')
 
        if key not in self.grid_cache:
            print 'computing grid'
            self.grid_cache[key] = predict_grid_3d(val_list, self.estimator, shape, False)
             
        return self.grid_cache[key]


    def _remove_field( self ):
        # remove everything related to the field and recreate (there should be a more efficient way of proceeding)
        for attr in ['field', 'x_plane', 'y_plane', 'z_plane', 'iso_surface' ]:

            obj = getattr( self, attr, None)
            if obj is not None:
                obj.remove()
                delattr(self,attr)
                
                
        if len(self.lut_manager_list) > 1:
            self.lut_manager_list = self.lut_manager_list[:1]
                
    def _add_field(self, shape ):
        x1_grid, x2_grid, x3_grid, y_mean, _y_std = self.compute_field(shape)

        if self.initialized:
            self.field.mlab_source.scalars = y_mean
            return



        
        self.field = mlab.pipeline.scalar_field( x1_grid, x2_grid, x3_grid, y_mean)

        self.iso_surface = mlab.pipeline.iso_surface(self.field, contours=[], opacity=0.3)
        self.lut_manager_list.append( self.iso_surface.module_manager.scalar_lut_manager )


        for axis in 'xyz':
            cut_plane = self._make_cut_plane(  '%s_axes'%axis)
            setattr(self,'%s_plane'%axis, cut_plane  )
#             cut_plane.ipw.enabled = '%s'%axis in self.cuts
        
        self.lut_manager_list.append( self.x_plane.module_manager.scalar_lut_manager )
        
        



    


    
    def _get_val_list(self):
        return tuple([ getattr(self,param_name) for param_name in self.param_names ])

    def _get_weights(self):
        return self.filter_(np.array(self._get_val_list())).reshape(-1)
    
    
    def _init_lut_manager(self):
        
        lut_manager = self.points.module_manager.scalar_lut_manager
        lut_manager.lut_mode = self.lut_mode
        
#         lut_manager.reverse_lut = True
        lut_manager.show_legend = True
        lut_manager.data_name = ""

        lut_manager.scalar_bar_representation.position  = [ 0.01, 0.03]
        lut_manager.scalar_bar_representation.position2 = [ 0.1, 0.4]

        self.lut_manager_list.append( lut_manager )
    
    def _add_points(self):
        if not hasattr(self,'X'):
            return

        
        x1, x2, x3 = self.X[:,[0,1,2]].T
        w = self._get_weights()
        
        self.points = mlab.quiver3d(
            x1,x2,x3, w, w, w, 
            scalars=self.y, mode='sphere', scale_factor=.01 )
        
        self.points.glyph.color_mode = 'color_by_scalar'
        self.points.glyph.glyph_source.glyph_source.center = [0, 0, 0]

    
        self._init_lut_manager()

    
    def _point_to_str(self, point):
        point_ = point[ :self.original_dim ]# bring back to the right dimensionality 
        hp_str_list = []
        for name, val in self.val_mapper(point_):
            if isinstance(val,float): val_str = "%.3g"%(val)
            else:                     val_str = str(val)
            hp_str_list.append( '%s: %s'%(name,val_str) )
        
        y_mean, y_std = self.estimator.predict_proba( np.array([point]).reshape(1,-1) )
        hp_str_list.append( '%s: %.3g'%(self.y_selected, y_mean ) )
        hp_str_list.append( ' +-: %.3g'%( y_std ) )
        return '\n'.join(hp_str_list)
    
    def _plane_cb(self,obj, evt):

        point = np.asarray(obj.GetCurrentCursorPosition())
        
        point[obj.GetPlaneOrientation()] = obj.GetSlicePosition()

#         self.scene.disable_render = True
        self.hp_val_text.text = self._point_to_str(point)
        self.hp_val_text.actor.visibility = True
#         self.scene.disable_render = False

    def _plane_cb_end(self,*argL):
        self.hp_val_text.actor.visibility = False


    
    @on_trait_change('cuts')
    def show_cuts(self):
        self.set_lut_mode()
        for axis in 'xyz':
            plane = getattr(self, '%s_plane'%axis)
            plane.ipw.enabled = '%s'%axis in self.cuts
#             plane.module_manager.scalar_lut_manager.lut  = self.lut_manager.lut
        self.set_lut_mode()

    def _make_cut_plane(self, plane_orientation='x_axes'):
        self.scene.disable_render = True
        mlab.gcf().scene.disable_render= True
        plane = mlab.pipeline.image_plane_widget(
            self.field,
            plane_orientation=plane_orientation)
        
        self.scene.disable_render = True

        plane.ipw.enabled = False
        plane.ipw.add_observer('InteractionEvent', self._plane_cb)
        plane.ipw.add_observer('EndInteractionEvent', self._plane_cb_end)
        plane.ipw.display_text = False
        plane.ipw.right_button_action = 0
        
        plane.ipw.use_continuous_cursor = True
        plane.ipw.margin_size_x = 0
        plane.ipw.margin_size_y = 0
        
#         plane.module_manager.scalar_lut_manager.lut  = self.lut_manager.lut
        
        
        return plane
    
    @on_trait_change('level_surface')
    def _set_contour(self):
        if self.level_surface == 0.:
            self.iso_surface.contour.contours = []
        else:
            self.iso_surface.contour.contours = [np.percentile(self.y, self.level_surface*100)] 
    

            
#     @on_trait_change('scene.activated')
    def init_view(self):
        
        if self.initialized:
            self.points.mlab_source.scalars = self.y
            return
    


        self._add_points()
#         self.compute_data()
#         self._add_field()
        
        self.hp_val_text = self.scene.mlab.text(0.01,0.98,'hp values',width=0.12)
        self.hp_val_text.actor.visibility = False
        self.hp_val_text.actor.text_scale_mode = 'viewport'
        self.hp_val_text.property.vertical_justification = 'top'
        
        axes = self.scene.mlab.axes(xlabel=self.x1_name, ylabel=self.x2_name, zlabel=self.x3_name )
        axes.label_text_property.opacity = 0
        axes.axes.corner_offset = 0.1
    
        if self.title is not None:
            self.scene.mlab.title(self.title, size=0.3,height=0.95)

        


# 
#     def _add_selection(self):
# #            self.selection= mlab.points3d(0.5,0.5,0.5, 0.05, opacity = 0.5 )
# #            self.selection.glyph.glyph.clamping = False
# 
#         self.selection = mlab.outline(line_width=3)
#         self.selection.outline_mode = 'cornered'
#         self.selection.bounds = (0.04, 0.06)*3
# 
#         self.glyph_points = self.points.glyph.glyph_source.glyph_source.output.points.to_array()
#         
#         picker = mlab.gcf().on_mouse_pick(self.picker_callback)
# #
# #            # Decrease the tolerance, so that we can more easily select a precise
# #            # point.
#         picker.tolerance = 0.01
#         
#     def picker_callback(self,picker):
#         """ Picker callback: this get called when on pick events.
#         """
#         if picker.actor in self.points.actor.actors:
#             # Find which data point corresponds to the point picked:
#             # we have to account for the fact that each data point is
#             # represented by a glyph with several points
#             point_id = picker.point_id/self.glyph_points.shape[0]
#             print 'id', picker.point_id, self.glyph_points.shape
#             # If the no points have been selected, we have '-1'
#             if point_id != -1:
#                 # Retrieve the coordinnates coorresponding to that data point
#                 x, y, z = self.x1[point_id], self.x2[point_id], self.x3[point_id]
#                 print point_id, x,y,z
#                 # Move the outline to the data point.
#                 w = 0.01
# #                    self.selection.mlab_source.reset( x = x, y = y, z = z )
#                 self.selection.bounds = (x-w, x+w, y-w, y+w, z-w, z+w)

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

if __name__ == "__main__":
    import viz