
import matplotlib
# We want matplotlib to use a wxPython backend
matplotlib.use('WXAgg', warn=False)
import wx


from matplotlib.backends.backend_wxagg import FigureCanvasWxAgg as FigureCanvas
from matplotlib.figure import Figure
from matplotlib.backends.backend_wx import NavigationToolbar2Wx

from traits.api import Instance, HasTraits, Range,List, Bool, on_trait_change, Button
from traitsui.api import CheckListEditor,Group, View, Item
from graalUtil.num import gaussConv
from traitsui.wx.editor import Editor
from traitsui.wx.basic_editor_factory import BasicEditorFactory

import numpy as np

class _MPLFigureEditor(Editor):

    scrollable  = True

    def init(self, parent):
        self.control = self._create_canvas(parent)
        self.set_tooltip()

    def update_editor(self):
        pass

    def _create_canvas(self, parent):
        """ Create the MPL canvas. """
        # The panel lets us add additional controls.
        panel = wx.Panel(parent, -1, style=wx.CLIP_CHILDREN)
        sizer = wx.BoxSizer(wx.VERTICAL)
        panel.SetSizer(sizer)
        # matplotlib commands to create a canvas
        mpl_control = FigureCanvas(panel, -1, self.value)
        sizer.Add(mpl_control, 1, wx.LEFT | wx.TOP | wx.GROW)
        toolbar = NavigationToolbar2Wx(mpl_control)
        sizer.Add(toolbar, 0, wx.EXPAND)
        self.value.canvas.SetMinSize((10,10))
        return panel

class MPLFigureEditor(BasicEditorFactory):
    klass = _MPLFigureEditor


#from fnmatch import fnmatch
#class PatternFilter:
#    
#    def __init__(self, *pattern_list ):
#        self.pattern_list = pattern_list
#    
#    def __call__(self, val ):
#        return any( [fnmatch(val, pattern) for pattern in self.pattern_list] )
#            
# 

color_converter = matplotlib.colors.ColorConverter()
class PlotSelector(HasTraits):
    
    smooth = Range(0.,5., 0.)

    name_list = List()
    name_selected = List()
    figure = Instance(Figure, ())
    legend = Bool(False)
    paused = Bool(False)

    def __init__(self,*args, **arg_dict):
        super(PlotSelector,self).__init__(*args, **arg_dict)
        self._plot_dict = {}
         
    def _legend_changed(self):
        if self.legend:
            self.axes.legend(loc='best')
        else:
            self.axes.legend().set_visible(False)
        self.figure.canvas.draw()
#        print 'legend'
    
    def plot(self, x, y, **arg_dict):
        name = arg_dict.get('label','plot_%d'%len(self._plot_dict))
        arg_dict['label'] = name
        self._plot_dict[name] = (x,y, arg_dict)
        self.name_list.append(name)
        self.name_selected.append(name)
        
    def clear(self):
        self._plot_dict = {}
        self.name_list = []
        self.name_selected = []

    @on_trait_change('name_selected,smooth,paused')
    def draw(self):
        if self.paused :
            return 


        self.figure.clf()
        self.axes = self.figure.add_subplot(111)

        for name in self.name_selected:
            x,y,arg_dict = self._plot_dict[name]
            if x is None: x = np.arange(y)
            if self.smooth > 0: 
                x,y = gaussConv(self.smooth, x, y )
                
            res= self.axes.plot(x,y,**arg_dict)
            color= color_converter.to_rgb( res[0].get_color() )
            print color

        if self.figure.canvas is not None:
            self.figure.canvas.draw()
    
    def default_traits_view(self):
        editor = CheckListEditor( name = 'name_list', cols = 1)
        
        self.view = View(
            Group(
                Item('figure', editor=MPLFigureEditor(), show_label=False),
                Group(
                    Item('smooth', style='custom', show_label=False),
                    Item('legend', style='simple', show_label=False ),
                    Item('name_selected', editor=editor, style='custom', show_label=False),
                    ),
                orientation='horizontal',
                scrollable=True,
            ),
            resizable = True,
            height=500,
            width = 600,
        )        
        return self.view
    
def test_MPLFigureEditor():
    # Create a window to demo the editor
    from traitsui.api import View, Item
    from numpy import sin, cos, linspace, pi

    class Test(HasTraits):

        figure = Instance(Figure, ())

        view = View(Item('figure', editor=MPLFigureEditor(),
                                show_label=False),
                        width=400,
                        height=300,
                        resizable=True)

        def __init__(self):
            super(Test, self).__init__()
            axes = self.figure.add_subplot(111)
            t = linspace(0, 2*pi, 200)
            axes.plot(sin(t)*(1+0.5*cos(11*t)), cos(t)*(1+0.5*cos(11*t)))

    Test().configure_traits()
    
def test_plot_selector():
    p = PlotSelector()
    
    x = np.linspace(-10,10,1000)
    p.paused = True
    p.plot(x,np.sin(x),label='sin')
    p.plot(x,np.cos(x),label='cos')
    p.paused = False
#    p.name_selected= ['sin','cos']
    p.configure_traits()


if __name__ == "__main__":
    test_plot_selector()