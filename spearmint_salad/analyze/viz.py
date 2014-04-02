#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''
Created on Nov 5, 2013

@author: alexandre
'''

import os
from spearmint_salad.analyze.mpl_plot import MPLFigureEditor
from matplotlib.figure import Figure

from traits.api import HasTraits, Any, Directory, List, Enum, Str, Instance, PrototypedFrom, Range
from traitsui.api import Item, Group, View,  CheckListEditor,  ListEditor

from traits.has_traits import on_trait_change
from operator import itemgetter
from fnmatch import fnmatch

from spearmint_salad.analyze.analyze_trace import HpInfo, plot_eval_info, plot_time, extract_metric_set
from spearmint_salad import pkl_trace, experiments_folder

from spearmint_salad.pkl_trace import get_column_dict
from graalUtil.num import gaussConv
from spearmint_salad.analyze.analyze_many_traces import plot_sign_test


# Define the demo class:    
class ExpList ( HasTraits ):
    """ Define the main DirectoryEditor demo class. """

    # Define a Directory trait to view:
    exp_folder = Directory(experiments_folder)
    filter = Str('*')
    exp_list = List(Str)
    exp_path_list =List()
    trace_list = List()
    
    exp_names = List(Str)
    
    def init(self):
        print 'init called'
    
    @on_trait_change('exp_list')
    def set_exp_path_list(self):
        self.exp_path_list = [ os.path.join( self.exp_folder, exp ) for exp in self.exp_list]
        
        trace_list_ = []
        for exp_path in self.exp_path_list:
            trace_list_ += load_trace_rec( exp_path ) 
        self.trace_list = trace_list_
    
    @on_trait_change('exp_folder,filter')
    def _load(self):
        print 'loading'


        exp_list = []
        for name in os.listdir(self.exp_folder):
            exp_path = os.path.join(self.exp_folder, name)
            if name.startswith('.'):
                continue
            
            if os.path.isdir(exp_path):
                if fnmatch( name, self.filter ):
                    exp_list.append( (name, os.path.getmtime(exp_path)) )

        
        if len(exp_list) == 0:
            self.exp_names = []
        else:
            exp_list.sort(  key=itemgetter(1) )
            self.exp_names = list(zip(*exp_list)[0][::-1])            
            self.exp_list = [self.exp_names[0]]


    def default_traits_view(self):
        editor = CheckListEditor( name = 'exp_names', cols = 1)
        
        self.view = View(
            "exp_folder",
            "filter",
            Group( Item('exp_list', editor=editor, style='custom', show_label=False, width=-20, emphasized=False),
                scrollable=True
                ),
            buttons   = ['OK'],
            resizable = True,
            height=500,
        )
        
        self._load()
        

        return self.view

def load_trace_rec(folder, filter_='trace*.pkl'):
    trace_path_list= []
    for root, _dirs, files in os.walk(folder):
        for file_ in files:
            if fnmatch( file_, filter_ ):
                trace_path_list.append( os.path.join( root, file_ ) ) 
    return trace_path_list



class TraceSelector(HasTraits):
    
    exp_list = Instance(HasTraits)
    trace_list = PrototypedFrom( 'exp_list' )
    trace_path = Enum(values='trace_list')

    def update(self):
        self.selected_trace()

#    @on_trait_change('trace_list')
#    def trace_list_changed(self):
##        print 'setting trace path to ', self.trace_list[0]
#        self.trace_path = self.trace_list[0]
#        
        
    @on_trait_change('trace_path')
    def selected_trace(self):
        if getattr(self, 'last_trace',None ) != self.trace_path:
            self.last_trace = self.trace_path
            trace = pkl_trace.TraceDBFile(self.trace_path)
            self.new_trace(trace)

    view = View(
        Item('trace_path', style='simple', width=-20),
        resizable=True)


class PatternFilter:
    
    def __init__(self, *pattern_list ):
        self.pattern_list = pattern_list
    
    def __call__(self, val ):
        return any( [fnmatch(val, pattern) for pattern in self.pattern_list] )
            
    



class TimeLine(TraceSelector):
    
    smooth = Range(0.,5., 0.)

    metric_list = List()
    metric_selector = List()
    figure = Instance(Figure, ())

    default_selected = PatternFilter('*')
    metric_pattern =  PatternFilter('*')
    
    
    def metric_filter(self, metric_list):
        return [ metric for metric in metric_list if self.metric_pattern(metric) ]

    def set_default_metrics(self):
        if not hasattr(self, 'initiated' ):
            self.initiated = True
            
            self.metric_selector = [ metric for metric in self.metric_list if self.default_selected(metric) ]

    def new_trace(self, trace):
        self.collection = trace.db.analyze
        metric_set = extract_metric_set(self.collection)
        metric_set.difference_update(['_id', 'i', 'hp_id', 'proc_id'] ) 
            
        metric_list = self.metric_filter( list(metric_set) )
        
        
        metric_list.sort()
        self.metric_list = metric_list
        
        self.set_default_metrics()
        
        self._plot()

#    def _smooth_changed(self):
#        pass

    @on_trait_change('metric_selector,smooth')
    def _plot(self):
        col_dict = get_column_dict(self.collection, 'i', *self.metric_selector)
        x = col_dict.pop('i')
        self.figure.clf()
        axes = self.figure.add_subplot(111)


        if self.smooth > 0: x, = gaussConv(self.smooth, x )

#        print x
        for label, y  in col_dict.items():
            if self.smooth > 0: y, = gaussConv(self.smooth, y )
#            print y 
            axes.plot(x,y, label=label)
        
        axes.legend(loc='best')
        axes.set_xlabel('iteration')
        self.figure.canvas.draw()
    
    def default_traits_view(self):
        editor = CheckListEditor( name = 'metric_list', cols = 1)
        
        self.view = View(
            Item('trace_path', style='simple', width=-20),
            Group(
                Item('figure', editor=MPLFigureEditor(), show_label=False),
                Group(
                    Item('smooth', style='custom', show_label=False),
                    Item('metric_selector', editor=editor, style='custom', show_label=False),
                    ),
                orientation='horizontal',
                scrollable=True,
            ),
            buttons   = ['OK'],
            resizable = True,
            height=500,
        )        

        return self.view
    


class BasicPlot(TraceSelector):
    
    figure = Instance(Figure, ())

    view = View(
        Item('trace_path', style='simple', width=-20),
        Item('figure', editor=MPLFigureEditor(), show_label=False), 
        resizable=True)
    
    def new_trace(self, trace):
        self.figure.clf()
        self._plot(trace)
        self.figure.canvas.draw()


class SgnTest(HasTraits):
    
    figure = Instance(Figure, ())
    exp_list = Instance(HasTraits)
    trace_list = PrototypedFrom( 'exp_list' )
    
    view = View(
        Item('figure', editor=MPLFigureEditor(), show_label=False))
    
    def _get_trace_list(self):
        return [ pkl_trace.TraceDBFile(trace_path) for trace_path in self.trace_list ]
        
    def update(self):
        self.figure.clf()
        axes = self.figure.add_subplot(111)
        trace_list = self._get_trace_list()
#        
#        key_pair_list = [ 
#            ('salad_risk.tst', 'argmin_risk.tst' ), 
#            ('salad_risk.greedy-05.tst', 'argmin_risk.tst' ),
#            ]
        
        key_pair_list = [ 
#             ('salad_risk.greedy-05.tst', 'salad_risk.tst' ), 
#             ('salad_risk.greedy-10.tst', 'salad_risk.tst' ),
            ('salad_risk.tst', 'argmin_risk.tst' ), 
#             ('salad_risk.greedy-10.tst', 'argmin_risk.tst' ),
            ]
        
        plot_sign_test(trace_list,key_pair_list, axes=axes)
        self.figure.canvas.draw()
   
#class PlotRisk(BasicPlot):
#    
#    def _plot(self, trace):
#        axes = self.figure.add_subplot(111)
#        plot_stats( trace.db.analyze, axes, 
#            'i', 'salad_risk.tst', 'argmin_risk.tst','salad_risk.val', 'argmin_risk.val' )

class PlotRisk(TimeLine):
    default_selected = PatternFilter('salad_risk.tst', 'argmin_risk.tst')
    metric_pattern = PatternFilter('salad_risk.tst', 'salad_risk.val', '*greedy*', 'argmin*', '*50%.tst' )  

class PlotEssr(TimeLine):
    default_selected = PatternFilter('salad*.tst')
    metric_pattern = PatternFilter('*%*', 'argmin*' )  

class PlotTime(BasicPlot):
    def _plot(self,trace):
        plot_time( trace, self.figure.add_subplot(111) )

from spearmint_salad.analyze.plot_nd import Plot3d_with_params
class PlotHP(TraceSelector):
    
    plot = Instance(Plot3d_with_params,())  
#    plot = Any()
    
    view = View(
        Item('trace_path', style='simple', width=-20),
        Item('plot', style='custom', show_label=False), 
        resizable=True )
    
    def new_trace(self, trace):
        print 'creating 3d plot'
        hp_info = HpInfo(trace)
        
        plot_eval_info(self.plot, hp_info, 'val.risk')

class Main(HasTraits):
    tabs = List(HasTraits)
    selected = Any

    @on_trait_change('selected')
    def new_tab(self):
        if hasattr(self.selected,'update'):
            self.selected.update()
        
    view = View(
        Item('tabs', style='custom', show_label=False, 
            editor=ListEditor(use_notebook=True, deletable=False, dock_style='tab',selected='selected')),
        resizable = True,
        height = 500,
        width= 500 )

exp_list = ExpList()

main = Main()
main.tabs = [ exp_list, 
    PlotRisk(exp_list = exp_list),
    PlotTime(exp_list = exp_list),
    PlotHP(exp_list = exp_list),
#     PlotEssr(exp_list = exp_list),
#     SgnTest(exp_list = exp_list),
    ]
main.configure_traits()



