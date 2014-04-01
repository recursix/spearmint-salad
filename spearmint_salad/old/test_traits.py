# -*- coding: utf-8 -*-
'''
Created on Oct 28, 2013

@author: alexandre
'''
from traits.api import HasTraits, Bool,Range
from traitsui.api import View, Handler

class TC_Handler(Handler):

    def setattr(self, info, object, name, value):
        Handler.setattr(self, info, object, name, value)
        info.object._updated = True

    def object__updated_changed(self, info):
        if info.initialized:
            info.ui.title += "*"

#class TestClass(HasTraits):
#    b1 = Bool
#    b2 = Bool
#    b3 = Bool
#    _updated = Bool(False)


#
#for e in dir(view1):
#    print e

#for attr in ['b1', "title", "handler", "buttons"]:
#    print attr, getattr(view1, attr, None)

#tc = TestClass()
#tc.add_trait( 'b4',Bool)

t = HasTraits()
nameL = []
for i in range(4):
    name = 'r%d'%i
    t.add_trait(name, Range(1,10,i) )
    nameL.append(name)

view1 = View(nameL,
             title="Alter Title",
             handler=TC_Handler(),
             buttons = ['OK', 'Cancel'])

t.configure_traits(view=view1)
