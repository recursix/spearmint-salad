#!/usr/bin/env python

from distutils.core import setup

setup(name='spearmint_salad',
      version='1.0',
      author='Alexandre Lacoste',
    packages=[
        'spearmint_salad', 
        'spearmint_salad.spearmint',
        'spearmint_salad.spearmint.chooser',
        'spearmint_salad.analyze',
        'spearmint_salad.example'],
     )