# -*- coding: utf-8 -*-
"""Create the 'gin' colormap from the Brainstorm package

"""
import matplotlib as mpl
from matplotlib.colors import LinearSegmentedColormap


cdict = {'red':   ((0.0, 1.0, 1.0),
                   (0.25, 0.0, 0.0),
                   (0.5, 0.0, 0.0),
                   (0.63, 1.0, 1.0),
                   (1.0, 1.0, 1.0)),

         'green': ((0.0, 1.0, 1.0),
                   (0.125, 1.0, 1.0),
                   (0.375, 0.0, 0.0),
                   (0.5, 0.0, 0.0),
                   (0.63, 0.0, 0.0),
                   (0.83, 1.0, 1.0),
                   (1.0, 1.0, 1.0)),

         'blue':  ((0.0, 1.0, 1.0),
                   (0.25, 1.0, 1.0),
                   (0.5, 0.0, 0.0),
                   (0.875, 0.0, 0.0),
                   (1.0, 1.0, 1.0))
         }

gin = LinearSegmentedColormap('gin', cdict, mpl.rcParams['image.lut'])
mpl.colormaps.register(name='gin', cmap=gin)
