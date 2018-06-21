from matplotlib.colors import LinearSegmentedColormap
import matplotlib.pyplot as plt


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

gin = LinearSegmentedColormap('gin', cdict)
plt.register_cmap(cmap=gin)
