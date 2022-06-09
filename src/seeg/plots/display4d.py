# -*- coding: utf-8 -*-
""" Class for plotting SEEG depth electrodes with time represented by changing
    colors of contacts

"""

from nilearn.plotting.cm import cold_hot
import numpy as np
import pyvista as pv

from ..utils import map_colors


class Display4D:

    def __init__(self, brain, depth_list, data, cmap=cold_hot) -> None:
        self.brain = brain
        self.depth_list = depth_list
        temp = np.zeros(data.shape[0]).reshape((data.shape[0], 1))
        self.data = np.hstack((temp, data))
        assert self.data.shape[0] == data.shape[0]
        assert self.data.shape[1] == data.shape[1] + 1
        self.cmap = cmap
        self.assign_colors()
        self.plotter = brain.plotter
        self.tstep_max = data.shape[1]
        self.tstep = 0

        self.plotter.add_key_event('l', self.increment)
        self.plotter.add_key_event('j', self.decrement)
        self.plotter.add_key_event('k', self.reset)

    def show(self):
        self.brain.show()
        self.plotter.app.exec_()

    def assign_colors(self):
        self.colors = [map_colors(column, self.cmap)[:, :3]
                       for column in self.data.T]

    def increment(self):
        print('increment')
        if self.tstep < self.tstep_max:
            self.tstep += 1
            self.update()

    def decrement(self):
        print('decrement')
        if self.tstep > 0:
            self.tstep -= 1
            self.update()

    def reset(self):
        print('reset')
        if self.tstep > 0:
            self.tstep = 0
            self.update()

    def update(self):
        colors = self.colors[self.tstep]
        idx = 0
        for depth in self.depth_list:
            for i in range(1, len(depth.actors)):
                depth.actors[i].GetProperty().SetColor(colors[idx])
                idx += 1

        self.plotter.update()
