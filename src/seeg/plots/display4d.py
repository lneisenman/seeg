# -*- coding: utf-8 -*-
""" Classes for plotting SEEG depth electrodes with time represented by
    changing colors of contacts

"""

from matplotlib.colors import Colormap
from mne.viz import Brain
from nilearn.plotting.cm import cold_hot
import numpy as np
import numpy.typing as npt

from .depths import Depth1, SILVER
from ..utils import map_colors


class Display4D:
    """Class to display time data on MNE Brain with SEEG electrodes.

    Data for each contact at each time point is displayed via color coding
    using the provided cmap. Hitting the 'l' key advances one time step.
    Hitting the 'j' key goes back one time step. Hitting the 'k' key returns
    to time 0 where all electrodes are silver.

    Parameters
    ----------
    brain : MNE Brain
        Brain object from MNE with depth electrodes
    depth_list : list
        list of depth electrodes included in the Brain object
    data : Numpy NDArray
        2-d array with time data. Rows correspond to electrodes
        and columns to time steps
    cmap : Matplotlib Colormap, optional
        diameter of the depth electrode (default = cold_hot from nilearn)
    threshold : int, optional
        percentage above which color codes are displayed (default = 75)
    """

    def __init__(self, brain: Brain, depth_list: list[Depth1],
                 data: npt.NDArray, cmap: Colormap = cold_hot,
                 threshold: int = 75):
        self.brain = brain
        self.depth_list = depth_list
        temp = np.zeros(data.shape[0]).reshape((data.shape[0], 1))
        self.data = np.hstack((temp, data))
        assert self.data.shape[0] == data.shape[0]
        assert self.data.shape[1] == data.shape[1] + 1
        self.cmap = cmap
        if not 0 <= threshold <= 100:
            raise ValueError('threshold should be an int between 0 and 100')

        self.threshold = threshold
        self.assign_colors()
        self.plotter = brain.plotter
        self.tstep_max = data.shape[1]
        self.tstep = 0
        self.update()

        self.plotter.add_key_event('l', self.increment)
        self.plotter.add_key_event('j', self.decrement)
        self.plotter.add_key_event('k', self.reset)

    def show(self) -> None:
        self.brain.show()
        self.plotter.app.exec_()

    def assign_colors(self) -> None:
        self.colors = [map_colors(column, self.cmap)[:, :3]     # type: ignore
                       for column in self.data.T]
        for i, column in enumerate(self.data.T):
            threshold = (np.min(column) +
                         self.threshold/100*(np.max(column) - np.min(column)))
            low = np.where(column < threshold)
            self.colors[i][low] = SILVER

        self.colors[0] = [SILVER]*self.data.shape[0]

    def increment(self) -> None:
        if self.tstep < self.tstep_max:
            self.tstep += 1
            self.update()

    def decrement(self) -> None:
        if self.tstep > 0:
            self.tstep -= 1
            self.update()

    def reset(self) -> None:
        if self.tstep > 0:
            self.tstep = 0
            self.update()

    def update(self) -> None:
        colors = self.colors[self.tstep]
        idx = 0
        for depth in self.depth_list:
            for i in range(1, len(depth.actors)):
                depth.actors[i].GetProperty().SetColor(colors[idx])
                idx += 1

        self.plotter.update()


class Display4DBP(Display4D):
    """Class to display bipolar time data on MNE Brain with SEEG electrodes.

    Data for each contact at each time point is displayed via color coding
    using the provided cmap. Hitting the 'l' key advances one time step.
    Hitting the 'j' key goes back one time step. Hitting the 'k' key returns
    to time 0 where all electrodes are silver.

    Parameters
    ----------
    brain : MNE Brain
        Brain object from MNE with depth electrodes
    depth_list : list
        list of depth electrodes included in the Brain object
    data : Numpy NDArray
        2-d array with time data. Rows correspond to electrodes
        and columns to time steps
    cmap : Matplotlib Colormap, optional
        diameter of the depth electrode (default = cold_hot from nilearn)
    threshold : int, optional
        percentage above which color codes are displayed (default = 75)
    """

    def update(self) -> None:
        colors = self.colors[self.tstep]
        idx = 0
        for depth in self.depth_list:
            for i in range(len(depth.actors_BP)):
                depth.actors_BP[i].GetProperty().SetColor(colors[idx])
                idx += 1

        self.plotter.update()
