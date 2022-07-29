# -*- coding: utf-8 -*-

import os

import mne
import numpy as np
import pyvista as pv

import seeg
from seeg.plots.depths import YELLOW


def test_sphere():
    p = pv.Plotter()
    seeg.draw_sphere(p, radius=3, center=(0, 0, 0), color=YELLOW,
                     opacity=0.25)
    p.show()


def test_cube():
    p = pv.Plotter()
    seeg.draw_cube(p, center=(0, 0, 0), x_len=1, y_len=1, z_len=1,
                   color=YELLOW, opacity=0.25)
    p.show()


def test_cyl():
    p = pv.Plotter()
    seeg.draw_cyl(p, tip=(0, 0, 0), base=(10, 10, 10), diam=3,
                  color=YELLOW, opacity=0.25)
    p.show()
