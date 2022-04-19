# -*- coding: utf-8 -*-

import os

import mne
import numpy as np
import pyvista as pv

import seeg


def test_sphere():
    p = pv.Plotter()
    seeg.draw_sphere(p, radius=3, center=(0, 0, 0), color=(1, 1, 0),
                     opacity=0.25)
    p.show()
