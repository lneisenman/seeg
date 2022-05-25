# -*- coding: utf-8 -*-

import os

import mne
import numpy as np

import seeg
from seeg.utils import map_colors


def test_napari(t_map, mri):
    seeg.plot_3d_epi_image_map(t_map, mri)
