# -*- coding: utf-8 -*-

import os

import mne
import numpy as np

import seeg
from seeg.plots.display4d import Display4D


def test1(subject_id, subjects_dir):
    locations = np.asarray([[25.8315, -9.93608, 8.57813],
                            [29.30372216, -9.53925532, 8.76840202],
                            [32.77594432, -9.14243064, 8.95867404],
                            [36.24816648, -8.74560596, 9.14894606],
                            [39.72038864, -8.34878128, 9.33921808]])
    depth = seeg.Depth('A', 5, locations, contact_len=2, spacing=1.5)
    brain = seeg.create_depths_plot([depth], subject_id, subjects_dir)
    data = np.ones(15).reshape((5, 3))
    data[0, 0] = 2
    data[1, 1] = 2
    data[2, 2] = 2
    display = seeg.Display4D(brain, [depth], data)
    display.show()
