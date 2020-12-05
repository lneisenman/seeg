# -*- coding: utf-8 -*-

import os

from mayavi import mlab
import numpy as np

import seeg


BASE_DIR = r'R:\Active\SEEG'
if not os.path.exists(BASE_DIR):
    BASE_DIR = r'C:\Users\leisenman\Box\SEEG'
    if not os.path.exists(BASE_DIR):
        BASE_DIR = r'C:\Users\eisenmanl\Box\SEEG'

SUBJECT_ID = 'seeg_brainstorm'
SUBJECTS_DIR = os.path.join(BASE_DIR, 'subjects')


def test_cube(eeg, mri, freqs, electrode_names):
    t_map = seeg.create_source_image_map(eeg, mri, freqs,
                                         low_freq=120, high_freq=200)
    depth_list = seeg.create_depths(electrode_names,
                                    eeg.baseline['eeg'].ch_names,
                                    eeg.electrodes)
    brain = seeg.create_depths_plot(depth_list, SUBJECT_ID, SUBJECTS_DIR)
    cras = seeg.read_cras(SUBJECT_ID, SUBJECTS_DIR)
    print(cras)
    seeg.draw_volume(mlab.gcf(), t_map, cras)
    mlab.show()
