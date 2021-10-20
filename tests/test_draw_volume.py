# -*- coding: utf-8 -*-

import os

import seeg


BASE_DIR = r'R:\Active\SEEG'
if not os.path.exists(BASE_DIR):
    BASE_DIR = r'C:\Users\leisenman\Box\SEEG'
    if not os.path.exists(BASE_DIR):
        BASE_DIR = r'C:\Users\eisenmanl\Box\SEEG'

SUBJECT_ID = 'seeg_brainstorm'
SUBJECTS_DIR = os.path.join(BASE_DIR, 'subjects')


def test_cube(eeg, electrode_names, t_map, affine):
    depth_list = seeg.create_depths(electrode_names,
                                    eeg.baseline['eeg'].ch_names,
                                    eeg.electrodes)
    brain = seeg.create_depths_plot(depth_list, SUBJECT_ID, SUBJECTS_DIR)
    seeg.draw_volume(brain.plotter.renderer, t_map, affine)
    brain.show()
    brain.plotter.app.exec_()
    