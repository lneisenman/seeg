# -*- coding: utf-8 -*-

import os

import mne
import numpy as np
import pytest

import seeg


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


def test_threshold(subject_id, subjects_dir):
    locations = np.asarray([[25.8315, -9.93608, 8.57813],
                            [29.30372216, -9.53925532, 8.76840202],
                            [32.77594432, -9.14243064, 8.95867404],
                            [36.24816648, -8.74560596, 9.14894606],
                            [39.72038864, -8.34878128, 9.33921808]])
    depth = seeg.Depth('A', 5, locations, contact_len=2, spacing=1.5)
    brain = seeg.create_depths_plot([depth], subject_id, subjects_dir)
    data = np.ones(15).reshape((5, 3))

    with pytest.raises(ValueError) as excinfo:
        display = seeg.Display4D(brain, [depth], data, threshold=110)
    with pytest.raises(ValueError) as excinfo:
        display = seeg.Display4D(brain, [depth], data, threshold=-10)


def test_show_depth_values(subject_id, subjects_dir, electrode_names,
                           electrodes, raw, T_x_inv):
    depth_list = seeg.create_depths(electrode_names, raw.info['ch_names'],
                                    electrodes)
    brain = seeg.create_depths_plot(depth_list, subject_id, subjects_dir)
    num_contacts = 0
    for depth in depth_list:
        num_contacts += depth.num_contacts

    data = np.ones(5*num_contacts).reshape((num_contacts, 5))
    data[10, 0] = 2
    data[11, 1] = 2
    data[12, 2] = 2
    data[13, 3] = 2
    data[14, 4] = 2
    data[16, 2] = 2
    data[17, 3] = 2
    data[18, 4] = 2
    display = seeg.Display4D(brain, depth_list, data)
    display.show()
