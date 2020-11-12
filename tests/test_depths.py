# -*- coding: utf-8 -*-

import os

from mayavi import mlab
import numpy as np
import pytest
from surfer import Brain

import seeg


BASE_DIR = r'R:\Active\SEEG'
if not os.path.exists(BASE_DIR):
    BASE_DIR = r'C:\Users\leisenman\Box\SEEG'
    if not os.path.exists(BASE_DIR):
        BASE_DIR = r'C:\Users\eisenmanl\Box\SEEG'

SUBJECT_ID = 'seeg_brainstorm'
SUBJECTS_DIR = os.path.join(BASE_DIR, 'subjects')


def test1():
    locations = np.asarray([[25.8315, -9.93608,  8.57813],
                            [27.81562695, -9.70932304,  8.68685687],
                            [29.30372216, -9.53925532,  8.76840203],
                            [31.28784911, -9.31249836,  8.87712890],
                            [32.77594432, -9.14243065,  8.95867405]])
    # depth = Depth('A', 5, locations, contact_len=1, spacing=1)

    locations = np.asarray([[25.8315, -9.93608, 8.57813],
                            [29.30372216, -9.53925532, 8.76840202],
                            [32.77594432, -9.14243064, 8.95867404],
                            [36.24816648, -8.74560596, 9.14894606],
                            [39.72038864, -8.34878128, 9.33921808]])
    depth = seeg.Depth('A', 5, locations, contact_len=2, spacing=1.5)
    brain = Brain(SUBJECT_ID, "rh", "pial", subjects_dir=SUBJECTS_DIR,
                  cortex='ivory', alpha=0.5)
    depth.draw(fig=mlab.gcf())
    depth.show_locations(fig=mlab.gcf())
    mlab.show()


def test_create_depths_plot(electrode_names, electrodes, raw):
    depth_list = seeg.create_depths(electrode_names, raw.info['ch_names'],
                                    electrodes)
    brain = seeg.create_depths_plot(depth_list, SUBJECT_ID, SUBJECTS_DIR)
    mlab.show()


def test_show_bipolar_values(electrode_names, electrodes, raw):
    depth_list = seeg.create_depths(electrode_names, raw.info['ch_names'],
                                    electrodes)
    brain = seeg.create_depths_plot(depth_list, SUBJECT_ID, SUBJECTS_DIR)
    values = 10*np.random.normal(size=len(raw.info['ch_names']))
    seeg.show_bipolar_values(depth_list, mlab.gcf(), values)
    mlab.show()


def test_create_depth_source_image_map(eeg, freqs, electrode_names,
                                       electrodes, raw):
    t_map = seeg.create_depth_source_image_map(eeg, freqs,
                                               low_freq=120, high_freq=200)
    print(t_map[0].min(), t_map[0].max())
    depth_list = seeg.create_depths(electrode_names, raw.info['ch_names'],
                                    electrodes)
    brain = seeg.create_depths_plot(depth_list, SUBJECT_ID, SUBJECTS_DIR)
    seeg.show_bipolar_values(depth_list, mlab.gcf(), t_map[0], radius=3)
    mlab.show()
