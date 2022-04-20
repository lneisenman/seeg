# -*- coding: utf-8 -*-

import os

import mne
import numpy as np

import seeg
from seeg.utils import map_colors


def test1(subject_id, subjects_dir):
    locations = np.asarray([[25.8315, -9.93608, 8.57813],
                            [29.30372216, -9.53925532, 8.76840202],
                            [32.77594432, -9.14243064, 8.95867404],
                            [36.24816648, -8.74560596, 9.14894606],
                            [39.72038864, -8.34878128, 9.33921808]])
    depth = seeg.Depth('A', 5, locations, contact_len=2, spacing=1.5)
    Brain = mne.viz.get_brain_class()
    brain = Brain(subject_id, "rh", "pial", subjects_dir=subjects_dir,
                  cortex='classic', alpha=0.5)
    depth.draw(brain.plotter)
    depth.show_locations(brain.plotter)
    brain.plotter.app.exec_()


def test_create_depths_plot(subject_id, subjects_dir, electrode_names,
                            electrodes, raw):
    depth_list = seeg.create_depths(electrode_names, raw.info['ch_names'],
                                    electrodes)
    num_contacts = 0
    for depth in depth_list:
        num_contacts += depth.num_contacts

    values = 10*np.random.normal(size=num_contacts)
    colors = seeg.map_colors(values)[:, :3]
    brain = seeg.create_depths_plot(depth_list, subject_id, subjects_dir,
                                    contact_colors=colors)
    brain.show()
    brain.plotter.app.exec_()


def test_show_depth_values(subject_id, subjects_dir, electrode_names,
                           electrodes, raw, T_x_inv):
    depth_list = seeg.create_depths(electrode_names, raw.info['ch_names'],
                                    electrodes)
    brain = seeg.create_depths_plot(depth_list, subject_id, subjects_dir)
    num_contacts = 0
    for depth in depth_list:
        num_contacts += depth.num_contacts

    values = 10*np.random.normal(size=num_contacts)
    radius = 1 + (values - np.min(values))/12
    seeg.show_depth_values(depth_list, brain.plotter, values, radius,
                           affine=T_x_inv)
    brain.show()
    brain.plotter.app.exec_()


def test_show_depth_bipolar_values(subject_id, subjects_dir, electrode_names,
                                   electrodes, raw, T_x_inv):
    depth_list = seeg.create_depths(electrode_names, raw.info['ch_names'],
                                    electrodes)
    brain = seeg.create_depths_plot(depth_list, subject_id, subjects_dir)
    values = 10*np.random.normal(size=len(raw.info['ch_names']))
    radius = 1 + (values - np.min(values))/12
    seeg.show_depth_bipolar_values(depth_list, brain.plotter, values, radius,
                                   affine=T_x_inv)
    brain.show()
    brain.plotter.app.exec_()


def test_create_depth_epi_image_map(subject_id, subjects_dir, eeg,
                                    electrode_names, electrodes, raw, T_x_inv):
    values = seeg.create_depth_epi_image_map(eeg, low_freq=120,
                                             high_freq=200)
    depth_list = seeg.create_depths(electrode_names, raw.info['ch_names'],
                                    electrodes)
    brain = seeg.create_depths_plot(depth_list, subject_id, subjects_dir)
    seeg.show_depth_bipolar_values(depth_list, brain.plotter, values[0],
                                   radius=3, affine=T_x_inv)
    brain.show()
    brain.plotter.app.exec_()
