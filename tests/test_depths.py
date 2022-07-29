# -*- coding: utf-8 -*-

import os

import mne
import numpy as np

import seeg
from seeg.utils import map_colors


def test1(subject_id, subjects_dir, T_x_inv):
    locations = np.asarray([[25.8315, -9.93608, 8.57813],
                            [29.30372216, -9.53925532, 8.76840202],
                            [32.77594432, -9.14243064, 8.95867404],
                            [36.24816648, -8.74560596, 9.14894606],
                            [39.72038864, -8.34878128, 9.33921808]])
    names = ['A1', 'A2', 'A3', 'A4', 'A5']
    depth = seeg.Depth('A', names, locations, contact_len=2, spacing=1.5)
    brain = seeg.create_depths_plot([depth], subject_id, subjects_dir)
    depth.show_locations(brain.plotter, affine=T_x_inv)
    brain.show()
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
                           electrodes, bads, raw, T_x_inv):
    depth_list = seeg.create_depths(electrode_names, raw.info['ch_names'],
                                    electrodes)
    brain = seeg.create_depths_plot(depth_list, subject_id, subjects_dir)
    num_contacts = 0
    for depth in depth_list:
        num_contacts += depth.num_contacts

    values = 10*np.random.normal(size=num_contacts)
    radius = 1 + (values - np.min(values))/12
    seeg.show_depth_values(depth_list, brain.plotter, values, radius, bads,
                           affine=T_x_inv)
    brain.show()
    brain.plotter.app.exec_()


def test_show_depth_bipolar_values(subject_id, subjects_dir, electrode_names,
                                   electrodes, bads, raw, T_x_inv):
    depth_list = seeg.create_depths(electrode_names, raw.info['ch_names'],
                                    electrodes)
    brain = seeg.create_depths_plot(depth_list, subject_id, subjects_dir)
    size = len(raw.info['ch_names']) - len(electrode_names) - 1
    values = 10*np.random.normal(size=size)
    radius = 1 + (values - np.min(values))/12
    seeg.show_depth_bipolar_values(depth_list, brain.plotter, values, radius,
                                   bads, affine=T_x_inv)
    brain.show()
    brain.plotter.app.exec_()


def test_create_depth_epi_image_map(subject_id, subjects_dir, eeg,
                                    electrode_names, electrodes, bads,
                                    T_x_inv):
    values = seeg.create_depth_epi_image_map(eeg, low_freq=120,
                                             high_freq=200)
    depth_list = seeg.create_depths(electrode_names,
                                    eeg.baseline['eeg'].info['ch_names'],
                                    electrodes)
    brain = seeg.create_depths_plot(depth_list, subject_id, subjects_dir)
    seeg.show_depth_bipolar_values(depth_list, brain.plotter, values[0, :],
                                   bads=bads, affine=T_x_inv)
    brain.show()
    brain.plotter.app.exec_()
