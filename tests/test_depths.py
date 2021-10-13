# -*- coding: utf-8 -*-

import os

import mne
import numpy as np

import seeg


BASE_DIR = r'R:\Active\SEEG'
if not os.path.exists(BASE_DIR):
    BASE_DIR = r'C:\Users\leisenman\Box\SEEG'
    if not os.path.exists(BASE_DIR):
        BASE_DIR = r'C:\Users\eisenmanl\Box\SEEG'

SUBJECT_ID = 'seeg_brainstorm'
SUBJECTS_DIR = os.path.join(BASE_DIR, 'subjects')


# def test1_mlab():
#     mne.viz.set_3d_backend('mayavi')
#     locations = np.asarray([[25.8315, -9.93608, 8.57813],
#                             [29.30372216, -9.53925532, 8.76840202],
#                             [32.77594432, -9.14243064, 8.95867404],
#                             [36.24816648, -8.74560596, 9.14894606],
#                             [39.72038864, -8.34878128, 9.33921808]])
#     depth = seeg.Depth('A', 5, locations, contact_len=2, spacing=1.5)
#     Brain = mne.viz.get_brain_class()
#     brain = Brain(SUBJECT_ID, "rh", "pial", subjects_dir=SUBJECTS_DIR,
#                   cortex='classic', alpha=0.5)
#     depth.draw(mlab.gcf().scene)
#     depth.show_locations(mlab.gcf().scene)
#     mlab.show()


def test1():
    locations = np.asarray([[25.8315, -9.93608, 8.57813],
                            [29.30372216, -9.53925532, 8.76840202],
                            [32.77594432, -9.14243064, 8.95867404],
                            [36.24816648, -8.74560596, 9.14894606],
                            [39.72038864, -8.34878128, 9.33921808]])
    depth = seeg.Depth('A', 5, locations, contact_len=2, spacing=1.5)
    Brain = mne.viz.get_brain_class()
    brain = Brain(SUBJECT_ID, "rh", "pial", subjects_dir=SUBJECTS_DIR,
                  cortex='classic', alpha=0.5)
    depth.draw(brain.plotter.renderer)
    depth.show_locations(brain.plotter.renderer)
    brain.show()


# def test_create_depths_plot_mlab(electrode_names, electrodes, raw):
#     mne.viz.set_3d_backend('mayavi')
#     depth_list = seeg.create_depths(electrode_names, raw.info['ch_names'],
#                                     electrodes)
#     brain = seeg.create_depths_plot(depth_list, SUBJECT_ID, SUBJECTS_DIR)
#     mlab.show()


def test_create_depths_plot(electrode_names, electrodes, raw):
    depth_list = seeg.create_depths(electrode_names, raw.info['ch_names'],
                                    electrodes)
    brain = seeg.create_depths_plot(depth_list, SUBJECT_ID, SUBJECTS_DIR)
    brain.show()


# def test_show_bipolar_values_mlab(electrode_names, electrodes, raw, T_x_inv):
#     mne.viz.set_3d_backend('mayavi')
#     depth_list = seeg.create_depths(electrode_names, raw.info['ch_names'],
#                                     electrodes)
#     brain = seeg.create_depths_plot(depth_list, SUBJECT_ID, SUBJECTS_DIR)
#     # print(type(brain))
#     # assert 1 == 0
#     values = 10*np.random.normal(size=len(raw.info['ch_names']))
#     seeg.show_bipolar_values(depth_list, mlab.gcf().scene, values,
#                              affine=T_x_inv)
#     mlab.show()


def test_show_bipolar_values(electrode_names, electrodes, raw, T_x_inv):
    depth_list = seeg.create_depths(electrode_names, raw.info['ch_names'],
                                    electrodes)
    brain = seeg.create_depths_plot(depth_list, SUBJECT_ID, SUBJECTS_DIR)
    values = 10*np.random.normal(size=len(raw.info['ch_names']))
    seeg.show_bipolar_values(depth_list, brain.plotter.renderer, values,
                             affine=T_x_inv)
    brain.show()


# def test_create_depth_epi_image_map_mlab(eeg, electrode_names,
#                                          electrodes, raw, T_x_inv):
#     mne.viz.set_3d_backend('mayavi')
#     values = seeg.create_depth_epi_image_map(eeg, low_freq=120,
#                                              high_freq=200)
#     # print(type(values), values[0])
#     # assert 1==0
#     depth_list = seeg.create_depths(electrode_names, raw.info['ch_names'],
#                                     electrodes)
#     brain = seeg.create_depths_plot(depth_list, SUBJECT_ID, SUBJECTS_DIR)
#     seeg.show_bipolar_values(depth_list, mlab.gcf().scene, values[0], radius=3,
#                              affine=T_x_inv)
#     mlab.show()


def test_create_depth_epi_image_map(eeg, electrode_names,
                                       electrodes, raw, T_x_inv):
    values = seeg.create_depth_epi_image_map(eeg, low_freq=120,
                                             high_freq=200)
    # print(type(values), values[0])
    # assert 1==0
    depth_list = seeg.create_depths(electrode_names, raw.info['ch_names'],
                                    electrodes)
    brain = seeg.create_depths_plot(depth_list, SUBJECT_ID, SUBJECTS_DIR)
    seeg.show_bipolar_values(depth_list, brain.plotter.renderer, values[0],
                             radius=3, affine=T_x_inv)
    brain.show()
