# -*- coding: utf-8 -*-

import os
import pytest

import seeg


pytest.skip('Skipping draw cube', allow_module_level=True)


def test_cube(subject_id, subjects_dir, eeg, electrode_names, t_map, affine):
    depth_list = seeg.create_depths(electrode_names,
                                    eeg.baseline['eeg'].ch_names,
                                    eeg.electrodes)
    brain = seeg.create_depths_plot(depth_list, subject_id, subjects_dir)
    seeg.draw_volume(brain.plotter, t_map, affine)
    brain.show()
    brain.plotter.app.exec_()
    