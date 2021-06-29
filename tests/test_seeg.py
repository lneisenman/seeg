# -*- coding: utf-8 -*-

"""
test_seeg
----------------------------------

Tests for `seeg` module.
"""

import matplotlib.pyplot as plt
from nilearn.input_data import NiftiMasker
from nilearn.mass_univariate import permuted_ols
from nilearn.plotting import plot_stat_map
import numpy as np
import pandas as pd
import pytest

import seeg


# HOME = r'C:\Users\eisenmanl\Documents\brainstorm_data_files'
# HOME = r'C:\Users\leisenman\Documents\brainstorm_db'


def test_create_epi_image_map(eeg, mri):
    t_map = seeg.create_epi_image_map(eeg, mri, low_freq=120, high_freq=200)
    seeg.plot_epi_image_map(t_map, mri)
    seeg.plot_epi_image_map(t_map, mri, cut_coords=(-38, -50, -12))
    plt.show()


def test_create_MT_epi_image_map(eeg, mri):
    t_map = seeg.create_epi_image_map(eeg, mri, low_freq=120, high_freq=200,
                                      method='multi')
    seeg.plot_epi_image_map(t_map, mri)
    seeg.plot_epi_image_map(t_map, mri, cut_coords=(-38, -50, -12))
    plt.show()


def test_EpiImage(eeg, mri):
    image = seeg.EpiImage(eeg, mri)
    image.plot(cut_coords=(-38, -50, -12))
    plt.show()


def test_EpiImage_MT(eeg, mri):
    image = seeg.EpiImage(eeg, mri, method='multi')
    image.plot(cut_coords=(-38, -50, -12))
    plt.show()


def test_SEEG(subject_id, subjects_dir, electrode_names, bads, electrode_file):
    study = seeg.Seeg(subject_id, subjects_dir, electrode_names, bads,
                      electrode_file=electrode_file)
    study.create_epi_image_map(120, 200)
    study.show_epi_image_map(cut_coords=(-38, -50, -12))


def test_setup_bipolar(raw, bads):
    anodes, cathodes, ch_names = seeg.setup_bipolar("v'", raw.ch_names, bads)
    print(anodes)
    print(cathodes)
    assert anodes == ["v'2", "v'3", "v'12", "v'13", "v'14"]  # v'1 is bad!
    assert cathodes == ["v'3", "v'4", "v'13", "v'14", "v'15"]
    assert ch_names == ["v'2-v'3", "v'3-v'4", "v'12-v'13", "v'13-v'14",
                        "v'14-v'15"]


def test_EEG(raw):
    test = seeg.EEG(['A'])
    test['baseline']['raw'] = raw
    print(test['baseline'])
    with pytest.raises(KeyError):
        print(test['bad key'])
