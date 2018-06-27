# -*- coding: utf-8 -*-

"""
test_seeg
----------------------------------

Tests for `seeg` module.
"""

from __future__ import (print_function, division, absolute_import,
                        unicode_literals)

import matplotlib.pyplot as plt
from nilearn.input_data import NiftiMasker
from nilearn.mass_univariate import permuted_ols
from nilearn.plotting import plot_stat_map
import numpy as np

import seeg


def test_brainstorm_seizure1():
    names = [r"l'", r"g'"]
    num_contacts = [9, 12]
    seizure1 = {'file_name': 'sz1.trc', 'bads': ["v'1", "f'1"],
                'electrodes': {'names': names, 'num_contacts': num_contacts},
                'baseline': {'start': 72.800, 'end': 77.800},
                'seizure': {'start': 110.8, 'end': 160.8}}
    seizure2 = {'file_name': 'sz2.trc', 'bads': ["v'1", "t'8"],
                'electrodes': {'names': names, 'num_contacts': num_contacts},
                'baseline': {'start': 103.510, 'end': 108.510},
                'seizure': {'start': 133.510, 'end': 183.510}}
    seizure3 = {'file_name': 'sz3.trc', 'bads': ["o'1", "t'8"],
                'electrodes': {'names': names, 'num_contacts': num_contacts},
                'baseline': {'start': 45.287, 'end': 50.287},
                'seizure': {'start': 110.287, 'end': 160.287}}
    seizures = [seizure1, seizure2, seizure3]
    seizures = [seizure1]
    freqs = np.arange(10, 220, 3)

    
    montage, __ = seeg.read_electrode_locations()
    
    for seizure in seizures:
        raw = seeg.read_eeg(montage.dig_ch_pos, seizure)
        seizure['baseline']['eeg'], seizure['seizure']['eeg'] = seeg.clip_eeg(seizure, raw)
        seizure['baseline']['bipolar'], seizure['seizure']['bipolar'] = seeg.create_bipolar(seizure)
        seizure['baseline']['power'], seizure['seizure']['power'] = seeg.calc_power(seizure, freqs)
        seizure['baseline']['ave_power'], seizure['seizure']['ave_power'] = seeg.ave_power_over_freq_band(seizure, freqs)
        seizure['baseline']['ex_power'], seizure['seizure']['ex_power'] = seeg.extract_power(seizure)
        seizure['baseline']['img'], seizure['seizure']['img'] = seeg.map_seeg_data(seizure, montage)

    
    mri = r'C:\Users\eisenmanl\Documents\brainstorm_data_files\tutorial_epimap\anat\MRI\3DT1pre_deface.nii'
    base_img = seizures[0]['baseline']['img']
    seiz_img = seizures[0]['seizure']['img']
    # base_img = nib.load('baseline.nii.gz')
    # seiz_img = nib.load('seizure.nii.gz')
    # base = base_img.get_data()
    # seiz = seiz_img.get_data()
    
    nifti_masker = NiftiMasker(memory='nilearn_cache', memory_level=1)  # cache options
    base_masked = nifti_masker.fit_transform(base_img)
    seiz_masked = nifti_masker.fit_transform(seiz_img)
    data = np.concatenate((base_masked, seiz_masked))
    labels = np.zeros(30, dtype=np.int)
    labels[15:] = 1
    neg_log_pvals, t_scores, _ = permuted_ols(
        labels, data,
        # + intercept as a covariate by default
        n_perm=10000, two_sided_test=True,
        n_jobs=2)  # can be changed to use more CPUs
    p_pt_img = nifti_masker.inverse_transform(neg_log_pvals)
    plot_stat_map(p_pt_img, mri, threshold=1.3)
    t_pt_img = nifti_masker.inverse_transform(t_scores)
    plot_stat_map(t_pt_img, mri, threshold=2)
    plt.show()
