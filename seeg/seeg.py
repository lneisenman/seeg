# -*- coding: utf-8 -*-

from __future__ import (print_function, division, absolute_import,
                        unicode_literals)

import matplotlib.pyplot as plt
from mayavi import mlab
import mne
import neo
import nibabel as nib
from nibabel.affines import apply_affine
import nilearn
from nilearn.input_data import NiftiMasker
from nilearn.mass_univariate import permuted_ols
from nilearn.plotting import plot_stat_map
import numpy as np
import numpy.linalg as npl
import os
import pandas as pd
from scipy import stats as sps

from .source_image import create_source_image
from .utils import create_montage, read_edf


class Seeg():
    '''Class to wrap EEG data, electrode locations and functions'''

    def __init__(self, subject, subject_dir, electrode_names=None, bads=None,
                 baseline_times=(0, 5), seizure_times=(0, 5)):
        self.subject_dir = subject_dir
        self.subject = subject
        self.subject_path = os.path.join(subject_dir, subject)
        self.mri_file = os.path.join(self.subject_path, 'mri/orig.mgz')
        self.baseline_eeg_file = os.path.join(self.subject_path,
                                              'eeg/Interictal.edf')
        self.seizure_eeg_file = os.path.join(self.subject_path,
                                             'eeg/Seizure_1.edf')
        self.electrode_file = os.path.join(self.subject_path, 'eeg/recon.fcsv')
        self.electrode_names = electrode_names
        self.recording = dict()
        self.recording['electrodes'] = electrode_names
        self.recording['bads'] = bads
        self.recording['baseline'] = dict()
        self.recording['baseline']['start'] = baseline_times[0]
        self.recording['baseline']['end'] = baseline_times[1]
        self.recording['seizure'] = dict()
        self.recording['seizure']['start'] = seizure_times[0]
        self.recording['seizure']['end'] = seizure_times[1]
        self.read_electrode_locations()
        self.montage, __ = create_montage(self.contacts)
        self.load_eeg()

    def read_electrode_locations(self):
        locations = pd.read_csv(self.electrode_file, skiprows=2)
        self.contacts = pd.DataFrame(columns=['contact', 'x', 'y', 'z'])
        self.contacts['contact'] = locations['label']
        self.contacts['x'] = locations['x']/1000
        self.contacts['y'] = locations['y']/1000
        self.contacts['z'] = locations['z']/1000

    def load_eeg(self):
        self.recording['baseline']['raw'] = \
            read_edf(self.baseline_eeg_file, self.contacts,
                     self.recording['bads'])
        self.recording['seizure']['raw'] = \
            read_edf(self.seizure_eeg_file, self.contacts,
                     self.recording['bads'])

    def create_source_image(self, freqs, low_freq, high_freq):
        self.img = create_source_image(self.recording, self.mri_file, freqs,
                                       self.montage, low_freq, high_freq)

    def show_source_img(self, threshold=2):
        plot_stat_map(self.img, self.mri_file, threshold=threshold)
