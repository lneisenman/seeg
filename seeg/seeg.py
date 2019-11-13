# -*- coding: utf-8 -*-

import os

import pandas as pd

from .epi_index import calculate_EI
from .plots.source_image import create_source_image_map, plot_source_image_map
from .utils import EEG, read_electrode_file, create_montage, read_edf


class Seeg():
    '''Class to wrap EEG data, electrode locations and functions'''

    def __init__(self, subject, subjects_dir, electrode_names=None, bads=None,
                 baseline_times=(0, 5), seizure_times=(0, 5), seiz_delay=0):
        self.subject = subject
        self.subjects_dir = subjects_dir
        self.electrode_names = electrode_names
        self.bads = bads
        self.baseline_times = baseline_times
        self.seizure_times = seizure_times
        self.seiz_delay = seiz_delay
        self.subject_path = os.path.join(subjects_dir, subject)
        self.mri_file = os.path.join(self.subject_path, 'mri/orig.mgz')
        self.baseline_eeg_file = os.path.join(self.subject_path,
                                              'eeg/Interictal.edf')
        self.seizure_eeg_file = os.path.join(self.subject_path,
                                             'eeg/Seizure1.edf')
        self.electrode_file = os.path.join(self.subject_path, 'eeg/recon.fcsv')
        self.recording = EEG(electrode_names, bads)
        self.read_electrode_locations()
        self.montage, __ = create_montage(self.contacts)
        self.load_eeg()

    def read_electrode_locations(self):
        self.contacts = read_electrode_file(self.electrode_file)

    def load_eeg(self):
        raw = read_edf(self.baseline_eeg_file, self.contacts, self.bads)
        self.recording.set_baseline(self.baseline_times[0],
                                    self.baseline_times[1], raw,
                                    self.baseline_eeg_file)
        raw = read_edf(self.seizure_eeg_file, self.contacts, self.bads)
        self.recording.set_seizure(self.seizure_times[0],
                                   self.seizure_times[1], raw,
                                   self.seizure_eeg_file)

    def create_source_image_map(self, freqs, low_freq, high_freq):
        self.t_map = create_source_image_map(self.recording, self.mri_file,
                                             freqs, self.montage, low_freq,
                                             high_freq, self.seiz_delay)

    def show_source_image_map(self, cut_coords=None, threshold=2):
        plot_source_image_map(self.t_map, self.mri_file, cut_coords, threshold)

    def calculate_EI(self, freqs, bias=1, threshold=1, tau=1, H=5):
        self.EI = calculate_EI(self.recording['seizure']['raw'], freqs,
                               bias=bias, threshold=threshold,
                               tau=tau, H=H)
