# -*- coding: utf-8 -*-

import os

import mne
import pandas as pd

from .epi_index import calculate_EI
from .plots.epi_image import EpiImage, create_epi_image_map, plot_epi_image_map
from .utils import (EEG, read_electrode_file, create_montage, read_edf,
                    load_eeg_data)


class Seeg():
    """ Class to contain SEEG recording information

    Parameters
    ----------
    subject : string
        name of subect folder in Freesurfer subjects_dir
    subjects_dir : string
        path to Freesurfer subjects_dir
    electrode_names : list of strings, optional
        names of electrodes, by default None
    bads : list of strings, optional
        names of bad channels, by default None
    electrode_file : string
        electrode data file name

    """

    def __init__(self, subject, subjects_dir, electrode_names, bads,
                 seizure=1, electrode_file='recon.fcsv', seiz_delay=0):
        self.subject = subject
        self.subjects_dir = subjects_dir
        self.electrode_names = electrode_names
        self.bads = bads
        self.subject_path = os.path.join(subjects_dir, subject)
        self.eeg_path = os.path.join(self.subject_path, 'eeg')
        self.mri_file = os.path.join(self.subject_path, 'mri/orig.mgz')
        self.electrode_file = electrode_file
        self.eeg = load_eeg_data(self.eeg_path, self.electrode_names,
                                 self.bads, seizure=seizure,
                                 electrode_file=electrode_file)[seizure-1]
        self.seiz_delay = seiz_delay
        # self.contacts = read_electrode_file(self.electrode_file)

    # def read_electrode_locations(self):
    #     """ read and process the file containing electrode
    #         locations

    #     """
    #     self.contacts = read_electrode_file(self.electrode_file)

    # def load_eeg(self):
    #     """ read baseline and seizure EEG data

    #     """
    #     raw = read_edf(self.baseline_eeg_file, self.contacts, self.bads)
    #     raw.set_annotations(mne.Annotations(self.baseline_time, 0, 'Seizure'))
    #     self.eeg.set_baseline(raw, file_name=self.baseline_eeg_file)
    #     raw = read_edf(self.seizure_eeg_file, self.contacts, self.bads)
    #     raw.set_annotations(mne.Annotations(self.seizure_time, 0, 'Seizure'))
    #     self.eeg.set_seizure(raw, file_name=self.seizure_eeg_file)

    def create_epi_image_map(self, low_freq, high_freq, D=3, dt=0.2,
                             method='welch'):
        """ calculate the source image map analagous to David et al 2011
            and the Brainstorm tutorial

        """
        self.epi_image = EpiImage(self.eeg, self.mri_file, low_freq, high_freq,
                                  self.seiz_delay, method, D, dt)
        # self.t_map = create_epi_image_map(self.eeg, self.mri_file,
        #                                   low_freq, high_freq,
        #                                   self.eeg.seiz_delay)

    def show_epi_image_map(self, cut_coords=None, threshold=2):
        """ Use matplotlib to display the source image map

        """
        self.epi_image.plot(cut_coords=cut_coords, threshold=threshold)

    def calculate_EI(self, freqs, bias=1, threshold=1, tau=1, H=5):
        """ calculate the epileptogenicity index as per Bartolomi et al 2008

        """
        self.EI = calculate_EI(self.eeg['seizure']['raw'], freqs,
                               bias=bias, threshold=threshold,
                               tau=tau, H=H)
