# -*- coding: utf-8 -*-

import os
from _pytest.config import filename_arg

import mne
import numpy as np
import nibabel as nib
import pytest

import seeg

BASE_DIR = r'R:\Active\SEEG'
if not os.path.exists(BASE_DIR):
    BASE_DIR = r'C:\Users\leisenman\Box\SEEG'
    if not os.path.exists(BASE_DIR):
        BASE_DIR = r'C:\Users\eisenmanl\Box\SEEG'

SUBJECT_ID = 'seeg_brainstorm'
SUBJECTS_DIR = os.path.join(BASE_DIR, 'subjects')
EEG_DIR = os.path.join(SUBJECTS_DIR, SUBJECT_ID, 'eeg')

ELECTRODE_NAMES = [r"y'", r"t'", r"u'", r"v'", r"x'", r"et'", r"b'", r"c'",
                   r"d'", r"e'", r"f'", r"l'", r"g'", r"s'", r"o'"]
BADS = ["v'1", "f'1"]

EEG_FILE = os.path.join(SUBJECTS_DIR, SUBJECT_ID, r'eeg\sz1.trc')
ELECTRODE_FILE = os.path.join(SUBJECTS_DIR, SUBJECT_ID,
                              r'eeg\elec_pos_patient.txt')


@pytest.fixture
def electrode_names():
    return ELECTRODE_NAMES


@pytest.fixture
def bads():
    return BADS


@pytest.fixture
def freqs():
    return np.arange(10, 220, 3)


@pytest.fixture
def electrodes():
    return seeg.read_electrode_file(ELECTRODE_FILE)


@pytest.fixture
def montage(electrodes):
    mont, __ = seeg.create_montage(electrodes)
    return mont


@pytest.fixture
def raw(electrodes):
    raw, __ = seeg.read_micromed_eeg(EEG_FILE, electrodes, BADS)
    return raw


@pytest.fixture
def mri():
    return os.path.join(SUBJECTS_DIR, SUBJECT_ID, r'mri\T1.mgz')


@pytest.fixture
def eeg():
    eeg = seeg.load_eeg_data(EEG_DIR, ELECTRODE_NAMES, BADS, seizure=1,
                             electrode_file='elec_pos_patient.txt')[0]
    return eeg


@pytest.fixture
def t_map():
    t_map = nib.load('tests/t_map.nii.gz')
    return t_map

@pytest.fixture
def Torig(mri):
    return mri.header.get_vox2ras_tkr()
