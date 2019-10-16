# -*- coding: utf-8 -*-

import os

import numpy as np
import pytest

import seeg

BASE_DIR = r'C:\Users\eisenmanl\Documents'
if not os.path.exists(BASE_DIR):
    BASE_DIR = os.path.join(r'C:\Users\leisenman\Documents',
                            r'brainstorm_db\tutorial_epimap')
else:
    BASE_DIR = os.path.join(BASE_DIR, r'brainstorm_data_files\tutorial_epimap')

ELECTRODE_NAMES = [r"y'", r"t'", r"u'", r"v'", r"x'", r"et'", r"b'", r"c'",
                   r"d'", r"e'", r"f'", r"l'", r"g'", r"s'", r"o'"]
BADS = ["v'1", "f'1"]
EEG_FILE = os.path.join(BASE_DIR, r'seeg\sz1.trc')
ELECTRODE_FILE = os.path.join(BASE_DIR,
                              r'anat\implantation\elec_pos_patient.txt')


@pytest.fixture
def electrode_names():
    return ELECTRODE_NAMES


@pytest.fixture
def seizure():
    seizure = {'bads': BADS,
               'electrodes': ELECTRODE_NAMES,
               'baseline': {'start': 72.800, 'end': 77.800,
                            'eeg_file_name': EEG_FILE},
               'seizure': {'start': 110.8, 'end': 160.8,
                           'eeg_file_name': EEG_FILE}
               }
    return seizure


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
    return os.path.join(BASE_DIR, r'anat\MRI\3DT1pre_deface.nii')