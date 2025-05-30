# -*- coding: utf-8 -*-

import os
# from _pytest.config import filename_arg

import mne
import numpy as np
import numpy.linalg as npl
import nibabel as nib
import pytest

import seeg

BASE_DIR = mne.datasets.misc.data_path()


ELECTRODE_NAMES = ['LA', 'LB', 'LC', 'LD', 'LE', 'LF',
                   'LG', 'LH', 'RA', 'RB']
BADS = ['LA09', 'LA10', 'LA11', 'LA12', 'LB09', 'LB10', 'LB11', 'LB12',
        'LD07', 'LD08', 'LE09', 'LE11', 'LE12', 'LG08', 'RB01', 'RB10',
        'RB11', 'RB12']

SUBJECT_ID = 'seeg_demo'
SUBJECTS_DIR = BASE_DIR
EEG_DIR = os.path.join(SUBJECTS_DIR, SUBJECT_ID, 'eeg')
EEG_FILE = os.path.join(EEG_DIR, 'baseline.edf')
ELECTRODE_FILE = os.path.join(EEG_DIR, 'measured_contacts.dat')


@pytest.fixture
def subject_id():
    return SUBJECT_ID


@pytest.fixture
def subjects_dir():
    return SUBJECTS_DIR


@pytest.fixture
def electrode_file():
    return ELECTRODE_FILE


@pytest.fixture
def implant(subject_id, subjects_dir):
    return seeg.Implantation(subject_id, subjects_dir)


@pytest.fixture
def electrode_names(implant):
    return implant.depths.name.values.tolist()


@pytest.fixture
def bads(implant):
    return implant.bads


@pytest.fixture
def freqs():
    return np.arange(10, 220, 3)


@pytest.fixture
def electrodes(implant):
    return implant.contacts


@pytest.fixture
def montage(electrodes):
    mont, __ = seeg.create_montage(electrodes)
    return mont


@pytest.fixture
def raw(electrodes):
    raw, __ = seeg.read_edf(EEG_FILE, electrodes, BADS)
    return raw


@pytest.fixture
def mri():
    return os.path.join(SUBJECTS_DIR, SUBJECT_ID, 'mri', 'T1.mgz')


@pytest.fixture
def eeg():
    eeg = seeg.load_eeg_data(EEG_DIR, ELECTRODE_NAMES, BADS, seizure=1,
                             electrode_file='measured_contacts.dat')[0]
    return eeg


@pytest.fixture
def t_map(eeg, mri):
    t_map = seeg.create_epi_image_map(eeg, mri, low_freq=120, high_freq=200)
    return t_map


@pytest.fixture
def image(mri):
    image = nib.load(mri)
    return image


@pytest.fixture
def Torig(image):
    Torig = image.header.get_vox2ras_tkr()
    return Torig


@pytest.fixture
def T_x_inv(image, Torig):
    inv = npl.inv(image.affine)
    T_x_inv = Torig@inv
    return T_x_inv


@pytest.fixture
def affine(T_x_inv, t_map):
    affine = T_x_inv@t_map.affine
    return affine


HIGH_FREQ = 100
BIAS = 0.25


@pytest.fixture
def synthetic():
    SAMPLES = 5000
    SFREQ = 500
    np.random.seed(10)
    time = np.arange(SAMPLES)/SFREQ
    signal = np.random.rand(SAMPLES) - .5
    for freq in [5, 25]:
        signal += np.sin(freq*2*np.pi*time)

    temp = 2*np.sin(50*2*np.pi*time)
    data = np.zeros((3, SAMPLES))
    data[0, :] = signal[:] * 1e-5
    temp[:1250] = 0
    signal += temp
    data[1, :] = signal[:] * 1e-5
    signal -= temp
    temp[:1500] = 0
    signal += temp
    data[2, :] = signal[:] * 1e-5

    info = mne.create_info(['A1', 'A2', 'A3'], SFREQ, ch_types='seeg',
                           verbose='error')
    synthetic = mne.io.RawArray(data, info)
    return synthetic
