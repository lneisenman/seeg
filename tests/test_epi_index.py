# -*- coding: utf-8 -*-

import matplotlib.pyplot as plt
import mne
import numpy as np
import pandas as pd
import pytest

import seeg


HIGH_FREQ = 100
BIAS = 0.25


@pytest.fixture
def raw():
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
    raw = mne.io.RawArray(data, info)
    return raw


def test_find_onsets(raw):
    ER = seeg.calc_ER(raw)
    U_n = seeg.cusum(ER, BIAS)
    print(U_n.shape)
    onsets = seeg.find_onsets(U_n, raw.ch_names)
    np.testing.assert_allclose(onsets['detection_time'], [np.nan, 2, 2.5])
    np.testing.assert_allclose(onsets['alarm_time'], [np.nan, 2.5, 3.])


def test_calculate_EI(raw):
    onsets = seeg.calculate_EI(raw)
    np.testing.assert_allclose(onsets['EI'], [0, 1., 0.667],
                               rtol=1e-3)
