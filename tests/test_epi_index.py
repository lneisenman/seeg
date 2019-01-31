# -*- coding: utf-8 -*-

import matplotlib.pyplot as plt
import mne
import numpy as np
import pandas as pd
import pytest

import seeg


# @pytest.fixture
def raw():
    SAMPLES = 2000
    SFREQ = 100
    np.random.seed(10)
    time = np.arange(SAMPLES)/SFREQ
    signal = np.random.rand(SAMPLES) - .5
    for freq in [5, 25]:
        signal += np.sin(freq*2*np.pi*time)

    temp = np.sin(25*2*np.pi*time)
    data = np.zeros((3, SAMPLES))
    data[0, :] = signal[:] * 1e-5
    temp[:500] = 0
    signal += temp
    data[1, :] = signal[:] * 1e-5
    signal -= temp
    temp[:600] = 0
    signal += temp
    data[2, :] = signal[:] * 1e-5

    info = mne.create_info(['A1', 'A2', 'A3'], SFREQ, ch_types='seeg',
                           verbose='error')
    raw = mne.io.RawArray(data, info)
    return raw


def test_raw(raw):
    ER = seeg.calc_ER(raw, np.arange(1, 51))
    U_n = seeg.cusum(ER)
    for i in range(ER.shape[0]):
        plt.plot(ER[i, :])

    plt.figure()
    for i in range(U_n.shape[0]):
        plt.plot(U_n[i, :])

    raw.plot(scalings='auto')
    plt.show()


def test_find_onsets(raw):
    ER = seeg.calc_ER(raw, np.arange(1, 51))
    U_n = seeg.cusum(ER)
    onsets = seeg.find_onsets(U_n, raw.info['sfreq'], raw.ch_names)
    np.testing.assert_allclose(onsets['detection'], [np.nan, 493, 592])
    np.testing.assert_allclose(onsets['alarm'], [np.nan, 496, 595])


def test_calculate_EI(raw):
    print(seeg.calcualte_EI(raw, np.arange(1, 51)))


if __name__ == '__main__':
    test_calculate_EI(raw())
