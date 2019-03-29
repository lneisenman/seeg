# -*- coding: utf-8 -*-

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
import pandas as pd
from scipy import stats as sps
import warnings

from . import gin
from . import source_image as srci
from . import utils


def calc_ER(raw, freqs):
    '''Calculate the ratio of beta+gamma to alpha+theta'''
    power = utils.calc_power(raw, freqs, freqs)
    numerator = np.sum(power[0, :, 13:, :], axis=1)
    denominator = np.sum(power[0, :, 4:12, :], axis=1)
    return numerator/denominator


def cusum(data, bias=1):
    ER_n = np.cumsum(data, axis=1)/np.arange(1, 1+data.shape[-1])
    U = data - ER_n - bias
    U_n = np.cumsum(U, axis=1)
    return U_n


def find_onsets(U_n, sfreq, ch_names, threshold=1):
    window_len = int(5*sfreq)
    step = int(window_len/2)
    seizure = False
    index = int(0)
    end = U_n.shape[-1]
    onsets = pd.DataFrame(index=ch_names, dtype=np.double,
                          columns=['min', 'detection', 'alarm'])
    while not seizure and (index < end - window_len):
        limit = index + window_len
        local_min = np.min(U_n[:, index:limit], axis=1)
        min_idx = np.argmin(U_n[:, index:limit], axis=1) + index
        for i, ch in enumerate(ch_names):
            test = U_n[i, min_idx[i]:limit] - local_min[i] - threshold
            idx = np.where(test > 0)
            if (len(idx[0]) > 0):
                onsets.loc[ch, 'min'] = local_min[i]
                onsets.loc[ch, 'detection'] = min_idx[i]
                onsets.loc[ch, 'alarm'] = min_idx[i] + idx[0][0]
                seizure = True

        index += step

    if seizure:
        detection = onsets.detection.min(skipna=True)
        channel = onsets['detection'].idxmin(skipna=True)
        index -= step
        limit = int(detection+step)
        for i, ch in enumerate(ch_names):
            if (ch != channel):
                new_min = np.min(U_n[i, min_idx[i]:limit])
                new_idx = \
                    np.argmin(U_n[i, min_idx[i]:limit]) + min_idx[i]
                test = U_n[i, new_idx:limit] - new_min - threshold
                idx = np.where(test > 0)
                if (len(idx[0]) > 0):
                    onsets.loc[ch, 'min'] = new_min
                    onsets.loc[ch, 'detection'] = new_idx
                    onsets.loc[ch, 'alarm'] = new_idx + idx[0][0]

    else:
        local_min = np.min(U_n[:, index:], axis=1)
        min_idx = np.argmin(U_n[:, index:], axis=1) + index
        for i, ch in enumerate(ch_names):
            test = U_n[i, min_idx[i]:] - local_min[i] - threshold
            idx = np.where(test > 0)
            if (len(idx[0]) > 0):
                onsets.loc[ch, 'min'] = local_min[i]
                onsets.loc[ch, 'detection'] = min_idx[i]
                onsets.loc[ch, 'alarm'] = min_idx[i] + idx[0][0]
                seizure = True

    if not seizure:
        warnings.warn('No seizures identified')

    return onsets


def calculate_EI(raw, freqs, bias=1, threshold=1, tau=1, H=5):
    ER = calc_ER(raw, freqs)
    U_n = cusum(ER, bias)
    onsets = find_onsets(U_n, raw.info['sfreq'], raw.ch_names, threshold)
    onsets['EI'] = 0
    N0 = int(onsets.detection.min(skipna=True))
    H_samples = int(H * raw.info['sfreq'])
    for i, ch in enumerate(raw.ch_names):
        N_di = onsets.loc[ch, 'detection']
        if not np.isnan(N_di):
            N_di = int(N_di)
            denom = ((N_di - N0)/raw.info['sfreq']) + 1
        else:
            N_di = N0 + 2*H_samples
            denom = ((N_di - N0)/raw.info['sfreq']) + 1
            N_di = N0

        onsets.loc[ch, 'EI'] = np.sum(ER[i, N_di:(N_di+H_samples)])/denom

    EI_max = onsets.EI.max()
    onsets.loc[:, 'EI'] = onsets.loc[:, 'EI']/EI_max
    return onsets
