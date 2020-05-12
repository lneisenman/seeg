# -*- coding: utf-8 -*-
""" Calculate epileptogenicity index as per Bartolomei, Brain 131:1818 2008 

"""

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

from . import utils


def calc_ER(raw, freqs, low=(4, 12), high=(13, 100)):
    '''Calculate the ratio of beta+gamma to theta+alpha'''
    power = utils.calc_power(raw, freqs, freqs)
    numerator = np.sum(power[0, :, high[0]:high[1], :], axis=1)
    denominator = np.sum(power[0, :, low[0]:low[1], :], axis=1)
    return numerator/denominator


def cusum(data, bias=0.1):
    ER_n = np.cumsum(data, axis=1)/np.arange(1, 1+data.shape[-1])
    U = data - ER_n - bias
    U_n = np.cumsum(U, axis=1)
    return U_n


def find_onsets(U_n, sfreq, ch_names, threshold=1, H=5, step_size=1):
    window_len = int(H*sfreq)
    step = int(step_size*sfreq)
    seizure = False
    index = int(0)
    end = U_n.shape[-1]
    onsets = pd.DataFrame(index=ch_names, dtype=np.double,
                          columns=['min', 'detection', 'alarm'])
    while not seizure and (index < end - window_len):
        limit = index + step
        local_min = np.min(U_n[:, index:limit], axis=1)
        min_idx = np.argmin(U_n[:, index:limit], axis=1) + index
        for i, ch in enumerate(ch_names):
            test = U_n[i, min_idx[i]:limit] - local_min[i] - threshold
            idx = np.where(test > 0)
            if (len(idx[0]) > 0):
                onsets.loc[ch, 'min'] = local_min[i]
                onsets.loc[ch, 'detection'] = (min_idx[i]/sfreq)
                onsets.loc[ch, 'alarm'] = (min_idx[i] + idx[0][0])/sfreq
                seizure = True

        index += step

    if seizure:
        detection = int(onsets['detection'].min(skipna=True) * sfreq)
        channel = onsets['detection'].idxmin(skipna=True)
        limit = int(detection + window_len)
        if limit > end:
            limit = end
            warnings.warn('Seizure detected at the end of the data. \
                           EI calculation may not be correct')
        for i, ch in enumerate(ch_names):
            if (ch != channel):
                new_min = np.min(U_n[i, detection:limit])
                new_idx = \
                    np.argmin(U_n[i, detection:limit]) + detection
                test = U_n[i, new_idx:limit] - new_min - threshold
                idx = np.where(test > 0)
                if (len(idx[0]) > 0):
                    onsets.loc[ch, 'min'] = new_min
                    onsets.loc[ch, 'detection'] = new_idx/sfreq
                    onsets.loc[ch, 'alarm'] = (new_idx + idx[0][0])/sfreq

    elif (index < window_len):
        local_min = np.min(U_n[:, index:], axis=1)
        min_idx = np.argmin(U_n[:, index:], axis=1) + index
        for i, ch in enumerate(ch_names):
            test = U_n[i, min_idx[i]:] - local_min[i] - threshold
            idx = np.where(test > 0)
            if (len(idx[0]) > 0):
                if not seizure:
                    warnings.warn('Seizure detected at the end of the data. \
                                   EI calculation may not be correct')
                onsets.loc[ch, 'min'] = local_min[i]
                onsets.loc[ch, 'detection'] = min_idx[i]/sfreq
                onsets.loc[ch, 'alarm'] = (min_idx[i] + idx[0][0])/sfreq
                seizure = True

    if not seizure:
        warnings.warn('No seizures identified')

    return onsets


def calculate_EI(raw, freqs, bias=0.1, threshold=1, tau=1, H=5):
    """ Calculate EI for all channels

    Parameters
    ----------
    raw : MNE io.Raw
        eeg data
    freqs : ndarray
        array of frequencies
    bias : float, optional
        bias for the Page-Hinkley CUSUM algorithm, by default 0.1
    threshold : int, optional
        threshold for the Page-Hinkley CUSUM algorithm, by default 1
    tau : int, optional
        tau for the EI calculation, by default 1
    H : int, optional
        duration over which ER is calculated, by default 5

    Returns
    -------
    onsets: pandas dataframe
        onsets for each channel in raw

    Notes
    -----

    EI is calculated as

    .. math:: EI_i=\frac{1}{N_{di} - N_0 + \tau}\sum_{n=N_{di}}^{N_{di}+H}ER[n],\quad \tau>0

    """
    ER = calc_ER(raw, freqs)
    U_n = cusum(ER, bias)
    onsets = find_onsets(U_n, raw.info['sfreq'], raw.ch_names, threshold, H)
    onsets['EI'] = 0
    N0 = int(onsets['detection'].min(skipna=True))
    H_samples = int(H * raw.info['sfreq'])
    recording_end = U_n.shape[-1]
    for i, ch in enumerate(raw.ch_names):
        N_di = onsets.loc[ch, 'detection']
        if not np.isnan(N_di):
            N_di = int(N_di)
            denom = ((N_di - N0)/raw.info['sfreq']) + tau
            end = N_di + H_samples
            if end > recording_end:
                onsets.loc[ch, 'EI'] = np.sum(ER[i, N_di:])/denom
            else:
                onsets.loc[ch, 'EI'] = np.sum(ER[i, N_di:end])/denom

    EI_max = onsets['EI'].max()
    onsets.loc[:, 'EI'] = onsets.loc[:, 'EI']/EI_max
    return onsets
