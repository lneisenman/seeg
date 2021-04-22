# -*- coding: utf-8 -*-

""" Calculate line length as per Esteller 2001
    http://ieeex plore.ieee.org/docum ent/10205 45/.

"""

import numpy as np

from .epi_index import cusum, find_onsets


def line_length(raw, window=1, step=0.25):
    '''
    calculate line length is segments of width window with time steps
    of time step

    Parameters
    ----------
    raw : MNE Raw
        EEG data
    window : float, optional
        window duration in seconds, by default 1
    step : float, optional
        step in seconds, by default 0.25

    Returns
    -------
    ndarray
        line length values for each channel
    ndarray
        standard deviation of the line length in the first interval
    '''
    data = raw.get_data()
    window_pnts = int(raw.info['sfreq']*window)
    step_pnts = int(step*raw.info['sfreq'])
    time_pnts = int(1+np.ceil((raw.n_times-window_pnts)/step_pnts))
    line_len = np.zeros((data.shape[0], time_pnts))
    ll = np.abs(np.diff(data))
    sd1 = np.std(ll[:, :window_pnts], axis=-1)
    for idx in range(time_pnts-1):
        start = idx*step_pnts
        end = start+window_pnts
        line_len[:, idx] = np.mean(ll[:, start:end], axis=-1)

    idx = time_pnts-1
    start = idx*step_pnts
    line_len[:, idx] = np.mean(ll[:, start:], axis=-1)

    return line_len, sd1


def ll_detect_seizure(raw, window=1, step=0.25, threshold=1):
    '''
    Find the seizure onset in each channel defined as the first time that
    line length is threshold standard deviations above the sd of the first
    segment

    Parameters
    ----------
    raw : MNE Raw
        EEG data
    window : float, optional
        window duration in seconds, by default 1
    step : float, optional
        step in seconds, by default 0.25
    threshold: float, optional
        number of standard deviations to use for the seizure threshold,
        by default 1

    Returns
    -------
    ndarray
        seizure onset time for each channel of raw
    ndarray
        line length values for each channel
    ndarray
        standard deviation of the line length in the first interval
    '''
    ll, sd = line_length(raw, window, step)
    thresholds = sd*threshold + ll[:, 0]
    sz = np.zeros_like(thresholds)
    for i, thresh in enumerate(thresholds):
        test = np.where(ll[i, :] > thresh)[0]
        if len(test) > 0:
            sz[i] = window/2 + (step*test[0])
        else:
            sz[i] = np.nan

    return sz + raw.times[0], ll, sd


def line_length_EI(raw, window=1, step=0.25, tau=1, H=5):
    '''
    Find the seizure onset in each channel defined as the first time that
    line length is threshold standard deviations above the sd of the first
    segment

    Parameters
    ----------
    raw : MNE Raw
        EEG data
    window : float, optional
        window duration in seconds, by default 1
    step : float, optional
        step in seconds, by default 0.25
    tau: float, optional
       tau for the EI calculation, by default 1
    H : float, optional
        time in seconds of the duration of the EI calculation, by default 5

    Returns
    -------
    ndarray
        seizure onset time for each channel of raw
    ndarray
        line length values for each channel
    ndarray
        standard deviation of the line length in the first interval
    '''
    ___, ll, sd = ll_detect_seizure(raw, window, step)
    bias = np.max(sd)/5
    threshold = bias*5
    U_n = cusum(ll, bias)
    onsets = find_onsets(U_n, raw.ch_names, threshold)
    onsets['LLEI_raw'] = 0
    N0 = onsets.detection_time.min(skipna=True)
    for i in range(len(raw.ch_names)):
        N_di = onsets.detection_time[i]
        if not np.isnan(N_di):
            denom = N_di - N0 + tau
            N_di_idx = int(onsets.detection_idx[i])
            end = N_di_idx + int(H/step)
            if end > ll.shape[-1]:
                onsets.loc[i, 'LLEI_raw'] = np.sum(ll[i, N_di_idx:])/denom
            else:
                onsets.loc[i, 'LLEI_raw'] = np.sum(ll[i, N_di_idx:end])/denom

    EI_max = onsets['LLEI_raw'].max()
    onsets['LLEI'] = onsets.LLEI_raw/EI_max
    return onsets
