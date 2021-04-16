""" Calculate line length as per Esteller 2001
    http://ieeex plore.ieee.org/docum ent/10205 45/.

"""

import mne
import numpy as np


def line_length(raw, window=1, step=0.25):
    '''
    calculate line length is segments of width window with time steps
    of time step

    Parameters
    ----------
    raw : MNE Raw
        EEG data
    window : float
        window duration in seconds
    step : float
        step in seconds

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
    window : float
        window duration in seconds
    step : float
        step in seconds
    threshold: float
        number of standard deviations to use for the seizure threshold

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
