""" Calculate line length as per Esteller 2001
    http://ieeex plore.ieee.org/docum ent/10205 45/.

"""

import mne
import numpy as np


def line_length(raw):
    '''
    calculate line length in 1 second windows with 75% overlap

    Parameters
    ----------
    raw : MNE Raw
        EEG data

    Returns
    -------
    ndarray
        line length values for each channel
    ndarray
        standard deviation of the line length in the first interval
    '''
    data = raw.get_data()
    window = int(raw.info['sfreq'])
    step = int(0.25*window)
    time_points = 4*(int(raw.n_times/window) - 1) + 1
    if raw.n_times%window > 0:
        print('uneven')
        time_points += np.ceil(4*(raw.n_times%window))
        print(f'time_pointes = {time_points}')

    line_len = np.zeros((data.shape[0], time_points))
    ll = np.abs(np.diff(data))
    sd1 = np.std(ll[:, :window], axis=-1)
    idx = 0
    start = 0
    end = start + window
    while end < ll.shape[-1]:
        line_len[:, idx] = np.mean(ll[:, start:end], axis=-1)
        idx += 1
        start += step
        end += step

    print(idx, start, end)
    if start < ll.shape[-1]-1:
        line_len[:, idx] = np.mean(ll[:, start:], axis=-1)

    return line_len, sd1
