# -*- coding: utf-8 -*-
""" Utility classes and functions

"""

from typing import Sequence

import matplotlib as mpl
import mne
from mne.io import Raw
import numpy as np
import numpy.typing as npt


def calc_power_multi(raw: Raw, window: float = 1,
                     step: float = 0.25) -> npt.NDArray:
    """ Calculate power in each channel using multitapers in sections,
        analagous to the Welch PSD method

    Parameters
    ----------
    raw : MNE Raw
        EEG data
    window : int
        number of seconds per segment
    step : float
        fraction of window to advance

    Returns
    -------
    density : ndarray
        power data
    """
    start = 0
    sfreq = int(raw.info['sfreq'])
    end = int(sfreq*window)
    pnts_per_step = int(sfreq*window*step)
    last_seg = (raw.n_times/sfreq) - (window/2)
    num_segs = len(np.arange(window/2, last_seg+(step*window/2), step*window))
    if sfreq % (1/step) > 0:
        num_segs += 1
    # if raw.n_times/sfreq - int(raw.n_times/sfreq) >= 1/sfreq:
    #     num_segs += 1

    density = np.zeros((len(raw.ch_names), int(sfreq/2), num_segs))
    i = 0
    data = raw.get_data()
    for i in range(num_segs-1):
        psd, ___ = mne.time_frequency.psd_array_multitaper(data[:, start:end],
                                                           sfreq,
                                                           verbose=False)
        # print(f'i = {i} and start = {start}')
        # print(f'density.shape = {density.shape}')
        # print(f'psd.shape = {psd.shape}')
        density[:, :, i] = psd[:, 1:]
        start += pnts_per_step
        end += pnts_per_step

    i = num_segs - 1
    psd, ___ = mne.time_frequency.psd_array_multitaper(data[:, start:], sfreq,
                                                       verbose=False)
    # print(f'i = {i} and start = {start}')
    # print(f'density.shape = {density.shape}')
    # print(f'psd.shape = {psd.shape}')
    # print(f'pnts_per_step = {pnts_per_step} and time = {raw.n_times}')
    density[:, :(psd.shape[-1]-1), i] = psd[:, 1:]

    return density


def calc_power_welch(raw: Raw, window: float = 1,
                     step: float = 0.25) -> npt.NDArray:
    """ Calculate power in each channel using the Welch method

        Parameters
        ----------
        raw : MNE Raw
            EEG data
        window : int
            number of seconds per segment
        step : float
            fraction of window to advance

        Returns
        -------
        density : ndarray
            power data
    """
    n_per_seg = int(raw.info['sfreq']*window)
    fmax = raw.info['sfreq']//2
    overlap = int((1 - step)*raw.info['sfreq'] + 0.5)
    # print(raw.info['sfreq'], step, 1-step)
    # print((1 - step)*raw.info['sfreq'])
    # print(n_per_seg, overlap)
    # assert 1 == 0
    psd, ___ = mne.time_frequency.psd_welch(raw, fmin=0, fmax=fmax,
                                            n_fft=n_per_seg,
                                            n_per_seg=n_per_seg,
                                            n_overlap=overlap, average=None)
    return psd  # type: ignore


def map_colors(values: Sequence,
               cmap: mpl.colors.Colormap = 'cold_hot') -> Sequence:
    vmin = np.min(values)
    vmax = np.max(values)
    if vmin < 0:
        if abs(vmin) > vmax:
            vmax = abs(vmin)
        else:
            vmin = -vmax

    vmin *= 1.1
    vmax *= 1.1
    norm = mpl.colors.Normalize(vmin, vmax)
    color_map = mpl.cm.get_cmap(cmap)
    return color_map(norm(values))  # type: ignore
