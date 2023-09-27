# -*- coding: utf-8 -*-
""" Utility classes and functions

"""

import os
from typing import Sequence

import matplotlib as mpl
import mne
from mne.io import Raw
import nibabel as nib
from nilearn.plotting.cm import cold_hot
import numpy as np
import numpy.typing as npt
import pandas as pd
# import scipy as sp
# import scipy.signal as sps


def strip_bad_channels(raw: Raw) -> Raw:
    """ Remove any channels in raw.info['bads']

    Parameters
    ----------
    raw : MNE Raw
        EEG data

    Returns
    -------
    stripped : Raw
        EEG data with bad channels removed
    """
    bads = raw.info['bads']
    if len(bads) == 0:
        return raw

    stripped = raw.copy().drop_channels(bads)
    return stripped


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
    sfreq = int(raw.info['sfreq']*window)
    overlap = int((1 - step)*raw.info['sfreq'] + 0.5)
    spectrum = raw.compute_psd(method='welch', n_fft=sfreq,
                               n_per_seg=sfreq, n_overlap=overlap,
                               average=None)
    return spectrum.get_data()  # type: ignore


def map_colors(values: Sequence,
               color_map: mpl.colors.Colormap = cold_hot) -> Sequence:
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
    return color_map(norm(values))  # type: ignore


def localize_electrodes(montage: mne.channels.DigMontage, SUBJECT_ID: str,
                        SUBJECTS_DIR: str) -> pd.DataFrame:
    fn = os.path.join(SUBJECTS_DIR, SUBJECT_ID,
                      r'mri/aparc+aseg.mgz')
    aseg = nib.load(fn)
    aseg_data = np.array(aseg.dataobj)      # type: ignore
    ids, colors = mne.read_freesurfer_lut()
    lut = {v: k for k, v in ids.items()}
    positions = montage.get_positions()
    mri_file = os.path.join(SUBJECTS_DIR, SUBJECT_ID, r'mri/T1.mgz')
    mri = nib.load(mri_file)
    inv = np.linalg.inv(mri.affine)         # type: ignore
    data = list()
    for ch in montage.ch_names:
        x, y, z = positions['ch_pos'][ch] * 1000
        i, j, k = np.round(mne.transforms.apply_trans(inv, (x, y, z))).astype(int)
        test = aseg_data[i, j, k]
        R, G, B, _ = colors[lut[test]]/256
        data.append([ch, x, y, z, i, j, k, test, lut[test], R, G, B])

    return pd.DataFrame(data, columns=['channel', 'x', 'y', 'z', 'i', 'j', 'k',
                                       'value', 'label', 'R', 'G', 'B'])

