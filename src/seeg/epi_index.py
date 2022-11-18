# -*- coding: utf-8 -*-
""" Calculate epileptogenicity index as per Bartolomei, Brain 131:1818 2008

"""

from typing import Sequence, Tuple
import warnings

from mne.io import Raw
import numpy as np
import numpy.typing as npt
import pandas as pd

from . import utils


def calc_ER(raw: Raw, low: Sequence = (4, 12), high: Sequence = (12, 127),
            window: float = 1, step: float = 0.25) -> npt.NDArray:
    """ Calculate the ratio of beta+gamma energy to theta+alpha energy

    This calculation is done using the Welch PSD in `window` second intervals
    and `step` second steps. The defaults match the defaults for the EI module
    in the [Anywave](https://meg.univ-amu.fr/wiki/AnyWave) package.


    Parameters
    ----------
    raw : MNE Raw
        EEG data
    freqs : ndarray
        array of frequencies
    low : tuple of floats, optional
        range of theta-alpha frequencies, by default (4, 12)
    high : tuple of floats, optional
        range of beta-gamma frequencies, by default (12, 127)
    window : float, optional
        duration in seconds of section for Welch PSD calculation, by default 1
    step : float, optional
        duration in seconds to step for Welch PSD calculation, by default 0.25

    Returns
    -------
    ndarray
        Energy ratio for each channel
    """
    psd = utils.calc_power_welch(raw, window, step)
    numerator = np.sum(psd[:, high[0]:high[1]+1, :], axis=1)
    denominator = np.sum(psd[:, low[0]:low[1]+1, :], axis=1)
    # print(f'psd shape = {psd.shape}')
    return numerator/denominator    # type: ignore


def cusum(ER: npt.NDArray, bias: float = 0.1) -> npt.NDArray:
    """ calculate the Page-Hinkley Cusum

    Parameters
    ----------
    ER : ndarray
        Energy ratio for each channel
    bias : float, optional
        a positive value contributing to the rate of decrease of U_n,
        by default 0.1

    Returns
    -------
    U_n : ndarray
        Page-Hinkley value for each channel
    """
    ER_n = np.cumsum(ER, axis=1)/np.arange(1, 1+ER.shape[-1])
    nu = np.ones_like(ER) * bias
    U = ER - ER_n - nu
    U_n = np.cumsum(U, axis=1)
    return U_n  # type: ignore


def _scan(U_n: npt.NDArray, ch_names: list,
          threshold: float) -> Tuple[bool, pd.DataFrame]:
    columns = ['channel', 'min', 'detection_idx', 'detection_time',
               'alarm_idx', 'alarm_time']
    onsets = pd.DataFrame(dtype=np.double, columns=columns)
    onsets['channel'] = ch_names
    seizure = False
    min_val = np.min(U_n, axis=1)
    min_idx = np.argmin(U_n, axis=1)
    for idx in np.where(min_idx < U_n.shape[-1])[0]:
        # print(idx, min_val[idx], min_idx[idx], U_n[idx, min_idx[idx]])
        test = np.where(U_n[idx, min_idx[idx]:] - min_val[idx] > threshold)[0]
        if len(test) > 0:
            onsets.loc[idx, 'min'] = min_val[idx]
            onsets.loc[idx, 'detection_idx'] = min_idx[idx]
            onsets.loc[idx, 'alarm_idx'] = test[0]+min_idx[idx]
            seizure = True

    return seizure, onsets


def find_onsets(U_n: npt.NDArray, ch_names: list, window: float, step: float,
                H: float, threshold: float = 1) -> pd.DataFrame:
    """ Calculate onset based on the Page-Hinkley CUSUM algorithm

    Parameters
    ----------
    U_n : ndarray
        Page-Hinkley Cusums for each channel
    ch_names : list of strings
        list of channel names
    window : float
        duration in seconds of section for Welch PSD calculation
    step : float
        duration in seconds to step for Welch PSD calculation
    H : float
        number of seconds over which the EI is calculated
    threshold : float , optional
        minimum increase above the local minimun to be considered
        significant, by default 1

    Returns
    -------
    onsets : DataFrame
        dataframe containing detection times and threshold times for each
        channel
    """
    start = window/2
    idx_start = 0
    EI_window = int(H/step)
    idx_end = EI_window
    seizure = False
    while (idx_end < U_n.shape[-1]) and not seizure:
        seizure, onsets = _scan(U_n[:, idx_start:idx_end], ch_names, threshold)
        if seizure:
            idx = np.argmin(onsets.detection_idx)
            idx_start += int(onsets.detection_idx[idx])
            idx_end = int(idx_start + EI_window)
            if idx_end >= U_n.shape[-1]:
                idx_end = int(U_n.shape[-1] - 1)

            ___, onsets = _scan(U_n[:, idx_start:idx_end], ch_names, threshold)
            onsets.detection_idx += idx_start
            onsets.detection_time = start + step*onsets.detection_idx
            onsets.alarm_idx += idx_start
            onsets.alarm_time = start + step*onsets.alarm_idx
        else:
            idx_start += 1
            idx_end += 1

    if not seizure:
        warnings.warn('No seizures identified')
        return None

    return onsets


def calculate_EI(raw: Raw, low: Sequence = (4, 12), high: Sequence = (12, 127),
                 window: float = 1, step: float = 0.25,
                 bias: float = 0.1, threshold: float = 1,
                 tau: float = 1, H: float = 5) -> pd.DataFrame:
    """ Calculate EI for all channels

    Parameters
    ----------
    raw : MNE io.Raw
        eeg data
    low : list of length 2, optional
        minimum and maximum low frequencies, by default (4, 12)
    high : list of length 2, optional
        minimum and maximum high frequencies, by default (12, 127)
    window : float, optional
        duration in seconds of section for Welch PSD calculation, by default 1
    step : float, optional
        duration in seconds to step for Welch PSD calculation, by default 0.25
    bias : float, optional
        bias for the Page-Hinkley CUSUM algorithm, by default 0.1
    threshold : float, optional
        threshold for the Page-Hinkley CUSUM algorithm, by default 1
    tau : float, optional
        tau for the EI calculation, by default 1
    H : float, optional
        duration in seconds over which EI is calculated, by default 5

    Returns
    -------
    onsets: pandas dataframe
        onsets for each channel in raw

    Notes
    -----

    EI is calculated as

    .. math:: EI_i=\frac{1}{N_{di} - N_0 + \tau}\sum_{n=N_{di}}^{N_{di}+H}ER[n],\quad \tau>0

    """
    EI_window = int(H/step)
    stripped = utils.strip_bad_channels(raw)
    ER = calc_ER(stripped, low, high, window, step)
    U_n = cusum(ER, bias)
    onsets = find_onsets(U_n, stripped.ch_names, window, step, H, threshold)
    onsets['EI_raw'] = 0
    N0 = onsets.detection_time.min(skipna=True)

    for i in range(len(stripped.ch_names)):
        N_di = onsets.detection_time[i]
        if not np.isnan(N_di):
            denom = N_di - N0 + tau
            N_di_idx = int(onsets.detection_idx[i])
            end = N_di_idx + EI_window + 1
            if end > ER.shape[-1]:
                onsets.loc[i, 'EI_raw'] = np.sum(ER[i, N_di_idx:])/denom
            else:
                onsets.loc[i, 'EI_raw'] = np.sum(ER[i, N_di_idx:end])/denom

    EI_max = onsets['EI_raw'].max()
    onsets['EI'] = onsets.EI_raw/EI_max
    return onsets
