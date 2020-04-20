# -*- coding: utf-8 -*-
""" Code for generating plots

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

from . import gin


def plot_eeg(eeg, depth, label_color='black'):
    """ Plots eeg data for a given depth electrode

    Arguments:
        eeg : MNE raw object
            contains EEG data to plot
        depth : string
            name of depth electrode

    Keyword Arguments:
        label_color : string
            color of axis labels (default: {'black'})

    Returns:
        matplotlib figure
    """
    channels = [channel for channel in eeg.ch_names if depth in channel]
    electrode = eeg.copy().pick_channels(channels)
    data, times = electrode.get_data(return_times=True)
    rows = len(channels)
    fig, ax = plt.subplots(rows, 1, sharex=True)
    for i in range(rows):
        ax[i].plot(times, data[i, :])
        ax[i].spines['right'].set_visible(False)
        ax[i].spines['top'].set_visible(False)
        ax[i].spines['bottom'].set_visible(False)
        ax[i].spines['left'].set_visible(False)
        ax[i].yaxis.set_major_locator(plt.NullLocator())
        ax[i].tick_params(bottom=False, left=False)
        ax[i].set_ylabel(channels[i], labelpad=10, rotation=0,
                         color=label_color)

    ax[-1].spines['bottom'].set_visible(True)
    ax[-1].tick_params(bottom=True, colors=label_color)
    ax[-1].set_xlabel('time (s)', color=label_color)
    return fig


def plot_power(power, ch_names, depth, label_color='black'):
    """ plot EEG power

    Parameters
    ----------
    power : ndarray
        array of EEG power
    ch_names : list
        list of channel names
    depth : string
        name of depth electrode whose power is being plotted
    label_color : str, optional
        color of axis labels, by default 'black'

    Returns
    -------
    matplotlib figure
        plot of EEG power
    """

    rows = [i for i in range(len(ch_names)) if depth in ch_names[i]]
    labels = [ch_names[row] for row in rows]
    fig, ax = plt.subplots(len(rows), 1, sharex=True)
    for i, row in enumerate(rows):
        ax[i].imshow(power[0, row, :, :]) #, cmap='gin')
        ax[i].set_ylabel(labels[i], labelpad=25, rotation=0, color=label_color)
        ax[i].yaxis.set_major_locator(plt.NullLocator())
        ax[i].tick_params(bottom=False, left=False)

    ax[-1].spines['bottom'].set_visible(True)
    ax[-1].tick_params(bottom=True, colors=label_color)
    ax[-1].set_xlabel('time (s)', color=label_color)
    return fig


def calc_z_scores(baseline, seizure):
    """ This function is meant to generate the figures shown in the  Brainstorm
    demo used to select the 120-200 Hz frequency band. It should also
    be similar to panel 2 in figure 1 in David et al 2011.

    This function will compute a z-score for each value of the seizure power
    spectrum using the mean and sd of the control power spectrum at each
    frequency. In the demo, the power spectrum is calculated for the 1st
    10 seconds of all three seizures and then averaged. Controls are
    similarly averaged

    Parameters
    ----------
    baseline : ndarray
        power spectrum of baseline EEG
    seizure : ndarray
        power spectrum of seizure EEG

    Returns
    -------
    ndarray
        seizure power spectrum scaled to a z-score by baseline power spectrum
        mean and SD
    """

    mean = np.mean(baseline, 1)
    sd = np.std(baseline, 1)
    z_scores = (seizure - mean)/sd
    return z_scores


def plot_z_scores(times, freqs, z_scores, ch_names, depth,
                  label_color='black'):
    """ Plots Z-scores

    Parameters
    ----------
    times : ndarray
        x-axis of plot
    freqs : ndarray
        y-axis of plot
    z_scores : ndarray
        array of Z-scores being plotted by color code
    ch_names : list
        list of channel names
    depth : string
        name of depth being plotted
    label_color : str, optional
        color of axis labels, by default 'black'

    Returns
    -------
    matplotlib figure
        color coded Z-score plot
    """

    rows = [i for i in range(len(ch_names)) if depth in ch_names[i]]
    labels = [ch_names[row] for row in rows]
    fig, ax = plt.subplots(len(rows), 1, sharex=True)
    for i, row in enumerate(rows):
        im = ax[i].pcolormesh(times, freqs, z_scores[0, row, :, :], cmap='hot')
        ax[i].set_ylabel(labels[i], labelpad=25, rotation=0, color=label_color)
        ax[i].yaxis.set_major_locator(plt.NullLocator())
        ax[i].tick_params(bottom=False, left=False)

    ax[-1].spines['bottom'].set_visible(True)
    ax[-1].tick_params(bottom=True, colors=label_color)
    ax[-1].set_xlabel('time (s)', color=label_color)
    cb = fig.colorbar(im, ax=ax)
    cb.ax.tick_params(axis='y', colors=label_color)
    return fig
