from __future__ import (print_function, division, absolute_import,
                        unicode_literals)

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


def plot_eeg(eeg, depth):
    channels = [channel for channel in eeg.ch_names if depth in channel]
    electrode = eeg.pick_channels(channels)
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
        ax[i].set_ylabel(channels[i], labelpad=10, rotation=0)

    ax[-1].spines['bottom'].set_visible(True)
    ax[-1].tick_params(bottom=True)
    ax[-1].set_xlabel('time (s)')
    return fig


def plot_power(power, ch_names, depth):
    rows = [i for i in range(len(ch_names)) if depth in ch_names[i]]
    labels = [ch_names[row] for row in rows]
    fig, ax = plt.subplots(len(rows), 1, sharex=True)
    for i, row in enumerate(rows):
        ax[i].imshow(power[0, row, :, :]) #, cmap='gin')
        ax[i].set_ylabel(labels[i], labelpad=25, rotation=0)
        ax[i].yaxis.set_major_locator(plt.NullLocator())
        ax[i].tick_params(bottom=False, left=False)

    ax[-1].spines['bottom'].set_visible(True)
    ax[-1].tick_params(bottom=True)
    ax[-1].set_xlabel('time (s)')
    return fig


def calc_z_scores(baseline, seizure):
    '''
    This function is meant to generate the figures shown in the  Brainstorm
    demo used to select the 120-200 Hz frequency band. It should also
    be similar to panel 2 in figure 1 in David et al 2011.

    This function will compute a z-score for each value of the seizure power
    spectrum using the mean and sd of the control power spectrum at each
    frequency. In the demo, the power spectrum is calculated for the 1st
    10 seconds of all three seizures and then averaged. Controls are
    similarly averaged
    '''

    mean = np.mean(baseline, 1)
    sd = np.std(baseline, 1)
    z_scores = (seizure - mean)/sd
    '''
    for index, electrode in enumerate(seizures[0]['seizure']['bipolar'].ch_names):
        baseline = np.zeros_like(seizures[0]['baseline']['power'][0, index, :, :])
        z_score = np.zeros_like(seizures[0]['seizure']['power'][0, index, :, :])
        for seizure in seizures:
            baseline += seizure['baseline']['power'][0, index, :, :]
            z_score += seizure['seizure']['power'][0, index, :, :]

        num = len(seizures)
        baseline /= num
        z_score /= num
        baseline = compress_data(baseline, 512, 1)
        z_score = compress_data(z_score, 512, 1)
        for i in range(z_score.shape[1]):
            z_score[:, i] -= mean
            z_score[:, i] /= sd

        z_scores[electrode] = z_score
    '''
    return z_scores


def plot_z_scores(times, freqs, z_scores, ch_names, depth):
    rows = [i for i in range(len(ch_names)) if depth in ch_names[i]]
    labels = [ch_names[row] for row in rows]
    fig, ax = plt.subplots(len(rows), 1, sharex=True)
    for i, row in enumerate(rows):
        im = ax[i].pcolormesh(times, freqs, z_scores[0, row, :, :], cmap='hot')
        ax[i].set_ylabel(labels[i], labelpad=25, rotation=0)
        ax[i].yaxis.set_major_locator(plt.NullLocator())
        ax[i].tick_params(bottom=False, left=False)

    ax[-1].spines['bottom'].set_visible(True)
    ax[-1].tick_params(bottom=True)
    ax[-1].set_xlabel('time (s)')
    #cbar_ax = fig.add_axes([0.85, 0.15, 0.05, 0.7])
    fig.colorbar(im, ax=ax)
    return fig
