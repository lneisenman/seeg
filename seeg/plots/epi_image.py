# -*- coding: utf-8 -*-
""" Functions to create an epileptogenicity image analagous to that from the
    Brainstorm demo of David et al. 2011

"""

import matplotlib.pyplot as plt
import nibabel as nib
from nibabel.affines import apply_affine
from nibabel import processing as nbp
import nilearn
from nilearn.input_data import NiftiMasker
from nilearn.mass_univariate import permuted_ols
from nilearn.plotting import plot_stat_map
from nilearn.plotting.cm import cold_hot
import numpy as np
import numpy.linalg as npl
from vispy.color import Colormap

from . import gin
from .. import utils


class EpiImage:

    def __init__(self, eeg, mri, low_freq=120, high_freq=200,
                 seiz_delay=0, step=3, num_steps=1):
        self.mri = mri
        self.low_freq = low_freq
        self.high_freq = high_freq
        self.seiz_delay = seiz_delay
        self.step = step
        self.t_maps = list()
        for i in range(num_steps):
            self.t_maps.append(create_source_image_map(eeg, mri,
                                                       low_freq, high_freq,
                                                       seiz_delay+(i*step)))

    def plot(self, cut_coords=None, threshold=2):
        image = plot_source_image_map(self.t_maps[0], self.mri, cut_coords,
                                      threshold)


def compress_data(data, old_freq, new_freq):
    """ compress data from old frequency to new frequency

    Parameters
    ----------
    data : ndarray
        2-d array of data
    old_freq : float
        old frequency
    new_freq : float
        new frequency

    Returns
    -------
    new : ndarray
        compressed data
    """

    num = round(0.5+(data.shape[1]*(new_freq/old_freq)))
    new = np.zeros((data.shape[0], num))
    ratio = old_freq/new_freq
    for i in range(num):
        start = int(i*ratio)
        end = int((i+1)*ratio)
        new[:, i] = np.mean(data[:, start:end], 1)

    return new


def ave_power_over_freq_band(eeg, low=120, high=200):
    """ returns the average power between the specified low and high
        frequencies

    Parameters
    ----------
    eeg : EEG | dict
        eeg data
    low : int, optional
        low frequency, by default 120
    high : int, optional
        high frequency, by default 200

    Returns
    -------
    baseline_ave_power : ndarray
        average power of baseline eeg in the specified range
    seizure_ave_power : ndarray
        average power of seizure eeg in the specified range
    """

    sfreq = eeg['baseline']['eeg'].info['sfreq']
    freqs = np.arange(sfreq//2)
    baseline_power = eeg['baseline']['power']
    shape = baseline_power.shape
    baseline_ave_power = np.zeros((shape[0], shape[1]))
    seizure_power = eeg['seizure']['power']
    shape = seizure_power.shape
    seizure_ave_power = np.zeros((shape[0], shape[1]))
    freq_index = np.where(np.logical_and(freqs >= low, freqs <= high))

    for i in range(len(eeg['seizure']['bipolar'].ch_names)):
        baseline_ave_power[i, :] = \
            np.mean(baseline_power[i, :, freq_index], 1)
        seizure_ave_power[i, :] = \
            np.mean(seizure_power[i, :, freq_index], 1)

    return baseline_ave_power, seizure_ave_power


def extract_power(eeg, D=3, dt=0.25, start=0):
    """ extract power vaules for image

    Parameters
    ----------
    seizure : EEG | dict
        eeg data
    D : int, optional
        epoch duration, by default 3
    dt : float, optional
        time step (seconds), by default 0.2
    start : int, optional
        time to start, by default 0

    Returns
    -------
    baseline_ex_power : ndarray
        baseline power
    seizure_ex_power : ndarray
        seizure power
    """
    assert int(D/dt)*dt == D
    num_steps = int(D/dt)
    seiz = eeg['seizure']['eeg']
    sfreq = seiz.info['sfreq']
    onset = seiz.annotations.onset[0] - (seiz.first_samp/sfreq) + start
    first = int(onset/dt)
    baseline_ex_power = eeg['baseline']['ave_power'][:, :num_steps]
    seizure_ex_power = eeg['seizure']['ave_power'][:, first:first+num_steps]
    return baseline_ex_power, seizure_ex_power


def create_volumes(eeg, mri):
    """ create Nifti1Images for the baseline and seizure data

    Parameters
    ----------
    seizure : EEG | dict
        eeg data
    mri : string
        path to MRI file

    Returns
    -------
    baseline_img : Nifti1Image
        image for baseline eeg data
    seizure_img : Nifti1Image
        image for seizure eeg data
    """
    img = nib.load(mri)
    resized = nbp.resample_to_output(img, voxel_sizes=3)
    shape = resized.dataobj.shape
    shape = np.append(shape, eeg['seizure']['ex_power'].shape[-1])
    affine = resized.affine.copy()
    baseline = np.zeros(shape)
    baseline_img = nib.Nifti1Image(baseline, affine)

    seiz = np.zeros(shape)
    seizure_img = nib.Nifti1Image(seiz, affine)
    return baseline_img, seizure_img


def voxel_coords(mri_coords, inverse):
    """ convert mri coordiantes to voxel coordinates in integers using the
        inverse affine

    Parameters
    ----------
    mri_coords : tuple
        MRI coordinates
    inverse : affine
        affine to convert to voxel coordiantes from MRI coordinates

    Returns
    -------
    tuple of integers
        voxel coordinates (integer)
    """
    coords = apply_affine(inverse, mri_coords)
    return coords.astype(np.int)


def map_seeg_data(eeg, mri):
    """map contact data to surrounding voxels and smooth

    Parameters
    ----------
    seizure : EEG | dict
        eeg data
    mri : string
        path to MRI file

    Returns
    -------
    base_img : Nifti1Image
        Image with data mapped from baseline eeg
    seiz_img : Nifti1Image
        Image with data mapped from seizure eeg
    """
    base_img, seiz_img = create_volumes(eeg, mri)
    base_data = base_img.get_data()
    seiz_data = seiz_img.get_data()
    affine = seiz_img.affine
    inverse = npl.inv(affine)
    electrodes = eeg['electrode_names']
    base_eeg = eeg['baseline']['eeg']
    bads = base_eeg.info['bads']
    montage = base_eeg.get_montage()
    coord_list = dict()
    contact_num = 0
    for electrode in electrodes:
        contacts = [i for i in base_eeg.ch_names if i.startswith(electrode)]
        num_contacts = utils.find_num_contacts(contacts, electrode)
        for i in range(1, num_contacts):
            anode = electrode + str(i)
            cathode = electrode + str(i+1)
            if (anode in contacts) and (cathode in contacts):
                if not ((anode in bads) or (cathode in bads)):
                    anode_idx = montage.ch_names.index(anode)
                    cathode_idx = montage.ch_names.index(cathode)
                    loc1 = montage.dig[anode_idx]['r']*1000
                    loc2 = montage.dig[cathode_idx]['r']*1000
                    coord_list[contact_num] = (loc1 + loc2)/2
                    contact_num += 1

    base_ave = eeg['baseline']['ex_power']
    seiz_ave = eeg['seizure']['ex_power']
    for i in coord_list.keys():
        x, y, z = voxel_coords(coord_list[i], inverse)
        base_data[x, y, z, :] = base_ave[i, :]
        seiz_data[x, y, z, :] = seiz_ave[i, :]

    base_img = nilearn.image.smooth_img(base_img, 6)
    seiz_img = nilearn.image.smooth_img(seiz_img, 6)
    return base_img, seiz_img


def calc_source_image_power_data(eeg, low_freq=120, high_freq=200, delay=0):
    """ calculate power data for SEEG source image as per David et. al. 2011

    Parameters
    ----------
    seizure : EEG | dict
        eeg data
    low_freq : int, optional
        low frequency cutoff, by default 120
    high_freq : int, optional
        high frequency cutoff, by default 200
    delay : float, optional
        time from beginning of seizure onset, by default 0

    Returns
    -------
    EEG | dict
        eeg data including power data
    """

    eeg['baseline']['bipolar'] = \
        utils.create_bipolar(eeg['baseline']['eeg'],
                             eeg['electrode_names'])
    eeg['seizure']['bipolar'] = \
        utils.create_bipolar(eeg['seizure']['eeg'],
                             eeg['electrode_names'])
    eeg['baseline']['power'] = \
        utils.calc_power_multi(eeg['baseline']['bipolar'])
    eeg['seizure']['power'] = \
        utils.calc_power_multi(eeg['seizure']['bipolar'])
    eeg['baseline']['ave_power'], eeg['seizure']['ave_power'] = \
        ave_power_over_freq_band(eeg, low=low_freq, high=high_freq)
    eeg['baseline']['ex_power'], eeg['seizure']['ex_power'] = \
        extract_power(eeg, start=delay)

    return eeg


def calc_sorce_image_from_power(eeg, mri, D=3, dt=0.25):
    """Create source image from power data

    Parameters
    ----------
    seizure : EEG | dict
        eeg data
    mri : string
        path to file containing MRI

    Returns
    -------
    Nifti1Image
        Image where voxel values represent corresponding t-values
    """
    eeg['baseline']['img'], eeg['seizure']['img'] = \
        map_seeg_data(eeg, mri)
    base_img = eeg['baseline']['img']
    seiz_img = eeg['seizure']['img']

    nifti_masker = NiftiMasker(memory='nilearn_cache', memory_level=1)
    base_masked = nifti_masker.fit_transform(base_img)
    seiz_masked = nifti_masker.fit_transform(seiz_img)
    data = np.concatenate((base_masked, seiz_masked))
    steps = int(D/dt)
    labels = np.zeros(2*steps, dtype=np.int)
    labels[steps:] = 1
    __, t_scores, _ = permuted_ols(
        labels, data,
        # + intercept as a covariate by default
        n_perm=10000, two_sided_test=True,
        n_jobs=2)  # can be changed to use more CPUs

    return nifti_masker.inverse_transform(t_scores)


def create_source_image_map(eeg, mri, low_freq=120, high_freq=200, delay=0):
    """ create the source image t-map as per David et. al. 2011'

    Parameters
    ----------
    seizure : EEG | dict
        structure containing EEG data
    mri : string
        Path to location of MRI file
    low_freq : int, optional
        low frequency cut-off, by default 120
    high_freq : int, optional
        high frequency cut-off, by default 200
    delay : int, optional
        delay from the beginning of the seizure, by default 0
    """

    eeg = calc_source_image_power_data(eeg, low_freq, high_freq, delay)
    return calc_sorce_image_from_power(eeg, mri)


def plot_source_image_map(t_map, mri, cut_coords=None, threshold=2):
    """ Use Nilearn to create a Matplotlib plot of the t_map superimposed
        on the MRI

    Parameters
    ----------
    t_map : Nifti1Image
        Image where voxel values represent t-value
    mri : string
        Path to location of MRI file
    cut_coords : tuple, optional
        x, y, z coordinates used to center the image, by default None
    threshod : float, optional
        minimum t-value to display, by default 0
   """
    plot_stat_map(t_map, mri, cut_coords=cut_coords, threshold=threshold)


def calc_depth_source_image_from_power(eeg):
    """ Calculate t-map image analagous to Brainstorm demo

    Parameters
    ----------
    seizure : EEG | dict
        baseline and seizure EEG data
    montage : MNE DigMontage
        EEG montage

    Returns
    -------
    ndarray
        t values
    """

    base = eeg['baseline']['ave_power'].T
    seiz = eeg['seizure']['ave_power'].T
    data = np.concatenate((base, seiz))
    labels = np.zeros(30, dtype=np.int)
    labels[15:] = 1
    __, t_scores, _ = permuted_ols(
        labels, data,
        # + intercept as a covariate by default
        n_perm=10000, two_sided_test=True,
        n_jobs=2)  # can be changed to use more CPUs

    return t_scores


def create_depth_source_image_map(eeg, low_freq=120,
                                  high_freq=200, delay=0):
    """ Calculate t-values for each individual (non-bad) depth contact

    Parameters
    ----------
    seizure : EEG | dict
        baseline and seizure EEG data
    freqs : ndarray
        array of frequencies
    low_freq : int, optional
        lower frequency limit for calculation, by default 120
    high_freq : int, optional
        upper frequency limit for calculation, by default 200
    seiz_delay : float, by default 0
        delay from the beginning of the seizure, by default 0

    Returns
    -------
    ndarray
        t-values for depth contacts
    """

    eeg = calc_source_image_power_data(eeg, low_freq,
                                       high_freq, delay)
    return calc_depth_source_image_from_power(eeg)


def plot_3d_source_image_map(t_map, mri):
    """ Use Napari to display a coronal 3D view of the t_map superimposed on
        the MRI

    Parameters
    ----------
    t_map : Nifti1Image
        Image where pixel values are corresponding t-values
    mri : string
        path to mri file
    """

    try:
        import napari
    except ImportError:
        print('This function requires napari')
        return

    img = nib.load(mri)
    temp = nib.Nifti1Image(t_map.get_fdata()[:, :, :, 0],
                           t_map.affine)
    resized = nbp.resample_from_to(temp, img)

    coronal_img_data = _set_coronal(img)
    coronal_map_data = _set_coronal(resized)

    controls = np.linspace(0, 1, 101)
    colors = [cold_hot(i) for i in controls]
    for i in range(48, 53):
        colors[i] = (colors[i][0], colors[i][1], colors[i][2], 0)

    cmap = Colormap(colors, controls)
    min_ = np.min(coronal_map_data)
    max_ = np.max(coronal_map_data)
    limits = (min_, -min_)
    if abs(min_) < max_:
        limits = (-max_, max_)
    with napari.gui_qt():
        viewer = napari.Viewer()
        viewer.add_image(coronal_img_data, name='image')
        viewer.add_image(coronal_map_data, name='t-map', opacity=0.5,
                         contrast_limits=limits, colormap=cmap)


def _set_coronal(img):
    ''' reorient image to display in napari as coronal slices '''
    canon = nib.funcs.as_closest_canonical(img)
    return np.fliplr(np.moveaxis(canon.get_fdata(), [1, 2], [0, 1]))
