# -*- coding: utf-8 -*-

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


def create_montage(electrodes, sfreq=1000, show=False):
    '''
    Create a montage from a pandas dataframe containing contact names and locations in meters

    electrodes : pandas dataframe with columns named "contact", "x", "y" and "z"
    '''
    dig_ch_pos = {electrodes['contact'][i]: np.asarray([electrodes['x'][i], electrodes['y'][i], electrodes['z'][i]])
                  for i in range(len(electrodes))}

    montage = mne.channels.DigMontage(dig_ch_pos=dig_ch_pos)
    names = list(electrodes['contact'].values)
    contact_info = mne.create_info(ch_names=names, sfreq=sfreq,
                                   ch_types='seeg', montage=montage)

    if show:
        sdir = r'C:\Users\eisenmanl\Desktop\ubuntu_shared\subjects'
        mne.viz.plot_alignment(contact_info, subject='seeg_brainstorm',
                               subjects_dir=sdir, surfaces=['pial'],
                               meg=False, coord_frame='head')
        mlab.view(200, 70)
        mlab.show()

    return montage, contact_info


def match_ch_type(name):
    out = 'seeg'
    if 'ecg' in name:
        out = 'ecg'
    if name in ['fz', 'cz']:
        out = 'eeg'
    return out


def read_micromed_eeg(dig_ch_pos, seizure, show=False):
    reader = neo.rawio.MicromedRawIO(filename=seizure['eeg_file_name'])
    reader.parse_header()
    ch_names = list(reader.header['signal_channels']['name'])
    dig_ch_pos = {k: v for k, v in dig_ch_pos.items() if k in ch_names}
    montage = mne.channels.DigMontage(dig_ch_pos=dig_ch_pos)
    ch_types = [match_ch_type(ch) for ch in ch_names]
    data = np.array(reader.get_analogsignal_chunk())
    data = reader.rescale_signal_raw_to_float(data).T
    data *= 1e-6  # putdata from microvolts to volts

    sfreq = reader.get_signal_sampling_rate()
    # print('read_eeg', sfreq)
    info = mne.create_info(ch_names, sfreq, ch_types=ch_types, montage=montage)
    raw = mne.io.RawArray(data, info)
    raw.info['bads'] = seizure['bads']
    if show:
        raw.plot()

    return raw


def clip_eeg(seizure, raw, show=False):
    start, end = seizure['baseline']['start'], seizure['baseline']['end']
    baseline = raw.copy().crop(start, end)

    start, end = seizure['seizure']['start'], seizure['seizure']['end']
    seizure = raw.copy().crop(start, end)
    if show:
        seizure.plot()

    return baseline, seizure


def find_num_contacts(contacts, electrode):
    start = len(electrode)
    numbers = [int(contact[start:]) for contact in contacts]
    return np.max(numbers)


def setup_bipolar(electrode, raw):
    contacts = [i for i in raw.ch_names if i.startswith(electrode)]
    anodes = list()
    cathodes = list()
    ch_names = list()
    num_contacts = find_num_contacts(contacts, electrode)
    bads = raw.info['bads']
    for i in range(num_contacts):
        anode = electrode + str(i)
        cathode = electrode + str(i+1)
        if (anode in contacts) and (cathode in contacts):
            if not ((anode in bads) or (cathodes in bads)):
                anodes.append(anode)
                cathodes.append(cathode)
                ch_names.append(anode + '-' + cathode)

    return anodes, cathodes, ch_names


def create_bipolar(seizure):
    anodes = list()
    cathodes = list()
    ch_names = list()
    electrodes = seizure['electrodes']
    baseline = seizure['baseline']['eeg']
    seiz = seizure['seizure']['eeg']
    for name in electrodes:
        temp = setup_bipolar(name, baseline)
        anodes.extend(temp[0])
        cathodes.extend(temp[1])
        ch_names.extend(temp[2])

    baseline_bp = mne.set_bipolar_reference(baseline, anodes, cathodes,
                                            ch_names, verbose=False)
    baseline_bp = baseline_bp.pick_channels(ch_names)
    baseline_bp = baseline_bp.reorder_channels(ch_names)
    seizure_bp = mne.set_bipolar_reference(seiz, anodes, cathodes,
                                           ch_names, verbose=False)
    seizure_bp = seizure_bp.pick_channels(ch_names)
    seizure_bp = seizure_bp.reorder_channels(ch_names)
    return baseline_bp, seizure_bp


def calc_power(seizure, freqs):
    n_cycles = freqs
    n_channels = seizure['seizure']['bipolar'].info['nchan']
    n_times = seizure['seizure']['bipolar'].n_times
    data = np.zeros((1, n_channels, n_times))
    data[0, :, :] = seizure['seizure']['bipolar'].get_data()[:, :]
    sfreq = seizure['seizure']['bipolar'].info['sfreq']
    power_fcn = mne.time_frequency.tfr_array_multitaper
    seizure_power = power_fcn(data, sfreq=sfreq, freqs=freqs,
                              n_cycles=n_cycles, output='power')

    n_channels = seizure['baseline']['bipolar'].info['nchan']
    n_times = seizure['baseline']['bipolar'].n_times
    data = np.zeros((1, n_channels, n_times))
    data[0, :, :] = seizure['baseline']['bipolar'].get_data()[:, :]
    baseline_power = power_fcn(data, sfreq=sfreq, freqs=freqs,
                               n_cycles=n_cycles, output='power')
    return baseline_power, seizure_power


def compress_data(data, old_freq, new_freq):
    ''' average points to compress
    assumes data is 2-D '''
    num = round(0.5+(data.shape[1]*(new_freq/old_freq)))
    new = np.zeros((data.shape[0], num))
    ratio = old_freq/new_freq
    for i in range(num):
        start = int(i*ratio)
        end = int((i+1)*ratio)
        new[:, i] = np.mean(data[:, start:end], 1)

    return new


def ave_power_over_freq_band(seizure, freqs, low=120, high=200):
    ''' returns the average power between low and high '''
    baseline_power = seizure['baseline']['power']
    shape = baseline_power.shape
    baseline_ave_power = np.zeros((shape[1], shape[3]))
    seizure_power = seizure['seizure']['power']
    shape = seizure_power.shape
    seizure_ave_power = np.zeros((shape[1], shape[3]))
    freq_index = np.where(np.logical_and(freqs >= low, freqs <= high))

    for i in range(len(seizure['seizure']['bipolar'].ch_names)):
        baseline_ave_power[i, :] = np.mean(baseline_power[0, i, freq_index, :], 1)
        seizure_ave_power[i, :] = np.mean(seizure_power[0, i, freq_index, :], 1)

    return baseline_ave_power, seizure_ave_power


def plot_ave_power(seizure, channel='g7-g8'):
    plt.figure()
    b_times = seizure['baseline']['bipolar'].times
    s_times = seizure['seizure']['bipolar'].times
    index = seizure['baseline']['bipolar'].ch_names.index(channel)
    plt.plot(b_times, seizure['baseline']['ave_power'][index, :],
             s_times, seizure['seizure']['ave_power'][index, :])


def extract_power(seizure, D=3, dt=0.2, start=0):
    assert int(D/dt)*dt == D
    sfreq = seizure['seizure']['bipolar'].info['sfreq']
    num_steps = int(D/dt)
    tstep0 = int((sfreq * start) + 1)
    baseline_ave_power = seizure['baseline']['ave_power']
    baseline_ex_power = np.zeros((baseline_ave_power.shape[0], num_steps))
    seizure_ave_power = seizure['seizure']['ave_power']
    seizure_ex_power = np.zeros((seizure_ave_power.shape[0], num_steps))
    for i in range(num_steps):
        t1 = int(i*dt*sfreq)
        t2 = int((i+1)*dt*sfreq)
        baseline_ex_power[:, i] = np.mean(baseline_ave_power[:, t1:t2], 1)
        t1 += tstep0
        t2 += tstep0
        seizure_ex_power[:, i] = np.mean(seizure_ave_power[:, t1:t2], 1)

    return baseline_ex_power, seizure_ex_power


def create_volumes(seizure):
    mri = r'C:\Users\eisenmanl\Documents\brainstorm_data_files\tutorial_epimap\anat\MRI\3DT1pre_deface.nii'
    img = nib.load(mri)
    shape = np.round(np.asarray(img.shape)/3).astype(np.int)
    shape = np.append(shape, seizure['baseline']['ex_power'].shape[-1])
    affine = img.affine.copy()
    affine[:3, :3] = img.affine[:3, :3] * 3
    baseline = np.zeros(shape)
    baseline_img = nib.Nifti1Image(baseline, affine)

    shape[-1] = seizure['seizure']['ex_power'].shape[-1]
    seiz = np.zeros(shape)
    seizure_img = nib.Nifti1Image(seiz, affine)
    return baseline_img, seizure_img


def voxel_coords(mri_coords, inverse):
    coords = apply_affine(inverse, mri_coords)
    return coords.astype(np.int)


def map_seeg_data(seizure, montage):
    base_img, seiz_img = create_volumes(seizure)
    base_data = base_img.get_data()
    seiz_data = seiz_img.get_data()
    affine = seiz_img.affine
    inverse = npl.inv(affine)
    electrodes = seizure['electrodes']
    eeg = seizure['baseline']['eeg']
    bads = eeg.info['bads']
    coord_list = dict()
    contact_num = -1
    for electrode in electrodes:
        contacts = [i for i in eeg.ch_names if i.startswith(electrode)]
        num_contacts = find_num_contacts(contacts, electrode)
        for i in range(1, num_contacts):
            contact_num += 1
            anode = electrode + str(i)
            cathode = electrode + str(i+1)
            if (anode in contacts) and (cathode in contacts):
                if not ((anode in bads) or (cathode in bads)):
                    loc1 = montage.dig_ch_pos[anode]*1000
                    loc2 = montage.dig_ch_pos[cathode]*1000
                    coord_list[contact_num] = (loc1 + loc2)/2
            
    base_ex = seizure['baseline']['ex_power']
    seiz_ex = seizure['seizure']['ex_power']
    for i in coord_list.keys():
        x, y, z = voxel_coords(coord_list[i], inverse)
        base_data[x, y, z, :] = base_ex[i, :]
        seiz_data[x, y, z, :] = seiz_ex[i, :]

    base_img = nilearn.image.smooth_img(base_img, 6)
    seiz_img = nilearn.image.smooth_img(seiz_img, 6)
    return base_img, seiz_img


def create_source_image(seizure, mri, freqs, raw_eeg, montage, low_freq=120,
                        high_freq=200, seiz_delay=0):
    ''' create and display the SEEG source image as per David et. al. 2011'''
    seizure['baseline']['eeg'], seizure['seizure']['eeg'] = clip_eeg(seizure, raw_eeg)
    seizure['baseline']['bipolar'], seizure['seizure']['bipolar'] = create_bipolar(seizure)
    seizure['baseline']['power'], seizure['seizure']['power'] = calc_power(seizure, freqs)
    seizure['baseline']['ave_power'], seizure['seizure']['ave_power'] = ave_power_over_freq_band(seizure, freqs, low=low_freq, high=high_freq)
    seizure['baseline']['ex_power'], seizure['seizure']['ex_power'] = extract_power(seizure, start=seiz_delay)
    seizure['baseline']['img'], seizure['seizure']['img'] = map_seeg_data(seizure, montage)
    base_img = seizure['baseline']['img']
    seiz_img = seizure['seizure']['img']
    
    nifti_masker = NiftiMasker(memory='nilearn_cache', memory_level=1)  # cache options
    base_masked = nifti_masker.fit_transform(base_img)
    seiz_masked = nifti_masker.fit_transform(seiz_img)
    data = np.concatenate((base_masked, seiz_masked))
    labels = np.zeros(30, dtype=np.int)
    labels[15:] = 1
    __, t_scores, _ = permuted_ols(
        labels, data,
        # + intercept as a covariate by default
        n_perm=10000, two_sided_test=True,
        n_jobs=2)  # can be changed to use more CPUs

    return nifti_masker.inverse_transform(t_scores)
