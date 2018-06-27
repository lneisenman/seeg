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


def read_electrode_locations(show=False):
    home = r'C:\Users\eisenmanl\Documents\brainstorm_data_files'
#    home = r'C:\Users\leisenman\Documents\brainstorm_db'
    electrode_file = r'\tutorial_epimap\anat\implantation\elec_pos_patient.txt'
    file_name = home + electrode_file
    # file_name = home + electrode_file
    sfreq = 1000
    electrodes = pd.read_table(file_name, header=None,
                               names=['contact', 'x', 'y', 'z'])
    # skip ecg locations
    seeg = electrodes[0:-2].copy()
    dig_ch_pos = {seeg['contact'][i]: np.asarray([seeg['x'][i]/1000, seeg['y'][i]/1000, seeg['z'][i]/1000]) for i in range(len(seeg))}

    montage = mne.channels.DigMontage(dig_ch_pos=dig_ch_pos)
    names = list(seeg['contact'].values)
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


def read_eeg(dig_ch_pos, seizure, show=False):
    home = r'C:\Users\eisenmanl\Documents'
#    home = r'C:\Users\leisenman\Documents'
    seiz = r'\tutorial_epimap\seeg'
    raw_fname = home + r'\brainstorm_data_files' + seiz + '\\' + seizure['file_name']
#    raw_fname = home + r'\brainstorm_db' + seiz + '\\' + seizure['file_name']
    reader = neo.rawio.MicromedRawIO(filename=raw_fname)
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


def setup_bipolar(electrode, first, last):
    anodes = [electrode + str(i) for i in range(first, last)]
    cathodes = [electrode + str(i) for i in range(first+1, last+1)]
    label = electrode[:-1]
    ch_names = [label + str(i) + '-' + label + str(i+1)
                for i in range(first, last)]
    return anodes, cathodes, ch_names



def create_bipolar(seizure):
    anodes = list()
    cathodes = list()
    ch_names = list()
    electrodes = seizure['electrodes']
    for name, num in zip(electrodes['names'], electrodes['num_contacts']):
        temp = setup_bipolar(name, 1, num)
        anodes.extend(temp[0])
        cathodes.extend(temp[1])
        ch_names.extend(temp[2])

    baseline = seizure['baseline']['eeg']
    seiz = seizure['seizure']['eeg']
    baseline_bp = mne.set_bipolar_reference(baseline, anodes, cathodes,
                                            ch_names, verbose=False)
    baseline_bp = baseline_bp.pick_channels(ch_names)
    seizure_bp = mne.set_bipolar_reference(seiz, anodes, cathodes,
                                           ch_names, verbose=False)
    seizure_bp = seizure_bp.pick_channels(ch_names)
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


def calc_z_scores(seizures, freq=1):
    '''
    This function is meant to generate the figures shown in the  Brainstorm
    demo used to select the 120-200 Hz frequency band. It should also
    be similar to figure 1 in David et al 2011.

    This function will compute a z-score for each value of the seizure power
    spectrum using the mean and sd of the control power spectrum at each
    frequency. In the demo, the power spectrum is calculated for the 1st
    10 seconds of all three seizures and then averaged. Controls are
    similarly averaged
    '''

    z_scores = dict()
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
        mean = np.mean(baseline, 1)
        sd = np.std(baseline, 1)
        for i in range(z_score.shape[1]):
            z_score[:, i] -= mean
            z_score[:, i] /= sd

        z_scores[electrode] = z_score

    return z_scores


def show_z_score(times, z_score):
    plt.figure()
    plt.pcolormesh(times, freqs, z_score, vmin=-10, vmax=10, cmap='gin')
    plt.xlim(0, 20)
    plt.colorbar()


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


def extract_power(seizure, D=3, dt=0.2, start=10):
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
    coord_list = list()
    for name, num in zip(electrodes['names'], electrodes['num_contacts']):
        for i in range(num-1):
            label = name + str(i+1)
            loc1 = montage.dig_ch_pos[label]*1000
            label = name + str(i+2)
            loc2 = montage.dig_ch_pos[label]*1000
            coord_list.append((loc1 + loc2)/2)
            
    base_ex = seizure['baseline']['ex_power']
    seiz_ex = seizure['seizure']['ex_power']
    for i in range(seizure['seizure']['ex_power'].shape[0]):
        x, y, z = voxel_coords(coord_list[i], inverse)
        # print(x, y, z)
        base_data[x, y, z, :] = base_ex[i, :]
        seiz_data[x, y, z, :] = seiz_ex[i, :]

    base_img = nilearn.image.smooth_img(base_img, 6)
    seiz_img = nilearn.image.smooth_img(seiz_img, 6)
    return base_img, seiz_img


if __name__ == '__main__':
    names = [r"l'", r"g'"]
    num_contacts = [9, 12]
    seizure1 = {'file_name': 'sz1.trc', 'bads': ["v'1", "f'1"],
                'electrodes': {'names': names, 'num_contacts': num_contacts},
                'baseline': {'start': 72.800, 'end': 77.800},
                'seizure': {'start': 110.8, 'end': 160.8}}
    seizure2 = {'file_name': 'sz2.trc', 'bads': ["v'1", "t'8"],
                'electrodes': {'names': names, 'num_contacts': num_contacts},
                'baseline': {'start': 103.510, 'end': 108.510},
                'seizure': {'start': 133.510, 'end': 183.510}}
    seizure3 = {'file_name': 'sz3.trc', 'bads': ["o'1", "t'8"],
                'electrodes': {'names': names, 'num_contacts': num_contacts},
                'baseline': {'start': 45.287, 'end': 50.287},
                'seizure': {'start': 110.287, 'end': 160.287}}
    seizures = [seizure1, seizure2, seizure3]
    seizures = [seizure1]
    freqs = np.arange(10, 220, 3)

    
    montage, contact_info = read_electrode_locations()
    
    for seizure in seizures:
        raw = read_eeg(montage.dig_ch_pos, seizure)
        seizure['baseline']['eeg'], seizure['seizure']['eeg'] = clip_eeg(seizure, raw)
        seizure['baseline']['bipolar'], seizure['seizure']['bipolar'] = create_bipolar(seizure)
        seizure['baseline']['power'], seizure['seizure']['power'] = calc_power(seizure)
        seizure['baseline']['ave_power'], seizure['seizure']['ave_power'] = ave_power_over_freq_band(seizure)
        seizure['baseline']['ex_power'], seizure['seizure']['ex_power'] = extract_power(seizure)
        seizure['baseline']['img'], seizure['seizure']['img'] = map_seeg_data(seizure, montage)

    
    mri = r'C:\Users\eisenmanl\Documents\brainstorm_data_files\tutorial_epimap\anat\MRI\3DT1pre_deface.nii'
    base_img = seizures[0]['baseline']['img']
    seiz_img = seizures[0]['seizure']['img']
    # base_img = nib.load('baseline.nii.gz')
    # seiz_img = nib.load('seizure.nii.gz')
    base = base_img.get_data()
    seiz = seiz_img.get_data()
    
    nifti_masker = NiftiMasker(memory='nilearn_cache', memory_level=1)  # cache options
    base_masked = nifti_masker.fit_transform(base_img)
    seiz_masked = nifti_masker.fit_transform(seiz_img)
    data = np.concatenate((base_masked, seiz_masked))
    labels = np.zeros(30, dtype=np.int)
    labels[15:] = 1
    neg_log_pvals, t_scores, _ = permuted_ols(
        labels, data,
        # + intercept as a covariate by default
        n_perm=10000, two_sided_test=True,
        n_jobs=2)  # can be changed to use more CPUs
    p_pt_img = nifti_masker.inverse_transform(neg_log_pvals)
    plot_stat_map(p_pt_img, mri, threshold=1.3)
    t_pt_img = nifti_masker.inverse_transform(t_scores)
    plot_stat_map(t_pt_img, mri, threshold=2)
    '''    
    z_scores = calc_z_scores(seizures)
    times = np.linspace(0, 50, 51)  # seizures[0]['seizure_bp'].times
    show_z_score(times, z_scores['g7-g8'])  # compare to fig in demo
    
    plot_ave_power(seizures[0])
    '''
    plt.show()
