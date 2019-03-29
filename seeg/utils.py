# -*- coding: utf-8 -*-

from mayavi import mlab
import mne
import neo
import numpy as np


def create_montage(electrodes, sfreq=1000, show=False):
    '''
    Create a montage from a pandas dataframe containing contact names
    and locations in meters

    electrodes : pandas dataframe with columns named "contact", "x", "y"
                 and "z"
    '''
    dig_ch_pos = {electrodes['contact'][i]:
                  np.asarray([electrodes['x'][i], electrodes['y'][i],
                             electrodes['z'][i]])
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


def read_micromed_eeg(dig_ch_pos, seizure, baseline=True, show=False):
    if baseline:
        file_name = seizure['baseline']['eeg_file_name']
    else:
        file_name = seizure['seizure']['eeg_file_name']

    reader = neo.rawio.MicromedRawIO(filename=file_name)
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


def read_edf(eeg_file, electrodes, bads=None, notch=False):
    raw = mne.io.read_raw_edf(eeg_file, preload=True)
    names = list()
    for contact in electrodes['contact']:
        name = 'EEG ' + contact + '-Org'
        if name in raw.ch_names:
            names.append(name)
        else:
            names.append(name+'-0')

    eeg = raw.copy().pick_channels(names)

    mapping = dict()
    for name in eeg.ch_names:
        label1 = name.split()[1]
        label2 = label1.split('-')[0]
        mapping[name] = label2

    eeg.rename_channels(mapping)
    eeg.info['bads'] = bads
    if notch:
        eeg.notch_filter(60)
    return eeg


def clip_eeg(seizure, show=False):
    start, end = seizure['baseline']['start'], seizure['baseline']['end']
    baseline = seizure['baseline']['raw'].copy().crop(start, end)

    start, end = seizure['seizure']['start'], seizure['seizure']['end']
    seizure = seizure['seizure']['raw'].copy().crop(start, end)
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


def create_bipolar(raw, electrodes):
    anodes = list()
    cathodes = list()
    ch_names = list()
    for name in electrodes:
        temp = setup_bipolar(name, raw)
        anodes.extend(temp[0])
        cathodes.extend(temp[1])
        ch_names.extend(temp[2])

    bipolar = mne.set_bipolar_reference(raw, anodes, cathodes,
                                        ch_names, verbose=False)
    bipolar = bipolar.pick_channels(ch_names)
    bipolar = bipolar.reorder_channels(ch_names)
    return bipolar


def calc_power(raw, freqs, n_cycles=7., output='power'):
    # n_cycles = freqs
    n_channels = raw.info['nchan']
    n_times = raw.n_times
    data = np.zeros((1, n_channels, n_times))
    data[0, :, :] = raw.get_data()[:, :]
    sfreq = raw.info['sfreq']
    return mne.time_frequency.tfr_array_multitaper(data, sfreq=sfreq,
                                                   freqs=freqs,
                                                   n_cycles=n_cycles,
                                                   output=output)
