# -*- coding: utf-8 -*-

from mayavi import mlab
import mne
import neo
import numpy as np
import pandas as pd


def read_electrode_file(file_name):
    if file_name[-4:] == 'fcsv':
        skiprows = 2
        sep = None
        names = None
        header = 'infer'
    else:
        skiprows = None
        sep = '\t'
        header = None
        names = ['label', 'x', 'y', 'z']

    contacts = pd.read_csv(file_name, skiprows=skiprows, sep=sep,
                           header=header, names=names)
    electrodes = pd.DataFrame(columns=['contact', 'x', 'y', 'z'])
    electrodes['contact'] = contacts['label']
    electrodes['x'] = contacts['x']/1000
    electrodes['y'] = contacts['y']/1000
    electrodes['z'] = contacts['z']/1000
    return electrodes


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


def read_micromed_eeg(file_name, electrodes, bads):
    reader = neo.rawio.MicromedRawIO(filename=file_name)
    reader.parse_header()
    ch_names = list(reader.header['signal_channels']['name'])
    # ch_types = [match_ch_type(ch) for ch in ch_names]
    data = np.array(reader.get_analogsignal_chunk())
    data = reader.rescale_signal_raw_to_float(data).T
    data *= 1e-6  # convert from microvolts to volts
    sfreq = reader.get_signal_sampling_rate()
    labels = ['contact', 'x', 'y', 'z']
    contacts = pd.DataFrame(columns=labels)
    contacts.loc[:, 'contact'] = ch_names
    for name in ch_names:
        contacts.loc[contacts['contact'] == name, ['x']] = \
            electrodes.loc[electrodes['contact'] == name, ['x']].values
        contacts.loc[contacts['contact'] == name, ['y']] = \
            electrodes.loc[electrodes['contact'] == name, ['y']].values
        contacts.loc[contacts['contact'] == name, ['z']] = \
            electrodes.loc[electrodes['contact'] == name, ['z']].values

    montage, info = create_montage(contacts, sfreq=sfreq)
    raw = mne.io.RawArray(data, info)
    raw.info['bads'] = bads
    return raw, montage


def read_edf(eeg_file, electrodes, bads=None, notch=False):
    raw = mne.io.read_raw_edf(eeg_file, preload=True)
    mapping = dict()
    LABELS = ['POL {}', 'EEG {}-Ref', 'EEG {}-Ref-0',
              'EEG {}-Org', 'EEG {}-Org-0']
    for contact in electrodes['contact']:
        for label in LABELS:
            if label.format(contact) in raw.ch_names:
                mapping[label.format(contact)] = contact
                break

    eeg = raw.copy().pick_channels(list(mapping.keys()))
    eeg.reorder_channels(list(mapping.keys()))
    eeg.rename_channels(mapping)
    eeg.info['bads'] = bads
    if notch:
        eeg.notch_filter(range(60, int(raw.info['sfreq']/2), 60))

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
