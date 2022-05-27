# -*- coding: utf-8 -*-
""" Utility classes and functions

"""
import os
from re import L

import matplotlib as mpl
import mne
import neo
import numpy as np
import pandas as pd


class EEG:
    """ Dict like class to contain EEG data

Parameters
    ----------
    electrode_names : list of string
        list of all electrode names
    bads : list of string
        list of bad electrode names

   """
    def __init__(self, electrode_names, bads=None):
        self.electrode_names = electrode_names
        self.bads = bads
        self.baseline = dict()
        self.seizure = dict()
        self.montage = None
        self.electrodes = None

    def __getitem__(self, key):
        if key == 'baseline':
            return self.baseline
        elif key == 'seizure':
            return self.seizure
        elif key == 'electrode_names':
            return self.electrode_names
        elif key == 'bads':
            return self.bads
        else:
            raise KeyError(f'{key} is not a valid key')

    def set_baseline(self, raw, pre=0, post=5, file_name=None):
        """ set parameters for baseline EEG

        """
        self.baseline['eeg'] = clip_eeg(raw, pre, post)
        self.baseline['file_name'] = file_name

    def set_seizure(self, raw, pre=5, post=20, file_name=None):
        """ set parameters for seizure EEG

        """
        self.seizure['raw'] = raw
        self.seizure['eeg'] = clip_eeg(raw, pre, post)
        self.seizure['file_name'] = file_name


def read_electrode_file(file_name):
    """ Read electrode names and coordinates (in meters)

    Parameters
    ----------
    file_name : string
        path to file

    Returns
    -------
    Pandas Dataframe
        Dataframe with columns for the contact name and x, y and z coordinates
    """
    if file_name[-4:] == 'fcsv':
        skiprows = 2
        sep = ','
        names = None
        header = 'infer'
    else:
        skiprows = None
        sep = '\s+'  # noqa
        header = None
        names = ['label', 'x', 'y', 'z']

    contacts = pd.read_table(file_name, skiprows=skiprows, sep=sep,
                             header=header, names=names, engine='python')
    electrodes = pd.DataFrame(columns=['contact', 'x', 'y', 'z'])
    electrodes['contact'] = contacts['label']
    electrodes['x'] = contacts['x']/1000
    electrodes['y'] = contacts['y']/1000
    electrodes['z'] = contacts['z']/1000
    return electrodes


def load_eeg_data(EEG_DIR, ELECTRODE_NAMES, BADS, seizure=1,
                  electrode_file='recon.fcsv', onset_file='onset_times.tsv'):

    electrode_file = os.path.join(EEG_DIR, electrode_file)
    electrodes = read_electrode_file(electrode_file)

    onset_times_file = os.path.join(EEG_DIR, onset_file)
    baseline_onsets, seizure_onsets = read_onsets(onset_times_file)

    studies = list(range(1, len(seizure_onsets)+1))
    if isinstance(seizure, int):
        if seizure > 0:
            studies = [seizure]

    eeg_list = list()

    for i in studies:
        j = i
        if len(baseline_onsets) == 1:
            j = 1

        fn = baseline_onsets.file_name[j]
        baseline_eeg_file = os.path.join(EEG_DIR, fn)
        baseline_onset_time = baseline_onsets.onset_time[j]

        fn = seizure_onsets.file_name[i]
        seizure_eeg_file = os.path.join(EEG_DIR, fn)
        seizure_onset_time = seizure_onsets.onset_time[i]

        read_eeg = read_micromed_eeg
        if fn[-3:] == 'edf':
            read_eeg = read_edf

        eeg = EEG(ELECTRODE_NAMES, BADS)
        raw, montage = read_eeg(baseline_eeg_file, electrodes, BADS)
        raw.set_annotations(mne.Annotations(baseline_onset_time, 0, 'Seizure'))
        eeg.set_baseline(raw, file_name=baseline_eeg_file)

        raw, montage = read_eeg(seizure_eeg_file, electrodes, BADS)
        raw.set_annotations(mne.Annotations(seizure_onset_time, 0, 'Seizure'))
        eeg.set_seizure(raw, file_name=seizure_eeg_file)

        eeg.montage = montage
        eeg.electrodes = electrodes

        eeg_list.append(eeg)

    return eeg_list


def create_montage(electrodes, sfreq=1000):
    """ Create a montage from a pandas dataframe containing contact names
    and locations in meters

    Parameters
    ----------
    electrodes : Pandas Dataframe
        pandas dataframe with columns named "contact", "x", "y" and "z"
    sfreq : int, optional
        EEG sampling frequency, by default 1000

    Returns
    -------
    montage : MNE montage
        the EEG montage
    contact_info : MNE info
        MNE info data structure
    """

    dig_ch_pos = {electrodes['contact'][i]:
                  np.asarray([electrodes['x'][i], electrodes['y'][i],
                             electrodes['z'][i]])
                  for i in range(len(electrodes))}

    montage = mne.channels.make_dig_montage(ch_pos=dig_ch_pos,
                                            coord_frame='head')
    names = list(electrodes['contact'].values)
    contact_info = mne.create_info(ch_names=names, sfreq=sfreq,
                                   ch_types='seeg').set_montage(montage)

    return montage, contact_info


def match_ch_type(name):
    """ return channel type based on channel name

    Parameters
    ----------
    name : string
        name of channel

    Returns
    -------
    out: string
        channel type
    """
    out = 'seeg'
    if 'ecg' in name:
        out = 'ecg'
    if name in ['fz', 'cz']:
        out = 'eeg'
    return out


def read_micromed_eeg(file_name, electrodes, bads):
    """ read micromed eeg file

    Parameters
    ----------
    file_name : string
        path to EEG file
    electrodes : list of strings
        names of electrodes
    bads : list of strings
        names of bad electrodes

    Returns
    -------
    raw : MNE Raw
        EEG data
    montage : MNE Montage
        EEG montage data
    """
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
    """ read data from edf file

    Parameters
    ----------
    eeg_file : string
        path to EEG file
    electrodes : list of strings
        names of electrodes
    bads : list of strings, optional
        names of bad channels, by default None
    notch : bool, optional
        apply notch filter to EEG data, by default False

    Returns
    -------
    eeg : MNE Raw
        eeg data
    """
    raw = mne.io.read_raw_edf(eeg_file, preload=True)
    mapping = dict()
    LABELS = ['POL {}', 'EEG {}-Ref', 'EEG {}-Ref-0',
              'EEG {}-Org', 'EEG {}-Org-0', '{}']
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

    montage, __ = create_montage(electrodes)
    return eeg, montage


def clip_eeg(raw, pre=5, post=20):
    """ Clip EEG file

    Parameters
    ----------
    raw : MNE Raw
        EEG data
    pre : float, optional
        how much time prior to the onset to clip, by default 5
    post : float, optional
        how much time after to the onset to clip, by default 20

    Returns
    -------
    NME Raw
    """
    onset = raw.annotations.onset[0]
    start = onset - pre
    if start < 0:
        start = 0

    end = onset + post
    if end > raw.times[-1]:
        end = raw.times[-1]

    clipped = raw.copy().crop(start, end)
    clipped.set_annotations(None)
    clipped.set_annotations(mne.Annotations(onset-start, 0, 'Seizure'))
    return clipped


def read_onsets(file_name='onset_times.tsv'):
    """ read `onset_times.tsv` file and return baseline and seizure data
        in separate dataframes

    Parameters
    ----------
    file_name : string
        path to `onset_times.tsv` file

    Returns
    -------
    baseline_onsets : pandas DataFrame
        baseline onset info
    seizure_onsets : pandas DataFrame
        seizure onset info
    """
    onset_times = pd.read_table(file_name, sep='\s+')   # noqa: ignore=W605
    onset_times['study_type'] = onset_times['study_type'].str.lower()
    grouped = onset_times.groupby('study_type')
    baseline_onsets = grouped.get_group('baseline').set_index('run')
    seizure_onsets = grouped.get_group('seizure').set_index('run')
    return baseline_onsets, seizure_onsets


def find_num_contacts(contacts, electrode):
    """ find the number of contacts on a given depth electrode

    Parameters
    ----------
    contacts : list of strings
        names of all contacts
    electrode : string
        electrode to be counted

    Returns
    -------
    int
        number of contacts on `electrode`
    """
    start = len(electrode)
    numbers = [int(contact[start:]) for contact in contacts]
    return np.max(numbers)


def setup_bipolar(electrode, ch_names, bads):
    """ create lists of names for resetting and EEG to a bipolar montage

    Parameters
    ----------
    electrode : string
        name of electrode
    ch_names : list of strings
        names of channels
    bads : list of strings
        names of bad channels

    Returns
    -------
    anodes : list of strings
        names of anode electrodes
    cathodes : list of strings
        names of cathode electrodes
    ch_names : list of strings
        names of bipolar channels
    """

    contacts = [i for i in ch_names if i.startswith(electrode)]
    anodes = list()
    cathodes = list()   # type: ignore
    ch_names = list()
    num_contacts = find_num_contacts(contacts, electrode)
    if contacts[0][-2] == '0':
        zero = True
    else:
        zero = False

    for i in range(num_contacts):
        if (i < 10) and zero:
            anode = electrode + '0' + str(i)
        else:
            anode = electrode + str(i)

        if (i < 9) and zero:
            cathode = electrode + '0' + str(i+1)
        else:
            cathode = electrode + str(i+1)

        if (anode in contacts) and (cathode in contacts):
            if not ((anode in bads) or (cathodes in bads)):
                anodes.append(anode)
                cathodes.append(cathode)
                ch_names.append(anode + '-' + cathode)

    return anodes, cathodes, ch_names


def create_bipolar(raw, electrodes):
    """ create EEG data in a bipolar montage

    Parameters
    ----------
    raw : MNE Raw
        EEG data
    electrodes : list of strings
        names of electrodes

    Returns
    -------
    bipolar : MNE Raw
        EEG data in a bipolar montage
    """
    anodes = list()
    cathodes = list()
    ch_names = list()
    for name in electrodes:
        temp = setup_bipolar(name, raw.ch_names, raw.info['bads'])
        anodes.extend(temp[0])
        cathodes.extend(temp[1])
        ch_names.extend(temp[2])

    bipolar = mne.set_bipolar_reference(raw, anodes, cathodes,
                                        ch_names, verbose=False)
    bipolar = bipolar.pick_channels(ch_names)
    bipolar = bipolar.reorder_channels(ch_names)
    return bipolar


def calc_power_multi(raw, window=1, step=0.25):
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


def calc_power_welch(raw, window=1, step=0.25):
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
    return psd


def map_colors(values, cmap='cold_hot'):
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
    return color_map(norm(values))
