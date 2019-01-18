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
