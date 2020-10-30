# -*- coding: utf-8 -*-

import os.path as op

import matplotlib.pyplot as plt
from mayavi import mlab
import mne
import pandas as pd

import seeg


data_path = mne.datasets.sample.data_path()
subjects_dir = op.join(data_path, 'subjects')
subject = 'sample'

misc_path = mne.datasets.misc.data_path()
seeg_path = op.join(misc_path, 'seeg')
eeg_file = op.join(seeg_path, 'sample_seeg.edf')
electrode_file = op.join(seeg_path, 'sample_seeg_electrodes.tsv')

eeg = mne.io.read_raw_edf(eeg_file)
electrodes = pd.read_table(electrode_file)
electrodes.rename(columns={'name': 'contact'}, inplace=True)
# electrodes['x'] = (electrodes['x']-10)/1000
# electrodes['y'] = (electrodes['y']-40)/1000
# electrodes['z'] = (electrodes['z']-20)/1000
electrodes['x'] /= 1000
electrodes['y'] /= 1000
electrodes['z'] /= 1000

ELECTRODE_NAMES = [r"L'", r"N'", r"F'", r"O'", r"G'", r"X'"]
ch_names = [electrodes.contact[i] for i in range(len(electrodes)) if
            electrodes.contact[i][:2] in ELECTRODE_NAMES]

depth_list = seeg.create_depths(ELECTRODE_NAMES, ch_names, electrodes)
brain = seeg.create_depths_plot(depth_list, subject, subjects_dir)
mlab.show()

# eeg.plot()
# plt.show()
