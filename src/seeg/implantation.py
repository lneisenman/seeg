# -*- coding: utf-8 -*-

from dataclasses import dataclass
import os

import numpy as np
import pandas as pd

from .eeg import read_electrode_file
from .plots.depths import create_depth_list


@dataclass
class Implantation():
    """ class to contain data about electrodes and contacts"""
    subject: str
    subjects_dir: str
    depths_file: str = 'depths.dat'
    contacts_file: str = 'measured_contacts.dat'
    bads_file: str = 'bads.dat'
    inactives_file: str = 'inactives.dat'

    def __post_init__(self) -> None:
        self.subject_path = os.path.join(self.subjects_dir, self.subject)
        self.eeg_path = os.path.join(self.subject_path, 'eeg')
        self.depths = read_data(os.path.join(self.eeg_path,
                                             self.depths_file),
                                ['name', 'diam', 'contact_len', 'spacing'])
        self.contacts = read_electrode_file(os.path.join(self.eeg_path,
                                                         self.contacts_file))
        self.bads = read_data(os.path.join(self.eeg_path, self.bads_file),
                              ['contact']).contact.values.tolist()
        self.inactives = read_data(os.path.join(self.eeg_path,
                                                self.inactives_file),
                                   ['contact']).contact.values.tolist()
        self.depth_list = create_depth_list(self.depths, self.inactives,
                                            self.contacts.contact.values,
                                            self.contacts)


def read_data(file_name: str, names: list | None = None) -> pd.DataFrame:
    """ Read whitespace separated data into a dataframe

    Parameters
    ----------
    file_name : string
        path to file
    names : list
        list of column names

    Returns
    -------
    Pandas Dataframe
        Dataframe
    """
    return pd.read_table(file_name, sep=r'\s+', names=names)
