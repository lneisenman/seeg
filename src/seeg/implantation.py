# -*- coding: utf-8 -*-

from dataclasses import dataclass
import os

import numpy as np
import pandas as pd

from .eeg import read_electrode_file
from .plots.depths import Depth


@dataclass
class Implantation():
    """ class to contain data about electrodes and contacts"""
    subject: str
    subjects_dir: str
    depths_file: str = 'depths.dat'
    contacts_file: str = 'measured_contacts.dat'
    bads_file: str = 'bads.dat'

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
        self.depth_list = create_depth_list(self.depths,
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


def create_depth_list(depths: pd.DataFrame, ch_names: list,
                      contacts: pd.DataFrame) -> list[Depth]:
    """Create a list of Depths

    For each electrode in `electrode_names` create a Depth and add to a list

    Parameters
    ----------
    electrode_names : list
        list of electrode names (strings)
    ch_names : list
        list of names of all contacts (strings) which are assumed to be in the
        form of electrode name followed by a number
    electrodes : Pandas DataFrame
        contains columns for contact name and x,y,z coordinates in meters

    Returns
    -------
    depth_list : list
        List of Depth's
    """

    depth_list = list()
    for depth in depths.itertuples():
        depth_contacts = contacts.loc[contacts.contact.str.startswith(depth.name), :]   # noqa
        active = [contact in ch_names for contact in depth_contacts.contact]
        locations = np.zeros((len(depth_contacts), 3))
        locations[:, 0] = depth_contacts.x.values
        locations[:, 1] = depth_contacts.y.values
        locations[:, 2] = depth_contacts.z.values
        locations *= 1000
        depth_list.append(Depth(depth.name, depth_contacts.contact.tolist(),
                                locations, depth.diam, depth.contact_len,
                                depth.spacing, active=active))

    return depth_list
