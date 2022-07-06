# -*- coding: utf-8 -*-

from dataclasses import dataclass, field
import os
from typing import Sequence, Tuple

from .epi_index import calculate_EI
from .plots.epi_image import EpiImage
from .eeg import load_eeg_data


@dataclass
class Seeg():
    """ Class to contain SEEG recording information    """
    subject: str
    subjects_dir: str
    electrode_names: list = field(default_factory=list)
    bads: list = field(default_factory=list)
    electrode_file: str = 'recon.fcsv'
    seiz_delay: float = 0

    def __post_init__(self) -> None:
        self.subject_path = os.path.join(self.subjects_dir, self.subject)
        self.eeg_path = os.path.join(self.subject_path, 'eeg')
        self.mri_file = os.path.join(self.subject_path, 'mri/orig.mgz')
        self.eeg_list = load_eeg_data(self.eeg_path, self.electrode_names,
                                      self.bads, seizure=0,
                                      electrode_file=self.electrode_file)

    def create_epi_image_map(self, low_freq: float, high_freq: float,
                             seizure: int = 1, D: float = 3, dt: float = 0.2,
                             method: str = 'welch') -> None:
        """ calculate the source image map analagous to David et al 2011
            and the Brainstorm tutorial

        """
        idx = seizure-1
        self.epi_image = EpiImage(self.eeg_list[idx], self.mri_file, low_freq,
                                  high_freq, self.seiz_delay, method, D, dt)
        # self.t_map = create_epi_image_map(self.eeg, self.mri_file,
        #                                   low_freq, high_freq,
        #                                   self.eeg.seiz_delay)

    def show_epi_image_map(self, cut_coords: Sequence = None,
                           threshold: float = 2) -> None:
        """ Use matplotlib to display the source image map

        """
        self.epi_image.plot(cut_coords=cut_coords, threshold=threshold)

    def calculate_EI(self, freqs: Sequence, bias: float = 1,
                     threshold: float = 1, tau: float = 1,
                     H: float = 5) -> None:
        """ calculate the epileptogenicity index as per Bartolomi et al 2008

        """
        self.EI = calculate_EI(self.eeg_list[0]['seizure']['raw'], freqs,
                               bias=bias, threshold=threshold,
                               tau=tau, H=H)
