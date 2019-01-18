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


class Seeg():
    pass
