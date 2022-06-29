# -*- coding: utf-8 -*-
""" Code to draw volume in PyVista figures

"""

from nibabel.affines import apply_affine
from nibabel.nifti1 import Nifti1Image
from nilearn.plotting.cm import cold_hot
import numpy as np
import numpy.typing as npt
import pyvista as pv

from . import draw


def draw_volume(plotter: pv.Plotter, t_map: Nifti1Image,
                affine: npt.NDArray) -> None:

    data = t_map.get_fdata()[:, :, :, 0]
    data -= np.min(data)
    data /= np.max(data)
    cutoff = 0.50*np.max(data)
    coords = np.transpose(np.nonzero(data > cutoff))

    xL, yL, zL = t_map.header.get_zooms()[:3]

    for coord in coords:
        cube = apply_affine(affine, coord)
        color = cold_hot(data[tuple(coord)])[:3]
        draw.draw_cube(plotter, cube, xL, yL, zL, color, 0.5)
