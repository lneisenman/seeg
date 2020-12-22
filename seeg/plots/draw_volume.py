# -*- coding: utf-8 -*-
""" Code to draw volume in Mayavi figures

"""

from nibabel.affines import apply_affine
from nilearn.plotting.cm import cold_hot
import numpy as np

from . import draw


def draw_volume(fig, t_map, affine):

    data = t_map.get_fdata()[:, :, :, 0]
    data -= np.min(data)
    data /= np.max(data)
    cutoff = 0.50*np.max(data)
    coords = np.transpose(np.nonzero(data > cutoff))

    xL, yL, zL = t_map.header.get_zooms()[:3]

    for coord in coords:
        cube = apply_affine(affine, coord)
        color = cold_hot(data[tuple(coord)])[:3]
        draw.draw_cube(fig, cube, xL, yL, zL, color, 0.5)

    return fig
