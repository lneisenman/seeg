# -*- coding: utf-8 -*-
""" Code to draw volume in Mayavi figures

"""

from seeg.plots.depths import read_cras
from nibabel.affines import apply_affine
from nilearn.plotting.cm import cold_hot
import numpy as np

from . import draw


def draw_volume(fig, t_map, cras=np.zeros(3)):
    print(f'affine = {t_map.affine}')
    print(np.diag(t_map.affine))
    xL, yL, zL = np.diag(t_map.affine)[:3]
    data = t_map.get_fdata()[:, :, :, 0]
    data -= np.min(data)
    data /= np.max(data)
    print(data.shape)
    cutoff = 0.50*np.max(data)
    print(f'cutoff = {cutoff}')
    coords = np.transpose(np.nonzero(data > cutoff))
    for coord in coords:
        print(f'coord = {coord}')
        cube = apply_affine(t_map.affine, coord)
        color = cold_hot(data[tuple(coord)])[:3]
        print(f'data = {data[tuple(coord)]}')
        print(f'color = {color}')
        draw.draw_cube(fig, cube+cras, xL, yL, zL, color, 0.5)

    return fig
