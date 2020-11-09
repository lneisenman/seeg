# -*- coding: utf-8 -*-
""" Code to draw volume in Mayavi figures

"""

from nibabel.affines import apply_affine
import numpy as np

from . import draw


def draw_volume(fig, t_map, cras=np.zeros(3)):
    print(t_map.affine)
    # print(t_map.header)
    center = apply_affine(t_map.affine, (0, 0, 0))
    print(center)
    draw.draw_cube(fig, center-cras, 3, 3, 3, (1, 0, 0), 0.5)

    return fig
