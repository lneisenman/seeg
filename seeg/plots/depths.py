# -*- coding: utf-8 -*-
""" Encapsulating class and utilities for plotting SEEG depth electrodes

"""

import os

import matplotlib as mpl
import mne
import nibabel as nib
from nibabel.affines import apply_affine
from nibabel.freesurfer import io as fio
import numpy as np
import numpy.linalg as npl
from scipy.optimize import minimize

from . import draw
from .. import utils


SILVER = mpl.colors.to_rgba('#C0C0C0')[:3]
GRAY = (0.5, 0.5, 0.5)
YELLOW = (1, 1, 0)
ROSA_COLOR_LIST = [(1.0, 0.0, 0.0),             # (255, 0, 0),
                   (0.0, 0.4392, 0.7529),       # (0, 112, 192),
                   (0.0, 0.6902, 0.3137),       # (0, 176, 80),
                   (0.4392, 0.1882, 0.6275),    # (112, 48, 160),
                   (0.898, 0.5686, 0.1647),     # (229, 145, 42),
                   (0.0, 0.6902, 0.549),        # (0, 176, 140),
                   (0.5725, 0.8157, 0.3137),    # (146, 208, 80),
                   (0.8392, 0.2706, 0.8196),    # (214, 69, 209),
                   (1.0, 0.9333, 0.6353),       # (255, 238, 162),
                   (0.9451, 0.7569, 0.9412),    # (241, 193, 240),
                   (0.7098, 0.4314, 0.08627),   # (181, 110, 22),
                   (0.702, 0.8392, 0.9922),     # (179, 214, 253)
                   ]


def rosa_colors():
    """Generate a repeating list of colors used by ROSA software for electrodes

    Yields
    ------
    next sequential RGB color from ROSA_COLOR_LIST
    """

    i = 0
    num_colors = len(ROSA_COLOR_LIST)
    while (True):
        yield ROSA_COLOR_LIST[i % num_colors]
        i += 1


class Depth():
    """Class to encapsulate data for individual SEEG electrodes.

    Data for each depth is stored and the contact locations are fit to
    a line. Idealized locations are calcuated from the fit and used for
    drawing the depth.

    Parameters
    ----------
    name : string
        name of Depth
    num_contacts : int
        number of contacts in the depth
    locations : array-like
        list of coordinates of each contact
    diam : float, optional
        diameter of the depth electrode (default = 0.8 mm)
    contact_len : float, optional
        length of each contact (default = 2 mm)
    spacing : float, optional
        distance between contacts (default = 1.5 mm)
    active : boolean or list of boolean, optional
        if a single value, all contacts have that value. Otherwise,
        list of each whether or not each contact is active
        (default = True)
    """

    def __init__(self, name, num_contacts, locations, diam=0.8, contact_len=2,
                 spacing=1.5, active=True):
        self.name = name
        self.num_contacts = num_contacts
        self.locations = locations
        self.diam = diam
        self.contact_len = contact_len
        self.spacing = spacing
        if isinstance(active, bool):
            active = [active]*num_contacts

        self.active = active
        self.fit_locations()

    def _calc_contacts(self, shift):
        """ calculate idealized contact locations based on fit of
        actual locations"""

        contacts = list()
        dist = self.contact_len + self.spacing
        for i in range(self.num_contacts):
            contacts.append(self.tip + shift*self.vector + i*dist*self.vector)

        return contacts

    def _distance(self, x):
        """ cost function for fitting line to contact locations"""

        shift = x[0]
        contacts = self._calc_contacts(shift)
        distance = 0
        for loc, cont in zip(self.locations, contacts):
            diff = loc - cont
            distance += np.linalg.norm(diff)

        return distance

    def fit_locations(self):
        """Fit actual contact locations to a line.

        Notes
        -----
        Finding the point on a given line that is closest to an arbitrary
        point in space involves finding the appropriate perpendicular line
        just like in 2 dimensions. I used this
        [formulation](https://math.stackexchange.com/questions/1521128/given-a-line-and-a-point-in-3d-how-to-find-the-closest-point-on-the-line)
        For a point P and a line defined by the points Q and R,
        the closest point on the line G is given by G=Q+t(R−Q)
        where t=(R−Q)⋅(Q−P)/(R−Q)⋅(R−Q).
        """

        ave = np.mean(self.locations, axis=0)
        centered = self.locations - ave
        __, __, w = np.linalg.svd(centered)
        self.vector = w[0]/np.linalg.norm(w[0])

        P = self.locations[0]
        Q = ave
        R = ave + self.vector

        R_Q = R - Q
        Q_P = Q - P
        numerator = np.dot(R_Q, Q_P)
        denominator = np.dot(R_Q, R_Q)

        t = numerator / denominator

        G = Q - t * R_Q
        self.tip = G
        fit = minimize(self._distance, 0, method='Powell')
        self.shift = fit.x
        # print('shift =', self.shift)
        v_shift = self.shift * self.vector
        dist = self.spacing + self.contact_len
        # print('dist =', dist)
        a = self.tip + v_shift - self.vector*self.contact_len/2
        b = self.tip + v_shift + self.vector*self.contact_len/2
        self.contacts = list()
        for i in range(self.num_contacts):
            self.contacts.append(((a + i*dist*self.vector),
                                  (b + i*dist*self.vector)))

        self.base = G + self.vector*(20 + self.num_contacts*dist)
        self.tip = a    # Make sure tip extends to end of distal contact

    def draw(self, scene, contact_colors=SILVER, depth_color=(1, 0, 0),
             affine=np.diag(np.ones(4))):
        """Draw fit of locations as a cylindrical depth on the provided scene

        Parameters
        ----------
        scene : VTK scene from mayavi or pyvista
            scene on which to draw the Depth
        contact_colors : RGB or list of RBG colors, optional
            If a single color is given, draw all contacts in that color.
            Otherwise list of colors for corresponding contacts
        depth_color : RGB color, optional
            Color used to draw the Depth
        affine : numpy ndarray (4x4)
            affine to convert coords to Freesurfer surface coordinates

        """
        if ((len(contact_colors) == len(self.contacts)) and
           len(contact_colors[0]) == 3):
            c_colors = contact_colors  # list of RGB colors for each contact
        elif len(contact_colors) == 3:
            c_colors = [contact_colors]*len(self.contacts)
        else:
            raise ValueError('contact_colors needs to be an RGB',
                             'value or a list of RGBs')

        tip = apply_affine(affine, self.tip)
        base = apply_affine(affine, self.base)
        draw.draw_cyl(scene, tip, base, self.diam, depth_color, 0.3)
        x, y, z = base
        draw.draw_text(scene, self.name, base, 36, color=depth_color)

        for i, contact in enumerate(self.contacts):
            if self.active[i]:
                tip = apply_affine(affine, contact[0])
                base = apply_affine(affine, contact[1])
                draw.draw_cyl(scene, tip, base, self.diam+0.25,
                              c_colors[i], 1)

    def show_locations(self, scene, colors=YELLOW,
                       affine=np.diag(np.ones(4))):
        """Draw actual contact locations as spheres on the provided scene

        Parameters
        ----------
        scene : VTK scene from mayavi or pyvista
            scene on which to draw the contact locations
        colors : RGB or list of RBG colors, optional
            If a single color is given, draw all contacts in that color.
            Otherwise list of colors for corresponding contacts
        affine : numpy ndarray (4x4)
            affine to convert coords to Freesurfer surface coordinates

        """
        if ((len(colors) == len(self.contacts)) and len(colors[0] == 3)):
            c_colors = colors  # list of RGB colors for each contact
        elif len(colors) == 3:
            c_colors = [colors]*len(self.contacts)
        else:
            raise ValueError('colors needs to be an RGB',
                             'value or a list of RGBs')

        radius = self.contact_len/1.5
        opacity = 0.3
        for i, location in enumerate(self.locations):
            if self.active[i]:
                loc = apply_affine(affine, location)
                draw.draw_sphere(scene, loc, radius, c_colors[i], opacity)


def create_depths(electrode_names, ch_names, electrodes):
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
    for name in electrode_names:
        contacts = electrodes.loc[electrodes.contact.str.startswith(name), :]
        active = [contact in ch_names for contact in contacts.contact]
        # contacts = df.loc[df.include.values, :]
        locations = np.zeros((len(contacts), 3))
        locations[:, 0] = contacts.x.values
        locations[:, 1] = contacts.y.values
        locations[:, 2] = contacts.z.values
        locations *= 1000
        depth = Depth(name, locations.shape[0], locations, active=active)
        depth_list.append(depth)

    return depth_list


def create_depths_plot(depth_list, subject_id, subjects_dir,
                       depth_colors=rosa_colors(), contact_colors=SILVER):
    """Create a MNE Brain and plot Depths. Returns the Brain

    Parameters
    ----------
    depth_list : list
        list of Depth's
    subject_id : string
        name of subject folder in Freesurfer subjects directory
    subjects_dir : string
        location of Freesurfer subjects directory
    depth_colors: list
        list of RGB tuples for the colors of the Depth's
    contact_colors: RGB tuple or list of tuples
        color of all contacts if a single value given. Otherwise list of
        colors for each contact

    Returns
    -------
    brain : MNE Brain
        MNE Brain showing transparent pial surface and Depths
    """

    mri_file = os.path.join(subjects_dir, subject_id, 'mri/T1.mgz')
    mri = nib.load(mri_file)
    mri_inv = npl.inv(mri.affine)
    Torig = mri.header.get_vox2ras_tkr()
    affine = Torig@mri_inv
    Brain = mne.viz.get_brain_class()
    brain = Brain(subject_id, 'both', 'pial', subjects_dir=subjects_dir,
                  cortex='classic', alpha=0.5)
    scene = brain.plotter.renderer

    if type(contact_colors[0]) is not float:
        c_list = True
        idx = 0
    else:
        c_list = False

    for depth, color in zip(depth_list, depth_colors):
        if c_list:
            c_colors = list()
            for i in range(depth.num_contacts):
                if depth.active[i]:
                    c_colors.append(contact_colors[idx])
                    idx += 1
                else:
                    c_colors.append(GRAY)

            depth.draw(scene, contact_colors=c_colors, depth_color=color,
                       affine=affine)
        else:
            depth.draw(scene, contact_colors=contact_colors, depth_color=color,
                       affine=affine)

    return brain


def show_bipolar_values(depth_list, scene, values, bads=[], radius=None,
                        cmap='cold_hot', affine=np.diag(np.ones(4))):
    """Plot contact values as color coded spheres on each Depth contact

    Plot contact values on the translucent pial surface in `fig` from
    the parameter `values` as sphere of radius `radius` on each contact

    Parameters
    ----------
    depth_list : list
        list of Depths
    scene : VTK scene
        figure of translucent pial surface on which to plot Depth's
    values : array-like
        values of each contact.
    bads : list, optional
        list of bad contacts
    radius : float, optional
        radius of spheres
    cmap : matplotlib colormap
        colormap for color coding spheres
    affine : numpy ndarray (4x4)
        affine to convert coords to Freesurfer surface coordinates

    """

    vmin = np.min(values)
    vmax = np.max(values)
    if vmin < 0:
        if abs(vmin) > vmax:
            vmax = abs(vmin)
        else:
            vmin = -vmax

    vmin *= 1.1
    vmax *= 1.1
    norm = mpl.colors.Normalize(vmin, vmax)
    color_map = mpl.cm.get_cmap(cmap)
    mapped_values = color_map(norm(values))
    for depth in depth_list:
        if radius is None:
            radius = depth.contact_len/1.5

        contacts = [depth.name + str(i+1) for i in range(depth.num_contacts)
                    if depth.active[i]]
        anodes, cathodes, __ = utils.setup_bipolar(depth.name, contacts,
                                                   bads)
        start = len(depth.name)
        val_idx = 0
        opacity = 0.3
        for i, (anode, cathode) in enumerate(zip(anodes, cathodes)):
            a_idx = int(anode[start:]) - 1
            c_idx = int(cathode[start:]) - 1
            an = (depth.contacts[a_idx][0] + depth.contacts[a_idx][1])/2
            ca = (depth.contacts[c_idx][0] + depth.contacts[c_idx][1])/2
            midpoint = apply_affine(affine, (an + ca)/2)
            color = (mapped_values[i+val_idx, 0], mapped_values[i+val_idx, 1],
                     mapped_values[i+val_idx, 2])
            draw.draw_sphere(scene, midpoint, radius, color, opacity)
            val_idx += len(anode)
