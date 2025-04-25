# -*- coding: utf-8 -*-
""" Encapsulating class and utilities for plotting SEEG depth electrodes

"""

from dataclasses import dataclass
from numbers import Number
import os
from typing import Sequence

from cycler import cycler
import matplotlib as mpl
import mne
import nibabel as nib
from nibabel.affines import apply_affine
from nilearn.plotting.cm import cold_hot
import numpy as np
import numpy.linalg as npl
import numpy.typing as npt
import pandas as pd
import pyvista as pv
from scipy.optimize import minimize

from . import draw
from ..eeg import setup_bipolar
from .. import utils


SILVER = mpl.colors.to_rgba('#C0C0C0')[:3]
GRAY = (0.5, 0.5, 0.5)
YELLOW = (1-1e-12, 1-1e-12, 0)
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


@dataclass
class Depth():
    """Class to encapsulate data for individual SEEG electrodes.

    Data for each depth is stored and the contact locations are fit to
    a line. Idealized locations are calcuated from the fit and used for
    drawing the depth.
    """
    name: str
    contact_names: list[str]
    locations: npt.NDArray
    diam: float = 0.8
    contact_len: float = 2
    spacing: float = 1.5
    active: bool | list[bool] = True

    def __post_init__(self) -> None:
        self.num_contacts = len(self.contact_names)
        if isinstance(self.active, bool):
            self.active = [self.active]*self.num_contacts

        self.blocks = pv.MultiBlock()
        self.actors = list()    # type: ignore
        self.fit_locations()

    def _calc_contacts(self, shift: float) -> list:
        """ calculate idealized contact locations based on fit of
        actual locations"""

        contacts = list()
        dist = self.contact_len + self.spacing
        for i in range(self.num_contacts):
            contacts.append(self.tip + shift*self.vector + i*dist*self.vector)

        return contacts

    def _distance(self, x: npt.NDArray) -> float:
        """ cost function for fitting line to contact locations"""

        shift = x[0]
        contacts = self._calc_contacts(shift)
        distance = 0
        for loc, cont in zip(self.locations, contacts):
            diff = loc - cont
            distance += np.linalg.norm(diff)    # type: ignore

        return distance

    def fit_locations(self) -> None:
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

    def draw(self, plotter: pv.Plotter,
             contact_colors: Sequence = SILVER,
             depth_color: tuple = (1, 0, 0),
             affine: npt.NDArray = np.diag(np.ones(4))) -> None:
        """Draw fit of locations as a cylindrical depth on the provided scene

        Parameters
        ----------
        plotter : pyvista plotter
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
        depth, actor = draw.draw_cyl(plotter, tip, base, self.diam,
                                     depth_color, 0.3)
        self.blocks['depth'] = depth
        self.actors.append(actor)
        x, y, z = base
        draw.draw_text(plotter, self.name, base, 36, color=depth_color)

        for i, contact in enumerate(self.contacts):
            if self.active[i]:  # type: ignore
                tip = apply_affine(affine, contact[0])
                base = apply_affine(affine, contact[1])
                cyl, actor = draw.draw_cyl(plotter, tip, base, self.diam+0.25,
                                           c_colors[i], 1)
                self.blocks[self.contact_names[i]] = cyl
                self.actors.append(actor)

    def show_bipolar(self, plotter: pv.Plotter,
                     contact_colors: Sequence = SILVER,
                     radii: float | Sequence | None = None,
                     opacity: float = 0.75, bads: list = [],
                     affine: npt.NDArray = np.diag(np.ones(4))) -> None:
        """Draw spheres between contacts on the depth in the provided scene

        Parameters
        ----------
        plotter : pyvista plotter
            scene on which to draw the Depth
        contact_colors : RGB or list of RBG colors, optional
            If a single color is given, draw all spheres in that color.
            Otherwise list of colors for corresponding spheres
        radii : float or numpy ndarray, optional
            radius of the sphere (default None)
        affine : numpy ndarray (4x4)
            affine to convert coords to Freesurfer surface coordinates

        """
        self.actors_BP = list()
        contacts = [self.contact_names[i] for i in range(self.num_contacts)
                    if self.active[i]]  # type: ignore
        anodes, cathodes, __ = setup_bipolar(self.name, contacts, bads)
        self.anodes = anodes
        self.cathodes = cathodes
        if (len(contact_colors) >= len(anodes)
                and len(contact_colors[0]) == 3):
            c_colors = contact_colors  # list of RGB colors for each sphere
        elif len(contact_colors) == 3:
            c_colors = [contact_colors]*len(anodes)
        else:
            raise ValueError('contact_colors needs to be an RGB',
                             'value or a list of RGBs')

        if radii is None:
            radii = self.contact_len/1.5

        if isinstance(radii, Number):
            radii = [radii]*len(anodes)

        if len(radii) < len(anodes):        # type: ignore
            raise ValueError('number of radii must at least equal the number'
                             ' of bipolar contacts')

        start = len(self.name)
        for i, (anode, cathode) in enumerate(zip(anodes, cathodes)):
            a_idx = int(anode[start:]) - 1
            c_idx = int(cathode[start:]) - 1
            an = (self.contacts[a_idx][0] + self.contacts[a_idx][1])/2
            ca = (self.contacts[c_idx][0] + self.contacts[c_idx][1])/2
            midpoint = apply_affine(affine, (an + ca)/2)
            sph, actor = draw.draw_sphere(plotter, midpoint,
                                          radii[i],      # type: ignore
                                          c_colors[i], opacity)
            name = 'C' + str(i) + '-C' + str(i+1)
            self.blocks[name] = sph
            self.actors_BP.append(actor)

    def show_locations(self, plotter: pv.Plotter,
                       colors: Sequence = YELLOW,
                       affine: npt.NDArray = np.diag(np.ones(4))) -> None:
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
            if self.active[i]:  # type: ignore
                loc = apply_affine(affine, location)
                draw.draw_sphere(plotter, loc, radius, c_colors[i], opacity)


@dataclass
class Highlights():
    """Class to contain info on contacts to highlight
    
    electrodes is a dict where electrodes are keys and a list of contacts is the value
    colors is either an RBG tuple for a single color or a dict where electrodes are keys and a list of colors is the value"""
    electrodes: dict
    colors: dict | tuple

    def __post_init__(self) -> None:
        if isinstance(self.colors, tuple):
            temp = dict()
            for electrode in self.electrodes.keys():
                temp[electrode] = [self.colors]*len(self.electrodes[electrode])

            self.colors = temp


def create_depths(electrode_names: list, ch_names: list,
                  electrodes: pd.DataFrame) -> list[Depth]:
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
        locations = np.zeros((len(contacts), 3))
        locations[:, 0] = contacts.x.values
        locations[:, 1] = contacts.y.values
        locations[:, 2] = contacts.z.values
        locations *= 1000
        depth = Depth(name, contacts.contact.tolist(), locations,
                      active=active)
        depth_list.append(depth)

    return depth_list


def create_depth_list(depths: pd.DataFrame, inactives: list, ch_names: list,
                      contacts: pd.DataFrame) -> list[Depth]:
    """Create a list of Depths

    For each electrode in `electrode_names` create a Depth and add to a list

    Parameters
    ----------
    electrode_names : list
        list of electrode names (strings)
    inactives: list
        list of inactive electrodes (no EEG recorded from these channels)
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
        active = [(contact in ch_names) and (contact not in inactives)
                  for contact in depth_contacts.contact]
        locations = np.zeros((len(depth_contacts), 3))
        locations[:, 0] = depth_contacts.x.values
        locations[:, 1] = depth_contacts.y.values
        locations[:, 2] = depth_contacts.z.values
        locations *= 1000
        depth_list.append(Depth(depth.name, depth_contacts.contact.tolist(),
                                locations, depth.diam, depth.contact_len,
                                depth.spacing, active=active))

    return depth_list


def create_depths_plot(depth_list: list[Depth], subject_id: str,
                       subjects_dir: str, alpha: float = 0.3,
                       depth_colors: list = ROSA_COLOR_LIST,
                       contact_colors: Sequence = SILVER) -> mne.viz.Brain:
    """Create a MNE Brain and plot Depths. Returns the Brain

    Parameters
    ----------
    depth_list : list
        list of Depth's
    subject_id : string
        name of subject folder in Freesurfer subjects directory
    subjects_dir : string
        location of Freesurfer subjects directory
    alpha: float
        opacity of brain (default = 0.3)
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
    mri_inv = npl.inv(mri.affine)           # type: ignore
    Torig = mri.header.get_vox2ras_tkr()    # type: ignore
    affine = Torig@mri_inv
    Brain = mne.viz.get_brain_class()
    brain = Brain(subject_id, 'both', 'pial', subjects_dir=subjects_dir,
                  cortex='classic', alpha=alpha, show=False)
    plotter = brain.plotter

    if not isinstance(contact_colors[0], float):
        c_list = True
        idx = 0
    else:
        c_list = False

    for depth, color in zip(depth_list, cycler(depth_color=depth_colors)()):
        if c_list:
            c_colors = list()
            for i in range(depth.num_contacts):
                if depth.active[i]:     # type: ignore
                    c_colors.append(contact_colors[idx])
                    idx += 1
                else:
                    c_colors.append(GRAY)

            depth.draw(plotter, contact_colors=c_colors, **color,
                       affine=affine)
        else:
            depth.draw(plotter, contact_colors=contact_colors,
                       **color, affine=affine)

    return brain


def show_depth_bipolar_values(depth_list: list[Depth], plotter: pv.Plotter,
                              values: Sequence,
                              radius: float | Sequence | None = None,
                              bads: list = [],
                              cmap: mpl.colors.Colormap = cold_hot,
                              affine: npt.NDArray =
                              np.diag(np.ones(4))) -> None:
    """Plot contact values as color coded spheres on each Depth contact

    Plot contact values on the translucent pial surface in `fig` from
    the parameter `values` as sphere of radius `radius` on each contact

    Parameters
    ----------
    depth_list : list
        list of Depths
    plotter : PyVista Plotter
        figure of translucent pial surface on which to plot Depth's
    values : array-like
        values of each contact.
    radius : float or array-like, optional
        radius of spheres
    bads : list, optional
        list of bad contacts
    cmap : matplotlib colormap
        colormap for color coding spheres
    affine : numpy ndarray (4x4)
        affine to convert coords to Freesurfer surface coordinates

    """

    if radius is not None and not isinstance(radius, Number):
        assert len(radius) == len(values), ('number of radii'  # type: ignore
                                            ' must equal number of values')

    mapped_values = utils.map_colors(values, cmap)[:, :3]   # type: ignore
    idx = 0
    for depth in depth_list:
        if radius is None:
            radii = [depth.contact_len/1.5]*len(values)
        elif isinstance(radius, Number):
            radii = [radius]*len(values)        # type: ignore
        else:
            radii = radius[idx:]                # type: ignore

        depth.show_bipolar(plotter, mapped_values[idx:],
                           radii=radii, bads=bads, affine=affine)
        idx += len(depth.anodes)


def show_depth_values(depth_list: list[Depth], plotter: pv.Plotter,
                      values: Sequence,
                      radius: float | Sequence | None = None,
                      bads: list = [],
                      cmap: mpl.colors.Colormap = cold_hot,
                      affine: npt.NDArray = np.diag(np.ones(4))) -> None:
    """Plot contact values as color coded spheres on each Depth contact

    Plot contact values on the translucent pial surface in `fig` from
    the parameter `values` as sphere of radius `radius` on each contact

    Parameters
    ----------
    depth_list : list
        list of Depths
    plotter : PyVista Plotter
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

    mapped_values = utils.map_colors(values, cmap)
    idx = 0
    opacity = 0.3
    if radius is None:
        radius = depth_list[0].contact_len/1.5

    if isinstance(radius, Number):
        radius = [radius]*len(values)

    if len(radius) != len(values):  # type: ignore
        raise ValueError('number of radii must equal number of values')

    for depth in depth_list:
        for i in range(depth.num_contacts):
            name = depth.name + str(i+1)
            if depth.active[i] and name not in bads:    # type: ignore
                start, end = depth.contacts[i]
                af_center = apply_affine(affine, (start+end)/2)
                color = mapped_values[idx, :3]  # type: ignore
                draw.draw_sphere(plotter, af_center,
                                 radius[idx],   # type: ignore
                                 color, opacity)
                idx += 1


def highlight_contacts(highlights: Highlights, depth_list: list[Depth],
                       plotter: pv.Plotter,
                       affine: npt.NDArray = np.diag(np.ones(4))) -> None:
    """Plot colored spheres over the contacts listed in `highlights`

    
    Parameters
    ----------
    highlights : Highlights
        dataclass containing contacts to be highlighted and corresponding colors
    depth_list : list
        list of Depths
    plotter : PyVista Plotter
        figure of translucent pial surface on which to plot spheres
    affine : numpy ndarray (4x4)
        affine to convert coords to Freesurfer surface coordinates

    """
    for depth in depth_list:
        if depth.name in highlights.electrodes.keys():
            for i, contact in enumerate(highlights.electrodes[depth.name]):
                coords = apply_affine(affine, np.sum(depth.contacts[contact-1], axis=0)/2)
                color = highlights.colors[depth.name][i]
                draw.draw_sphere(plotter, coords, depth.contact_len/1.5, color, 0.5)
