# -*- coding: utf-8 -*-
""" Encapsulating class and utilities for plotting SEEG depth electrodes"""

import os

import matplotlib as mpl
from mayavi import mlab
from nibabel.freesurfer import io as fio
from nilearn.plotting.cm import cold_hot
import numpy as np
from scipy.optimize import minimize
from surfer import Brain
from tvtk.api import tvtk
from tvtk.common import configure_input_data

from . import gin
from .. import utils

SILVER = mpl.colors.to_rgba('#C0C0C0')[:3]
YELLOW = (1, 1, 0)
ROSA_COLOR_LIST = [(1.0, 0.0, 0.0),             # (255, 0, 0),
                   (0.0, 0.4392, 0.7529),       # (0, 112, 192),
                   (0.0, 0.6902, 0.3137),       # (0, 176, 80),
                   (0.4392, 0.1882, 0.6275),    # (112, 48, 160),
                   (0.898, 0.5686, 0.1647),     # (229, 145, 42),
                   (0.0, 0.6902, 0.549),         # (0, 176, 140),
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
                 spacing=1.5, active=True, cras=np.zeros(3)):
        self.name = name
        self.num_contacts = num_contacts
        self.locations = locations - cras
        self.diam = diam
        self.contact_len = contact_len
        self.spacing = spacing
        if isinstance(active, bool):
            active = [active]*num_contacts

        self.active = active
        self.cras = cras
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

    def draw(self, fig=None, contact_colors=SILVER, depth_color=(1, 0, 0)):
        """Draw fit of locations as a cylindrical depth

        Parameters
        ----------
        fig : mayavi mlab figure
            figure on which to draw the Depth
        contact_colors : RGB or list of RBG colors, optional
            If a single color is given, draw all contacts in that color.
            Otherwise list of colors for corresponding contacts
        depth_color : RGB color, optional
            Color used to draw the Depth

        Returns
        -------
        fig : mayavi mlab figure
            figure showing cylindrical depth electrode
        """

        if ((len(contact_colors) == len(self.contacts)) and
           len(contact_colors[0] == 3)):
            c_colors = contact_colors  # list of RGB colors for each contact
        elif len(contact_colors) == 3:
            c_colors = [contact_colors]*len(self.contacts)
        else:
            raise ValueError('contact_colors needs to be an RGB',
                             'value or a list of RGBs')
        if fig is None:
            fig = mlab.figure()

        lineSource = tvtk.LineSource(point1=self.tip, point2=self.base)
        lineMapper = tvtk.PolyDataMapper()
        configure_input_data(lineMapper, lineSource.output)
        lineSource.update()
        line_prop = tvtk.Property(opacity=0)
        lineActor = tvtk.Actor(mapper=lineMapper, property=line_prop)
        fig.scene.add_actor(lineActor)

        # Create a tube around the line
        tubeFilter = tvtk.TubeFilter()
        configure_input_data(tubeFilter, lineSource.output)
        tubeFilter.radius = self.diam/2
        tubeFilter.number_of_sides = 50
        tubeFilter.update()
        tubeMapper = tvtk.PolyDataMapper()
        configure_input_data(tubeMapper, tubeFilter.output)
        p = tvtk.Property(opacity=0.3, color=depth_color)
        tubeActor = tvtk.Actor(mapper=tubeMapper, property=p)
        fig.scene.add_actor(tubeActor)

        for i, contact in enumerate(self.contacts):
            if self.active[i]:
                contactSource = tvtk.LineSource(point1=contact[0],
                                                point2=contact[1])
                contactMapper = tvtk.PolyDataMapper()
                configure_input_data(contactMapper, contactSource.output)
                contactSource.update()
                contact_prop = tvtk.Property(opacity=0)
                contactActor = tvtk.Actor(mapper=lineMapper,
                                          property=contact_prop)
                fig.scene.add_actor(contactActor)

                contactFilter = tvtk.TubeFilter()
                configure_input_data(contactFilter, contactSource.output)
                contactFilter.radius = (self.diam/2) + 0.125
                contactFilter.number_of_sides = 50
                contactFilter.update()
                contact_tubeMapper = tvtk.PolyDataMapper()
                configure_input_data(contact_tubeMapper, contactFilter.output)
                p_contact = tvtk.Property(opacity=1, color=c_colors[i])
                contact_tubeActor = tvtk.Actor(mapper=contact_tubeMapper,
                                               property=p_contact)
                fig.scene.add_actor(contact_tubeActor)

        return fig

    def show_locations(self, fig=None, colors=YELLOW):
        """Draw actual contact locations as spheres

        Parameters
        ----------
        fig : mayavi mlab figure, optional
            figure on which to draw the Depth
        colors : RGB or list of RBG colors, optional
            If a single color is given, draw all contacts in that color.
            Otherwise list of colors for corresponding contacts

        Returns
        -------
        fig : mayavi mlab figure
            figure showing cylindrical depth electrode
        """

        if fig is None:
            fig = mlab.figure()

        if ((len(colors) == len(self.contacts)) and len(colors[0] == 3)):
            c_colors = colors  # list of RGB colors for each contact
        elif len(colors) == 3:
            c_colors = [colors]*len(self.contacts)
        else:
            raise ValueError('colors needs to be an RGB',
                             'value or a list of RGBs')

        radius = self.contact_len/1.5
        for i, location in enumerate(self.locations):
            if self.active[i]:
                sphereSource = tvtk.SphereSource(center=location,
                                                 radius=radius)
                sphereMapper = tvtk.PolyDataMapper()
                configure_input_data(sphereMapper, sphereSource.output)
                sphereSource.update()
                sphere_prop = tvtk.Property(opacity=0.3, color=c_colors[i])
                sphereActor = tvtk.Actor(mapper=sphereMapper,
                                         property=sphere_prop)
                fig.scene.add_actor(sphereActor)

        return fig


def create_depths(electrode_names, ch_names, electrodes, cras=np.zeros(3)):
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
    cras : numpy nd-array of length 3, optional
        correction for some Freesurfer file coordinates

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
        depth = Depth(name, locations.shape[0], locations, active=active,
                      cras=cras)
        depth_list.append(depth)

    return depth_list


def create_depths_plot(depth_list, subject_id, subjects_dir,
                       depth_colors=rosa_colors(), contact_colors=SILVER):
    """Create a pysurfer Brain and plot Depths. Returns the Brain

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
    brain : pysurfer Brain
        pysurfer Brain showing transparent pial surface and Depths
    """

    brain = Brain(subject_id, 'both', 'pial', subjects_dir=subjects_dir,
                  cortex='ivory', alpha=0.5)
    fig = mlab.gcf()
    for depth, color in zip(depth_list, depth_colors):
        depth.draw(fig=fig, contact_colors=contact_colors, depth_color=color)

    return brain


def show_bipolar_values(depth_list, fig, values, bads=[], radius=None,
                        cmap='cold_hot'):
    """Plot contact values as color coded spheres on each Depth contact

    Plot contact values on the translucent pial surface in `fig` from
    the parameter `values` as sphere of radius `radius` on each contact

    Parameters
    ----------
    depth_list : list
        list of Depths
    fig : Mayavi mlab figure
        figure of translucent pial surface on which to plot Depth's
    values : array-like
        values of each contact.
    bads : list, optional
        list of bad contacts
    radius : float, optional
        radius of spheres
    cmap : matplotlib colormap
        colormap for color coding spheres

    Returns
    -------
    fig : mayavi mlab figure
        figure of translucent pial surface with depth electrodes
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
    print(mapped_values.shape)
    # mpl.pyplot.scatter(range(len(values)), values, color=mapped_values)
    # mpl.pyplot.show()
    print(fig, fig.scene)
    for depth in depth_list:
        if radius is None:
            radius = depth.contact_len/1.5

        contacts = [depth.name + str(i+1) for i in range(depth.num_contacts)
                    if depth.active[i]]
        anodes, cathodes, __ = utils.setup_bipolar(depth.name, contacts,
                                                   bads)
        start = len(depth.name)
        val_idx = 0
        for i, (anode, cathode) in enumerate(zip(anodes, cathodes)):
            a_idx = int(anode[start:]) - 1
            c_idx = int(cathode[start:]) - 1
            an = (depth.contacts[a_idx][0] + depth.contacts[a_idx][1])/2
            ca = (depth.contacts[c_idx][0] + depth.contacts[c_idx][1])/2
            midpoint = (an + ca)/2
            sphereSource = tvtk.SphereSource(center=midpoint, radius=radius)
            sphereMapper = tvtk.PolyDataMapper()
            configure_input_data(sphereMapper, sphereSource.output)
            sphereSource.update()
            color = (mapped_values[i+val_idx, 0], mapped_values[i+val_idx, 1],
                     mapped_values[i+val_idx, 2])
            print(color)
            sphere_prop = tvtk.Property(opacity=0.3, color=color)
            sphereActor = tvtk.Actor(mapper=sphereMapper, property=sphere_prop)
            fig.scene.add_actor(sphereActor)
            val_idx += len(anode)

    return fig


def read_cras(subject_id, subjects_dir):
    """read the RAS at the center of the volume.

    Reading the RAS at the center of the volume allow for correction
    of surfaces that are not centered at (0, 0, 0)

    Parameters
    ----------
    subject_id : string
        name of subject folder in Freesurfer subjects directory
    subjects_dir : string
        location of Freesurfer subjects directory

    Returns
    -------
    cras : ndarray
        RAS at the center of the volume.
    """

    file_name = os.path.join(subjects_dir, subject_id, 'surf/lh.pial')
    coords, faces, info = fio.read_geometry(file_name, read_metadata=True)
    cras = info['cras']
    return cras
