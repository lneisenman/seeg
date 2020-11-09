# -*- coding: utf-8 -*-
""" Code to draw spheres, cylinders and cubes in Mayavi figures

"""

from tvtk.api import tvtk
from tvtk.common import configure_input_data


def draw_sphere(fig, center, radius, color, opacity):
    sphereSource = tvtk.SphereSource(center=center, radius=radius)
    sphereMapper = tvtk.PolyDataMapper()
    configure_input_data(sphereMapper, sphereSource.output)
    sphereSource.update()
    sphere_prop = tvtk.Property(opacity=opacity, color=color)
    sphereActor = tvtk.Actor(mapper=sphereMapper, property=sphere_prop)
    fig.scene.add_actor(sphereActor)

    return fig


def draw_cyl(fig, tip, base, diam, color, opacity):
    lineSource = tvtk.LineSource(point1=tip, point2=base)
    lineMapper = tvtk.PolyDataMapper()
    configure_input_data(lineMapper, lineSource.output)
    lineSource.update()
    line_prop = tvtk.Property(opacity=0)
    lineActor = tvtk.Actor(mapper=lineMapper, property=line_prop)
    fig.scene.add_actor(lineActor)

    # Create a tube around the line
    tubeFilter = tvtk.TubeFilter()
    configure_input_data(tubeFilter, lineSource.output)
    tubeFilter.radius = diam/2
    tubeFilter.number_of_sides = 50
    tubeFilter.update()
    tubeMapper = tvtk.PolyDataMapper()
    configure_input_data(tubeMapper, tubeFilter.output)
    p = tvtk.Property(opacity=opacity, color=color)
    tubeActor = tvtk.Actor(mapper=tubeMapper, property=p)
    fig.scene.add_actor(tubeActor)

    return fig


def draw_cube(fig, center, x_length, y_length, z_length, color, opacity):
    cube = tvtk.CubeSource(center=center, x_length=x_length, y_length=y_length,
                           z_length=z_length)
    cube_mapper = tvtk.PolyDataMapper()
    configure_input_data(cube_mapper, cube.output)
    cube.update()
    p = tvtk.Property(opacity=opacity, color=color)
    cube_actor = tvtk.Actor(mapper=cube_mapper, property=p)
    fig.scene.add_actor(cube_actor)

    return fig
