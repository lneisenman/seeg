# -*- coding: utf-8 -*-
""" Code to draw spheres, cylinders and cubes in Pyvista figures

"""

import pyvista as pv
import vtk


def draw_sphere(plotter, center, radius, color, opacity):
    sphere = pv.Sphere(radius=radius, center=center)
    plotter.add_mesh(sphere, opacity=opacity, color=color)


def draw_cyl(plotter, tip, base, diam, color, opacity):
    line = pv.Line(tip, base)
    tube = line.tube(radius=diam/2)
    plotter.add_mesh(tube, opacity=opacity, color=color)


def draw_cube(plotter, center, x_len, y_len, z_len, color, opacity):
    cube = pv.Cube(center, x_len, y_len, z_len)
    plotter.add_mesh(cube, opacity=opacity, color=color)


def draw_text(plotter, text, location, size, color):
    textActor = vtk.vtkBillboardTextActor3D()
    textActor.SetInput(text)
    textActor.SetPosition(*location)
    textActor.GetTextProperty().SetFontSize(size)
    textActor.GetTextProperty().SetColor(*color)

    plotter.renderer.AddActor(textActor)
