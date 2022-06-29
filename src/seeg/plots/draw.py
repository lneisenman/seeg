# -*- coding: utf-8 -*-
""" Code to draw spheres, cylinders and cubes in Pyvista figures

"""

from typing import Sequence, Tuple
import pyvista as pv
import vtk


def draw_sphere(plotter: pv.Plotter, center: Sequence,
                radius: float, color: Sequence,
                opacity: float) -> Tuple[pv.PolyData, vtk.vtkActor]:
    sphere = pv.Sphere(radius=radius, center=center)
    actor = plotter.add_mesh(sphere, opacity=opacity, color=color)
    return sphere, actor


def draw_cyl(plotter: pv.Plotter, tip: Sequence, base: Sequence,
             diam: float, color: Sequence,
             opacity: float) -> Tuple[pv.PolyData, vtk.vtkActor]:
    line = pv.Line(tip, base)
    tube = line.tube(radius=diam/2)
    actor = plotter.add_mesh(tube, opacity=opacity, color=color)
    return tube, actor


def draw_cube(plotter: pv.Plotter, center: Sequence, x_len: float,
              y_len: float, z_len: float, color: Sequence,
              opacity: float) -> Tuple[pv.PolyData, vtk.vtkActor]:
    cube = pv.Cube(center, x_len, y_len, z_len)
    actor = plotter.add_mesh(cube, opacity=opacity, color=color)
    return cube, actor


def draw_text(plotter: pv.Plotter, text: str, location: Sequence,
              size: int, color: Sequence) -> None:
    textActor = vtk.vtkBillboardTextActor3D()
    textActor.SetInput(text)
    textActor.SetPosition(*location)
    textActor.GetTextProperty().SetFontSize(size)
    textActor.GetTextProperty().SetColor(*color)

    plotter.renderer.AddActor(textActor)
