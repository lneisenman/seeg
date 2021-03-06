# -*- coding: utf-8 -*-
""" Code to draw spheres, cylinders and cubes in Mayavi and Pyvista figures

"""

import pyvista
import vtk


def draw_sphere(scene, center, radius, color, opacity):
    sphereSource = vtk.vtkSphereSource()
    sphereSource.SetCenter(*center)
    sphereSource.SetRadius(radius)

    sphereMapper = vtk.vtkPolyDataMapper()
    sphereMapper.SetInputConnection(sphereSource.GetOutputPort())

    sphereActor = vtk.vtkActor()
    sphereActor.SetMapper(sphereMapper)
    sphereActor.GetProperty().SetColor(*color)
    sphereActor.GetProperty().SetOpacity(opacity)

    scene.add_actor(sphereActor)


def draw_cyl(scene, tip, base, diam, color, opacity):
    lineSource = vtk.vtkLineSource()
    lineSource.SetPoint1(*tip)
    lineSource.SetPoint2(*base)
    lineMapper = vtk.vtkPolyDataMapper()
    lineMapper.SetInputConnection(lineSource.GetOutputPort())

    lineActor = vtk.vtkActor()
    lineActor.SetMapper(lineMapper)
    lineActor.GetProperty().SetOpacity(0)

    tubeFilter = vtk.vtkTubeFilter()
    tubeFilter.SetInputConnection(lineSource.GetOutputPort())
    tubeFilter.SetRadius(diam/2)
    tubeFilter.SetNumberOfSides(50)
    tubeFilter.Update()

    tubeMapper = vtk.vtkPolyDataMapper()
    tubeMapper.SetInputConnection(tubeFilter.GetOutputPort())

    tubeActor = vtk.vtkActor()
    tubeActor.SetMapper(tubeMapper)
    tubeActor.GetProperty().SetColor(*color)
    tubeActor.GetProperty().SetOpacity(opacity)

    scene.add_actor(tubeActor)


def draw_cube(scene, center, x_len, y_len, z_len, color, opacity):
    cubeSource = vtk.vtkCubeSource()
    cubeSource.SetCenter(center)
    cubeSource.SetXLength(x_len)
    cubeSource.SetYLength(y_len)
    cubeSource.SetZLength(z_len)

    cubeMapper = vtk.vtkPolyDataMapper()
    cubeMapper.SetInputConnection(cubeSource.GetOutputPort())

    cubeActor = vtk.vtkActor()
    cubeActor.SetMapper(cubeMapper)
    cubeActor.GetProperty().SetColor(*color)
    cubeActor.GetProperty().SetOpacity(opacity)

    scene.add_actor(cubeActor)


def draw_text(scene, text, location, size, color):
    textActor = vtk.vtkBillboardTextActor3D()
    textActor.SetInput(text)
    textActor.SetPosition(*location)
    textActor.GetTextProperty().SetFontSize(size)
    textActor.GetTextProperty().SetColor(*color)
    
    if isinstance(scene, pyvista.Renderer):
        scene.AddActor(textActor)
    else:
        scene.add_actor(textActor)
