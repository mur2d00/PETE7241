import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
import os
import sys
import pandas as pd
import vtk
from vtk.util import numpy_support as VN
import pyvista as pv


Directory = "/Users/murad/Library/CloudStorage/OneDrive-LouisianaStateUniversity/MLPractice/isosurfaces/"
outputfile = Directory + "surface_1.vtp"
print ('Loading', outputfile)
reader = vtk.vtkXMLPolyDataReader()
reader.SetFileName(outputfile)
reader.Update()
data_vtk = reader.GetOutput()
n_points = data_vtk.GetNumberOfPoints()
x_vtk = np.zeros((n_points,1))
y_vtk = np.zeros((n_points,1))
z_vtk = np.zeros((n_points,1))
VTKpoints = vtk.vtkPoints()

for i in range(n_points):
    pt_iso  =  data_vtk.GetPoint(i)
    x_vtk[i] = pt_iso[0]   
    y_vtk[i] = pt_iso[1]
    z_vtk[i] = pt_iso[2]
VTKpoints.InsertPoint(i, pt_iso[0], pt_iso[1], pt_iso[2])

point_data = vtk.vtkUnstructuredGrid()
point_data.SetPoints(VTKpoints)
x  = np.reshape(x_vtk , (np.size(x_vtk [:]),1)) 
y  = np.reshape(y_vtk , (np.size(y_vtk [:]),1))
z  = np.reshape(z_vtk , (np.size(z_vtk [:]),1))

# Extract Ua, Ub, gradUa, gradUb, pa, pff, muI from vtk file
pointData = reader.GetOutput().GetPointData()
Ub = pointData.GetArray('U.b')
Ua = pointData.GetArray('U.a')
pa1 = pointData.GetArray('pa')
muI1 = pointData.GetArray('muI')


UbX = np.zeros((n_points,1))
UbY = np.zeros((n_points,1))
UbZ = np.zeros((n_points,1))
UaX = np.zeros((n_points,1))
UaY = np.zeros((n_points,1))
UaZ = np.zeros((n_points,1))
pa  = np.zeros((n_points,1))
muI = np.zeros((n_points,1))


for i in range(n_points):
        UbX[i] = Ub.GetComponent(i,0)
        UbY[i] = Ub.GetComponent(i,0)
        UbZ[i] = Ub.GetComponent(i,0)
        UaX[i] = Ua.GetComponent(i,0)
        UaY[i] = Ua.GetComponent(i,0)
        UaZ[i] = Ua.GetComponent(i,0)
        pa[i] = pa1.GetValue(i)
        muI[i] = muI1.GetValue(i)



##Plot the vtk data
grid = pv.read("/Users/murad/Library/CloudStorage/OneDrive-LouisianaStateUniversity/MLPractice/isosurfaces/surface_1.vtp")
grid.plot(scalars='U.b', component=0, cmap='turbo', cpos='xy')
