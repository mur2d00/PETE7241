import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
import os
import sys
import pandas as pd
import vtk
from vtk.util import numpy_support as VN
import pyvista as pv



#

Directory = "/Users/murad/Library/CloudStorage/OneDrive-LouisianaStateUniversity/MLPractice/isosurfaces/"
outputfile = Directory + "surface_1.vtp"

print ('Loading', outputfile)

reader = vtk.vtkXMLPolyDataReader()
reader.SetFileName(outputfile)
reader.Update()
data_vtk = reader.GetOutput()
n_points = data_vtk.GetNumberOfPoints()
print ('n_points of the mesh:' ,n_points)
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

#Plot the vtk data
data = np.genfromtxt("/Users/murad/Library/CloudStorage/OneDrive-LouisianaStateUniversity/MLPractice/output_1.txt", delimiter= ' ');
x_coord = data[:,0]
y_coord = data[:,1]
z_coord = data[:,2]
grid = pv.read("/Users/murad/Library/CloudStorage/OneDrive-LouisianaStateUniversity/MLPractice/isosurfaces/surface_1.vtp")
grid.plot(scalars='U.b', component=0, cmap='turbo', cpos='xy')


