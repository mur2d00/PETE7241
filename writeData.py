from netCDF4 import Dataset
import fluidfoam
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import os
import timeit



os.system('cls' if os.name == 'nt' else 'clear')
start = timeit.default_timer()

print("########## Reading Mesh ##########")
x, y, z = fluidfoam.readof.readmesh('/work/nazari/2CylsedFoam/', structured=True, precision=12)
nx, ny, nz = x.shape

print("########## Reading Concentration File ##########")
from fluidfoam import readscalar
sol = '/work/nazari/2CylsedFoam/'
time = '2'
alpha1 = fluidfoam.readscalar(sol, time, "alpha.a", True, precision=12)
alpha_a = (np.mean(np.mean(alpha1, 0), 1))

print("########## Reading Velocity File ##########")
ua1 = fluidfoam.readvector(sol, time, "U.a", True, precision=12)
ua_a = (np.mean(np.mean(ua1, 3), 1)) #Averaged by x and z  !!!!!
ua_abox = np.zeros(ua1.shape)
for j, xi in enumerate(x[:, 0, 0]):
            for k, zi in enumerate(z[0, 0, :]):
                ua_abox[:, j, :, k] = ua_a             
# Calculating u'                
ubprim1 = ua1 - ua_abox

urms_a = np.sqrt(np.mean(np.mean((ubprim1[0, :, :, :] * ubprim1[0, :, :, :]), 2),0))
vrms_a = np.sqrt(np.mean(np.mean((ubprim1[1, :, :, :] * ubprim1[1, :, :, :]), 2),0))
wrms_a = np.sqrt(np.mean(np.mean((ubprim1[2, :, :, :] * ubprim1[2, :, :, :]), 2),0))


print("########## Reading Bed Surface Elevation ##########")
#isoalphaa file contains isosurfaces of the bed (alpha.a = 0.5) in x,y,z coordinate format
bedelevation = pd.read_csv(r"/work/nazari/2CylsedFoam/postProcessing/surfaces/isoalphaa_1.raw", delimiter=' ')
xDir = np.linspace (-1,1,100)
yDir = np.linspace (-0.3,0.3,100)
x,y = np.meshgrid(xDir,yDir)
plt.contourf(x,y,bedelevation[:, 2])



stop = timeit.default_timer()
print('Time elapsed: ', round(stop - start), 'seconds')  
