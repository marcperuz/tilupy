# -*- coding: utf-8 -*-
"""
Created on Tue Jul 11 18:37:05 2023

@author: peruzzetto
"""

import os
import numpy as np

import swmb.dem
import swmb.cmd
import swmb.models.shaltop.initsimus as shinit

# %% Folder of simulations
tmp = os.path.abspath(__file__)
folder_simus = os.path.join(tmp.split('bin')[0], 'data',
                            'frankslide', 'shaltop')

# %% Rasters (asc format, should also work with tif files)
folder_data = os.path.join(tmp.split('bin')[0], 'data',
                            'frankslide', 'rasters')
raster_mass = os.path.join(folder_data, 'Frankslide_pile.asc')
raster_topo = os.path.join(folder_data, 'Frankslide_topography.asc')

# %% Read topo and mass
x, y, z, dx = swmb.dem.read_raster(raster_topo)
x, y, m, dx = swmb.dem.read_raster(raster_mass)
nx = len(x)
ny = len(y)

# %% Compute initial volume
[Fx, Fy] = np.gradient(z, y, x, edge_order=2)
c = (1+Fx**2+Fy**2)**(-1/2)
vol_init = np.sum(m*dx**2)

#%% Convert vertical thickness to thickness normal to the topography
m = m*c

# %% Write Topography and mass file for Shaltop    
file_z = os.path.join(folder_simus, 'topography.d')
np.savetxt(file_z,
           #np.reshape(z, (z.size, 1), order='F'),
           np.reshape(np.flip(z, axis=0), (z.size, 1)),
           fmt='%.12G')

file_m = os.path.join(folder_simus, 'init_mass.d')
np.savetxt(file_m,
           #np.reshape(m, (m.size, 1), order='F'),
           np.reshape(np.flip(m, axis=0), (m.size, 1)),
           fmt='%.12G')

# %% Initiate simulations parameters (names are the same as in Shaltop parameter file)
params = dict(nx=nx, ny=ny, per=x[-1]-x[0],
              pery=y[-1]-y[0],
              tmax=100, # Simulation maximum time in seconds (not comutation time)
              dt_im=10, # Time interval (s) between snapshots recordings
              file_z_init = 'topography.d', # Name of topography input file
              file_m_init = 'init_mass.d',# name of init mass input file
              initz=0, # Topography is read from file
              ipr=0, # Initial mass is read from file
              eps0=1e-13, #Minimum value for thicknesses and velocities
              icomp=1, # choice of rheology (Coulomb with constant basal friction)
              x0=235, # Min x value (used for plots after simulation is over)
              y0=188) # Min y value (used for plots after simulation is over)

# %% Prepare simulations for Coulomb and different values of delta
deltas = [15, 20, 25]
for delta in deltas:
    params_txt = 'delta_{:05.2f}'.format(delta).replace('.', 'p')
    params['folder_output'] = params_txt # Specify folder chere outputs are stored
    params['delta1'] = delta # Specify the friction coefficient
    #Write parameter file
    shinit.write_params_file(params, directory=folder_simus,
                             file_name=params_txt + '.txt')
    #Create folder for results (not done by shlatop!)
    os.makedirs(os.path.join(folder_simus, params_txt), exist_ok=True)
    
# %% RUN SIMULATIONS
# in simulations folder, in prompt :
    # shaltop "" delta_15p00.txt
    # shaltop "" delta_20p00.txt
    # shaltop "" delta_25p00.txt

# %% Plot results
topo_kwargs = dict(contour_step=10, step_contour_bold=100)
swmb.cmd.plot_results('shaltop', 'h', 'delta_*.txt', folder_simus,
                      save=True, display_plot=False, figsize=(15/2.54, 15/2.54),
                      minval=0.1, maxval=100,
                      topo_kwargs=topo_kwargs)
swmb.cmd.plot_results('shaltop', 'h_max', 'delta_*.txt', folder_simus,
                      save=True, display_plot=False, figsize=(15/2.54, 15/2.54),
                      cmap_intervals=[0.1, 5, 10, 25, 50, 100],
                      topo_kwargs=topo_kwargs)

