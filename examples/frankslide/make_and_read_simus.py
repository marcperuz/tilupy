# -*- coding: utf-8 -*-
"""
Created on Tue Jul 11 18:37:05 2023

@author: peruzzetto
"""

import os
import numpy as np

import swmb.raster
import swmb.cmd
import swmb.download_data
import swmb.models.shaltop.initsimus as shinit

# !!! You may want to change folder_base !!!
FOLDER_BASE = os.path.dirname(os.path.abspath(__file__))

# %% Import rasters from github
# You can skip this step if you cloned the github repository, ad just copy/past
# the files in FOLDER_BASE
folder_data = os.path.join(FOLDER_BASE, 'rasters')
os.makedirs(folder_data, exist_ok=True)
#raster_topo and raster_mass are the paths to the topography and initial mass
#rasters
raster_topo = swmb.download_data.import_frankslide_dem(folder_out=folder_data)
raster_mass = swmb.download_data.import_frankslide_pile(folder_out=folder_data)

# %% Create folder for shaltop simulations
folder_simus = os.path.join(FOLDER_BASE, 'shaltop')
os.makedirs(folder_simus, exist_ok=True)

# %% Read topo and mass
shinit.raster_to_shaltop_txtfile(raster_topo,
                                 os.path.join(folder_simus, 'topography.d'))
axes_props = shinit.raster_to_shaltop_txtfile(raster_mass,
                                              os.path.join(folder_simus, 'init_mass.d'))

# %% Initiate simulations parameters (names are the same as in Shaltop parameter file)
params = dict(nx=axes_props['nx'], ny=axes_props['ny'],
              per=axes_props['nx']*axes_props['dx'],
              pery=axes_props['ny']*axes_props['dy'],
              tmax=100, # Simulation maximum time in seconds (not comutation time)
              dt_im=10, # Time interval (s) between snapshots recordings
              file_z_init = 'topography.d', # Name of topography input file
              file_m_init = 'init_mass.d',# name of init mass input file
              initz=0, # Topography is read from file
              ipr=0, # Initial mass is read from file
              hinit_vert=1, # Initial is given as vertical thicknesses and 
              # must be converted to thicknesses normal to topography
              eps0=1e-13, #Minimum value for thicknesses and velocities
              icomp=1, # choice of rheology (Coulomb with constant basal friction)
              x0=1000, # Min x value (used for plots after simulation is over)
              y0=2000) # Min y value (used for plots after simulation is over)

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
# topo_kwargs = dict(contour_step=10, step_contour_bold=100)
# swmb.cmd.plot_results('shaltop', 'h', 'delta_*.txt', folder_simus,
#                       save=True, display_plot=False, figsize=(15/2.54, 15/2.54),
#                       minval=0.1, maxval=100,
#                       topo_kwargs=topo_kwargs)
# swmb.cmd.plot_results('shaltop', 'h_max', 'delta_*.txt', folder_simus,
#                       save=True, display_plot=False, figsize=(15/2.54, 15/2.54),
#                       cmap_intervals=[0.1, 5, 10, 25, 50, 100],
#                       topo_kwargs=topo_kwargs)

