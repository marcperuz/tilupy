# -*- coding: utf-8 -*-
"""
Created on Tue Jul 11 18:37:05 2023

@author: peruzzetto
"""

import os

# Read an write rasters
import tilupy.raster
# Functions to download examples of elevation and initial mass rasters
import tilupy.download_data
#Submodule used to prepare Shaltop simulations
import tilupy.models.shaltop.initsimus as shinit

# !!! You may want to change folder_base !!!
FOLDER_BASE = os.path.dirname(os.path.abspath(__file__))

# %% Import rasters from github
# You can skip this step if you cloned the github repository, ad just copy/past
# the files in FOLDER_BASE
folder_data = os.path.join(FOLDER_BASE, 'rasters')
os.makedirs(folder_data, exist_ok=True)
#raster_topo and raster_mass are the paths to the topography and initial mass
#rasters
raster_topo = tilupy.download_data.import_frankslide_dem(folder_out=folder_data)
raster_mass = tilupy.download_data.import_frankslide_pile(folder_out=folder_data)

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
    params['folder_output'] = params_txt # Specify folder where outputs are stored
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


