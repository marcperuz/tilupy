# -*- coding: utf-8 -*-
"""
Created on Tue Jul 18 11:27:46 2023

@author: peruzzetto
"""

import os
import matplotlib.pyplot as plt

import tilupy.cmd
import tilupy.read

# !!! You may want to change folder_base !!!
FOLDER_BASE = os.path.dirname(os.path.abspath(__file__))
folder_simus = os.path.join(FOLDER_BASE, 'shaltop')

# %% Get results

# Initiate results
res = tilupy.read.get_results('shaltop', folder_base=folder_simus,
                              file_params = 'delta_25p00.txt')
# Get x, y and z (topography) arrays
x, y, z = res.x, res.y, res.z
plt.imshow(z)
plt.show()

# %% Get simulated thicknesses
res_name = 'h' 
h_res = res.get_output(res_name) # Thicknesses recorded at different times
# h_res.d is a 3D numpy array of dimension (len(x) x len(y) x len(h_res.t))
plt.imshow(h_res.d[:, :, -1]) # Plot final thickness.
plt.show()
t = h_res.t # Get times of simulation outputs.

# %% Get simulated maximum thickness
res_name = 'h_max'
h_max_res = res.get_output(res_name) #h_max_res is read directly from simulation
# results when possible, and is deduced from res.get_output('h') otherwise
# h_max_res.d is a 2D numpy array of dimension (len(x) x len(y))
plt.imshow(h_max_res.d)
plt.show()

# %% Plot results

# Process a single simulation
params_files = 'delta_2*p00.txt' 

# Process all simulations with param file matching pattern
# params_files = 'delta*.txt'

topo_kwargs = dict(contour_step=10, step_contour_bold=100)
tilupy.cmd.plot_results('shaltop', 'h', params_files, folder_simus,
                        save=True, display_plot=False, figsize=(15/2.54, 15/2.54),
                        vmin=0.1, vmax=100,
                        topo_kwargs=topo_kwargs)
tilupy.cmd.plot_results('shaltop', 'h_max', params_files, folder_simus,
                        save=True, display_plot=False, figsize=(15/2.54, 15/2.54),
                        cmap_intervals=[0.1, 5, 10, 25, 50, 100],
                        topo_kwargs=topo_kwargs)

# %% Convert results to rasters
tilupy.cmd.to_raster('shaltop', 'h_max', params_files,
                     folder_simus, fmt='tif')