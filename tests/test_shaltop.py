# -*- coding: utf-8 -*-
"""
Created on Wed Jul 19 15:39:38 2023

@author: peruzzetto
"""

import os

import tilupy.models.shaltop.initsimus as shinit


def test_shaltop_raster_to_input(folder_data):
    
    raster_topo = os.path.join(folder_data, 'frankslide', 'rasters', 
                               'Frankslide_topography.asc')
    
    folder_simus = os.path.join(folder_data, 'frankslide', 'shaltop')
    
    file_topo_sh = os.path.join(folder_simus, 'topography.d')
    
    if os.path.isfile(file_topo_sh):
        os.remove(file_topo_sh)
    
    shinit.raster_to_shaltop_txtfile(raster_topo, file_topo_sh)
    
    assert os.path.isfile(file_topo_sh)    

def test_shaltop_make_read_param_file(folder_data):
    
    params = dict(nx=201, ny=201,
                  per=201*20,
                  pery=201*20,
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
    
    deltas = [15, 20, 25]
    folder_simus = os.path.join(folder_data, 'frankslide', 'shaltop')
    files_created = True
    
    for delta in deltas:
        params_txt = 'delta_{:05.2f}'.format(delta).replace('.', 'p')
        param_file_path = os.path.join(folder_simus,params_txt + '.txt')
        if os.path.isfile(param_file_path):
            os.remove(param_file_path)
        params['folder_output'] = params_txt # Specify folder where outputs are stored
        params['delta1'] = delta # Specify the friction coefficient
        #Write parameter file
        shinit.write_params_file(params, directory=folder_simus,
                                 file_name=params_txt + '.txt')
        
        files_created = files_created & os.path.isfile(param_file_path)
        if not files_created:
            break
        
        assert files_created
