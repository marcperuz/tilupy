#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 26 11:03:56 2021

@author: peruzzetto
"""

import os
import numpy as np
import itertools
import platform

readme_param_match = dict(tmax='tmax',
                          CFL='cflhyp',
                          h_min='eps0',
                          dt_im_output='dt_im')

def readme_to_params(folder_data):

    params=dict()
    with open(os.path.join(folder_data, 'README.txt'), 'r') as f:
        for line in f:
            (key,val)=line.split()
            params[key] = val
    return params

def write_params(params):
    txt = ''
    for key in params:
        txt += '{:s}={:s} '.format(key, params[key])
    return txt[:-1]

def make_simus(law, rheol_params, folder_data, folder_out):
    """
    Write shaltop initial file for simple slope test case

    Parameters
    ----------
    deltas : TYPE
        DESCRIPTION.
    folder_in : TYPE
        DESCRIPTION.
    folder_out : TYPE
        DESCRIPTION.

    Returns
    -------
    None.

    """

    # Parameters from README.txt file
    params_readme = readme_to_params(folder_data)
    # Parameters for simulation
    params = dict()
    # Directory where run script will be created
    folder_law = os.path.join(folder_out, law)
    os.makedirs(folder_law, exist_ok=True)

    # Get topography and initial mass, and write them in Shaltop format
    zfile = os.path.join(folder_data, 'topo.asc')
    mfile = os.path.join(folder_data, 'mass.asc')

    ## Prepare simulation parameters

    # Topography and initial mass
    params['elevation'] = 'elev'
    params['hrelease'] = 'minit'

    # Ambient parameters
    params["ambient"] = '0,0,0,0'
    # Control parameters
    params["controls"] = '0,0,0,0,0,0'

    # Thresholds parameters
    # (min thicknees for output, min flow knetic energy for output,
    # min flow pressure for output, min thickness in simulation)
    h0 = np.float(params_readme['h_min'])
    params["thresholds"]="{:.2E},{:.2E},{:.2E},{:.2E}".format(h0, h0, h0, h0)

    # CFL parameters
    # (CFL condition, initial time step when CFL not applicable)
    params['cfl']="{:.2E},0.0001".format(np.float(params_readme['CFL']))

    # Output time parameters
    params['time'] = "{:.3f},{:.2f}".format(np.float(params_readme['dt_im_output']),
                                            np.float(params_readme['tmax']))

    #Simulation phases
    params['phases'] = 's'

    ### Write bash run file
    file_txt = ""
    file_path = os.path.join(folder_law, 'run.avaflow.sh')

    # Set region
    grid_params = dict()
    with open(zfile, 'r') as fid:
        for i in range(6):
            tmp = fid.readline().split()
            grid_params[tmp[0]] = float(tmp[1])
    e = grid_params['cellsize']*grid_params['ncols']
    n = grid_params['cellsize']*grid_params['nrows']
    tmp = 'g.region s=0 w=0 e={:.2f} n={:.2f} res={:.4f}\n'
    file_txt += tmp.format(e, n, grid_params['cellsize'])
    file_txt += 'g.region -s\n\n'

    # Read ascii files for mass and topography
    file_txt += 'r.in.gdal --overwrite input=../../topo.asc output=elev\n'
    file_txt += 'r.in.gdal --overwrite input=../../mass.asc output=minit\n'

    # Prepare rheological params
    param_names = [param for param in rheol_params]
    param_vals = [rheol_params[param] for param in rheol_params]
    n_params = len(param_names)

    # prefix for output folder
    prefix = ''
    for param_name in param_names:
        if param_name.startswith('delta'):
            prefix += param_name + '_{:05.2f}_'
        elif param_name == 'ksi':
            prefix += param_name + '_{:06.1f}_'
    prefix = prefix[:-1]

    for param_set in zip(*param_vals):

        simu_prefix = prefix.format(*param_set).replace('.', 'p')
        params['prefix'] = simu_prefix
        if law == 'coulomb':
            params['friction'] = '{:.2f},{:.2f}'.format(param_set[0],
                                                         param_set[0])
        file_txt += '\n'
        file_txt += 'start_time=`date +%s`\n'
        file_txt += 'r.avaflow -e ' + write_params(params) + '\n'
        file_txt += 'end_time=`date +%s`\n'
        file_txt += 'elapsed_time=$(($end_time - $start_time))\n'
        file_txt += ('string_time="${start_time} ' +
                     simu_prefix + ' ${elapsed_time}"\n')
        file_txt += 'echo ${string_time} >> simulation_duration.txt\n\n'


    with open(file_path, "w") as fid:
        fid.write(file_txt)
