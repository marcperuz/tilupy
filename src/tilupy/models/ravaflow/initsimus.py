#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 26 11:03:56 2021

@author: peruzzetto
"""

import os
import numpy as np

import tilupy.notations

readme_param_match = dict(tmax='tmax',
                          CFL='cflhyp',
                          h_min='eps0',
                          dt_im_output='dt_im')


def readme_to_params(folder_data):
    """
    Transform readme to parameters dictionnary.

    Parameters
    ----------
    folder_data : TYPE
        DESCRIPTION.

    Returns
    -------
    params : TYPE
        DESCRIPTION.

    """
    params = dict()
    with open(os.path.join(folder_data, 'README.txt'), 'r') as f:
        for line in f:
            (key, val) = line.split()
            params[key] = val
    return params


def write_params(params):
    """
    Write parameters to string for ravaflow input.

    Parameters
    ----------
    params : TYPE
        DESCRIPTION.

    Returns
    -------
    TYPE
        DESCRIPTION.

    """
    txt = ''
    for key in params:
        txt += '{:s}={:s} '.format(key, params[key])
    return txt[:-1]


def make_simus(law, rheol_params, folder_files, folder_out, readme_file):
    """
    Write shaltop initial file for simple slope test case.

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
    params_readme = tilupy.notations.readme_to_params(readme_file)
    # Parameters for simulation
    params = dict()
    # Directory where run script will be created
    folder_law = os.path.join(folder_out, law)
    os.makedirs(folder_law, exist_ok=True)

    # Prepare simulation parameters

    # Topography and initial mass
    params['elevation'] = 'elev'
    params['hrelease'] = 'minit'

    # Control parameters
    params["controls"] = '0,0,0,0,0,0'

    # Thresholds parameters
    # (min thicknees for output, min flow knetic energy for output,
    # min flow pressure for output, min thickness in simulation)
    h0 = np.float(params_readme['h_min'])
    params["thresholds"] = "{:.2E},{:.2E},{:.2E},{:.2E}".format(h0, h0, h0, h0)

    # CFL parameters
    # (CFL condition, initial time step when CFL not applicable)
    params['cfl'] = "{:.2E},0.0001".format(np.float(params_readme['CFL']))

    # Output time parameters
    params['time'] = "{:.3f},{:.2f}".format(np.float(params_readme['dt_im_output']),
                                            np.float(params_readme['tmax']))

    # Simulation phases
    params['phases'] = 's'

    # Write bash run file
    file_txt = ""
    file_path = os.path.join(folder_law, 'run.avaflow.sh')

    # Set region
    grid_params = dict()
    with open(os.path.join(folder_files, 'topo.asc'), 'r') as fid:
        for i in range(6):
            tmp = fid.readline().split()
            grid_params[tmp[0]] = float(tmp[1])
    e = grid_params['cellsize']*grid_params['ncols']
    n = grid_params['cellsize']*grid_params['nrows']
    tmp = 'g.region s=0 w=0 e={:.2f} n={:.2f} res={:.4f}\n'
    file_txt += tmp.format(e, n, grid_params['cellsize'])
    file_txt += 'g.region -s\n\n'

    # Read ascii files for mass and topography
    # Get relative path to topo and mass file, from the fodler
    # where simulations will be run. Find position of folder_files in
    # folder_out path. Simulations are run in folder_out/law/
    path = os.path.normpath(folder_out)
    ffs = path.split(os.sep)
    _, folder_topo = os.path.split(folder_files)
    n_up = len(ffs) - ffs.index(folder_topo)
    path_up = '../'*n_up
    zfile = path_up + 'topo.asc'
    mfile = path_up + 'mass.asc'
    file_txt += 'r.in.gdal --overwrite input={:s} output=elev\n'.format(zfile)
    file_txt += 'r.in.gdal --overwrite input={:s} output=minit\n'.format(mfile)

    # prefix for output folder
    prefixs = tilupy.notations.make_rheol_string(rheol_params, law)

    param_names = [param for param in rheol_params]

    txt_friction = '{:.2f},{:.2f},{:.2f}'

    for i in range(len(rheol_params[param_names[0]])):

        simu_prefix = prefixs[i]
        if law == 'coulomb':
            delta1 = rheol_params['delta1'][i]
            params['friction'] = txt_friction.format(delta1, delta1, 0)

        if law == 'voellmy':
            delta1 = rheol_params['delta1'][i]
            ksi = rheol_params['ksi'][i]
            params['friction'] = txt_friction.format(delta1, delta1,
                                                     np.log10(ksi))

        params['prefix'] = simu_prefix

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
