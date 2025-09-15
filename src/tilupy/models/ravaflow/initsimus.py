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


def readme_to_params(folder_data: str) -> dict:
    """Read a README.txt file and convert it to a parameters dictionary.

    Each line of the README file should contain a key and a value separated
    by whitespace. The function parses the file and returns a dictionary
    mapping keys to their corresponding string values.
    
    Parameters
    ----------
    folder_data : str
        Path to the folder containing the README.txt file.

    Returns
    -------
    dict
        Dictionary where keys are parameter names (str) and values are
        their corresponding values (str) read from the file.
    """
    params = dict()
    with open(os.path.join(folder_data, 'README.txt'), 'r') as f:
        for line in f:
            (key, val) = line.split()
            params[key] = val
    return params


def write_params(params: dict) -> str:
    """Convert a parameters dictionary to a formatted string for RavaFlow input.

    Generates a single string in the form "key1=val1 key2=val2 ...",
    suitable for passing as command-line arguments to RavaFlow.

    Parameters
    ----------
    params : dict
        Dictionary of parameters, where keys and values are strings.

    Returns
    -------
    str
        Formatted string containing all parameters in "key=value" format,
        separated by spaces.
    """
    txt = ''
    for key in params:
        txt += '{:s}={:s} '.format(key, params[key])
    return txt[:-1]


def make_simus(law: str, rheol_params: dict, folder_files: str, folder_out: str, readme_file: str) -> None:
    """Write ravaflow initial file for simple slope test case

    Reads simulation parameters from a README file and input
    ASCII rasters, sets topography and mass, prepares numerical and rheological
    parameters, and generates a shell script to execute simulations. Supports
    multiple combinations of rheology parameters.
    
    Parameters
    ----------
    law : str
        Rheological law to use, e.g., "coulomb" or "voellmy".
    rheol_params : dict of list
        Dictionary of rheology parameters, each key maps to a list of values
        defining multiple simulation runs.
    folder_files : str
        Path to folder containing input ASCII files "topo.asc" and "mass.asc".
    folder_out : str
        Path to the folder where output simulation folders and run scripts
        will be created.
    readme_file : str
        Path to the README.txt file containing simulation metadata and parameters.
        
    Returns
    -------
    None
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
