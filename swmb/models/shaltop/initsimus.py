#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 25 15:18:31 2021

@author: peruzzetto
"""

import os
import numpy as np

import swmb.notations

README_PARAM_MATCH = dict(tmax='tmax',
                          CFL='cflhyp',
                          h_min='eps0',
                          dt_im_output='dt_im')

SHALTOP_LAW_ID = dict(coulomb=1,
                      voellmy=8,
                      bingham=6,
                      muI=7)


def read_ascii(file):
    """
    Read ascii grid file to numpy ndarray.

    Parameters
    ----------
    file : TYPE
        DESCRIPTION.

    Returns
    -------
    None.

    """
    dem = np.loadtxt(file, skiprows=6)
    dem = np.flip(dem, axis=0).T
    grid = {}
    with open(file, 'r') as fid:
        for i in range(6):
            tmp = fid.readline().split()
            grid[tmp[0]] = float(tmp[1])
    try:
        x0 = grid['xllcenter']
        y0 = grid['yllcenter']
    except KeyError:
        x0 = grid['xllcorner']
        y0 = grid['yllcorner']
    nx = int(grid['ncols'])
    ny = int(grid['nrows'])
    dx = dy = grid['cellsize']
    x = np.linspace(x0, x0+(nx-1)*dx, nx)
    y = np.linspace(y0, y0+(ny-1)*dy, ny)

    return x, y, dem, dx


def write_params_file(params, sup_data={}, directory=None,
                      file_name='params.txt'):
    """
    Write params file for shaltop simulations

    Parameters
    ----------
    params : TYPE
        DESCRIPTION.
    sup_data : TYPE, optional
        DESCRIPTION. The default is {}.
    directory : TYPE, optional
        DESCRIPTION. The default is None.
    file_name : TYPE, optional
        DESCRIPTION. The default is 'params.txt'.

    Returns
    -------
    None.

    """

    if directory is None:
        directory = os.getcwd()
    with open(os.path.join(directory, file_name), 'w') as file_params:
        for name in params:
            val = params[name]
            if type(val) == int or type(val) == np.int64:
                file_params.write('{:s} {:d}\n'.format(name, val))
            if type(val) == float or type(val) == np.float64:
                file_params.write('{:s} {:.8G}\n'.format(name, val))
            if type(val) == str:
                file_params.write('{:s} {:s}\n'.format(name, val))


def make_simus(law, rheol_params, folder_data, folder_out, readme_file):
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
    # Get topography and initial mass, and write them in Shaltop format
    zfile = os.path.join(folder_data, 'topo.asc')
    mfile = os.path.join(folder_data, 'mass.asc')
    x, y, z, dx = read_ascii(zfile)
    _, _, m, _ = read_ascii(mfile)
    np.savetxt(os.path.join(folder_out, 'z.d'), z.T.flatten())
    np.savetxt(os.path.join(folder_out, 'm.d'), m.T.flatten())

    # Get simulation parameters from README.txt and raster .asc files
    params = swmb.notations.readme_to_params(readme_file, README_PARAM_MATCH)
    params['nx'] = len(x)
    params['ny'] = len(y)
    params['per'] = dx*len(x)
    params['pery'] = dx*len(y)
    params['file_m_init'] = '../m.d'
    params['file_z_init'] = '../z.d'

    # Folder for rheological law, and set params accordingly
    folder_law = os.path.join(folder_out, law)
    params['icomp'] = SHALTOP_LAW_ID[law]

    param_names = [param for param in rheol_params]

    texts = swmb.notations.make_rheol_string(rheol_params, law)

    # Run shaltop file
    run_shaltop_file = os.path.join(folder_law, 'run_shaltop.sh')
    file_txt = ""

    for i in range(len(rheol_params[param_names[0]])):

        simu_text = texts[i]
        for param_name in param_names:
            params[param_name] = rheol_params[param_name][i]
        params['folder_output'] = simu_text
        os.makedirs(os.path.join(folder_law, simu_text), exist_ok=True)
        write_params_file(params, directory=folder_law,
                          file_name=simu_text+'.txt')
        file_txt += 'start_time=`date +%s`\n'
        file_txt += 'shaltop "" ' + simu_text + '.txt\n'
        file_txt += 'end_time=`date +%s`\n'
        file_txt += 'elapsed_time=$(($end_time - $start_time))\n'
        file_txt += ('string_time="${start_time} ' +
                     simu_text + ' ${elapsed_time}"\n')
        file_txt += 'echo ${string_time} >> simulation_duration.txt\n\n'

    with open(run_shaltop_file, "w") as fid:
        fid.write(file_txt)
