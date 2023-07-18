#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun  1 15:27:36 2021

@author: peruzzetto
"""
import os

import numpy as np
import tilupy.read

# Dictionnary with results names lookup table, to match code output names
LOOKUP_NAMES = dict(h='hflow',
                    u='vflow', ux='vflowx', uy='vflowy',
                    hu='momentum', hu2='ek',
                    vol='vol', ekint='ekint')

# Classify results
STATES_OUTPUT = ['h', 'ux', 'uy']
forces = ['facc', 'fcurv', 'ffric', 'fgrav', 'fpression']
FORCES_OUTPUT = []
for axis in ['x', 'y']:
    for f in forces:
        FORCES_OUTPUT.append(f + axis)


def read_params(file):
    """
    Read simulation parameters from file.

    Parameters
    ----------
    file : str
        Parameters file.

    Returns
    -------
    params : dict
        Dictionnary with parameters

    """
    params = dict()
    if not file.endswith('.txt'):
        prefix = file.sep('/')[-1].sep('_')[0]
        file = os.path.join(file, prefix+'_files',
                            prefix+'_documentation.txt')

    with open(file, 'r') as f:
        for line in f:
            (key, val) = line.split('\t')
            try:
                params[key] = float(val)
            except ValueError:
                if val == 'TRUE':
                    params[key] = True
                if val == 'FALSE':
                    params[key] = False
                else:
                    params[key] = val

    return params


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
    # dem = np.flip(dem, axis=0).T
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

    return x, y, dem


def read_asciis(file_prefix, folder=None, ind=None, nodigit=False):
    """
    Read mutiple ascii grid file matching prefix.

    Parameters
    ----------
    file : TYPE
        DESCRIPTION.
    ind : TYPE, optional
        DESCRIPTION. The default is None.

    Returns
    -------
    None.

    """
    if folder is None:
        folder = os.getcwd()

    files_tmp = os.listdir(folder)
    files = []
    for ff in files_tmp:
        # test is file begins with file_prefix
        if ff.startswith(file_prefix) and ff.endswith('.asc'):
            letter = ff.split(file_prefix)[1][0]
            # After file_prefix, character must be either '.' or a digit
            if nodigit:
                if letter == '.':
                    files.append(ff)
            else:
                if letter.isdigit() or letter == '.':
                    files.append(ff)

    files.sort()

    if ind is not None:
        if ind == 'final':
            files = [files[-1]]
        elif ind == 'initial':
            files = [files[0]]
        else:
            files = [ff for ff in files if int(ff[-8:-4]) in ind]

    files = [os.path.join(folder, ff) for ff in files]

    dem = np.loadtxt(files[0], skiprows=6)
    if len(files) > 1:
        dem = dem[:, :, np.newaxis]
    for ff in files[1:]:
        _, _, dem2 = np.loadtxt(ff, skiprows=6)
        dem = np.concatenate((dem, dem2[:, :, np.newaxis]), axis=2)

    dem = np.squeeze(dem)

    return dem


class Results(tilupy.read.Results):
    """Results of shaltop simulations."""

    def __init__(self, text_rheol=None, folder_base=None, **varargs):
        """
        Init simulation results.

        Parameters
        ----------
        file_params : str
            File where simulation parameters will be read

        """
        self.code = 'ravaflow'
        self.prefix = text_rheol
        self.folder_base = folder_base
        file_params = os.path.join(folder_base,
                                   self.prefix + '_results',
                                   self.prefix + '_files',
                                   self.prefix + '_documentation.txt')
        self.params = read_params(file_params)

        self.folder_ascii = os.path.join(folder_base,
                                         self.prefix + '_results',
                                         self.prefix + '_ascii')
        self.folder_files = os.path.join(folder_base,
                                         self.prefix + '_results',
                                         self.prefix + '_files')

        self.set_axes()

        # Get time of outputs
        with open(os.path.join(self.folder_files,
                               self.prefix+'_summary.txt')) as fid:
            self.tim = []
            for line in fid:
                res = line.split()
                if len(res) > 0:
                    if res[0].isdigit():
                        self.tim.append(float(res[4]))

    def set_axes(self, x=None, y=None, **varargs):
        """Set x and y axes."""
        dd = self.params['Cell size']
        x = np.arange(self.params['Western boundary']+dd/2,
                      self.params['Eastern boundary'],
                      dd)
        y = np.arange(self.params['Southern boundary']+dd/2,
                      self.params['Northern boundary'],
                      dd)
        self.x = x
        self.y = y
        self.nx = len(x)
        self.ny = len(y)

    def set_zinit(self, zinit=None):
        """Set zinit, initial topography."""
        path_zinit = os.path.join(self.folder_ascii,
                                  self.prefix+'_elev.asc')
        self.zinit = np.loadtxt(path_zinit, skiprows=6)

    def get_temporal_output(self, name, d=None, t=None, **varargs):
        """
        Read 2D time dependent simulation results.

        Parameters
        ----------
        name : str
            Name of output.
        d : ndarray, optional
            Data to be read. If None, will be computed.
            The default is None.
        t : ndarray, optional
            Time of results snapshots (1D array), matching last dimension of d.
            If None, will be computed. The default is None.
        **varargs : TYPE
            DESCRIPTION.

        """
        # Read thicknesses or velocity components
        file_prefix = self.prefix + '_' + LOOKUP_NAMES[name]
        d = read_asciis(file_prefix, folder=self.folder_ascii)

        return tilupy.read.TemporalResults(name, d, self.tim)

    def get_static_output(self, name, stat,
                          d=None, from_file=False, **varargs):
        """
        Read 2D time dependent simulation results.

        Parameters
        ----------
        name : str
            Name of output.
        d : ndarray, optional
            Data to be read. Last dimension is for time.
            If None, will be computed.
            The default is None.
        **varargs : TYPE
            DESCRIPTION.

        """
        if stat in ['initial', 'final']:
            file_prefix = self.prefix + '_' + LOOKUP_NAMES[name]
            d = read_asciis(file_prefix, folder=self.folder_ascii, ind=stat)
        elif stat == 'max' and from_file:
            file = '{:s}_{:s}_{:s}.asc'.format(self.prefix,
                                               LOOKUP_NAMES[name],
                                               stat)
            file = os.path.join(self.folder_ascii, file)
            d = np.loadtxt(file, skiprows=6)
        else:
            data = self.get_temporal_output(name)
            d = data.get_temporal_stat(stat).d

        return tilupy.read.StaticResults(name+stat, d)
