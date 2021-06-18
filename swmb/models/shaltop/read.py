#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun  1 15:27:36 2021

@author: peruzzetto
"""
import os

import numpy as np
import swmb.read

# Dictionnary with results names lookup table, to match code output names
LOOKUP_NAMES = dict(h='rho',
                    u='unorm', ux='u', uy='ut',
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
    if file is None:
        params = None
    else:
        params = dict()
        with open(file, 'r') as f:
            for line in f:
                (key, val) = line.split(' ')
                try:
                    params[key] = float(val)
                except ValueError:
                    params[key] = val.rstrip()
        params['nx'] = int(params['nx'])
        params['ny'] = int(params['ny'])

    return params


def read_file_bin(file, nx, ny):
    """Read shaltop .bin result file."""
    data = np.fromfile(file, dtype=np.float32)
    nbim = int(np.size(data)/nx/ny)
    data = np.reshape(data, (nx, ny, nbim), order='F')
    data = np.transpose(np.flip(data, axis=1), (1, 0, 2))

    return data


def read_file_init(file, nx, ny):
    """Read shaltop initial .d files."""
    data = np.loadtxt(file)
    data = np.reshape(data, (nx, ny), order='F')
    data = np.transpose(np.flip(data, axis=1), (1, 0))
    return data


class Results(swmb.read.Results):
    """Results of shaltop simulations."""

    def __init__(self, text_rheol, folder_base, **varargs):
        """
        Init simulation results.

        Parameters
        ----------
        file_params : str
            File where simulation parameters will be read

        """
        file_params = os.path.join(folder_base, text_rheol + '.txt')
        self.folder_base = folder_base
        self.code = 'shaltop'
        self.params = read_params(file_params)
        self.set_axes()

        # Folder where results are stored
        if 'folder_output' not in self.params:
            self.folder_output = os.path.join(self.folder_base,
                                              'data2')
        else:
            self.folder_output = os.path.join(self.folder_base,
                                              self.params['folder_output'])

        # Get time of outputs
        self.tim = np.loadtxt(os.path.join(self.folder_output, 'time_im.d'))

    def set_axes(self, x=None, y=None, **varargs):
        """Set x and y axes."""
        if 'x0' in self.params and 'y0' in self.params:
            x0 = self.params['x0']
            y0 = self.params['y0']
        elif 'x0' in varargs and 'y0' in varargs:
            x0 = varargs['x0']
            y0 = varargs['y0']
        else:
            x0 = 0
            y0 = 0

        nx = self.params['nx']
        ny = self.params['ny']
        dx = self.params['per']/nx
        dy = self.params['pery']/ny
        x = dx*np.arange(nx)+dx/2
        y = dy*np.arange(ny)+dy/2

        try:
            coord_pos = varargs['coord_pos']
        except KeyError:
            coord_pos = 'bottom_left'

        if coord_pos == 'bottom_left':
            x = x+x0
            y = y+y0
        elif coord_pos == 'upper_right':
            x = x+x0-x[-1]
            y = y+y0-y[-1]
        elif coord_pos == 'upper_left':
            x = x+x0
            y = y+y0-y[-1]
        elif coord_pos == 'lower_right':
            x = x+x0-x[-1]
            y = y+y0

        self.x = x
        self.y = y
        self.nx = nx
        self.ny = ny

    def set_zinit(self, zinit=None):
        """Set zinit, initial topography."""
        path_zinit = os.path.join(self.folder_base,
                                  self.params['file_z_init'])
        zinit = read_file_init(path_zinit, self.nx, self.ny)
        self.zinit = zinit

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
        if name in STATES_OUTPUT:
            file = os.path.join(self.folder_output,
                                LOOKUP_NAMES[name] + '.bin')
            d = read_file_bin(file, self.nx, self.ny)
            t = self.tim

        # Compute the velocity
        if name == 'u':
            d = self.get_u()
            t = self.tim

        if name in ['hu', 'hu2']:
            fileh = os.path.join(self.folder_output,
                                 'rho.bin')
            h = read_file_bin(fileh, self.nx, self.ny)
            u = self.get_u()
            if name == 'hu':
                d = h*u
            elif name == 'hu2':
                d = h*u**2
            t = self.tim

        return swmb.read.TemporalResults(name, d, t)

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
        if from_file:
            file = os.path.join(self.folder_output,
                                LOOKUP_NAMES[name] + stat + '.bin')
            d = np.squeeze(read_file_bin(file, self.nx, self.ny))
        else:
            data = self.get_temporal_output(name)
            d = data.get_temporal_stat(stat).d

        return swmb.read.StaticResults(name+stat, d)

    def get_u(self):
        """ Compute velocity norm from results """
        file = os.path.join(self.folder_output,
                            'u' + '.bin')
        u = read_file_bin(file, self.nx, self.ny)
        file = os.path.join(self.folder_output,
                            'ut' + '.bin')
        ut = read_file_bin(file, self.nx, self.ny)

        [Fx, Fy] = np.gradient(self.zinit, self.x, self.y)
        u = u*self.costh[:, :, np.newaxis]
        ut = ut*self.costh[:, :, np.newaxis]
        d = np.sqrt(u**2 + ut**2 + (Fx[:, :, np.newaxis]*u
                                    + Fy[:, :, np.newaxis]*ut)**2)

        return d
