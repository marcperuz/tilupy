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
LOOKUP_NAMES = dict(h='rho', hvert='rho',
                    u='unorm', ux='u', uy='ut',
                    hu='momentum', hu2int='ek',
                    vol='vol')

# Classify results
STATES_OUTPUT = ['h', 'ux', 'uy', 'hvert']
forces = ['facc', 'fcurv', 'ffric', 'fgrav', 'fpression']
FORCES_OUTPUT = []
for axis in ['x', 'y']:
    for f in forces:
        FORCES_OUTPUT.append(f + axis)
        
INTEGRATED_OUTPUT = ['ek', 'ep', 'etot']


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

def get_axes(**varargs):
    """Set x and y axes."""
    if 'x0' in varargs :
        x0 = varargs['x0']
    else:
        x0 = 0
    if 'y0' in varargs:
        y0 = varargs['y0']
    else:
        y0 = 0

    nx = varargs['nx']
    ny = varargs['ny']
    dx = varargs['per']/nx
    dy = varargs['pery']/ny
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

    return x, y

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


class Results(tilupy.read.Results):
    """Results of shaltop simulations."""

    def __init__(self, file_params=None, folder_base=None, **varargs):
        """
        Init simulation results.

        Parameters
        ----------
        file_params : str
            File where simulation parameters will be read

        """
        super().__init__()
        
        if folder_base is None:
            folder_base = os.getcwd()
        if file_params is None:
            file_params = 'params.txt'
            
        if '.' not in file_params:
            file_params = file_params + '.txt'
            
        file_params = os.path.join(folder_base, file_params)
        
        params = read_params(file_params)
        x, y = get_axes(**params)
        
        varargs.update(dict(code='shaltop',
                            htype='normal',
                            params=params,
                            x=x, y=y,
                            nx=len(x), ny=len(y)))
        
        for key in varargs:
            setattr(self, key, varargs[key])

        self.folder_base = folder_base
        # Folder where results are stored
        if 'folder_output' not in self.params:
            self.folder_output = os.path.join(self.folder_base,
                                              'data2')
        else:
            self.folder_output = os.path.join(self.folder_base,
                                              self.params['folder_output'])

        # Get time of outputs
        self.tim = np.loadtxt(os.path.join(self.folder_output, 'time_im.d'))

    
        
    @property
    def zinit(self):
        """ Compute or get cos(slope) of topography """
        if self._zinit is None:
            self.set_zinit()
        return self._zinit

    def set_zinit(self, zinit=None):
        """Set zinit, initial topography."""
        path_zinit = os.path.join(self.folder_base, self.folder_output,
                                  'z.bin')
        if not os.path.isfile(path_zinit) and 'file_z_init' in self.params:
            path_zinit = os.path.join(self.folder_base,
                                      self.params['file_z_init'])
            self._zinit = read_file_init(path_zinit, self.nx, self.ny)  
        else:
            self._zinit = np.squeeze(read_file_bin(path_zinit, self.nx, self.ny))

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
            if name == 'hvert':
                d = d/self.costh[:, :, np.newaxis]
            t = self.tim
            
        # Raed integrated kinetic energy
        if name in ['hu2int']:
            array = np.loadtxt(os.path.join(self.folder_output,
                                            LOOKUP_NAMES[name] + '.d'))
            d = array[:, 1]
            t = array[:, 0]

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
            
        if 'h_thresh' in varargs and varargs['h_thresh'] is not None and d.ndim==3:
            d = tilupy.read.use_thickness_threshold(self, d,
                                                  varargs['h_thresh'])

        return tilupy.read.TemporalResults(name, d, t)

    def get_static_output(self, name,
                          d=None, from_file=True, **varargs):
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
        if name in tilupy.read.COMPUTED_STATIC_DATA_2D:
            state, stat = name.split('_')
            if stat in ['final', 'initial']:
                hh = self.get_temporal_output(state)
                if state == 'hvert':
                    hh.d = hh.d/self.costh[:, :, np.newaxis]
                return hh.get_temporal_stat(stat)
            if from_file:
                file = os.path.join(self.folder_output,
                                    LOOKUP_NAMES[state] + stat + '.bin')
                if os.path.isfile(file):
                    d = np.squeeze(read_file_bin(file, self.nx, self.ny))
                else:
                    print(file + ' was not found, ' + name + '_' + stat
                          + ' computed from temporal output.')
                    from_file = False
            if not from_file:
                data = self.get_temporal_output(state)
                if state == 'hvert':
                    hh.d = hh.d/self.costh
                d = data.get_temporal_stat(stat).d
        else:
             raise(NotImplementedError())   

        return tilupy.read.StaticResults(name, d)

    def get_u(self):
        """ Compute velocity norm from results """
        file = os.path.join(self.folder_output,
                            'u' + '.bin')
        u = read_file_bin(file, self.nx, self.ny)
        file = os.path.join(self.folder_output,
                            'ut' + '.bin')
        ut = read_file_bin(file, self.nx, self.ny)

        [Fx, Fy] = np.gradient(self.zinit, self.y, self.x)
        u = u*self.costh[:, :, np.newaxis]
        ut = ut*self.costh[:, :, np.newaxis]
        d = np.sqrt(u**2 + ut**2 + (Fx[:, :, np.newaxis]*u
                                    + Fy[:, :, np.newaxis]*ut)**2)

        return d
