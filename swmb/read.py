#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun  1 14:20:52 2021

@author: peruzzetto
"""

import matplotlib.pyplot as plt
import numpy as np

import os

import swmb.notations as notations
import swmb.plot as plt_fn

RAW_STATES = ['h', 'ux', 'uy']

TEMPORAL_DATA_0D = ['hu2int', 'vol']
TEMPORAL_DATA_1D = ['']
TEMPORAL_DATA_2D = ['h', 'u', 'ux', 'uy', 'hu', 'hu2']
STATIC_DATA_0D = []
STATIC_DATA_1D = []
STATIC_DATA_2D = []

NP_OPERATORS = ['max', 'mean', 'std', 'sum', 'min']
OTHER_OPERATORS = ['final', 'initial']

for stat in NP_OPERATORS + OTHER_OPERATORS:
    for name in TEMPORAL_DATA_2D:
        STATIC_DATA_2D.append(name+stat)


class TemporalResults:
    """ Time dependent result of simulation """

    def __init__(self, name, d, t, x=None, y=None, z=None):
        # 0d, 1d or 2d result, plus one dimension for time.
        self.d = d
        # 1d array with times, matching last dimension of self.d
        self.t = t
        # Name of data (e.g. h, u, hu, ...)
        self.name = name

    def get_temporal_stat(self, stat):
        """ Statistical analysis along temporal dimension """
        if stat in NP_OPERATORS:
            dnew = getattr(np, stat)(self.d, axis=-1)
        elif stat == 'final':
            dnew = self.d[..., -1]
        elif stat == 'initial':
            dnew = self.d[..., 0]
        return StaticResults(name+stat, dnew)

    def spatial_integration(self, axis=(0, 1), cellsize=None):
        """ Spatial integration along one or two axes """
        if cellsize is None:
            dnew = np.sum(self.d, axis=axis)
        else:
            dnew = np.sum(self.d*cellsize, axis=axis)
        self.d = dnew
        
    def plot(self, axe=None, figsize=None, folder_out=None,
             x=None, y=None, z=None, **kwargs):
        """ Plot results as time dependent"""
        
        if axe is None:
            fig, axe = plt.subplots(1, 1, figsize=figsize)
        
        if self.d.ndim == 1:
            axe.plot(self.t, self.d, **kwargs)
            axe.set_xlabel('Time (s)')
            axe.set_ylabel(notations.LABELS[self.name])
            
        elif self.d.ndim == 2:
            raise NotImplementedError('Plot of 1D data as time functions not implemented yet')
            
        elif self.d.ndim == 3:
            if x is None or y is None or z is None:
                raise TypeError('x, y or z data missing')
            plt_fn.plot_maps(x, y, z, self.d, self.t,
                             self.name, folder_out=folder_out, 
                             figsize=figsize, **kwargs)


class StaticResults:
    """ Result of simulation without time dependence"""

    def __init__(self, name, d):
        # 1d or 2d array
        self.d = d
        # Name of data
        self.name = name


class Results:
    """ Results of thin-layer model simulation

    This class is the parent class for all simulation results, whatever the
    kind of input data. Methods and functions for processing results are given
    here. Reading results from code specific outputs is done in inhereited
    classes.
    """

    def __init__(self, x, y, code=None, zinit=None, t=None, htype='normal',
                 params=None, h_thresh=None):
        """
        Create from given topography, thicknesses and velocities.

        It is not recommended

        Parameters
        ----------
        x : ndarray
            1D array for x-axis
        y : ndarray
            1D array for y-axis
        zinit : ndarray
            2D array giving the topography (size len(x)*len(y))
        htype : str, optional
            Direction in which flow height is given, 'normal' to topography
            or 'vertical' (default 'normal')
        params : dict, optional
            Dictionnary with simulation parameters. The default is None.
        """

        self.x = x
        self.y = y
        self.nx = len(x)
        self.ny = len(y)
        self.code = code
        self.htype = htype
        self.params = params
        self.h_thresh = h_thresh
        
        self._zinit = zinit
        self._costh = None
        self._h = None
        

    @property
    def zinit(self):
        """ Get initial topography """
        return self._zinit
    
    @property
    def h(self):
        """ Get initial topography """
        if self._h is None:
            self._h = self.get_temporal_output('h').d
        return self._h
    
    def get_costh(self):
        """Get cos(slope) of topography"""
        [Fx, Fy] = np.gradient(self.zinit, self.y, self.x)
        costh = 1/np.sqrt(1 + Fx**2 + Fy**2)
        return costh

    @property
    def costh(self):
        """ Compute or get cos(slope) of topography """
        if self._costh is None:
            self._costh = self.get_costh()
        return self._costh

    def get_temporal_output(self, name):
        return TemporalResults(name, None, None)

    def get_static_output(self, name, stat):
        return StaticResults(name+stat, None)
    
    def plot(self, name, save=True, h_thresh=None, **kwargs):
        
        if save:
            folder_out = os.path.join(self.folder_output, 'plots')
            os.makedirs(folder_out, exist_ok=True)
            kwargs['folder_out'] = folder_out
            
        if name in TEMPORAL_DATA_2D:
            data = self.get_temporal_output(name)
            if h_thresh is None:
                h_thresh = self.h_thresh
            if h_thresh is not None:
               data.d[self.h<h_thresh] = np.nan 
        elif name in STATIC_DATA_2D:
            data = self.get_static_output(name)
        
        if 'x' not in kwargs:
            kwargs['x'] = self.x
        if 'y' not in kwargs:
            kwargs['y'] = self.y
        if 'z' not in kwargs:
            kwargs['z'] = self.zinit
            
        data.plot(**kwargs)
