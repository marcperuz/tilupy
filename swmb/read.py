#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun  1 14:20:52 2021

@author: peruzzetto
"""

import numpy as np

RAW_STATES = ['h', 'ux', 'uy']

TEMPORAL_DATA_0D = ['hu2int', 'vol']
TEMPORAL_DATA_1D = ['']
TEMPORAL_DATA_2D = ['h', 'ux', 'uy', 'hu', 'hu2']
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

    def __init__(self, name, d, t):
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

    def __init__(self, x, y, zinit, t=None, h=None, u=None, htype='normal',
                 params=None):
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
        t : ndarray, optional
            1D array giving times of outputs (default None)
        h : ndarray, optional
            3D array giving the heights, size len(x)*len(y)*len(t)
            (default None)
        u : ndarray, optional
            3D array giving the norm of velocity, size len(x)*len(y)*len(t)
            (default None)
        htype : str, optional
            Direction in which flow height is given, 'normal' to topography
            or 'vertical' (default 'normal')
        params : dict, optional
            Dictionnary with simulation parameters. The default is None.
        """

        self.x = x
        self.y = y
        self.zinit = zinit
        self.t = t
        self.htype = htype
        self.h = TemporalResults(self.t, h)
        self.u = TemporalResults(self.t, h)
        self.params = None
        self._costh = None

    def get_costh(self):
        """Get cos(slope) of topography"""
        [Fx, Fy] = np.gradient(self.zinit, self.x, self.y)
        costh = 1/np.sqrt(1 + Fx**2 + Fy**2)

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
