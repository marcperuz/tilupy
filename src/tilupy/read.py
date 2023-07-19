#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun  1 14:20:52 2021

@author: peruzzetto
"""

import matplotlib.pyplot as plt
import numpy as np

import os
import importlib

import tilupy.notations as notations
import tilupy.plot as plt_fn
import tilupy.raster

RAW_STATES = ['hvert', 'h', 'ux', 'uy']

TEMPORAL_DATA_0D = ['hu2int', 'vol']
TEMPORAL_DATA_1D = ['']
TEMPORAL_DATA_2D = ['hvert', 'h', 'u', 'ux', 'uy', 'hu', 'hu2']
STATIC_DATA_0D = []
STATIC_DATA_1D = []
STATIC_DATA_2D = []

TOPO_DATA_2D = ['z', 'zinit', 'costh']

NP_OPERATORS = ['max', 'mean', 'std', 'sum', 'min']
OTHER_OPERATORS = ['final', 'initial']

COMPUTED_STATIC_DATA_2D = []

for stat in NP_OPERATORS + OTHER_OPERATORS:
    for name in TEMPORAL_DATA_2D:
        COMPUTED_STATIC_DATA_2D.append(name + '_' + stat)
        
STATIC_DATA_2D += COMPUTED_STATIC_DATA_2D

DATA_NAMES = TEMPORAL_DATA_0D + TEMPORAL_DATA_1D + TEMPORAL_DATA_2D
DATA_NAMES += STATIC_DATA_0D + STATIC_DATA_1D + STATIC_DATA_2D

class TemporalResults:
    """ Time dependent result of simulation """

    def __init__(self, name, d, t, x=None, y=None, z=None):
        # 0d, 1d or 2d result, plus one dimension for time.
        self.d = d
        # 1d array with times, matching last dimension of self.d
        self.t = t
        # Name of data (e.g. h, u, hu, ...)
        self.name = name
        # x and y arrays
        self.x = x
        self.y = y
        # topography
        self.z = z

    def get_temporal_stat(self, stat):
        """ Statistical analysis along temporal dimension """
        if stat in NP_OPERATORS:
            dnew = getattr(np, stat)(self.d, axis=-1)
        elif stat == 'final':
            dnew = self.d[..., -1]
        elif stat == 'initial':
            dnew = self.d[..., 0]
        return StaticResults(self.name + '_' + stat, dnew)

    def spatial_integration(self, axis=(0, 1), cellsize=None):
        """ Spatial integration along one or two axes """
        if cellsize is None:
            dnew = np.sum(self.d, axis=axis)
        else:
            dnew = np.sum(self.d*cellsize, axis=axis)
        self.d = dnew
        
    def plot(self, axe=None, figsize=None, folder_out=None, fmt='png', dpi=150,
             x=None, y=None, z=None, sup_plt_fn=None, sup_plt_fn_args=None,
             **kwargs):
        """ Plot results as time dependent"""
        
        if axe is None:
            fig, axe = plt.subplots(1, 1, figsize=figsize)
        
        if self.d.ndim == 1:
            axe.plot(self.t, self.d, **kwargs)
            axe.set_xlabel('Time (s)')
            axe.set_ylabel(notations.LABELS[self.name])
            
        elif self.d.ndim == 2:
            raise NotImplementedError(('Plot of 1D data as time',
                                       ' functions not implemented yet'))
            
        elif self.d.ndim == 3:
            if x is None or y is None or z is None:
                raise TypeError('x, y or z data missing')
            plt_fn.plot_maps(x, y, z, self.d, self.t,
                             self.name, folder_out=folder_out, 
                             figsize=figsize, fmt=fmt, dpi=dpi,
                             sup_plt_fn=sup_plt_fn,
                             sup_plt_fn_args=sup_plt_fn_args,
                             **kwargs)
            axe, fig = None, None
            
        if self.d.ndim != 3 and sup_plt_fn is not None:
            if sup_plt_fn_args is None:
                sup_plt_fn_args = dict()
            sup_plt_fn(axe, **sup_plt_fn_args)
            
        return axe, fig
    
    def save(self, folder=None, file_name=None, fmt='asc', time=None,
             x=None, y=None, **kwargs):
        
        if self.d.ndim == 1:
            raise NotImplementedError()
            
        elif self.d.ndim == 2:
            raise NotImplementedError()
            
        elif self.d.ndim == 3:
            if x is None:
                x = self.x
            if y is None:
                y = self.y
            if x is None or y is None:
                raise ValueError('x et y arrays must not be None')
            
            if file_name is None:
                file_name = self.name
                
            if folder is not None:
                file_name = os.path.join(folder, file_name)
                
            if time is not None:
                if isinstance(time, str):
                    if time == 'final':
                        inds = [self.d.shape[2]-1]
                    elif time == 'initial':
                        inds = [0]
                else:
                    inds = [np.argmin(time - np.abs(np.array(self.t)-time))]
            
            for i in range(inds):
                file_out = file_name + '_{:04d}.'.format(i) + fmt
                tilupy.raster.write_raster(x, y, self.d[:, :, i], file_out,
                                      fmt=fmt, **kwargs)
                


class StaticResults:
    """ Result of simulation without time dependence"""

    def __init__(self, name, d, x=None, y=None, z=None):
        # 1d or 2d array
        self.d = d
        # Name of data
        self.name = name
        # x and y arrays
        self.x = x
        self.y = y
        # topography
        self.z = z
        
    def plot(self, axe=None, figsize=None,
             folder_out=None, suffix=None, prefix=None,
             fmt='png', dpi=150,
             x=None, y=None, z=None, 
             sup_plt_fn=None, sup_plt_fn_args=None, **kwargs):
        """Plot results as map"""
        
        if axe is None:
            fig, axe = plt.subplots(1, 1, figsize=figsize)
        else:
            fig = axe.figure
            
        if x is None:
            x = self.x
        if y is None:
            y = self.y
        if z is None:
            z = self.z
        
        if x is None or y is None or z is None:
            raise TypeError('x, y or z data missing')
            
        axe = plt_fn.plot_data_on_topo(x, y, z, self.d, axe=axe,
                                       figsize=figsize,
                                       **kwargs)
        if sup_plt_fn is not None:
            if sup_plt_fn_args is None:
                sup_plt_fn_args = dict()
            sup_plt_fn(axe, **sup_plt_fn_args)
        
        if folder_out is not None:
            file_name = self.name
            if suffix is not None:
                file_name = file_name + '_' + suffix
            if prefix is not None:
                file_name = prefix + '_' + file_name
            file_out = os.path.join(folder_out,file_name + '.' + fmt)
            axe.figure.tight_layout(pad=0.1)
            axe.figure.savefig(file_out, dpi=dpi, bbox_inches='tight',
                               pad_inches=0.05)
        
        return axe, fig
    
    def save(self, folder=None, file_name=None, fmt='txt', time=None,
             x=None, y=None, **kwargs):
        
        if x is None:
            x = self.x
        if y is None:
            y = self.y
        
        if x is None or y is None:
            raise ValueError('x et y arrays must not be None')
            
        if file_name is None:
            file_name = self.name + '.' + fmt
            
        if folder is not None:
            file_name = os.path.join(folder, file_name)
            
        tilupy.raster.write_raster(x, y, self.d, file_name,
                              fmt=fmt, **kwargs)
            
        

class Results:
    """ Results of thin-layer model simulation

    This class is the parent class for all simulation results, whatever the
    kind of input data. Methods and functions for processing results are given
    here. Reading results from code specific outputs is done in inhereited
    classes.
    """

    def __init__(self, *args, **kwargs):
        """
        Call init function used for code in inherited classes.

        It is not recommended

        Parameters
        ----------
        code : string
            Name of code from which results must be read.
        """
        self._h_max = None
        self._h = None
        self._costh = None
        self._zinit = None
        

    @property
    def zinit(self):
        """ Get initial topography """
        return self._zinit
    
    @property
    def z(self):
        """ Alias for zinit"""
        return self.zinit
    
    @property
    def h(self):
        """ Get thickness """
        if self._h is None:
            self._h = self.get_temporal_output('h').d
        return self._h
    
    @property
    def h_max(self):
        """ Get maximum thickness """
        if self._h_max is None:
            self._h_max = self.get_static_output('h', 'max').d
        return self._h_max
    
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

    def get_temporal_output(self, name, h_thresh=None):
        return TemporalResults(name, None, None, h_thresh=h_thresh)

    def get_static_output(self, name, **kwargs):
        return StaticResults(name, None, **kwargs)
    
    def get_output(self, name, **kwargs):
        if name in TEMPORAL_DATA_2D:
            data = self.get_temporal_output(name, **kwargs)
        elif name in STATIC_DATA_2D:
            state, stat = name.split('_')
            data = self.get_static_output(name, **kwargs)
        return data
    
    def plot(self, name, save=True, folder_out=None, dpi=150, fmt='png',
             h_thresh=None, from_file=False, display_plot=True,
             **kwargs):
        
        assert(name in DATA_NAMES)
        
        if save:
            if folder_out is None:
                folder_out = os.path.join(self.folder_output, 'plots')
            os.makedirs(folder_out, exist_ok=True)
            kwargs['folder_out'] = folder_out
            kwargs['dpi'] = dpi
            kwargs['fmt'] = fmt
            
        if not display_plot:
            backend = plt.get_backend()
            plt.switch_backend('Agg')
        
        data = self.get_output(name, from_file=from_file, h_thresh=h_thresh)
            
        if name in TEMPORAL_DATA_2D + STATIC_DATA_2D:
            if 'colorbar_kwargs' not in kwargs:
                kwargs['colorbar_kwargs'] = dict()
            if 'label' not in kwargs['colorbar_kwargs']:
                labels = notations.LABELS
                kwargs['colorbar_kwargs']['label'] = labels[name]
                
        
        if 'x' not in kwargs:
            kwargs['x'] = self.x
        if 'y' not in kwargs:
            kwargs['y'] = self.y
        if 'z' not in kwargs:
            kwargs['z'] = self.zinit
            
        axe, fig = data.plot(**kwargs)
        
        if not display_plot:
            plt.switch_backend(backend)
        
        return axe, fig
    
    def save(self, name, folder=None, file_name=None, fmt='txt',
             from_file=True, **kwargs):
        
        if folder is None:
            folder = os.path.join(self.folder_output, 'processed')
            os.makedirs(folder, exist_ok=True)
        
        if name in DATA_NAMES:
            data = self.get_output(name, from_file=from_file)
            if data.d.ndim>1:
                if 'x' not in kwargs:
                    kwargs['x'] = self.x
                if 'y' not in kwargs:
                    kwargs['y'] = self.y
                
            data.save(folder=folder, file_name=file_name, fmt=fmt,
                      **kwargs)
            
        elif name in TOPO_DATA_2D:
            if file_name is None:
                file_name = name
            file_out = os.path.join(folder, file_name)
            tilupy.raster.write_raster(self.x, self.y, getattr(self, name),
                                  file_out, fmt=fmt, **kwargs)
            
    
def get_results(code, **kwargs):
    """
    Get simulation results for a given code. This function calls the 
    appropriate module.

    Parameters
    ----------
    code : TYPE
        DESCRIPTION.
    **kwargs : TYPE
        DESCRIPTION.

    Returns
    -------
    None.

    """
    module = importlib.import_module('tilupy.models.'+code+'.read')
    return module.Results(**kwargs)

def use_thickness_threshold(simu, array, h_thresh):
    
    thickness = simu.get_temporal_output('h')
    array[thickness.d<h_thresh] = 0
    return array