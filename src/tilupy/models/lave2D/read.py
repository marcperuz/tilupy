# -*- coding: utf-8 -*-
"""
Created on Wed Jun  5 11:07:44 2024

@author: peruzzetto
"""

import os

import numpy as np
import tilupy.read

from scipy.interpolate import RegularGridInterpolator


STATES_OUTPUT = ["h", "u"]
"""State variables of the flow available with lave2D.

Implemented states :

    - h : flow thickness (normal to the surface)
    - u : flow velocity (norm)
"""


class Results(tilupy.read.Results):
    """Results of lave2D simulations.

    This class is the results class for lave2D. Reading results from lave2D outputs 
    are done in this class.
    
    This class has all the global and quick attributes of the parent class. The quick 
    attributes are only computed if needed and can be deleted to clean memory.
    
    In addition to these attributes, there are those necessary for the operation of 
    reading the lave2D results.
    
    Global attributes:
    ------------------
        _code : str
            Name of the code that generated the result.
        _folder : str
            Path to find code files (like parameters).
        _folder_output :
            Path to find the results of the code.
        _zinit : numpy.ndarray
            Surface elevation of the simulation.
        _tim : list
            Lists of recorded time steps.
        _x : numpy.ndarray
            X-coordinates of the simulation.
        _y : numpy.ndarray
            Y-coordinates of the simulation.
    
    Quick access attributes:
    ------------------------
        _h : tilupy.read.TemporalResults2D
            Fluid height over time.
        _h_max : tilupy.read.TemporalResults0D
            Max fluid hieght over time.
        _u : tilupy.read.TemporalResults2D
            Norm of fluid velocity over time.
        _u_max : tilupy.read.TemporalResults0D
            Max norm of fluid velocity over time.
        _costh : numpy.ndarray
            Value of cos[theta] at any point on the surface.
            
    Specific attributes:
    --------------------
        _name : str
            Name of the lave2D project.
        _grid : str
            - If _grid=="cell", use cell, output nx and ny sizes are the size of topography raster, minus 1.
            - If _grid=="edges", use edges, output will have same dimension as topography raster.
        _raster : str
            Path to the raster file.
        _params : dict
            Dictionary storing all simulation parameters.
        _dx : float
            Cell size in the X direction.
        _dy : float
            Cell size in the Y direction.
        _nx : int
            Number of cells in the X direction.
        _ny : int
            Number of cells in the Y direction.
        
    Parameters:
    -----------
    folder : str
        Path to the folder containing the simulation files.
    name : str
        Simulation/Project name.
    raster : str
        Raster name.
    grid : str, optional
        - If grid=="cell", use cell, output nx and ny sizes are the size of topography raster, minus 1.
        - If grid=="edges" (default), use edges, output will have same dimension as topography raster.
    """
    def __init__(self, folder, name, raster, grid="edges"):
        super().__init__()
        self._code = "lave2D"
        
        if folder is None:
            folder = os.getcwd()
        self._folder = folder
        self._folder_output = folder  
        # Results are in the same folder than simulation data
        
        self._name = name
        
        if not raster.endswith(".asc"):
            raster = raster + ".asc"
        self._raster = raster
        
        self._x, self._y, self._zinit = tilupy.raster.read_ascii(os.path.join(folder, self._raster))
        self._nx, self._ny = len(self._x), len(self._y)
        self._dx = self._x[1] - self._x[0]
        self._dy = self._y[1] - self._y[0]
                
        # grid=='edges' -> use edges, output will have same dimension as
        # topography raster
        # grid=='cell' -> use cell, output nx and ny sizes are the size of
        # topography raster, minus 1
        self._grid = grid
        if self._grid == "cells":
            fz = RegularGridInterpolator((self._y, self._x),
                                         self._zinit,
                                         method="linear",
                                         bounds_error=False,
                                         fill_value=None,
                                         )
            self._x = self._x[1:] - self._dx / 2
            self._y = self._y[1:] - self._dy / 2
            x_mesh, y_mesh = np.meshgrid(self._x, self._y)
            self._zinit = fz((y_mesh, x_mesh))
        
        self._params = dict()
    
        self._params["nx"] = len(self._x)
        self._params["ny"] = len(self._y)

        # Rheology
        rheol = np.loadtxt(os.path.join(folder, self._name + ".rhe"))
        self._params["tau/rho"] = rheol[0]
        self._params["K/tau"] = rheol[1]

        # Numerical parameters
        with open(os.path.join(folder, "DONLEDD1.DON"), "r") as fid:
            for line in fid:
                name = line[:34].strip(" ")
                value = line[34:]
                if len(value) == 1:
                    value = int(value)
                else:
                    value = float(value)
                self._params[name] = value


    def _extract_output(self, 
                        name: str, 
                        **kwargs
                        ) -> tilupy.read.TemporalResults2D | tilupy.read.TemporalResults0D | tilupy.read.AbstractResults:
        """Result extraction for lave2D files.

        Parameters
        ----------
        name : str
            Wanted output. Can access to variables in :data:`STATES_OUTPUT`.

        Returns
        -------
        tilupy.read.TemporalResults2D | tilupy.read.TemporalResults0D | tilupy.read.AbstractResults
            Wanted output. If no output computed, return an object of :class:`tilupy.read.AbstractResults`.
        """
        d = None
        t = None
        notation = None
        
        # read initial mass
        h_init = np.loadtxt(os.path.join(self._folder, self._name + ".cin"), skiprows=1)
        
        # Read results
        file_res = os.path.join(self._folder, self._name + ".asc")
        n_times = int(self._params["tmax"] / self._params["dtsorties"])
        tim = [0]
        
        if self._grid == "edges":
            ny_out = self._params["ny"] - 1
            x_out = self._x[1:] - self._dx / 2
            y_out = self._y[1:] - self._dy / 2
            x_mesh, y_mesh = np.meshgrid(self._x, self._y)
            h_init = h_init.reshape(self._params["ny"] - 1, self._params["nx"] - 1)
            fh = RegularGridInterpolator((y_out, x_out),
                                         h_init,
                                         method="linear",
                                         bounds_error=False,
                                         fill_value=None)
            h_init = fh((y_mesh, x_mesh))
        else:
            ny_out = self._params["ny"]
            h_init = h_init.reshape(self._params["ny"], self._params["nx"])
        
        x_mesh_out, y_mesh_out = np.meshgrid(self._x[1:] - self._dx / 2, 
                                             self._y[1:] - self._dy / 2)

        h = np.zeros((self._params["ny"], self._params["nx"], n_times + 1))
        h[:, :, 0] = np.flip(h_init, axis=0)
        u = np.zeros((self._params["ny"], self._params["nx"], n_times + 1))
        
        with open(file_res, "r") as fid:
            lines = fid.readlines()
            n_lines = 2 * (ny_out) + 8  # Number of lines per time step
            for i_time in range(n_times):
                i_start = n_lines * i_time
                i_start_h = i_start + 7
                i_stop_h = i_start + ny_out + 7
                i_start_u = i_start + ny_out + 8
                i_stop_u = i_start + 2 * ny_out + 8
                tim.append(float(lines[i_start]))
                h_out = np.loadtxt(lines[i_start_h:i_stop_h])
                u_out = np.loadtxt(lines[i_start_u:i_stop_u])
                
                if self._grid == "edges":
                    fh = RegularGridInterpolator((y_out, x_out),
                                                 h_out,
                                                 method="linear",
                                                 bounds_error=False,
                                                 fill_value=None)
                    h[:, :, i_time + 1] = fh((y_mesh, x_mesh))
                    
                    fu = RegularGridInterpolator((y_out, x_out),
                                                 u_out,
                                                 method="linear",
                                                 bounds_error=False,
                                                 fill_value=None)
                    u[:, :, i_time + 1] = fu((y_mesh, x_mesh))
                else:
                    h[:, :, i_time + 1] = h_out
                    u[:, :, i_time + 1] = u_out
            
        self._tim = tim
        
        available_outputs = {"h": h[:],
                             "u": u[:],
                             }

        if name in STATES_OUTPUT:
            d = available_outputs[name]
            t = self._tim
        
        if t is None:
            return tilupy.read.AbstractResults(name, d, notation=notation)

        else:
            if d.ndim == 3:
                return tilupy.read.TemporalResults2D(name, 
                                                     d, 
                                                     t, 
                                                     notation=notation, 
                                                     x=self._x, 
                                                     y=self._y, 
                                                     z=self._zinit)
            if d.ndim == 1:
                return tilupy.read.TemporalResults0D(name, 
                                                     d, 
                                                     t, 
                                                     notation=notation)
        return None