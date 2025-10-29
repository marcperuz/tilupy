# -*- coding: utf-8 -*-

import os

import numpy as np
import tilupy.read

from scipy.interpolate import RegularGridInterpolator


# Classify results
AVAILABLE_OUTPUT = ["h", "hvert", "u", "hu", "hu2"]
"""All output available for a shaltop simulation.

Implemented states:

    - h : Flow thickness (normal to the surface)
    - u : Norm of the velocity (from ux and uy)
    
Output computed from other output:
    
    - hvert : True vertical flow thickness
    - hu : Momentum flux (from h and u)
    - hu2 : Convective momentum flux (from h and u)
"""


class Results(tilupy.read.Results):
    """Results of lave2D simulations.

    This class is the results class for lave2D. Reading results from lave2D outputs 
    are done in this class.
    
    This class has all the global and quick attributes of the parent class. The quick 
    attributes are only computed if needed and can be deleted to clean memory.
    
    In addition to these attributes, there are those necessary for the operation of 
    reading the lave2D results.
    
    Parameters
    ----------
        folder : str
            Path to the folder containing the simulation files.
        name : str
            Simulation/Project name.
        raster : str
            Raster name.
        grid : str, optional
            - If grid=="cell", use cell, output nx and ny sizes are the size of topography raster, minus 1.
            - If grid=="edges" (default), use edges, output will have same dimension as topography raster.
    
    Attributes
    ----------
        _name : str
            Name of the lave2D project.
        _grid : str
            - If _grid=="cell", use cell, output nx and ny sizes are the size of topography raster, minus 1.
            - If _grid=="edges", use edges, output will have same dimension as topography raster.
        _raster : str
            Path to the raster file.
        _params : dict
            Dictionary storing all simulation parameters.
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
        
        # Create self._tim
        self._extract_output("X")


    def _extract_output(self, 
                        name: str, 
                        **kwargs
                        ) -> tilupy.read.TemporalResults2D | tilupy.read.AbstractResults:
        """Result extraction for lave2D files.

        Parameters
        ----------
        name : str
            Wanted output. Can access to variables in :data:`AVAILABLE_OUTPUT`.

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
        
        if self._tim is None: 
            self._tim = np.array(tim)
        
        extracted_outputs = {"h": h[:],
                             "u": u[:],
                             }

        if name in ["h", "u"]:
            d = extracted_outputs[name]
            t = self._tim
        
        if name == "hu":
            d = extracted_outputs['h'] * extracted_outputs['u']
            t = self._tim

        elif name == "hu2":
            d = extracted_outputs['h'] * extracted_outputs['u'] * extracted_outputs['u']
            t = self._tim
        
        elif name == "hvert":
            if self._costh is None:
                self._costh = self.compute_costh()
            d = extracted_outputs["h"] / self._costh[:, :, np.newaxis]
            t = self._tim
        
        # if name == "ek":
        #     if self._costh is None:
        #             self._costh = self.compute_costh()
            
        #     d = []
        #     for i in range(len(self._tim)):
        #         d.append(np.sum((available_outputs['h'][:, :, i] 
        #                          * available_outputs['u'][:, :, i] 
        #                          * available_outputs['u'][:, :, i])
        #                         / self._costh[:, :]))
        #     d = np.array(d)
        #     t = self._tim
                 
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
        return None


    def _read_from_file(self, *args, **kwargs):
        """Not useful"""
        return "No _read_from_file for Lave2D."