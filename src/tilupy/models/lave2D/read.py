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


class Results(tilupy.read.Results):
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


    # def read_resfile(self):
    #     # read initial mass
    #     h_init = np.loadtxt(os.path.join(self._folder, self._name + ".cin"), skiprows=1)
        
    #     # Read results
    #     file_res = os.path.join(self._folder, self._name + ".asc")
    #     n_times = int(self._params["tmax"] / self._params["dtsorties"])
    #     self._tim = [0]
        
    #     if self._grid == "edges":
    #         ny_out = self._params["ny"] - 1
    #         x_out = self._x[1:] - self._dx / 2
    #         y_out = self._y[1:] - self._dy / 2
    #         x_mesh, y_mesh = np.meshgrid(self._x, self._y)
    #         h_init = h_init.reshape(self._params["ny"] - 1, self._params["nx"] - 1)
    #         fh = RegularGridInterpolator((y_out, x_out),
    #                                       h_init,
    #                                       method="linear",
    #                                       bounds_error=False,
    #                                       fill_value=None,
    #                                       )
    #         h_init = fh((y_mesh, x_mesh))
    #     else:
    #         ny_out = self._params["ny"]
    #         h_init = h_init.reshape(self._params["ny"], self._params["nx"])
        
    #     x_mesh_out, y_mesh_out = np.meshgrid(
    #         self._x[1:] - self._dx / 2, self._y[1:] - self._dy / 2
    #     )

    #     self._h = np.zeros((self._params["ny"], self._params["nx"], n_times + 1))
    #     self._h[:, :, 0] = np.flip(h_init, axis=0)
    #     self._u = np.zeros((self._params["ny"], self._params["nx"], n_times + 1))
    #     with open(file_res, "r") as fid:
    #         lines = fid.readlines()
    #         n_lines = 2 * (ny_out) + 8  # Number of lines per time step
    #         for i_time in range(n_times):
    #             i_start = n_lines * i_time
    #             i_start_h = i_start + 7
    #             i_stop_h = i_start + ny_out + 7
    #             i_start_u = i_start + ny_out + 8
    #             i_stop_u = i_start + 2 * ny_out + 8
    #             self._tim.append(float(lines[i_start]))
    #             h_out = np.loadtxt(lines[i_start_h:i_stop_h])
    #             u_out = np.loadtxt(lines[i_start_u:i_stop_u])
    #             if self._grid == "edges":
    #                 fh = RegularGridInterpolator(
    #                     (y_out, x_out),
    #                     h_out,
    #                     method="linear",
    #                     bounds_error=False,
    #                     fill_value=None,
    #                 )
    #                 self._h[:, :, i_time + 1] = fh((y_mesh, x_mesh))
    #                 fu = RegularGridInterpolator(
    #                     (y_out, x_out),
    #                     u_out,
    #                     method="linear",
    #                     bounds_error=False,
    #                     fill_value=None,
    #                 )
    #                 self._u[:, :, i_time + 1] = fu((y_mesh, x_mesh))
    #             else:
    #                 self._h[:, :, i_time + 1] = h_out
    #                 self._u[:, :, i_time + 1] = u_out

    
    # def _extract_output(self, name, **kwargs):
    #     d = None
    #     t = None
    #     notation = None

    #     if name in ["h", "u"]:
    #         name = "_" + name       
    #         d = getattr(self, name)
                        
    #         if d is None:
    #             self.read_resfile()
    #             d = getattr(self, name)
            
    #         t = self._tim
    #         return tilupy.read.TemporalResults2D(name, 
    #                                              d, 
    #                                              t, 
    #                                              notation=notation, 
    #                                              x=self._x, 
    #                                              y=self._y, 
    #                                              z=self._zinit
    #                                              )
    #     elif name in ["ux", 'uy']:
    #         return None
        

    def _extract_output(self, name, **kwargs):
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
                                        fill_value=None,
                                        )
            h_init = fh((y_mesh, x_mesh))
        else:
            ny_out = self._params["ny"]
            h_init = h_init.reshape(self._params["ny"], self._params["nx"])
        
        x_mesh_out, y_mesh_out = np.meshgrid(
            self._x[1:] - self._dx / 2, self._y[1:] - self._dy / 2
        )

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
                                                    fill_value=None,
                                                    )
                    h[:, :, i_time + 1] = fh((y_mesh, x_mesh))
                    fu = RegularGridInterpolator((y_out, x_out),
                                                    u_out,
                                                    method="linear",
                                                    bounds_error=False,
                                                    fill_value=None,
                                                    )
                    u[:, :, i_time + 1] = fu((y_mesh, x_mesh))
                else:
                    h[:, :, i_time + 1] = h_out
                    u[:, :, i_time + 1] = u_out
            
        self._tim = tim
        
        if name == "h":
            d = h[:]
            t = self._tim
            
        if name == "u":
            d = u[:]
            t = self._tim
        
        if name in ["ux", "uy"]:
            return None
        
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