# -*- coding: utf-8 -*-
"""
Created on Wed Jun  5 11:07:44 2024

@author: peruzzetto
"""

import os

import numpy as np
import tilupy.read

from scipy.interpolate import RegularGridInterpolator


class Results(tilupy.read.Results):
    def __init__(self, folder, name, raster, grid="edges"):
        super().__init__()

        self.folder = folder
        self.folder_output = folder  # Results are in the same folder
        # than simulation data
        self.name = name
        # grid=='edges' -> use edges, output will have same dimension as
        # topography raster
        # grid=='cell' -> use cell, output nx and ny sizes are the size of
        # topography raster, minus 1
        self.grid = grid

        if not raster.endswith(".asc"):
            raster = raster + ".asc"
        self.raster = raster

        self.params = dict()

        self.x, self.y, self._zinit = tilupy.raster.read_ascii(
            os.path.join(folder, self.raster)
        )
        self.dx = self.x[1] - self.x[0]
        self.dy = self.y[1] - self.y[0]
        if self.grid == "cells":
            fz = RegularGridInterpolator(
                (self.y, self.x),
                self._zinit,
                method="linear",
                bounds_error=False,
                fill_value=None,
            )
            self.x = self.x[1:] - self.dx / 2
            self.y = self.y[1:] - self.dy / 2
            x_mesh, y_mesh = np.meshgrid(self.x, self.y)
            self._zinit = fz((y_mesh, x_mesh))

        self.params["nx"] = len(self.x)
        self.params["ny"] = len(self.y)

        # Rheology
        rheol = np.loadtxt(os.path.join(folder, self.name + ".rhe"))
        self.params["tau/rho"] = rheol[0]
        self.params["K/tau"] = rheol[1]

        # Numerical parameters
        with open(os.path.join(self.folder, "DONLEDD1.DON"), "r") as fid:
            for line in fid:
                name = line[:34].strip(" ")
                value = line[34:]
                if len(value) == 1:
                    value = int(value)
                else:
                    value = float(value)
                self.params[name] = value

    def read_resfile(
        self,
    ):
        # read initial mass
        h_init = np.loadtxt(
            os.path.join(self.folder, self.name + ".cin"), skiprows=1
        )

        # Read results
        file_res = os.path.join(self.folder, self.name + ".asc")
        n_times = int(self.params["tmax"] / self.params["dtsorties"])
        self._tim = [0]
        if self.grid == "edges":
            ny_out = self.params["ny"] - 1
            x_out = self.x[1:] - self.dx / 2
            y_out = self.y[1:] - self.dy / 2
            x_mesh, y_mesh = np.meshgrid(self.x, self.y)
            h_init = h_init.reshape(
                self.params["ny"] - 1, self.params["nx"] - 1
            )
            fh = RegularGridInterpolator(
                (y_out, x_out),
                h_init,
                method="linear",
                bounds_error=False,
                fill_value=None,
            )
            h_init = fh((y_mesh, x_mesh))
        else:
            ny_out = self.params["ny"]
            h_init = h_init.reshape(self.params["ny"], self.params["nx"])
        # x_mesh_out, y_mesh_out = np.meshgrid(
        #     self.x[1:] - self.dx / 2, self.y[1:] - self.dy / 2
        # )

        self._h = np.zeros((self.params["ny"], self.params["nx"], n_times + 1))
        self._h[:, :, 0] = np.flip(h_init, axis=0)
        self._u = np.zeros((self.params["ny"], self.params["nx"], n_times + 1))
        with open(file_res, "r") as fid:
            lines = fid.readlines()
            n_lines = 2 * (ny_out) + 8  # Number of lines per time step
            for i_time in range(n_times):
                i_start = n_lines * i_time
                i_start_h = i_start + 7
                i_stop_h = i_start + ny_out + 7
                i_start_u = i_start + ny_out + 8
                i_stop_u = i_start + 2 * ny_out + 8
                self._tim.append(float(lines[i_start]))
                h_out = np.loadtxt(lines[i_start_h:i_stop_h])
                u_out = np.loadtxt(lines[i_start_u:i_stop_u])
                if self.grid == "edges":
                    fh = RegularGridInterpolator(
                        (y_out, x_out),
                        h_out,
                        method="linear",
                        bounds_error=False,
                        fill_value=None,
                    )
                    self._h[:, :, i_time + 1] = fh((y_mesh, x_mesh))
                    fu = RegularGridInterpolator(
                        (y_out, x_out),
                        u_out,
                        method="linear",
                        bounds_error=False,
                        fill_value=None,
                    )
                    self._u[:, :, i_time + 1] = fu((y_mesh, x_mesh))
                else:
                    self._h[:, :, i_time + 1] = h_out
                    self._u[:, :, i_time + 1] = u_out

    @property
    def h(self):
        if self._h is None:
            self.read_resfile()
        return self._h

    @property
    def u(self):
        if self._u is None:
            self.read_resfile()
        return self._u

    @property
    def tim(self):
        if self._tim is None:
            self.read_resfile()
        return self._tim

    def _get_output(self, name, **kwargs):
        d = None
        t = None
        notation = None

        if name in ["h", "u"]:
            d = getattr(self, name)
            t = self.tim
            return tilupy.read.TemporalResults2D(
                name, d, t, notation=notation, x=self.x, y=self.y, z=self.z
            )
