#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun  3 12:29:52 2021

@author: peruzzetto
"""

import os
import numpy as np


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

    return x, y, dem, dx


def write_ascii(x, y, z, file_out):
    """
    Write raster as ascii file.

    Parameters
    ----------
    x : TYPE
        DESCRIPTION.
    y : TYPE
        DESCRIPTION.
    z : TYPE
        DESCRIPTION.

    Returns
    -------
    None.

    """
    nx = z.shape[1]
    ny = z.shape[0]
    cellsize = x[1] - x[0]
    header_txt = 'ncols {:.0f}\nnrows {:.0f}\nxllcorner 0\nyllcorner 0\n'
    header_txt += 'cellsize {:.4f}\nnodata_value -99999'
    header_txt = header_txt.format(nx, ny, cellsize)
    np.savetxt(file_out, z, header=header_txt, comments='')
