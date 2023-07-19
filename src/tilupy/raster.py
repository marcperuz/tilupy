#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun  3 12:29:52 2021

@author: peruzzetto
"""

import numpy as np
import importlib

def read_raster(file):
    if file.endswith('.asc') or file.endswith('.txt'):
        return read_ascii(file)
    elif file.endswith('.tif') or file.endswith('.tif'):
        return read_tiff(file)

def read_tiff(file):
    
    import rasterio
    with rasterio.open(file, 'r') as src:
        dem = src.read(1)
        ny, nx = dem.shape
        x = np.linspace(src.bounds.left, src.bounds.right, nx)
        y = np.linspace(src.bounds.bottom, src.bounds.top, ny)
    return x, y, dem
    
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

    return x, y, dem

def write_tiff(x, y, z, file_out, **kwargs):
    """
    Write raster as tif file

    Parameters
    ----------
    x : TYPE
        DESCRIPTION.
    y : TYPE
        DESCRIPTION.
    z : TYPE
        DESCRIPTION.
    file_out : TYPE
        DESCRIPTION.

    Returns
    -------
    None.

    """
    import rasterio
    from rasterio.transform import Affine
    if 'driver' not in kwargs:
        kwargs['driver'] = 'GTiff'
    res = (x[-1]-x[0])/(len(x)-1)
    transform = Affine.translation(x[0] - res / 2, y[-1] - res / 2)\
        * Affine.scale(res, -res)
    with rasterio.open(
            file_out,
            'w',
            height=z.shape[0],
            width=z.shape[1],
            count=1,
            dtype=z.dtype,
            transform=transform,
            **kwargs) as dst:
        dst.write(z, 1)
    

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
    header_txt = 'ncols {:.0f}\nnrows {:.0f}\nxllcorner {:.5f}\nyllcorner {:.5f}\n'
    header_txt += 'cellsize {:.4f}\nnodata_value -99999'
    header_txt = header_txt.format(nx, ny, x[0], y[0], cellsize)
    np.savetxt(file_out, z, header=header_txt, comments='')
    
def write_raster(x, y, z, file_out, fmt=None, **kwargs):
    
    # File format read from file_out overrides fmt
    fmt = file_out.split('.')
    if len(fmt)>1:
        fmt = fmt[-1]
    else:
        file_out = file_out + '.' + fmt
    
    if fmt not in ['asc', 'ascii', 'txt', 'tif', 'tiff']:
        raise ValueError('File format not implemented in write_raster')
        
    if fmt.startswith('tif'):
        if importlib.util.find_spec('rasterio') is None:
            print(('rasterio is required to write tif files.',
                   ' Switching to asc format'))
            fmt = 'asc'
        
    if fmt in ['asc', 'ascii', 'txt']:
        write_ascii(x, y, z, file_out)
    elif fmt in ['tif', 'tiff']:
        write_tiff(x, y, z, file_out, **kwargs)
    else:
        raise NotImplementedError()
