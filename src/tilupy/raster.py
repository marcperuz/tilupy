#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import importlib


def read_raster(file: str) -> list[np.ndarray]:
    """Convert a raster file (tif or asc) into numpy array.

    Parameters
    ----------
    file : str
        Path to the raster file.

    Returns
    -------
    list[np.ndarray]
        X and Y coordinates and data values in numpy array. 
    """
    if file.endswith(".asc") or file.endswith(".txt"):
        return read_ascii(file)
    elif file.endswith(".tif") or file.endswith(".tif"):
        return read_tiff(file)


def read_tiff(file: str) -> list[np.ndarray]:
    """Read and convert a tiff file into numpy array.

    Parameters
    ----------
    file : str
        Path to the tiff file.

    Returns
    -------
    list[np.ndarray]
        X and Y coordinates and data values in numpy array. 
    """
    import rasterio

    with rasterio.open(file, "r") as src:
        dem = src.read(1)
        ny, nx = dem.shape
        x = np.linspace(src.bounds.left, src.bounds.right, nx)
        y = np.linspace(src.bounds.bottom, src.bounds.top, ny)
    return x, y, dem


def read_ascii(file: str) -> list[np.ndarray]:
    """Read and convert a ascii file into numpy array.

    Parameters
    ----------
    file : str
        Path to the ascii file.

    Returns
    -------
    list[np.ndarray]
        X and Y coordinates and data values in numpy array. 
    """
    dem = np.loadtxt(file, skiprows=6)
    grid = {}
    with open(file, "r") as fid:
        for i in range(6):
            tmp = fid.readline().split()
            grid[tmp[0]] = float(tmp[1])
    try:
        x0 = grid["xllcenter"]
        y0 = grid["yllcenter"]
    except KeyError:
        x0 = grid["xllcorner"]
        y0 = grid["yllcorner"]
    nx = int(grid["ncols"])
    ny = int(grid["nrows"])
    dx = dy = grid["cellsize"]
    x = np.linspace(x0, x0 + (nx - 1) * dx, nx)
    y = np.linspace(y0, y0 + (ny - 1) * dy, ny)

    return x, y, dem


def write_tiff(x: np.ndarray, 
               y: np.ndarray, 
               z: np.ndarray, 
               file_out: str, 
               **kwargs
               ) -> None:
    """Write tif file from numpy array.

    Parameters
    ----------
    x : np.ndarray
        X coordinates.
    y : np.ndarray
        Y coordinates.
    z : np.ndarray
        Elevation values.
    file_out : str
        Name of the output folder.
    """
    import rasterio
    from rasterio.transform import Affine

    if "driver" not in kwargs:
        kwargs["driver"] = "GTiff"
    res = (x[-1] - x[0]) / (len(x) - 1)
    transform = Affine.translation(x[0] - res / 2, y[-1] - res / 2) * Affine.scale(res, -res)
    
    with rasterio.open(file_out,
                       "w",
                       height=z.shape[0],
                       width=z.shape[1],
                       count=1,
                       dtype=z.dtype,
                       transform=transform,
                       **kwargs) as dst:
        dst.write(z, 1)


def write_ascii(x: np.ndarray, 
                y: np.ndarray, 
                z: np.ndarray, 
                file_out: str, 
                ) -> None:
    """Write ascii file from numpy array.

    Parameters
    ----------
    x : np.ndarray
        X coordinates.
    y : np.ndarray
        Y coordinates.
    z : np.ndarray
        Elevation values.
    file_out : str
        Name of the output folder.
    """
    nx = z.shape[1]
    ny = z.shape[0]
    cellsize = x[1] - x[0]
    header_txt = ("ncols {:.0f}\nnrows {:.0f}\nxllcorner {:.5f}\nyllcorner {:.5f}\n")
    header_txt += "cellsize {:.4f}\nnodata_value -99999"
    header_txt = header_txt.format(nx, ny, x[0], y[0], cellsize)
    np.savetxt(file_out, z, header=header_txt, comments="")


def write_raster(x: np.ndarray, 
                 y: np.ndarray, 
                 z: np.ndarray, 
                 file_out: str,
                 fmt: str = None,
                 **kwargs
                 ) -> None:
    """Write raster file from numpy array.

    Parameters
    ----------
    x : np.ndarray
        X coordinates.
    y : np.ndarray
        Y coordinates.
    z : np.ndarray
        Elevation values.
    file_out : str
        Name of the output folder.
    fmt : str
        Wanted format : "asc", "ascii", "txt", "tif", "tiff".
    
    Raises
    ------
    ValueError
        If invalid format.
    """
    # File format read from file_out overrides fmt
    fmt_tmp = file_out.split(".")
    if len(fmt_tmp) > 1:
        fmt = fmt_tmp[-1]
    else:
        if fmt is None:
            fmt = "asc"
        file_out = file_out + "." + fmt

    if fmt not in ["asc", "ascii", "txt", "tif", "tiff"]:
        raise ValueError("File format not implemented in write_raster")

    if fmt.startswith("tif"):
        if importlib.util.find_spec("rasterio") is None:
            print(("rasterio is required to write tif files.",
                   " Switching to asc format",))
            fmt = "asc"

    if fmt in ["asc", "ascii", "txt"]:
        write_ascii(x, y, z, file_out)
    elif fmt in ["tif", "tiff"]:
        write_tiff(x, y, z, file_out, **kwargs)
    else:
        raise NotImplementedError()
