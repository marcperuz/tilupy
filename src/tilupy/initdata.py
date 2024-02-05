#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May 21 12:10:46 2021

@author: peruzzetto
"""
import os
import numpy as np

import tilupy.make_topo
import tilupy.make_mass
import tilupy.plot


def make_constant_slope(
    folder_out,
    theta=10,
    m_radius=50,
    m_height=50,
    m_x=None,
    m_y=None,
    xmax=800,
    ymax=550,
    cellsize=2.5,
):
    """Init topography and mass for simulation on constant slope."""
    # Create mesh
    x = np.linspace(0, xmax, int(np.round(xmax / cellsize)))
    y = np.linspace(0, ymax, int(np.round(ymax / cellsize)))
    xmesh, ymesh = np.meshgrid(x, y)

    # Topography slope
    slope = np.tan(np.deg2rad(theta))

    # Topography array
    z = -slope * (xmesh - xmax)

    # Create initial mass
    if m_x is None:
        m_x = 3 * m_radius
    if m_y is None:
        m_y = ymax / 2
    m = (
        1
        - (xmesh - m_x) ** 2 / m_radius**2
        - (ymesh - m_y) ** 2 / m_radius**2
    )
    m = np.maximum(m * m_height, 0)

    # Write headers for ascii files
    nx = z.shape[1]
    ny = z.shape[0]
    header_txt = "ncols {:.0f}\nnrows {:.0f}\nxllcorner 0\nyllcorner 0\n"
    header_txt += "cellsize {:.4f}\nnodata_value -99999"
    header_txt = header_txt.format(nx, ny, cellsize)

    for a, name in zip([z, m], ["topo", "mass"]):
        np.savetxt(
            os.path.join(folder_out, name + ".asc"),
            a,
            header=header_txt,
            comments="",
        )


def gray99_topo_mass(
    dx=0.1, dy=0.1, save=False, folder_out=None, res_type="true_normal"
):
    # Initiate topography
    X, Y, Z = tilupy.make_topo.gray99(dx=dx, dy=dy)

    # Initiate initial mass. It is a spherical calotte above the topography,
    # in Gray et al 99 (p. 1859) the resulting mass has a height of 0.22 m and a radius
    # of 0.32 m (more precisely it is the length in the downslope direction)
    # The correspondig radius of the sphere, and the offset from the topography
    # in the topography normal direction (norm_offset) are deduced from these
    # parameters. See also Gig 3 in Wieland, Gray and Hutter (1999)

    x0 = 0.06 * np.cos(np.deg2rad(40))
    hmass = 0.22
    wmass = 0.32
    radius = (wmass**2 + hmass**2) / (2 * hmass)
    norm_offset = (wmass**2 - hmass**2) / (2 * hmass)
    # Z = -np.tile(X, [len(Y), 1])*np.tan(np.deg2rad(20))
    M = tilupy.make_mass.calotte(
        X, Y, Z, x0, 0, radius, norm_offset=norm_offset, res_type=res_type
    )

    return X, Y, Z, M


if __name__ == "__main__":
    x, y, z, m = gray99_topo_mass(dx=0.01, dy=0.01)
    tilupy.plot.plot_data_on_topo(x, y, z, m, topo_kwargs=dict(level_min=0.1))
