#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May 21 12:10:46 2021

@author: peruzzetto
"""
import os
import numpy as np


def make_constant_slope(folder_out, theta=10, m_radius=50, m_height=50,
                        m_x=None, m_y=None,
                        xmax=800, ymax=550, cellsize=2.5):
    """Init topography and mass for simulation on constant slope."""
    # Create mesh
    x = np.linspace(0, xmax, int(np.round(xmax/cellsize)))
    y = np.linspace(0, ymax, int(np.round(ymax/cellsize)))
    xmesh, ymesh = np.meshgrid(x, y)

    # Topography slope
    slope = np.tan(np.deg2rad(theta))

    # Topography array
    z = -slope*(xmesh-xmax)

    # Create initial mass
    if m_x is None:
        m_x = 3*m_radius
    if m_y is None:
        m_y = ymax/2
    m = 1-(xmesh - m_x)**2/m_radius**2 - (ymesh - m_y)**2/m_radius**2
    m = np.maximum(m*m_height, 0)

    # Write headers for ascii files
    nx = z.shape[1]
    ny = z.shape[0]
    header_txt = 'ncols {:.0f}\nnrows {:.0f}\nxllcorner 0\nyllcorner 0\n'
    header_txt += 'cellsize {:.4f}\nnodata_value -99999'
    header_txt = header_txt.format(nx, ny, cellsize)

    for a, name in zip([z, m], ['topo', 'mass']):
        np.savetxt(os.path.join(folder_out, name+'.asc'), a,
                   header=header_txt,
                   comments='')
