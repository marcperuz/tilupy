# -*- coding: utf-8 -*-
"""
Created on Fri Aug  4 19:27:09 2023

@author: peruzzetto
"""

import numpy as np


def calotte(x, y, z, x0, y0, radius, norm_offset=0, res_type='projected_normal'):
    """
    Construct mass on topography as volume between sphere and topography.

    Parameters
    ----------
    x : np.array
        xaxis array, with length nx
    y : np.array
        yaxis array, with length ny
    z : np.array
        array of altitudes, of size (ny, nx). z[0, 0] has coordinates
        (x[0], y[-1])
    x0 : float
        x position of the calotte
    y0 : float
        y position of the calotte
    radius : float
        radius of the shpere
    norm_offset : float, optional
        downwards offset between the sphere center and the topography, in the
        direction normal to the topography. The default is 0.
    res_type : string, optional
        Type of thickness output. 'true_normal': real thickness in the
        direction normal to the topography. 'vertical': Thickness in the
        vertical direction. 'projected_normal': Thickness normal to the
        topography is computed from the vertical thickness projected on 
        the axe normal to the topography. The default is 'projected_normal'.

    Returns
    -------
    m : np.array
        array of mass height, in the direction normal to topography

    """
    z = np.flip(z, axis=0).T

    xmesh, ymesh = np.meshgrid(x, y, indexing='ij')
    nx = len(x)
    ny = len(y)

    # Get altitude of mass center on topography
    i0 = np.unravel_index(np.argmin(np.abs(x-x0), axis=None), (nx,))
    j0 = np.unravel_index(np.argmin(np.abs(y-y0), axis=None), (ny,))
    z0 = z[i0, j0]
    # Topography gradient
    [Fx, Fy] = np.gradient(z, x, y, edge_order=2)
    Fz = np.ones((nx, ny))
    c = 1/np.sqrt(1+Fx**2+Fy**2)
    Fx = -Fx*c
    Fy = -Fy*c
    Fz = Fz*c
    # Correct position from offset (shpere is moved downward,
    # perpendicular to topography)
    x0 = x0-norm_offset*Fx[i0, j0]
    y0 = y0-norm_offset*Fy[i0, j0]
    z0 = z0-norm_offset*Fz[i0, j0]

    # Compute mass height only where relevant (ie around the mass center)
    dist_to_mass = (xmesh-x0)**2+(ymesh-y0)**2
    ind = (dist_to_mass <= radius**2)

    B = 2*(Fx*(xmesh-x0)+Fy *
           (ymesh-y0)+Fz*(z-z0))
    C = (xmesh-x0)**2+(ymesh-y0)**2+(z-z0)**2-radius**2
    D = B**2-4*C

    # Intersection between shpere and normal to the topography, solution of
    # t**2+B*t+C=0
    m = np.zeros((nx, ny))

    if res_type == 'true_normal':
        # B = 2*(Fx[ind]*(xmesh[ind]-x0)+Fy[ind] *
        #        (ymesh[ind]-y0)+Fz[ind]*(z[ind]-z0))
        # C = (xmesh[ind]-x0)**2+(ymesh[ind]-y0)**2+(z[ind]-z0)**2-radius**2
        # D = B**2-4*C
        B = 2*(Fx*(xmesh-x0)+Fy *
               (ymesh-y0)+Fz*(z-z0))
        C = (xmesh-x0)**2+(ymesh-y0)**2+(z-z0)**2-radius**2
        D = B**2-4*C
        ind = D > 0
        t1 = (-B-np.sqrt(D))/2
        t2 = (-B+np.sqrt(D))/2
        ind2 = t1*t2 < 0
        m[ind2] = np.maximum(t1[ind2], t2[ind2])

    # Vertical thickness of calotte.
    if res_type in ['vertical', 'projected_normal']:
        zs = z0 + np.sqrt(radius**2 - (xmesh - x0)**2 - (ymesh - y0)**2)
        zi = z0 - np.sqrt(radius**2 - (xmesh - x0)**2 - (ymesh - y0)**2)
        ind = (z < zs) & (z > zi)
        m[ind] = zs[ind] - z[ind]
        if res_type == 'projected_normal':
            m = m * c

    m = np.flip(m.T, axis=0)

    return m
