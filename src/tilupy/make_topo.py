# -*- coding: utf-8 -*-
"""
Created on Fri Aug  4 12:09:41 2023

@author: peruzzetto
"""

import numpy as np
import scipy

import tilupy.plot


def gray99(nx=None, ny=None, xmin=-0.4, x1=1.75, x2=2.15, xmax=3.2, ymax=0.5, R=1.1,
           theta1=40, theta2=0, maxz=None, dx=0.01, dy=0.01, plot=False):
    """
    Construct channel as in Gray et all 99.

    Input coordinates are in curvilinear coordinates along the reference
    topography following the channel bottom. Output coordinates are in the
    fixed cartesian frame.
    Parameters
    ----------
    nx : int
        Size of the grid in x direction
    ny : int
        Size of the y direction
    xmin : float, optional
        Minimum x coordinate. The default is -0.4.
    x1 : float, optional
        Min coordinate of the channel outlet (transition zone).
        The default is 1.75.
    x2 : float, optional
        Max coordinate of the channel outlet (transition zone).
        The default is 2.15.
    xmax : float, optional
        Maximum x coordinate. The default is 3.2.
    ymax : float, optional
        Maximum y coordinate the final yxais spans from -ymax to xmax.
        The default is 0.5.
    R : float, optional
        Radius of curvature of the channel. The default is 1.1.
    theta1 : float, optional
        Slope of the channel in degree. The default is 40.
    theta2 : float, optional
        Slope after the channel. The default is 0.
    maxz : float, optional
        Maximum z coordinate. The default is None.
    plot : boolean, optional
        Plot result. The default is False.
    dx : float, optional
        Cell size of the x axis. if specified, nx is recomputed. Default: None
    dy : float, optional
        Cell size of the y axis. if specified, ny is recomputed. Default: None


    Returns
    -------
    Xout : float nx*ny array
        Mesh of X coordinates in the cartesian frame
    Yout : float nx*ny array
        Mesh of Y coordinates in the cartesian frame.
    Zout : float nx*ny array
        Mesh of Z coordinates in the cartesian frame.

    """
    theta1 = np.deg2rad(theta1)
    theta2 = np.deg2rad(theta2)

    if nx is None:
        x = np.arange(xmin, xmax+dx/2, dx)
        nx = len(x)
    else:
        x = np.linspace(xmin, xmax, nx)

    if ny is None:
        y = np.arange(-ymax, ymax+dy/2, dy)
        ny = len(y)
    else:
        y = np.linspace(-ymax, ymax, ny)

    xcurv = np.tile(x.reshape((nx, 1)), (1, ny))
    ycurv = np.tile(y.reshape((1, ny)), (nx, 1))

    # Superficial topography : channel
    # alpha=1/(2*R)*np.sin(0.5*np.pi*(x-x2)/(x1-x2))**2
    # alpha=1/(2*R)*np.abs(x-x2)**1/np.abs(x1-x2)**1
    alpha = 1/(2*R)*(3*((x-x2)/(x1-x2))**2-2*((x-x2)/(x1-x2))**3)
    # alpha=0*x
    alpha[x > x2] = 0
    alpha[x < x1] = 1/(2*R)
    alpha = np.tile(alpha.reshape((nx, 1)), (1, ny))

    zchannel = alpha*np.abs(ycurv)**2
    # plt.figure()
    # plt.imshow(zchannel)

    # del alpha

    if not maxz:
        maxz = R/2

    zchannel[zchannel > maxz] = maxz

    # Base topography in curvilinear system. The transition zone between x1 and x2 is
    # a cylindre
    zbase = -np.sin(theta2)*(x-xmax)

    ind = (x <= x2) & (x >= x1)
    angle = (x[ind]-x1)/(x2-x1)*(theta2-theta1)+theta1
    R2 = (x2-x1)/(theta1-theta2)
    z2 = -np.sin(theta2)*(x2-xmax)
    zbase[ind] = R2*(1-np.cos(angle))-R2*(1-np.cos(theta2))+z2

    ind = x <= x1
    z1 = R2*(1-np.cos(theta1))-R2*(1-np.cos(theta2))+z2
    zbase[ind] = -np.sin(theta1)*(x[ind]-x1)+z1
    zbase = np.tile(zbase.reshape((nx, 1)), (1, ny))

    # Conversion in fixed cartesian frame
    zd = np.gradient(zbase, x[1]-x[0], edge_order=2, axis=0)
    Xd = np.sqrt(1-zd**2)
    X = scipy.integrate.cumtrapz(Xd, x, axis=0, initial=0)
    X = X+xmin*np.cos(theta1)

    # plt.figure()
    # plt.plot(X[:,0])

    del Xd

    # Topography conversion in fixed cartesian frame
    [Fx, Fy] = np.gradient(zbase, X[:, 0], ycurv[0, :], edge_order=2)
    Fz = np.ones(zbase.shape)
    costh = 1/np.sqrt(Fx**2+Fy**2+1)  # Slope angle
    Fx = -Fx*costh
    Fy = -Fy*costh
    Fz = Fz*costh
    Z = zbase+zchannel*Fz
    Xmesh = X+zchannel*Fx
    Ymesh = ycurv+zchannel*Fy

    # Reconstruction of regular cartesian mesh
    Xout = np.linspace(Xmesh.min(), Xmesh.max(), nx)
    Xout = np.tile(Xout.reshape((nx, 1)), (1, ny))
    Yout = np.linspace(Ymesh.min(), Ymesh.max(), ny)
    Yout = np.tile(Yout.reshape((1, ny)), (nx, 1))
    Zout = scipy.interpolate.griddata((Xmesh.reshape(nx*ny), Ymesh.reshape(nx*ny)),
                                      Z.reshape(nx*ny),
                                      (Xout, Yout), method='cubic')
    Ztmp = scipy.interpolate.griddata((Xmesh.reshape(nx*ny), Ymesh.reshape(nx*ny)),
                                      Z.reshape(nx*ny),
                                      (Xout, Yout), method='nearest')
    ind = np.isnan(Zout)
    Zout[ind] = Ztmp[ind]

    del Ztmp
    # fz=scipy.interpolate.Rbf(Xmesh,Ymesh,Z)
    # Zout=fz(Xout,Yout)

    if plot:

        tilupy.plot.plot_topo(Zout.T, Xout[:, 1], Yout[1, :])
    #     fig=plt.figure()
    #     axe=fig.gca(projection='3d')

    #     #axe.plot_surface(Xmesh, Ymesh, Z, rstride=1, cstride=1, antialiased=True)
    #     axe.plot_surface(Xout, Yout, Zout, rstride=1, cstride=1, antialiased=True)
    #     #axe.scatter(Xmesh,Ymesh,Z)

    #     axe.set_xlabel('X')
    #     axe.set_ylabel('Y')
    #     plot_functions.set_axes_equal(axe)
    #     plt.tight_layout()
    #     plt.show()

    return Xout[:, 1], Yout[1, :], Zout.T


if __name__ == '__main__':

    X, Y, Z = gray99(plot=True)
