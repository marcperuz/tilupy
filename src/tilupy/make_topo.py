# -*- coding: utf-8 -*-
"""
Created on Fri Aug  4 12:09:41 2023

@author: peruzzetto
"""

import numpy as np
import scipy

import tilupy.plot


def gray99(
    nx=None,
    ny=None,
    xmin=-0.4,
    x1=1.75,
    x2=2.15,
    xmax=3.2,
    ymax=0.6,
    R=1.1,
    theta1=40,
    theta2=0,
    maxz=None,
    dx=0.01,
    dy=0.01,
    plot=False,
):
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
    dx : float, optional
        Cell size of the x axis. if specified, nx is recomputed. Default: 0.01
    dy : float, optional
        Cell size of the y axis. if specified, ny is recomputed. Default: 0.01
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
        x = np.arange(xmin, xmax + dx / 2, dx)
        nx = len(x)
    else:
        x = np.linspace(xmin, xmax, nx)

    if ny is None:
        y = np.arange(-ymax, ymax + dy / 2, dy)
        ny = len(y)
    else:
        y = np.linspace(-ymax, ymax, ny)

    ycurv = np.tile(y.reshape((1, ny)), (nx, 1))

    # Superficial topography : channel
    # alpha=1/(2*R)*np.sin(0.5*np.pi*(x-x2)/(x1-x2))**2
    # alpha=1/(2*R)*np.abs(x-x2)**1/np.abs(x1-x2)**1
    alpha = (
        1
        / (2 * R)
        * (3 * ((x - x2) / (x1 - x2)) ** 2 - 2 * ((x - x2) / (x1 - x2)) ** 3)
    )
    alpha[x > x2] = 0
    alpha[x < x1] = 1 / (2 * R)
    alpha = np.tile(alpha.reshape((nx, 1)), (1, ny))

    zchannel = alpha * np.abs(ycurv) ** 2
    # plt.figure()
    # plt.imshow(zchannel)

    # del alpha

    if not maxz:
        maxz = R / 2

    zchannel[zchannel > maxz] = maxz

    # Base topography in curvilinear system.
    # The transition zone between x1 and x2 is a cylindre
    zbase = -np.sin(theta2) * (x - xmax)

    ind = (x <= x2) & (x >= x1)
    angle = (x[ind] - x1) / (x2 - x1) * (theta2 - theta1) + theta1
    R2 = (x2 - x1) / (theta1 - theta2)
    z2 = -np.sin(theta2) * (x2 - xmax)
    zbase[ind] = R2 * (1 - np.cos(angle)) - R2 * (1 - np.cos(theta2)) + z2

    ind = x <= x1
    z1 = R2 * (1 - np.cos(theta1)) - R2 * (1 - np.cos(theta2)) + z2
    zbase[ind] = -np.sin(theta1) * (x[ind] - x1) + z1
    zbase = np.tile(zbase.reshape((nx, 1)), (1, ny))

    # Conversion in fixed cartesian frame
    zd = np.gradient(zbase, x[1] - x[0], edge_order=2, axis=0)
    Xd = np.sqrt(1 - zd**2)
    X = scipy.integrate.cumtrapz(Xd, x, axis=0, initial=0)
    X = X + xmin * np.cos(theta1)

    # plt.figure()
    # plt.plot(X[:,0])

    del Xd

    # Topography conversion in fixed cartesian frame
    [Fx, Fy] = np.gradient(zbase, X[:, 0], ycurv[0, :], edge_order=2)
    Fz = np.ones(zbase.shape)
    costh = 1 / np.sqrt(Fx**2 + Fy**2 + 1)  # Slope angle
    Fx = -Fx * costh
    Fy = -Fy * costh
    Fz = Fz * costh
    Z = zbase + zchannel * Fz
    Xmesh = X + zchannel * Fx
    Ymesh = ycurv + zchannel * Fy

    # Reconstruction of regular cartesian mesh
    Xout = np.linspace(Xmesh.min(), Xmesh.max(), nx)
    Xout = np.tile(Xout.reshape((nx, 1)), (1, ny))
    Yout = np.linspace(Ymesh.min(), Ymesh.max(), ny)
    Yout = np.tile(Yout.reshape((1, ny)), (nx, 1))
    Zout = scipy.interpolate.griddata(
        (Xmesh.reshape(nx * ny), Ymesh.reshape(nx * ny)),
        Z.reshape(nx * ny),
        (Xout, Yout),
        method="cubic",
    )
    Ztmp = scipy.interpolate.griddata(
        (Xmesh.reshape(nx * ny), Ymesh.reshape(nx * ny)),
        Z.reshape(nx * ny),
        (Xout, Yout),
        method="nearest",
    )
    ind = np.isnan(Zout)
    Zout[ind] = Ztmp[ind]

    del Ztmp
    # fz=scipy.interpolate.Rbf(Xmesh,Ymesh,Z)
    # Zout=fz(Xout,Yout)

    if plot:
        if theta2 == 0:
            blod, thin = tilupy.plot.get_contour_intervals(
                np.nanmin(Zout), np.nanmax(Zout)
            )
            level_min = thin
        else:
            level_min = None
        tilupy.plot.plot_topo(
            Zout.T, Xout[:, 1], Yout[1, :], level_min=level_min
        )

    return Xout[:, 1], Yout[1, :], Zout.T


def channel(
    nx=None,
    ny=None,
    dx=None,
    dy=None,
    xmin=-0.4,
    xmax=3.6,
    ymax=0.5,
    xstart_channel=0.65,
    xend_channel=2.3,
    xstart_trans=0.4,
    xend_trans=2.75,
    R=1.1,
    bend=0.2,
    nbends=1,
    theta_start=40,
    theta_channel=40,
    theta_end=0,
    plot=False,
    maxh=None,
    interp_method="linear",
):
    """
    Generate channel with potential multiple bends. Input coordinates are
    curvilinear along the flattened topography.

    Parameters
    ----------
    nx : int
        Size of the grid in x direction
    ny : int
        Size of the y direction
    dx : float, optional
        Cell size of the x axis. if specified, nx is recomputed. Default: 0.01
    dy : float, optional
        Cell size of the y axis. if specified, ny is recomputed. Default: 0.01
    xmin : float, optional
        Minimum x coordinate. The default is -0.4.
    xmax : float, optional
        Maximum x coordinate. The default is 3.2.
    ymax : float, optional
        Maximum y coordinate the final yxais spans from -ymax to xmax.
        The default is 0.5.
    xstart_channel : float, optional
        Start of the channel. The default is 0.65.
    xend_channel : float, optional
        end of the channel. The default is 2.3.
    xstart_trans : TYPE, optional
        start of the transition zone before the channel start.
        The default is 0.4.
    xend_trans : TYPE, optional
        End of the transition zone after the channel end. The default is 2.75.
    R : float, optional
        Radius of curvature of the channel. The default is 1.1.
    bend : float, optional
        Width of the channel bend. The default is 0.2.
    nbends : ind, optional
        Number of bends. The default is 1.
    theta_start : float, optional
        Slope before the channel. The default is 40.
    theta_channel : float, optional
        Slope of the channel. The default is 40.
    theta_end : float, optional
        Slope after the channel. The default is 0.
    plot : bool, optional
        Plot generated topography. The default is False.
    maxh : float, optional
        Depth of the channel. The default is None.
    interp_method : string, optional
        Interpolation method for converting the topography from curvilinear
        coordinates to cartesian coordinates. The default is 'linear'.

    Returns
    -------
    TYPE
        DESCRIPTION.

    """

    theta_start = np.deg2rad(theta_start)
    theta_channel = np.deg2rad(theta_channel)
    theta_end = np.deg2rad(theta_end)

    if ny is None and dy is None:
        dy = ymax / 100

    if nx is None and dx is None:
        if dy is not None:
            dx = dy
        else:
            raise ValueError("nx or dx must be specified as input")

    # x and y coordinates in the flattened topography
    if nx is None:
        xtopo = np.arange(xmin, xmax + dx / 2, dx)
        nx = len(xtopo)
    else:
        xtopo = np.linspace(xmin, xmax, nx)

    if ny is None:
        ytopo = np.arange(-ymax, ymax + dy / 2, dy)
        ny = len(ytopo)
    else:
        ytopo = np.linspace(-ymax, ymax, ny)

    xtopo = np.tile(xtopo[:, np.newaxis], (1, ny))
    ytopo = np.tile(ytopo[np.newaxis, :], (nx, 1))

    # Height above flattened topography is a channel
    # in alpha(x)*(y-thalweg(x))**2,

    # alpha(x) is 1/2R in the channel, and depends on a transition
    # function in the transition zones
    def trans_function(x, x1, x2):
        xx = 3 * ((x - x2) / (x1 - x2)) ** 2 - 2 * ((x - x2) / (x1 - x2)) ** 3
        return xx

    alpha = np.zeros((nx, ny))
    ind = (xtopo > xstart_channel) & (xtopo < xend_channel)
    alpha[ind] = 1 / (2 * R)
    ind = (xtopo > xstart_trans) & (xtopo <= xstart_channel)
    alpha[ind] = (
        1 / (2 * R) * trans_function(xtopo[ind], xstart_channel, xstart_trans)
    )
    ind = (xtopo > xend_channel) & (xtopo <= xend_trans)
    alpha[ind] = (
        1 / (2 * R) * trans_function(xtopo[ind], xend_channel, xend_trans)
    )

    # the thalweg is centered on y=0 outside [xstart_channel,xend_channel]. Inbetween,
    # it is given by a cos**2
    def end_bend(x, x1, x2):
        yy = (bend / 2) * (1 + np.cos(np.pi * (x - x2) / (x1 - x2)))
        return yy

    def mid_bend(x, x1, x2):
        yy = bend * np.cos(np.pi * (x - x1) / (x2 - x1))
        return yy

    thalweg = np.zeros((nx, ny))

    if nbends > 0:
        step = (xend_channel - xstart_channel) / nbends

        ind = (xtopo > xstart_channel) & (xtopo < xstart_channel + step / 2)
        thalweg[ind] = end_bend(
            xtopo[ind], xstart_channel, xstart_channel + step / 2
        )
        ind = (xtopo >= xend_channel - step / 2) & (xtopo < xend_channel)
        thalweg[ind] = (-1) ** (nbends + 1) * end_bend(
            xtopo[ind], xend_channel, xend_channel - step / 2
        )
        if nbends > 1:
            ind = (xtopo >= xstart_channel + step / 2) & (
                xtopo < xend_channel - step / 2
            )
            thalweg[ind] = mid_bend(
                xtopo[ind],
                xstart_channel + step / 2,
                xstart_channel + (3 / 2) * step,
            )

    htopo = alpha * (ytopo - thalweg) ** 2

    if not maxh:
        maxh = R / 2

    htopo[htopo > maxh] = maxh

    # Reconstruction of bz the basal topography. The real topo is given by
    # bz+\vec{n}*htopo. Slopes of bz are given by theta_* outside the transition
    # zones. We use a cylinder shape inbetween. This is done by computing the slope
    # angle of bz, and using then -sin(slope_angle)=d(bz)/d(xtopo)

    slope_angle = np.zeros((nx, ny))
    ind = xtopo < xstart_trans
    slope_angle[ind] = theta_start
    ind = xtopo >= xend_trans
    slope_angle[ind] = theta_end
    ind = (xtopo >= xstart_channel) & (xtopo < xend_channel)
    slope_angle[ind] = theta_channel

    ind = (xtopo >= xstart_trans) & (xtopo < xstart_channel)
    slope_angle[ind] = (xtopo[ind] - xstart_trans) / (
        xstart_channel - xstart_trans
    )
    slope_angle[ind] = (
        slope_angle[ind] * (theta_channel - theta_start) + theta_start
    )

    ind = (xtopo >= xend_channel) & (xtopo < xend_trans)
    slope_angle[ind] = (xtopo[ind] - xend_trans) / (xend_channel - xend_trans)
    slope_angle[ind] = (
        slope_angle[ind] * (theta_channel - theta_end) + theta_end
    )

    bz = scipy.integrate.cumtrapz(
        -np.sin(slope_angle), xtopo, axis=0, initial=0
    )
    bz = bz - np.min(bz)

    # Get the coordinates of (xtopo,ytopo) in the cartesian reference frame
    # by=ytopo
    bx = scipy.integrate.cumtrapz(
        np.cos(slope_angle), xtopo, axis=0, initial=0
    )
    bx = bx + xmin * np.cos(theta_start)

    # Vector normal to topography in cartesian coordinates
    # (nx,ny,nz)=(-sin(theta),0,cos(theta))
    # as the topography does vary in the y direction
    # The real topography is thus given in cartesian coordinates by
    # (xcart,ycart,zcart)=(bx,by,bz)+htopo(nx,ny,nz)
    xcart = bx + htopo * np.sin(slope_angle)
    zcart = bz + htopo * np.cos(slope_angle)

    # Reconstruct regular mesh for interpolation
    Xout = np.linspace(xcart[0, 0], xcart[-1, 0], nx)
    Yout = ytopo
    Xout = np.tile(Xout[:, np.newaxis], (1, ny))
    Zout = scipy.interpolate.griddata(
        (xcart.reshape(nx * ny), ytopo.reshape(nx * ny)),
        zcart.reshape(nx * ny),
        (Xout, Yout),
        method=interp_method,
    )
    Ztmp = scipy.interpolate.griddata(
        (xcart.reshape(nx * ny), ytopo.reshape(nx * ny)),
        zcart.reshape(nx * ny),
        (Xout, Yout),
        method="nearest",
    )
    ind = np.isnan(Zout)
    Zout[ind] = Ztmp[ind]

    if plot:
        if theta_end == 0:
            blod, thin = tilupy.plot.get_contour_intervals(
                np.nanmin(Zout), np.nanmax(Zout)
            )
            level_min = thin
        else:
            level_min = None
        tilupy.plot.plot_topo(
            Zout.T, Xout[:, 1], Yout[1, :], level_min=level_min
        )

    return Xout[:, 1], Yout[1, :], Zout.T, thalweg


if __name__ == "__main__":
    # %% Test gray99
    X, Y, Z = gray99(plot=True)

    # %% Test synthetic channel
    bend = 0.25
    R = 0.2

    nx = 600
    ny = 300

    xmin = 0.1
    xmax = 4.5
    xstart_trans = -0.3
    xstart_channel = 0.2
    xend_channel = 2.3
    xend_trans = 2.75
    ymax = 1

    theta_start = 10
    theta_channel = 10
    theta_end = 0
    x, y, z, t = channel(
        nx,
        ny,
        xmin=xmin,
        xmax=xmax,
        ymax=ymax,
        xstart_channel=xstart_channel,
        xend_channel=xend_channel,
        xstart_trans=xstart_trans,
        theta_start=theta_start,
        theta_end=theta_end,
        theta_channel=theta_channel,
        R=R,
        bend=bend,
        maxh=R,
        plot=True,
    )
