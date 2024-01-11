# -*- coding: utf-8 -*-
"""
Created on Mon Oct 10 14:47:17 2022

@author: peruzzetto
"""

import numpy as np
import matplotlib.pyplot as plt

import shapely.geometry as geom
import shapely.ops


def CSI(pred, obs):
    ipred = pred > 0
    iobs = obs > 0

    TP = np.sum(ipred * iobs)
    FP = np.sum(ipred * ~iobs)
    FN = np.sum(~ipred * iobs)

    return TP / (TP + FP + FN)


def diff_runout(
    x_contour, y_contour, point_ref, section=None, orientation="W-E"
):
    npts = len(x_contour)
    contour = geom.LineString(
        [(x_contour[i], y_contour[i]) for i in range(npts)]
    )
    point = geom.Point(point_ref)
    if section is None:
        return point.distance(contour)
    elif isinstance(section, np.ndarray):
        section = geom.LineString(section)

    assert isinstance(section, geom.LineString)
    section = revert_line(section, orientation)
    intersections = section.intersection(contour)
    if isinstance(intersections, geom.MultiPoint):
        intersections = geom.LineString(section.intersection(contour))
    intersections = np.array(intersections.coords)
    if orientation == "W-E":
        i = np.argmax(intersections[:, 0])
    if orientation == "E-W":
        i = np.argmin(intersections[:, 0])
    if orientation == "S-N":
        i = np.argmax(intersections[:, 1])
    if orientation == "N-S":
        i = np.argmin(intersections[:, 1])
    intersection = geom.Point(intersections[i, :])

    #######
    # plt.figure()
    # cont = np.array(contour.coords)
    # sec = np.array(section.coords)
    # inter = np.array(intersection.coords)
    # pt = np.array(point.coords)
    # plt.plot(cont[:,0], cont[:,1],
    #           sec[:,0], sec[:,1],
    #           pt[:,0], pt[:,1], 'o',
    #           inter[:,0], inter[:,1],'x')
    #######

    return section.project(intersection) - section.project(point)


def revert_line(line, orientation="W-E"):
    pt_init = line.coords[0]
    pt_end = line.coords[-1]
    if orientation == "W-E":
        if pt_init[0] > pt_end[0]:
            line = shapely.ops.substring(line, 1, 0, normalized=True)
    elif orientation == "E-W":
        if pt_init[0] < pt_end[0]:
            line = shapely.ops.substring(line, 1, 0, normalized=True)
    if orientation == "S-N":
        if pt_init[1] > pt_end[1]:
            line = shapely.ops.substring(line, 1, 0, normalized=True)
    elif orientation == "N-S":
        if pt_init[1] < pt_end[1]:
            line = shapely.ops.substring(line, 1, 0, normalized=True)

    return line


def get_contour(x, y, z, zlevels, indstep=1, maxdist=30, closed_contour=True):
    # Add sup_value at the border of the array, to make sure contour
    # lines are closed
    if closed_contour:
        z2 = z.copy()
        ni = z2.shape[0]
        nj = z2.shape[1]
        z2 = np.vstack([np.zeros((1, nj)), z2, np.zeros((1, nj))])
        z2 = np.hstack([np.zeros((ni + 2, 1)), z2, np.zeros((ni + 2, 1))])
        dxi = x[1] - x[0]
        dxf = x[-1] - x[-2]
        dyi = y[1] - y[0]
        dyf = y[-1] - y[-2]
        x2 = np.insert(np.append(x, x[-1] + dxf), 0, x[0] - dxi)
        y2 = np.insert(np.append(y, y[-1] + dyf), 0, y[0] - dyi)
    else:
        x2, y2, z2 = x, y, z

    backend = plt.get_backend()
    plt.switch_backend("Agg")
    plt.figure()
    ax = plt.gca()
    cs = ax.contour(x2, y2, np.flip(z2, 0), zlevels)
    nn1 = 1
    v1 = np.zeros((1, 2))
    xcontour = {}
    ycontour = {}
    for indlevel in range(len(zlevels)):
        for p in cs.collections[indlevel].get_paths():
            if p.vertices.shape[0] > nn1:
                v1 = p.vertices
                nn1 = p.vertices.shape[0]
        xc = [v1[::indstep, 0]]
        yc = [v1[::indstep, 1]]
        if maxdist is not None and not closed_contour:
            ddx = np.abs(v1[0, 0] - v1[-1, 0])
            ddy = np.abs(v1[0, 1] - v1[-1, 1])
            dd = np.sqrt(ddx**2 + ddy**2)
            if dd > maxdist:
                xc[0] = None
                yc[0] = None
        xcontour[zlevels[indlevel]] = xc[0]
        ycontour[zlevels[indlevel]] = yc[0]
    plt.switch_backend(backend)
    return xcontour, ycontour


def format_path_linux(path):
    """
    Change a Windows-type path to a path formatted for Linux. \\ are changed
    to /, and partitions like "C:" are changed to "/mnt/c/"

    Parameters
    ----------
    path : string
        String with the path to be modified.

    Returns
    -------
    path2 : string
        Formatted path.

    """
    if path[1] == ":":
        path2 = "/mnt/{:s}/".format(path[0].lower()) + path[2:]
    else:
        path2 = path
    path2 = path2.replace("\\", "/")
    if " " in path2:
        path2 = '"' + path2 + '"'
    return path2
