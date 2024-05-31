# -*- coding: utf-8 -*-
"""
Created on Fri May 31 11:23:25 2024

@author: peruzzetto
"""

import numpy as np
import tilupy.raster


def make_edges_matrices(nx, ny):
    """
    Numbering edges for a regular rectangular grid.

    Considering a matrix M with whape (ny, nx),the convention is that
    M[0, 0] corresponds to the lower left corner of the matrix, M[0, -1]
    to the lower right corner, M[-1, 0] to the upper left and M[-1, -1] to
    the upper right. Edges are numbered cell by cell, counter clockwise :
    bottom, right, up, left. The first cell is M[0, 0], then cells are
    processed line by line.

    :param nx: Number of faces in the X direction
    :type nx: int
    :param ny: Number of faces in the Y direction
    :type ny: int
    :return: DESCRIPTION
    :rtype: TYPE

    """

    # Numbers of horizontal edges
    h_edges = np.zeros((ny + 1, nx))
    # Numbers of vertical edges
    v_edges = np.zeros((ny, nx + 1))

    # Number of edges on first completed line
    n_edges_l1 = 4 + 3 * (nx - 1)
    # Number of edges on folowwing lines
    n_edges_l = 3 + 2 * (nx - 1)

    # Fill first line of h_edges
    h_edges[0, 0] = 1
    h_edges[0, 1:] = np.arange(nx - 1) * 3 + 5
    # Fill second line of h_edges
    h_edges[1, :] = h_edges[0, :] + 2

    try:
        # Fill first column of h_edges
        h_edges[2:, 0] = np.arange(ny - 1) * n_edges_l + n_edges_l1 + 2
        # Fill the rest of the table
        tmp = np.concatenate(([3], np.arange(nx - 2) * 2 + 5))
        h_edges[2:, 1:] = h_edges[2:, 0][:, np.newaxis] + tmp[np.newaxis, :]
    except IndexError:
        pass

    # Fill first  2 columns of v_edges
    v_edges[0, 0] = 4
    try:
        v_edges[1:, 0] = np.arange(ny - 1) * n_edges_l + n_edges_l1 + 3
    except IndexError:
        pass
    v_edges[:, 1] = v_edges[:, 0] - 2
    # Fill the rest of v_edges
    try:
        v_edges[0, 2:] = 2 + np.concatenate(([4], np.arange(nx - 2) * 3 + 7))
        tmp = np.concatenate(([3], np.arange(nx - 2) * 2 + 5))
        v_edges[1:, 2:] = v_edges[1:, 1][:, np.newaxis] + tmp[np.newaxis, :]
    except IndexError:
        pass

    h_edges = np.flip(h_edges, axis=0).astype(int)
    v_edges = np.flip(v_edges, axis=0).astype(int)
    return h_edges, v_edges


if __name__ == "__main__":
    h_edges, v_edges = make_edges_matrices(1, 1)


class ModellingDomain:
    """
    Mesh of simulation topography
    """

    def __init__(
        self,
        raster=None,
        xmin=0,
        ymin=0,
        nx=None,
        ny=None,
        dx=1,
        dy=1,
    ):
        if raster is not None:
            if isinstance(raster, np.ndarray()):
                self.z = raster
                self.nx = raster.shape[1]
                self.ny = raster.shape[0]
                self.dx = dx
                self.dy = dy
                self.x = np.arange(
                    xmin, xmin + (self.nx - 1) * self.dx + self.dx / 2, self.dx
                )
                self.y = np.arange(
                    ymin, ymin + (self.ny - 1) * self.dy + self.dy / 2, self.dy
                )
            if isinstance(raster, str):
                self.x, self.y, self.z = tilupy.raster.read_raster(raster)
                self.nx = len(self.x)
                self.ny = len(self.y)
                self.dx = self.x[1] - self.x[0]
                self.dy = self.y[1] - self.y[0]
        else:
            self.z = raster
            self.nx = nx
            self.ny = ny
            self.dx = dx
            self.dy = dy
            if (
                xmin is not None
                and ymin is not None
                and nx is not None
                and ny is not None
                and dx is not None
                and dy is not None
            ):
                self.x = np.arange(xmin, xmin + dx * nx + dx / 2, dx)
                self.y = np.arange(ymin, ymin + dy * ny + dy / 2, dy)

        self.h_edges = None
        self.v_edges = None

        if self.nx is not None and self.ny is not None:
            self.set_edges()

    def set_edges(self):
        """
        Set horizontal and vertical edges number
        :return: DESCRIPTION
        :rtype: TYPE

        """
        self.h_edges, self.v_edges = make_edges_matrices(self.nx, self.ny)

    def get_edge(self, xcoord, ycoord, cardinal):
        """
        Get edge number for given coordinates and cardinal direction
        :param xcoord: DESCRIPTION
        :type xcoord: TYPE
        :param ycoord: DESCRIPTION
        :type ycoord: TYPE
        :param cardinal: DESCRIPTION
        :type cardinal: TYPE
        :return: DESCRIPTION
        :rtype: TYPE

        """
        assert (self.h_edges is not None) and (
            self.v_edges is not None
        ), "h_edges and v_edges must be computed to get edge number"
        ix = np.argmin(np.abs(self.x - xcoord))
        iy = np.argmin(np.abs(self.y - ycoord))
        if cardinal == "S":
            return self.h_edges[-iy - 1, ix]
        elif cardinal == "N":
            return self.h_edges[-iy - 2, ix]
        elif cardinal == "W":
            return self.v_edges[-iy - 1, ix]
        elif cardinal == "E":
            return self.v_edges[-iy - 1, ix + 1]

    def get_edges(self, xcoords, ycoords, cardinal, from_extremities=True):
        """
        Get edges numbers for a set of coordinates and cardinals direction.
        xcoords and ycoords can be the min and max x and y coordinates of a line
        if from_extremities is True.
        :param xcoords: DESCRIPTION
        :type xcoords: TYPE
        :param ycoords: DESCRIPTION
        :type ycoords: TYPE
        :param cardinal: DESCRIPTION
        :type cardinal: TYPE
        :param from_extremities: DESCRIPTION, defaults to True
        :type from_extremities: TYPE, optional
        :return: DESCRIPTION
        :rtype: TYPE

        """
        if from_extremities:
            d = np.sqrt(
                (xcoords[1] - xcoords[0]) ** 2 + (ycoords[1] - ycoords[0]) ** 2
            )
            npts = int(np.ceil(d / min(self.dx, self.dy)))
            xcoords = np.linspace(xcoords[0], xcoords[1], npts + 1)
            ycoords = np.linspace(ycoords[0], ycoords[1], npts + 1)
        print(xcoords)
        if self.h_edges is None or self.v_edges is None:
            self.set_edges
        res = dict()
        # cardinal is a string with any combination of cardinal directions
        for card in cardinal:
            edges_num = []
            for xcoord, ycoord in zip(xcoords, ycoords):
                edges_num.append(self.get_edge(xcoord, ycoord, card))
            res[card] = list(set(edges_num))  # Duplicate edges are removed
            res[card].sort()
        return res
