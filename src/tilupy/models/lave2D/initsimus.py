# -*- coding: utf-8 -*-
"""
Created on Fri May 31 11:23:25 2024

@author: peruzzetto
"""

import numpy as np


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
