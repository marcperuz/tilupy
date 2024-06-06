# -*- coding: utf-8 -*-
"""
Created on Fri May 31 11:23:25 2024

@author: peruzzetto
"""

import warnings
import numpy as np
import os

from scipy.interpolate import RegularGridInterpolator

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
            if isinstance(raster, np.ndarray):
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
        self.h_edges, self.v_edges = make_edges_matrices(
            self.nx - 1, self.ny - 1
        )

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
        ix = np.argmin(np.abs(self.x[:-1] + self.dx / 2 - xcoord))
        iy = np.argmin(np.abs(self.y[:-1] + self.dx / 2 - ycoord))
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


class Simu:
    def __init__(self, folder, name):
        os.makedirs(folder, exist_ok=True)
        self.folder = folder
        self.name = name
        self.x = None
        self.y = None
        self.z = None

    def set_topography(self, z, x=None, y=None, file_out=None):
        if isinstance(z, str):
            self.x, self.y, self.z = tilupy.raster.read_raster(z)
        else:
            self.x, self.y, self.z = x, y, z

        if file_out is None:
            file_out = "toposimu.asc"
        else:
            fname = file_out.split(".")[0]
            if len(fname) > 8:
                warnings.warns(
                    """Topography file name is too long,
                    only 8 first characters are retained"""
                )
                fname = fname[:8]
            elif len(fname) < 8:
                warnings.warns(
                    """Topography file name is too short and is adapted,
                    exactly 8 characters needed"""
                )
                fname = fname.ljust(8, "0")
            file_out = fname + ".asc"

        tilupy.raster.write_ascii(
            self.x, self.y, self.z, os.path.join(self.folder, file_out)
        )

    def set_numeric_params(
        self,
        tmax,
        dtsorties,
        paray=0.00099,
        dtinit=0.01,
        cfl_const=1,
        CFL=0.2,
        alpha_cor_pente=1.0,
    ):
        self.tmax = tmax
        self.dtsorties = dtsorties

        with open(os.path.join(self.folder, "DONLEDD1.DON"), "w") as fid:
            fid.write(
                "paray".ljust(34, " ") + "{:.12f}\n".format(paray).lstrip("0")
            )
            fid.write("tmax".ljust(34, " ") + "{:.12f}\n".format(tmax))
            fid.write("dtinit".ljust(34, " ") + "{:.12f}\n".format(dtinit))
            fid.write(
                "cfl constante ?".ljust(34, " ") + "{:.0f}\n".format(cfl_const)
            )
            fid.write("CFL".ljust(34, " ") + "{:.12f}\n".format(CFL))
            fid.write("Go,VaLe1,VaLe2".ljust(34, " ") + "{:.0f}\n".format(3))
            fid.write("secmbr0/1".ljust(34, " ") + "{:.0f}\n".format(1))
            fid.write(
                "CL variable (si=0) ?".ljust(34, " ") + "{:d}\n".format(0)
            )
            fid.write(
                "dtsorties".ljust(34, " ") + "{:.12f}\n".format(dtsorties)
            )
            fid.write(
                "alpha cor pentes".ljust(34, " ")
                + "{:.12f}".format(alpha_cor_pente)
            )

    def set_rheology(self, tau_rho, K_tau=0.3):
        """
        Set Herschel-Bulkley rheology parameters

        Rheological parameters are tau/rho and K/tau. See
        -Coussot, P., 1994. Steady, laminar, flow of concentrated mud
        suspensions in open channel. Journal of Hydraulic Research 32, 535–559.
        doi.org/10.1080/00221686.1994.9728354 --> Eq 8 and text after eq 22
        for the default value of K/tau
        -Rickenmann, D. et al., 2006. Comparison of 2D debris-flow
        simulation models with field events. Computational Geosciences 10,
        241–264. doi.org/10.1007/s10596-005-9021-3 --> Eq 9


        :param tau_rho: DESCRIPTION
        :type tau_rho: TYPE
        :param K_rau: DESCRIPTION
        :type K_rau: TYPE
        :return: DESCRIPTION
        :rtype: TYPE

        """
        with open(os.path.join(self.folder, self.name + ".rhe"), "w") as fid:
            fid.write("{:.3f}\n".format(tau_rho))
            fid.write("{:.3f}".format(K_tau))

    def set_boundary_conditions(
        self,
        xcoords,
        ycoords,
        cardinal,
        discharges,
        times=None,
        thicknesses=None,
        tmax=9999,
        discharge_from_volume=False,
    ):
        try:
            modelling_domain = ModellingDomain(
                self.z,
                self.x[0],
                self.y[0],
                dx=self.x[1] - self.x[0],
                dy=self.y[1] - self.y[0],
            )
        except AttributeError:
            print("Simulation topography has not been set")
            raise

        edges = modelling_domain.get_edges(
            xcoords, ycoords, cardinal, from_extremities=True
        )
        edges = edges[cardinal]

        if times is None:
            # If no time vector is given, discharges is interpreted as a volume
            # from which an empirical peak discharge is computed
            peak_discharge = 0.0188 * discharges**0.79
            times = [0, 2 * discharges / peak_discharge, tmax]
            discharges = [peak_discharge, 0, 0]

        n_times = len(times)
        n_edges = len(edges)

        assert n_times == len(
            discharges
        ), "discharges and time vectors must be the same length"

        if thicknesses is None:
            thicknesses = [1 for i in range(n_times)]

        if cardinal in ["W", "E"]:
            dd = self.y[1] - self.y[0]
        else:
            dd = self.x[1] - self.x[0]

        with open(os.path.join(self.folder, self.name + ".cli"), "w") as fid:
            fid.write(
                "{:d}\n".format(n_times)
            )  # Number of time steps in the hydrogram
            for i, time in enumerate(times):
                fid.write("{:.4f}\n".format(time))  # Write time
                fid.write("{:d}\n".format(n_edges))  # Write n_edges
                qn = discharges[i] / (n_edges * dd)
                qt = 0
                for i_edge, edge in enumerate(edges):
                    fid.write(
                        "\t{:d} {:.7E} {:.7E} {:.7E}\n".format(
                            edge, thicknesses[i], qn, qt
                        )
                    )

    def set_init_mass(self, mass_raster, h_min=0.01):
        x, y, m = tilupy.raster.read_ascii(mass_raster)
        # The input raster is given as the topography for the grid corners,
        # but results are then written on grid cell centers
        x2 = x[1:] - (x[1] - x[0]) / 2
        y2 = y[1:] - (y[1] - y[0]) / 2
        fm = RegularGridInterpolator(
            (y, x),
            m,
            method="linear",
            bounds_error=False,
            fill_value=None,
        )
        x_mesh, y_mesh = np.meshgrid(x2, y2)
        m2 = fm((y_mesh, x_mesh))
        m2[m2 < h_min] = h_min
        np.savetxt(
            os.path.join(self.folder, self.name + ".cin"),
            m2.flatten(),
            header="0.000000E+00",
            comments="",
            fmt="%.10E",
        )


if __name__ == "__main__":
    domain = ModellingDomain(xmin=0, ymin=0, nx=4, ny=3, dx=1, dy=1)
    domain.set_edges()
    res = domain.get_edges([1.5, 2.5], [0, 0], "S")
