# -*- coding: utf-8 -*-

import warnings
import numpy as np
import os
import shutil

from scipy.interpolate import RegularGridInterpolator

import tilupy.raster


def make_edges_matrices(nx: int, ny: int) -> list[np.ndarray, np.ndarray]:
    """Numbering edges for a regular rectangular grid.

    Considering a matrix M with whape (ny, nx),the convention is that
    M[0, 0] corresponds to the lower left corner of the matrix, M[0, -1]
    to the lower right corner, M[-1, 0] to the upper left and M[-1, -1] to
    the upper right. Edges are numbered cell by cell, counter clockwise :
    bottom, right, up, left. The first cell is M[0, 0], then cells are
    processed line by line.

    Parameters
    ----------
    nx : int
        Number of faces in the X direction
    ny : int
        Number of faces in the Y direction

    Returns
    -------
    h_edges : numpy.ndarray
        Matrix containing the global edge indices of all horizontal edges
        (bottom and top edges of the cells). Rows correspond to y-levels,
        columns to x-positions.
    v_edges : numpy.ndarray
        Matrix containing the global edge indices of all vertical edges
        (left and right edges of the cells). Rows correspond to y-levels,
        columns to x-positions.
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


class ModellingDomain:
    """Mesh of simulation topography
    
    This class represents a structured 2D mesh built either from a raster
    or from user-specified dimensions and spacing.
    It stores the topographic data (z), the grid geometry (x, y, dx, dy, nx, ny),
    and the numbering of edges used for discretization of the domain.
    
    Parameters
    ----------
        raster : numpy.ndarray, str, or None, optional
            - If numpy.ndarray: used directly as the elevation grid (z).
            - If str: path to a raster file readable by :func:`tilupy.raster.read_raster`.
            - If None: grid is generated from :data:`nx`, :data:`ny`, :data:`dx`, :data:`dy`.
        xmin : float, optional
            Minimum X coordinate of the grid, used if :data:`raster` is None, 
            by default 0.0.
        ymin : float, optional
            Minimum Y coordinate of the grid, used if :data:`raster` is None, 
            by default 0.0.
        nx : int, optional
            Number of grid points in the X direction, used if :data:`raster` is None, 
            by default None.
        ny : int, optional
            Number of grid points in the Y direction, used if :data:`raster` is None, 
            by default None.
        dx : float, optional
            Grid spacing along X, by default 1.0.
        dy : float, optional
            Grid spacing along Y, by default 1.0.
    
    Attributes
    ----------
        _z : numpy.ndarray
            Elevation values of the mesh nodes ([ny, nx]).
        _nx : int
            Number of grid points along the X direction.
        _ny : int
            Number of grid points along the Y direction.
        _dx : float
            Grid resolution in the X direction.
        _dy : float
            Grid resolution in the Y direction.
        _x : numpy.ndarray
            Array of X coordinates.
        _y : numpy.ndarray
            Array of Y coordinates.
        _h_edges : numpy.ndarray
            Matrix of horizontal edge numbers ([ny, nx-1]).
        _v_edges : numpy.ndarray
            Matrix of vertical edge numbers ([ny-1, nx]).
    """
    def __init__(self,
                 raster: np.ndarray = None,
                 xmin: float = 0,
                 ymin: float = 0,
                 nx: int = None,
                 ny: int = None,
                 dx: float = 1,
                 dy: float = 1,
                 ):
        if raster is not None:
            if isinstance(raster, np.ndarray):
                self._z = raster
                self._nx = raster.shape[1]
                self._ny = raster.shape[0]
                self._dx = dx
                self._dy = dy
                self._x = np.arange(xmin, 
                                    xmin + (self._nx - 1) * self._dx + self._dx / 2, 
                                    self._dx)
                self._y = np.arange(ymin, 
                                    ymin + (self._ny - 1) * self._dy + self._dy / 2, 
                                    self._dy)
                
            if isinstance(raster, str):
                self._x, self._y, self._z = tilupy.raster.read_raster(raster)
                self._nx = len(self._x)
                self._ny = len(self._y)
                self._dx = self._x[1] - self._x[0]
                self._dy = self._y[1] - self._y[0]
        else:
            self._z = raster
            self._nx = nx
            self._ny = ny
            self._dx = dx
            self._dy = dy
            if (xmin is not None
                and ymin is not None
                and nx is not None
                and ny is not None
                and dx is not None
                and dy is not None
                ):
                self._x = np.arange(xmin, xmin + dx * nx + dx / 2, dx)
                self._y = np.arange(ymin, ymin + dy * ny + dy / 2, dy)

        self._h_edges = None
        self._v_edges = None

        if self._nx is not None and self._ny is not None:
            self.set_edges()


    def set_edges(self) -> None:
        """Compute and assign horizontal and vertical edge number.
        
        Calls :func:`tilupy.initsimus.make_edges_matrices` to build the edge numbering
        for the grid defined by nx and ny. 
        Results are stored in attributes :attr:`_h_edges` and :attr:`_v_edges`.
        """
        self._h_edges, self._v_edges = make_edges_matrices(self._nx - 1, self._ny - 1)


    def get_edge(self, xcoord: float, ycoord: float, cardinal: str) -> int:
        """Get edge number for given coordinates and cardinal direction

        Parameters
        ----------
        xcoord : float
            X coordinate of the point.
        ycoord : float
            Y coordinate of the point.
        cardinal : str
            Cardinal direction of the edge ('N', 'S', 'E', 'W'):
            
                - 'N': top edge of the cell
                - 'S': bottom edge of the cell
                - 'E': right edge of the cell
                - 'W': left edge of the cell
        
        Returns
        -------
        int
            Edge number corresponding to the requested location and direction.
        
        Raises
        ------
        AssertionError
            If :meth:`set_edges` has not been called before (i.e. :attr:`_h_edges` and :attr:`_v_edges` are None).
        """
        assert (self._h_edges is not None) and (self._v_edges is not None), "h_edges and v_edges must be computed to get edge number"
        ix = np.argmin(np.abs(self._x[:-1] + self._dx / 2 - xcoord))
        iy = np.argmin(np.abs(self._y[:-1] + self._dx / 2 - ycoord))
        if cardinal == "S":
            return self._h_edges[-iy - 1, ix]
        elif cardinal == "N":
            return self._h_edges[-iy - 2, ix]
        elif cardinal == "W":
            return self._v_edges[-iy - 1, ix]
        elif cardinal == "E":
            return self._v_edges[-iy - 1, ix + 1]


    def get_edges(self, xcoords: np.ndarray, ycoords: np.ndarray, cardinal: str, from_extremities: bool=True) -> dict[str: list[int]]:
        """Get edges numbers for a set of coordinates and cardinals direction.
        
        If :data:`from_extremities` is True, xcoords and ycoords are treated as the extremities of a segment. 

        Parameters
        ----------
        xcoords : np.ndarray or list
            X coordinates of points. If :data:`from_extremities` is True, indicated (xmin, xmax).
        ycoords : np.ndarray or list
            Y coordinates of points. If :data:`from_extremities` is True, indicated (ymin, ymax).
        cardinal : str
            Cardinal directions to extract, any combination of {'N', 'S', 'E', 'W'}.
            Example: "NS" will return both north and south edges.
        from_extremities : bool, optional
            If True, :data:`xcoords` and :data:`ycoords` are considered as the endpoints
            of a line, by default True

        Returns
        -------
        dict[str: list[int]]
            Dictionary where keys are the requested cardinals and values are
            sorted lists of unique edge numbers.
        """
        if from_extremities:
            d = np.sqrt((xcoords[1] - xcoords[0]) ** 2 + (ycoords[1] - ycoords[0]) ** 2)
            npts = int(np.ceil(d / min(self._dx, self._dy)))
            xcoords = np.linspace(xcoords[0], xcoords[1], npts + 1)
            ycoords = np.linspace(ycoords[0], ycoords[1], npts + 1)
            
        if self._h_edges is None or self._v_edges is None:
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
    """Simulation configuration and input file generator.
    
    This class manages the setup of lave2D. It creates the necessary input files (topography, numeric
    parameters, rheology, boundary conditions, initial mass) that will be read by lave2D.
    
    Parameters
    ----------
        folder : str
            Directory where simulation files are written. Created if it does not exist.
        name : str
            Base name of the simulation (used as file prefix).
    
    Attributes
    ----------
        _folder : str
            Path to the directory where simulation input and output files are stored.
        _name : str
            Base name of the simulation, used as a prefix for generated files. Must have 8 characters.
        _x : numpy.ndarray
            Array of X coordinates of the topography grid.
        _y : numpy.ndarray
            Array of Y coordinates of the topography grid.
        _z : numpy.ndarray
            2D array of topographic elevations.
        _tmax : float
            Maximum simulation time.
        _dtsorties : float
            Time step for output results.
    """
    def __init__(self, folder: str, name: str):
        os.makedirs(folder, exist_ok=True)
        self._folder = folder
        self._name = name
        self._x = None
        self._y = None
        self._z = None


    def set_topography(self, z: np.ndarray | str, 
                       x: np.ndarray=None, 
                       y: np.ndarray=None, 
                       file_out: str=None
                       ) -> None:
        """Set simulation topography and write it as an ASCII grid.

        Parameters
        ----------
        z : ndarray or str
            Topography values as a 2D NumPy array, or path to a raster file.
        x : ndarray, optional
            X coordinates of the grid, ignored if :data:`z` is a file path.
        y : ndarray, optional
            Y coordinates of the grid, ignored if :data:`z` is a file path.
        file_out : str, optional
            Output filename (add ".asc"), by defaults "toposimu.asc".
            Filenames are truncated or padded to 8 characters.

        Returns
        -------
        None
        """
        if isinstance(z, str):
            self._x, self._y, self._z = tilupy.raster.read_raster(z)
        else:
            self._x, self._y, self._z = x, y, z

        if file_out is None:
            file_out = "toposimu.asc"
        else:
            fname = file_out.split(".")[0]
            if len(fname) > 8:
                warnings.warns("""Topography file name is too long,
                               only 8 first characters are retained"""
                               )
                fname = fname[:8]
            elif len(fname) < 8:
                warnings.warns("""Topography file name is too short and is adapted,
                               exactly 8 characters needed"""
                               )
                fname = fname.ljust(8, "0")
            file_out = fname + ".asc"

        tilupy.raster.write_ascii(self._x, 
                                  self._y, 
                                  self._z, 
                                  os.path.join(self._folder, file_out))


    def set_numeric_params(self,
                           tmax: float,
                           dtsorties: float,
                           paray: float=0.00099,
                           dtinit: float=0.01,
                           cfl_const: int=1,
                           CFL: float=0.2,
                           alpha_cor_pente: float=1.0,
                           ) -> None:
        """Set numerical simulation parameters and write them to file.

        Parameters
        ----------
        tmax : float
            Maximum simulation time.
        dtsorties : float
            Time step for output results.
        paray : float, optional
            Hydraulic roughness parameter, by default 0.00099.
        dtinit : float, optional
            Initial time step, by default 0.01.
        cfl_const : int, optional
            If 1, CFL condition is constant, by default 1.
        CFL : float, optional
            Courant-Friedrichs-Lewy number, by default 0.2.
        alpha_cor_pente : float, optional
            Slope correction coefficient, by default 1.0.

        Returns
        -------
        None
        """
        self._tmax = tmax
        self._dtsorties = dtsorties

        with open(os.path.join(self._folder, "DONLEDD1.DON"), "w") as fid:
            fid.write("paray".ljust(34, " ") + "{:.12f}\n".format(paray).lstrip("0"))
            fid.write("tmax".ljust(34, " ") + "{:.12f}\n".format(tmax))
            fid.write("dtinit".ljust(34, " ") + "{:.12f}\n".format(dtinit))
            fid.write("cfl constante ?".ljust(34, " ") + "{:.0f}\n".format(cfl_const))
            fid.write("CFL".ljust(34, " ") + "{:.12f}\n".format(CFL))
            fid.write("Go,VaLe1,VaLe2".ljust(34, " ") + "{:.0f}\n".format(3))
            fid.write("secmbr0/1".ljust(34, " ") + "{:.0f}\n".format(1))
            fid.write("CL variable (si=0) ?".ljust(34, " ") + "{:d}\n".format(0))
            fid.write("dtsorties".ljust(34, " ") + "{:.12f}\n".format(dtsorties))
            fid.write("alpha cor pentes".ljust(34, " ") + "{:.12f}".format(alpha_cor_pente))


    def set_rheology(self, tau_rho: float, K_tau: float=0.3) -> None:
        r"""Set Herschel-Bulkley rheology parameters

        Rheological parameters are tau/rho and K/tau. See :
        
            - Coussot, P., 1994. Steady, laminar, flow of concentrated mud suspensions in open channel. Journal of Hydraulic Research 32, 535-559. doi.org/10.1080/00221686.1994.9728354 
                --> Eq 8 and text after eq 22 for the default value of K/tau
            - Rickenmann, D. et al., 2006. Comparison of 2D debris-flow simulation models with field events. Computational Geosciences 10, 241-264. doi.org/10.1007/s10596-005-9021-3 
                --> Eq 9

        tau_rho : float
            Yield stress divided by density (:math:`\tau/\rho`).
        K_tau : float, optional
            Consistency index divided by yield stress (:math:`K/\tau`).
            By default 0.3, following Rickenmann (2006).
        
        Returns
        -------
        None
        """
        with open(os.path.join(self._folder, self._name + ".rhe"), "w") as fid:
            fid.write("{:.3f}\n".format(tau_rho))
            fid.write("{:.3f}".format(K_tau))


    def set_boundary_conditions(self,
                                xcoords: list,
                                ycoords: list,
                                cardinal: str,
                                discharges: list | float,
                                times: list=None,
                                thicknesses: list=None,
                                tmax: float=9999,
                                discharge_from_volume: bool=False,
                                ) -> None:
        r"""Define and write inflow hydrograph boundary conditions.

        Parameters
        ----------
        xcoords : list
            X coordinates of boundary location (two values for segment ends).
        ycoords : list
            Y coordinates of boundary location (two values for segment ends).
        cardinal : str 
            Boundary orientation ('N', 'S', 'E', 'W').
        discharges : list or float
            If :data:`times` is None: interpreted as inflow volume (:math:`m^3`).
            Else: Interpreted as discharge values (:math:`m^3/s`) at each time step.
        times : list, optional
            Time vector associated with discharges. If None, a synthetic
            hydrograph is built from volume. By default None.
        thicknesses : list, optional
            Flow thicknesses for each time step. If None, put 1m everywhere. By default None.
        tmax : float, optional
            Maximum time for synthetic hydrograph generation, by default 9999.
        discharge_from_volume : bool, optional
            If True, discharges are computed from inflow volume, by default False. Not used.

        Returns
        -------
        None
        
        Raises
        ------
        AttributeError
            If no topography not been set.
        """
        try:
            modelling_domain = ModellingDomain(self._z,
                                               self._x[0],
                                               self._y[0],
                                               dx=self._x[1] - self._x[0],
                                               dy=self._y[1] - self._y[0])
        except AttributeError:
            print("Simulation topography has not been set")
            raise

        edges = modelling_domain.get_edges(xcoords, 
                                           ycoords, 
                                           cardinal, 
                                           from_extremities=True)
        edges = edges[cardinal]

        if times is None:
            # If no time vector is given, discharges is interpreted as a volume
            # from which an empirical peak discharge is computed
            peak_discharge = 0.0188 * discharges**0.79
            times = [0, 2 * discharges / peak_discharge, tmax]
            discharges = [peak_discharge, 0, 0]

        n_times = len(times)
        n_edges = len(edges)

        assert n_times == len(discharges), "discharges and time vectors must be the same length"

        if thicknesses is None:
            thicknesses = [1 for i in range(n_times)]

        if cardinal in ["W", "E"]:
            dd = self._y[1] - self._y[0]
        else:
            dd = self._x[1] - self._x[0]

        with open(os.path.join(self._folder, self._name + ".cli"), "w") as fid:
            fid.write("{:d}\n".format(n_times))  # Number of time steps in the hydrogram
            for i, time in enumerate(times):
                fid.write("{:.4f}\n".format(time))  # Write time
                fid.write("{:d}\n".format(n_edges))  # Write n_edges
                qn = discharges[i] / (n_edges * dd)
                qt = 0
                for i_edge, edge in enumerate(edges):
                    fid.write("\t{:d} {:.7E} {:.7E} {:.7E}\n".format(edge, 
                                                                     thicknesses[i], 
                                                                     qn, 
                                                                     qt))


    def set_init_mass(self, mass_raster: str, h_min: float=0.01) -> None:
        """Set initial debris-flow mass from a raster file.

        Reads an ASCII raster of initial thickness and interpolates it onto
        the simulation grid cell centers.

        Parameters
        ----------
        mass_raster : str
            Path to raster file containing initial mass thickness.
        h_min : float, optional
            Minimum non-zero thickness assigned to the grid. Default is 0.01.

        Returns
        -------
        None
        """
        x, y, m = tilupy.raster.read_ascii(mass_raster)
        # The input raster is given as the topography for the grid corners,
        # but results are then written on grid cell centers
        x2 = x[1:] - (x[1] - x[0]) / 2
        y2 = y[1:] - (y[1] - y[0]) / 2
        
        y2 = y2[::-1]
        
        fm = RegularGridInterpolator((y, x),
                                     m,
                                     method="linear",
                                     bounds_error=False,
                                     fill_value=None)
        x_mesh, y_mesh = np.meshgrid(x2, y2)
        m2 = fm((y_mesh, x_mesh))
        m2[m2 < h_min] = h_min
        np.savetxt(os.path.join(self._folder, self._name + ".cin"),
                   m2.flatten(),
                   header="0.000000E+00",
                   comments="",
                   fmt="%.10E")


def write_simu(raster_topo: str, 
               raster_mass: str, 
               tmax: float, 
               dt_im: float,
               simulation_name: str,
               lave2D_exe_folder: str,
               rheology_type: str,
               rheology_params: dict,
               folder_out: str = None,
               ) -> None:
    """
    Prepares all input files required for a Lave2D simulation and saves them in a dedicated folder.

    Parameters
    ----------
    raster_topo : str
        Name of the ASCII topography file.
    raster_mass : str
        Name of the ASCII initial mass file.
    tmax : float
        Maximum simulation time.
    dt_im : float
        Output image interval (in time steps).
    simulation_name : str
        Simulation/project name.
    lave2D_exe_folder : str
        Path to the folder containing "Lave2_Arc.exe" and "vf2marc.exe".
    rheology_type : str
        Rheology to use for the simulation. 
    rheology_params : dict
        Necessary parameters for the rheology. For this case:
            - tau_rho
            - K_tau
    folder_out : str, optional
        Output folder where simulation inputs will be stored.
    
    Returns
    -------
    None
        
    Raises
    ------
    ValueError
        If the rheology isn't Herschel_Bulkley.
    """
    if folder_out is None:
        folder_out = "."
    
    if rheology_type != "Herschel_Bulkley":
        raise ValueError("Rheology type must be 'Herschel_Bulkley'.")
    
    # output_file = os.path.join(folder_out, "lave2D")

    os.makedirs(folder_out, exist_ok=True)
    
    shutil.copy2(os.path.join(lave2D_exe_folder, "Lave2_Arc.exe"),
                 folder_out)
    shutil.copy2(os.path.join(lave2D_exe_folder, "vf2marc.exe"), 
                 folder_out)
    
    simu_lave2D = Simu(folder_out, simulation_name)
    simu_lave2D.set_topography(raster_topo)
    simu_lave2D.set_init_mass(raster_mass)
    simu_lave2D.set_rheology(rheology_params["tau_rho"], rheology_params["K_tau"])
    simu_lave2D.set_numeric_params(tmax, dt_im)

    ## Not used in simulation, but the .cli file is needed 
    simu_lave2D.set_boundary_conditions([0, 0],  # X min and max coords for input flux
                                        [1, 2],  # Y min and max coord for input flux
                                        "W",  # Cardinal direction (Flow from East to West)
                                        [0, 0],  # Discharge hydrogram
                                        times=[0, tmax + 1],  # Corresponding times
                                        thicknesses=[0, 0],  # Thickness hydrogramm
                                        )


"""
if __name__ == "__main__":
    h_edges, v_edges = make_edges_matrices(1, 1)


if __name__ == "__main__":
    domain = ModellingDomain(xmin=0, ymin=0, nx=4, ny=3, dx=1, dy=1)
    domain.set_edges()
    res = domain.get_edges([1.5, 2.5], [0, 0], "S")
"""