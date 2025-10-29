#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os

import numpy as np
import tilupy.read

from tilupy import notations

# Dictionnary with results names lookup table, to match code output names
LOOKUP_NAMES = dict(h="rho",
                    hvert="rho",
                    ux="u",
                    uy="ut",
                    u="unorm",
                    hu="momentum",
                    ek="ek",
                    vol="vol",
                    ep="ep",
                    etot="etot",
                    )
"""Dictionary of correspondance, to match code output names."""

# Classify results
AVAILABLE_OUTPUT = ["h", "hvert", "u", "hu", "hu2", "ux", "uy"]
"""All output available for a shaltop simulation.

Implemented states:

    - h : Flow thickness (normal to the surface)
    - hvert : True vertical flow thickness
    - ux : Velocity component in X direction
    - uy : Velocity component in Y direction

Output computed from other output:

    - u : Norm of the velocity (from ux and uy)
    - hu : Momentum flux (from h and u)
    - hu2 : Convective momentum flux (from h and u)
"""

READ_FROM_FILE_OUTPUT = ["h_max", "u_max", "hu_max", "hu2_int"]
"""Output read from the file.

Read output:

    - h_max : Maximum value of h integrating all time steps
    - u_max : Maximum value of u integrating all time steps
    - hu_max : Maximum value of hu integrating all time steps
    - hu2_int : Integral of values of hu2 for each time steps
"""

FORCES_OUTPUT = []
"""Force-related output variables, including components in x, y, and additional forces.
   
   Generated from:
   
        - faccx, faccy : acceleration force
        - fcurvx, fcurvy : curvature force
        - ffricx, ffricy : friction force
        - fgravx, fgravy : gravitational force
        - finertx, finerty : inertial force
        - fpressionx, fpressiony : pressure force
        
   Additional variables:
    
        - shearx, sheary, shearz : shear stresses
        - normalx, normaly, normalz : normal stresses
        - pbottom : basal pressure
        - pcorrdt : pressure correction (time step)
        - pcorrdiv : pressure correction (divergence)
"""
 
for force in ["facc", "fcurv", "ffric", "fgrav", "finert", "fpression"]:
    FORCES_OUTPUT.append(force + "x")
    FORCES_OUTPUT.append(force + "y")

FORCES_OUTPUT += ["shearx",
                  "sheary",
                  "shearz",
                  "normalx",
                  "normaly",
                  "normalz",
                  "pbottom",
                  "pcorrdt",
                  "pcorrdiv"]

INTEGRATED_OUTPUT = ["ek", "ep", "etot"]
"""Integrated energy quantities.

Implemented energy:

    - ek : kinetic energy
    - ep : potential energy
    - etot : total energy
"""


def read_params(file: str) -> dict:
    """Read simulation parameters from file.

    Parameters
    ----------
    file : str
        Path to the parameters file.

    Returns
    -------
    params : dict
        Dictionnary with parameters
    """
    if file is None:
        params = None
    else:
        params = dict()
        with open(file, "r") as f:
            lines = filter(None, (line.rstrip() for line in f))
            for line in lines:
                (key, val) = line.split(" ")
                try:
                    params[key] = float(val)
                except ValueError:
                    params[key] = val.rstrip()
        params["nx"] = int(params["nx"])
        params["ny"] = int(params["ny"])

    return params


def read_file_bin(file: str, 
                  nx: int, 
                  ny: int
                  ) -> np.ndarray:
    """Read shaltop .bin result file.

    Parameters
    ----------
    file : str
        Path to the .bin file.
    nx : int
        Number of cells in the X direction.
    ny : int
        Number of cells in the Y direction.

    Returns
    -------
    numpy.ndarray
        Values in the .bin file.
    """
    data = np.fromfile(file, dtype=np.float32)
    nbim = int(np.size(data) / nx / ny)
    data = np.reshape(data, (nx, ny, nbim), order="F")
    data = np.transpose(np.flip(data, axis=1), (1, 0, 2))
    return data


def read_file_init(file: str, nx: int, ny: int) -> np.ndarray:
    """Read shaltop initial .d files.

    Parameters
    ----------
    file : str
        Path to the .d file.
    nx : int
        Number of cells in the X direction.
    ny : int
        Number of cells in the Y direction.

    Returns
    -------
    numpy.ndarray
        Values in the .d file.
    """
    data = np.loadtxt(file)
    data = np.reshape(data, (nx, ny), order="F")
    data = np.transpose(np.flip(data, axis=1), (1, 0))
    return data


class Results(tilupy.read.Results):
    """Results of shaltop simulations.

    This class is the results class for shaltop. Reading results from shaltop outputs 
    are done in this class.
    
    This class has all the global and quick attributes of the parent class. The quick 
    attributes are only computed if needed and can be deleted to clean memory.
    
    In addition to these attributes, there are those necessary for the operation of 
    reading the shaltop results.
            
    Parameters
    ----------
        folder : str
            Path to the folder containing the simulation files.
        file_params : str
            Name of the simulation parameters file.
    
    Attributes
    ----------
        _htype : str
            Always "normal".
        _tforces : list
            Times of output.
        _params : dict
            Dictionary of the simulation parameters.
    """
    def __init__(self, folder=None, file_params=None, **varargs):
        super().__init__()
        self._code = "shaltop"
        
        try:
            if "folder_base" in varargs:
                raise UserWarning("Variable name has changed: 'folder_base' -> 'folder'")
        except UserWarning as w:
            print(f"[WARNING] {w}")
        if "folder_base" in varargs :
            folder = varargs["folder_base"]

        if folder is None:
            folder = os.getcwd()
        self._folder = folder 
            
        if file_params is None:
            file_params = "params.txt"
        if "." not in file_params:
            file_params = file_params + ".txt"
        file_params = os.path.join(folder, file_params)
        self._params = read_params(file_params)
        
        # Folder where results are stored
        if "folder_output" not in self._params:
            self._folder_output = os.path.join(self._folder, 
                                               "data2")
        else:
            self._folder_output = os.path.join(self._folder, 
                                               self._params["folder_output"])

        self._x, self._y = self.get_axes(**self._params)
        self._nx, self._ny = len(self._x), len(self._y)
        self._dx = self._x[1] - self._x[0]
        self._dy = self._y[1] - self._y[0]
        self._zinit = self.get_zinit()

        self._htype = "normal"

        # Get time of outputs
        self._tim = np.loadtxt(os.path.join(self._folder_output, "time_im.d"))
        file_tforces = os.path.join(self._folder_output, "time_forces.d")
        if os.path.isfile(file_tforces):
            self._tforces = np.loadtxt(os.path.join(self._folder_output, 
                                                   "time_forces.d"))
        else:
            self._tforces = []


    def get_zinit(self, zinit=None) -> np.ndarray:
        """Get zinit, the initial topography.

        Returns
        -------
        numpy.ndarray
            The initial topography.
        """
        path_zinit = os.path.join(self._folder_output, "z.bin")
        if not os.path.isfile(path_zinit) and "file_z_init" in self._params:
            path_zinit = os.path.join(self._folder, self._params["file_z_init"])
            zinit = read_file_init(path_zinit, self._nx, self._ny)
        else:
            zinit = np.squeeze(read_file_bin(path_zinit, self._nx, self._ny))
        return zinit


    def get_axes(self, **varargs) -> tuple[np.ndarray, np.ndarray]:
        """Get X and Y axes.

        varargs : dict
            All parameters needed to compute the axes :
            :data:`x0`, :data:`y0`, :data:`nx`, :data:`ny`, :data:`per`, 
            :data:`pery` and :data:`coord_pos`.
        
        Returns
        -------
        tuple[numpy.ndarray, numpy.ndarray]
            Values of X and Y axes.
        """
        if "x0" in varargs:
            x0 = varargs["x0"]
        else:
            x0 = 0
        if "y0" in varargs:
            y0 = varargs["y0"]
        else:
            y0 = 0

        nx = varargs["nx"]
        ny = varargs["ny"]
        dx = varargs["per"] / nx
        dy = varargs["pery"] / ny
        x = dx * np.arange(nx) + dx / 2
        y = dy * np.arange(ny) + dy / 2

        try:
            coord_pos = varargs["coord_pos"]
        except KeyError:
            coord_pos = "bottom_left"

        if coord_pos == "bottom_left":
            x = x + x0
            y = y + y0
        elif coord_pos == "upper_right":
            x = x + x0 - x[-1]
            y = y + y0 - y[-1]
        elif coord_pos == "upper_left":
            x = x + x0
            y = y + y0 - y[-1]
        elif coord_pos == "lower_right":
            x = x + x0 - x[-1]
            y = y + y0

        return x, y


    def get_u(self) -> np.ndarray:
        """Compute velocity norm from results.

        Returns
        -------
        numpy.ndarray
            Values of flow velocity (norm).
        """
        file = os.path.join(self._folder_output, "u" + ".bin")
        u = read_file_bin(file, self._nx, self._ny)
        file = os.path.join(self._folder_output, "ut" + ".bin")
        ut = read_file_bin(file, self._nx, self._ny)

        if self._costh is None:
            self._costh = self.compute_costh()
        # print(np.shape(self._zinit))
        # print(self._zinit)
        # print(np.shape(self._y))
        # print(np.shape(self._x))

        [Fx, Fy] = np.gradient(self._zinit, np.flip(self._y), self._x)
        u = u * self._costh[:, :, np.newaxis]
        ut = ut * self._costh[:, :, np.newaxis]
        d = np.sqrt(u**2 + ut**2 + (Fx[:, :, np.newaxis] * u + Fy[:, :, np.newaxis] * ut) ** 2)
        
        return d


    def _read_from_file(self, 
                        name: str, 
                        operator: str, 
                        axis: str=None, 
                        **kwargs
                        ) -> tilupy.read.StaticResults2D | tilupy.read.TemporalResults0D:
        """Read output from specific files.
        
        Can access to data in :data:`READ_FROM_FILE_OUTPUT`.

        Parameters
        ----------
        name : str
            Wanted output. Can access to: "u", "hu", "h" and "hu2".
        operator : str
            Wanted operator. Can be "max" for every output except "hu2" or "int" only for "hu2".
        axis : str, optional
            Optional axis. Can be "t". By default None.

        Returns
        -------
        tilupy.read.StaticResults2D | tilupy.read.TemporalResults0D
            Wanted output.
        """
        res = None

        if name in ["u", "hu", "h"]:
            if operator in ["max"] and axis in [None, "t"]:
                file = os.path.join(self._folder_output, 
                                    LOOKUP_NAMES[name] + operator + ".bin")
                d = np.squeeze(read_file_bin(file, self._nx, self._ny))
                res = tilupy.read.StaticResults2D("_".join([name, operator]), 
                                                  d, 
                                                  x=self._x, 
                                                  y=self._y, 
                                                  z=self._zinit)

        if (name, operator) == ("hu2", "int"):
            array = np.loadtxt(os.path.join(self._folder_output, "ek.d"))
            d = array[:, 1]
            t = array[:, 0]
            res = tilupy.read.TemporalResults0D(name, d, t)

        return res


    def _extract_output(self, 
                        name: str, 
                        **kwargs
                        )-> tilupy.read.TemporalResults2D | tilupy.read.TemporalResults0D | tilupy.read.AbstractResults:
        """Result extraction for shaltop files.

        Parameters
        ----------
        name : str
            Wanted output. Can access to variables in :data:`AVAILABLE_OUTPUT`, :data:`INTEGRATED_OUTPUT` 
            and :data:`FORCES_OUTPUT`.

        Returns
        -------
        tilupy.read.TemporalResults2D | tilupy.read.TemporalResults0D | tilupy.read.AbstractResults
            Wanted output. If no output computed, return an object of :class:`tilupy.read.AbstractResults`.
        """
        # Read thicknesses or velocity components
        d = None
        t = None
        notation = None

        if name in ["h", "ux", "uy", "hvert"]:
            if self._costh is None:
                self._costh = self.compute_costh()
                
            file = os.path.join(self._folder_output, 
                                LOOKUP_NAMES[name] + ".bin")
            
            d = read_file_bin(file, self._nx, self._ny)
            
            if name == "hvert":
                d = d / self._costh[:, :, np.newaxis]
            if name in ["ux", "uy"]:
                d = d * self.compute_costh()[:, :, np.newaxis]
            t = self._tim

        if name == "u":
            d = self.get_u()
            t = self._tim

        if name in ["hu", "hu2"]:
            fileh = os.path.join(self._folder_output, "rho.bin")
            h = read_file_bin(fileh, self._nx, self._ny)
            u = self.get_u()
            if name == "hu":
                d = h * u
            elif name == "hu2":
                d = h * u**2
            t = self._tim

        if name in INTEGRATED_OUTPUT:
            array = np.loadtxt(os.path.join(self._folder_output, 
                                            LOOKUP_NAMES[name] + ".d"))
            d = array[:, 1]
            if "density" in self._params:
                density = self._params["density"]
            else:
                density = 1
            d = d * density
            t = array[:, 0]

        if name in FORCES_OUTPUT:
            file = os.path.join(self._folder_output, name + ".bin")
            d = read_file_bin(file, self._nx, self._ny)
            t = self._tforces
            notation = notations.Notation(name,
                                          long_name=name,
                                          unit=notations.Unit(Pa=1, kg=-1, m=3),
                                          symbol=name)

        if d is None:
            file = os.path.join(self._folder_output, name)
            if os.path.isfile(file + ".bin"):
                d = read_file_bin(file + ".bin", self._nx, self._ny)
                t = self._tim
            elif os.path.isfile(file + ".d"):
                d = np.loadtxt(file + ".d")

        if ("h_thresh" in kwargs
            and kwargs["h_thresh"] is not None
            and d.ndim == 3):
            d = tilupy.read.use_thickness_threshold(self, 
                                                    d, 
                                                    kwargs["h_thresh"])

        if t is None:
            return tilupy.read.AbstractResults(name, d, notation=notation)

        else:
            if d.ndim == 3:
                return tilupy.read.TemporalResults2D(name, 
                                                     d, 
                                                     t, 
                                                     notation=notation, 
                                                     x=self._x, 
                                                     y=self._y, 
                                                     z=self._zinit)
            if d.ndim == 1:
                return tilupy.read.TemporalResults0D(name, 
                                                     d, 
                                                     t, 
                                                     notation=notation)
        return None

    # def get_temporal_output(self, name, d=None, t=None, **varargs):
    #     """
    #     Read 2D time dependent simulation results.

    #     Parameters
    #     ----------
    #     name : str
    #         Name of output.
    #     d : ndarray, optional
    #         Data to be read. If None, will be computed.
    #         The default is None.
    #     t : ndarray, optional
    #         Time of results snapshots (1D array), matching last dimension of d.
    #         If None, will be computed. The default is None.
    #     **varargs : TYPE
    #         DESCRIPTION.

    #     """
    #     # Read thicknesses or velocity components
    #     if name in STATES_OUTPUT:
    #         file = os.path.join(
    #             self.folder_output, LOOKUP_NAMES[name] + ".bin"
    #         )
    #         d = read_file_bin(file, self.nx, self.ny)
    #         if name == "hvert":
    #             d = d / self.costh[:, :, np.newaxis]
    #         t = self.tim

    #     # Raed integrated kinetic energy
    #     if name in ["hu2_int"]:
    #         array = np.loadtxt(
    #             os.path.join(self.folder_output, LOOKUP_NAMES[name] + ".d")
    #         )
    #         d = array[:, 1]
    #         t = array[:, 0]

    #     # Compute the velocity
    #     if name == "u":
    #         d = self.get_u()
    #         t = self.tim

    #     if name in ["hu", "hu2"]:
    #         fileh = os.path.join(self.folder_output, "rho.bin")
    #         h = read_file_bin(fileh, self.nx, self.ny)
    #         u = self.get_u()
    #         if name == "hu":
    #             d = h * u
    #         elif name == "hu2":
    #             d = h * u**2
    #         t = self.tim

    #     if (
    #         "h_thresh" in varargs
    #         and varargs["h_thresh"] is not None
    #         and d.ndim == 3
    #     ):
    #         d = tilupy.read.use_thickness_threshold(
    #             self, d, varargs["h_thresh"]
    #         )

    #     return tilupy.read.TemporalResults2D(name, d, t)

    # def get_static_output(self, name, d=None, from_file=True, **varargs):
    #     """
    #     Read 2D time dependent simulation results.

    #     Parameters
    #     ----------
    #     name : str
    #         Name of output.
    #     d : ndarray, optional
    #         Data to be read. Last dimension is for time.
    #         If None, will be computed.
    #         The default is None.
    #     **varargs : TYPE
    #         DESCRIPTION.

    #     """
    #     if name in tilupy.read.COMPUTED_STATIC_DATA_2D:
    #         state, stat = name.split("_")
    #         if stat in ["final", "initial"]:
    #             hh = self.get_temporal_output(state)
    #             if state == "hvert":
    #                 hh.d = hh.d / self.costh[:, :, np.newaxis]
    #             return hh.get_temporal_stat(stat)
    #         if from_file:
    #             file = os.path.join(
    #                 self.folder_output, LOOKUP_NAMES[state] + stat + ".bin"
    #             )
    #             if os.path.isfile(file):
    #                 d = np.squeeze(read_file_bin(file, self.nx, self.ny))
    #             else:
    #                 print(
    #                     file
    #                     + " was not found, "
    #                     + name
    #                     + "_"
    #                     + stat
    #                     + " computed from temporal output."
    #                 )
    #                 from_file = False
    #         if not from_file:
    #             data = self.get_temporal_output(state)
    #             if state == "hvert":
    #                 hh.d = hh.d / self.costh
    #             d = data.get_temporal_stat(stat).d
    #     else:
    #         raise (NotImplementedError())

    #     return tilupy.read.StaticResults(name, d)