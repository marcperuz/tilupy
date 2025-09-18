#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun  1 14:20:52 2021

@author: peruzzetto
"""
from __future__ import annotations
from abc import abstractmethod

import matplotlib
import matplotlib.pyplot as plt
import numpy as np

import os
import importlib
import warnings

import tilupy.notations as notations
import pytopomap.plot as plt_fn
import tilupy.plot as plt_tlp
import tilupy.raster


RAW_STATES = ["hvert", "h", "ux", "uy"]
"""Raw states at the output of a model.

Implemented states :

    - hvert : Fluid thickness taken vertically
    - h : Fluid thickness taken normal to topography
    - ux : X-component of fluid velocity
    - uy : Y-component of fluid velocity
"""

TEMPORAL_DATA_0D = ["ek", "vol"]
"""Time-varying 0D data.
   
Implemented 0D temporal data :

    - ek : kinetic energy
    - vol : Fluid volume
    
Also combine all the assembly possibilities between TEMPORAL_DATA_2D and NP/OTHER_OPERATORS, at each point xy following this format:

[TEMPORAL_DATA_2D]_[NP/OTHER_OPERATORS]_xy

For instance with h :
    - h_max_xy
    - h_mean_xy
    - h_std_xy
    - h_sum_xy
    - h_min_xy
    - h_final_xy
    - h_init_xy
    - h_int_xy
"""

TEMPORAL_DATA_1D = [""]
"""Time-varying 1D data.

Combine all the assembly possibilities between TEMPORAL_DATA_2D, NP/OTHER_OPERATORS and with an axis like this:

[TEMPORAL_DATA_2D]_[NP/OTHER_OPERATORS]_[x/y]

For instance with h :
    - h_max_x
    - h_max_y
    - h_mean_x
    - h_mean_y
    - h_std_x
    - h_std_y
    - h_sum_x
    - h_sum_y
    - h_min_x
    - h_min_y
    - h_final_x
    - h_final_y
    - h_init_x
    - h_init_y
    - h_int_x
    - h_int_y
"""

TEMPORAL_DATA_2D = ["hvert", "h", "u", "ux", "uy", "hu", "hu2"]
"""Time-varying 2D data.
   
Implemented 2D temporal data :

    - hvert : Fluid height taken vertically
    - h : Fluid height taken normal to topography
    - u : Fluid velocity
    - ux : X-component of fluid velocity
    - uy : Y-component of fluid velocity
    - hu : Volume flow rate
    - hu2 : Quadratic flow, convective term in equations
"""

STATIC_DATA_0D = []
"""Static 0D data."""

STATIC_DATA_1D = []
"""Static 1D data."""

STATIC_DATA_2D = []
"""Static 2D data.

Combine all the assembly possibilities between TEMPORAL_DATA_2D, NP/OTHER_OPERATORS and like this:

[TEMPORAL_DATA_2D]_[NP/OTHER_OPERATORS]

For instance with h :
    - h_max : Maximum value of fluid height
    - h_mean : Mean of fluid height
    - h_std : Standard deviation of fluid height
    - h_sum : Sum of fluid height
    - h_min : Minimum value of fluid height
    - h_final : Final value of fluid height
    - h_init : Initial value of fluid height
    - h_int : Integrated value of fluid height
"""

TOPO_DATA_2D = ["z", "zinit", "costh"]
"""Data related to topography.

Implemented topographic data :

    - z : Elevation value of topography
    - zinit : Initial elevation value of topography (same as z if the topography doesn't change during the flow)
    - costh : Cosine of the angle between the vertical and the normal to the relief. Factor to transform vertical height (hvert) into normal height (h).
"""

NP_OPERATORS = ["max", "mean", "std", "sum", "min"]
"""Statistical operators.
   
Implemented operators :

    - max : Maximum value
    - mean : Mean
    - std : Standard deviation
    - sum : Sum
    - min : Minimum value
"""

OTHER_OPERATORS = ["final", "init", "int"]
"""Other operators.

Implemented operators :

    - final : Final value
    - init : Initial value
    - int : Integrated value
"""

TIME_OPERATORS = ["final", "init", "int"]
"""Time-related operators.

Implemented operators :

    - final : Final value
    - init : Initial value
    - int : Integrated value
"""

for stat in NP_OPERATORS + OTHER_OPERATORS:
    for name in TEMPORAL_DATA_2D:
        STATIC_DATA_2D.append(name + "_" + stat)

for stat in NP_OPERATORS + OTHER_OPERATORS:
    for name in TEMPORAL_DATA_2D:
        TEMPORAL_DATA_0D.append(name + "_" + stat + "_xy")
        
for stat in NP_OPERATORS + OTHER_OPERATORS:
    for name in TEMPORAL_DATA_2D:
        for axis in ["x", "y"]:
            TEMPORAL_DATA_1D.append(name + "_" + stat + "_" + axis)
        

TEMPORAL_DATA = TEMPORAL_DATA_0D + TEMPORAL_DATA_1D + TEMPORAL_DATA_2D
"""Assembling all temporal data.

TEMPORAL_DATA_0D + TEMPORAL_DATA_1D + TEMPORAL_DATA_2D
"""

STATIC_DATA = STATIC_DATA_0D + STATIC_DATA_1D + STATIC_DATA_2D
"""Assembling all static data.

STATIC_DATA_0D + STATIC_DATA_1D + STATIC_DATA_2D
"""

DATA_NAMES = TEMPORAL_DATA_0D + TEMPORAL_DATA_1D + TEMPORAL_DATA_2D + STATIC_DATA_0D + STATIC_DATA_1D + STATIC_DATA_2D
"""Assembling all data.

TEMPORAL_DATA + STATIC_DATA
"""


class AbstractResults:
    """Abstract class for TemporalResults and StaticResults.

    Attributes:
    -----------
        _name : str
            Name of the property.
        _d : numpy.ndarray
            Values of the property.
        _notation : tilupy.Notation
            Instance of the class Notation.

    Parameters:
    -----------
        name : str
            Name of the property.
        d : numpy.ndarray
            Values of the property.
        notation : dict, optional
            Dictionnary of argument for creating an instance of the class Notation. 
            If None use the function get_notation. By default None.
        kwargs
    """

    def __init__(self, name: str, 
                 d: np.ndarray, 
                 notation: dict = None, 
                 **kwargs):
        self._name = name
        self._d = d
        if isinstance(notation, dict):
            self._notation = notations.Notation(**notation)
        elif notation is None:
            self._notation = notations.get_notation(name)
        else:
            self._notation = notation
        self.__dict__.update(kwargs)


    @property
    def d(self) -> np.ndarray:
        """Get data values.

        Returns
        -------
        np.ndarray
            Attribute self._d.
        """
        return self._d
        
    
    @property
    def name(self) -> str:
        """Get data name.

        Returns
        -------
        str
            Attribute self._name.
        """
        return self._name
    
    
    @property
    def notation(self) -> tilupy.notations.Notation:
        """Get data notation.

        Returns
        -------
        tilupy.Notation
            Attribute self._notation.
        """
        return self._notation


class TemporalResults(AbstractResults):
    """Abstract class for time dependent result of simulation.

    Attributes:
    -----------
        _name : str
            Name of the property.
        _d : numpy.ndarray
            Values of the property.
        _notation : tilupy.Notation
            Instance of the class Notation.
        _t : numpy.ndarray
            Time steps.
            
    Parameters:
    -----------
        name : str
            Name of the property.
        d : numpy.ndarray
            Values of the property.
        t : numpy.ndarray
            Time steps, must match the last dimension of d.
        notation : dict, optional
            Dictionnary of argument for creating an instance of the class Notation. 
            If None use the function get_notation. By default None.
    """
    def __init__(self, name: str, d: np.ndarray, t: np.ndarray, notation: dict=None):
        super().__init__(name, d, notation=notation)
        # 1d array with times, matching last dimension of self.d
        self._t = t


    def get_temporal_stat(self, stat: str) -> tilupy.read.StaticResults2D | tilupy.read.StaticResults1D:
        """Statistical analysis along temporal dimension.

        Parameters
        ----------
        stat : str
            Statistical operator to apply. Must be implemented in NP_OPERATORS or in
            OTHER_OPERATORS.

        Returns
        -------
        tilupy.StaticResults2D or tilupy.StaticResults1D
            Static result object depending on the dimensionality of the data.
        """
        if stat in NP_OPERATORS:
            dnew = getattr(np, stat)(self._d, axis=-1)
        elif stat == "final":
            dnew = self._d[..., -1]
        elif stat == "init":
            dnew = self._d[..., 0]
        elif stat == "int":
            dnew = np.trapezoid(self._d, x=self._t)

        notation = notations.add_operator(self._notation, stat, axis="t")

        if dnew.ndim == 2:
            return StaticResults2D(self._name + "_" + stat,
                                   dnew,
                                   notation=notation,
                                   x=self.x,
                                   y=self.y,
                                   z=self.z,
                                   )
        elif dnew.ndim == 1:
            return StaticResults1D(self._name + "_" + stat,
                                   dnew,
                                   notation=notation,
                                   coords=self._coords,
                                   coords_name=self._coords_name,
                                   )

    @abstractmethod
    def get_spatial_stat(self, stat, axis):
        """Abstract method for statistical analysis along spatial dimension.

        Parameters
        ----------
        stat : str
            Statistical operator to apply. Must be implemented in NP_OPERATORS or in
            OTHER_OPERATORS.
        axis : str
            Axis where to do the analysis.
        """
        pass


    @abstractmethod
    def plot(*arg, **kwargs):
        """Abstract method to plot the temporal evolution of the results."""
        pass


    @abstractmethod
    def save(*arg, **kwargs):
        """Abstract method to save the temporal results."""
        pass


    @property
    def t(self) -> np.ndarray:
        """Get times.

        Returns
        -------
        numpy.ndarray
            Attribute self._t
        """
        return self._t


class TemporalResults0D(TemporalResults):
    """
    Class for simulation results described where the data is one or multiple scalar functions of time. 
    Inherits from TemporalResults.
    
    Attributes:
    -----------
        _name : str
            Name of the property.
        _d : numpy.ndarray
            Values of the property.
        _notation : tilupy.Notation
            Instance of the class Notation.
        _t : numpy.ndarray
            Time steps.
        _scalar_names : list[str]
            List of names of the scalar fields.
    
    Parameters:
    -----------
        name : str
            Name of the property.
        d : numpy.ndarray
            Values of the property. The last dimension correspond to time. 
            It can be a one dimensionnal Nt array, or a two dimensionnal [Nd, Nt] array,
            where Nt is the legnth of t, and Nd correspond to the number of scalar values 
            of interest (e.g. X and Y coordinates of the center of mass / front).
        t : numpy.ndarray
            Time steps, must match the last dimension of d (size Nt).
        scalar_names : list[str]
            List of length Nd containing the names of the scalar fields (one
            name per row of d)
        notation : dict, optional
            Dictionnary of argument for creating an instance of the class Notation. 
            If None use the function get_notation. By default None.
    """

    def __init__(self, name: str, 
                 d: np.ndarray, 
                 t: np.ndarray, 
                 scalar_names: list[str]=None, 
                 notation: dict=None):
        super().__init__(name, d, t, notation=notation)
        self._scalar_names = scalar_names


    def plot(self, axe: matplotlib.axes._axes.Axes=None, 
             figsize: tuple[float]=None, 
             **kwargs
             ) -> matplotlib.axes._axes.Axes:
        """Plot the temporal evolution of the 0D results.

        Parameters
        ----------
        axe : matplotlib.axes._axes.Axes, optional
            Existing matplotlib window, if None create one. By default None.
        figsize : tuple[float], optional
            Size of the figure, by default None.

        Returns
        -------
        matplotlib.axes._axes.Axes
            The created plot.
        """
        if axe is None:
            fig, axe = plt.subplots(1, 1, figsize=figsize, layout="constrained")

        if isinstance(self._d, np.ndarray):
            data = self._d.T
        else:
            data = self._d
        axe.plot(self._t, data, label=self._scalar_names)
        axe.set_xlabel(notations.get_label("t"))
        axe.set_ylabel(notations.get_label(self._notation))

        return axe


    def save(self):
        """Save the temporal 0D results.

        Raises
        ------
        NotImplementedError
            Not implemented yet.
        """
        raise NotImplementedError("Saving method for TemporalResults0D not implemented yet")


    def get_spatial_stat(self, *arg, **kwargs):
        """Statistical analysis along spatial dimension for 0D results.

        Raises
        ------
        NotImplementedError
            Not implemented because irrelevant.
        """
        raise NotImplementedError("Spatial integration of Spatialresults0D is not implemented because non relevant")


    @property
    def scalar_names(self) -> list[str]:
        """Get list of names of the scalar fields.

        Returns
        -------
        list[str]
            Attribute self._scalar_names.
        """
        return self._scalar_names
    

class TemporalResults1D(TemporalResults):
    """
    Class for simulation results described by one dimension for space and one dimension for time. 
    Inherits from TemporalResults.
    
    Attributes:
    -----------
        _name : str
            Name of the property.
        _d : numpy.ndarray
            Values of the property.
        _notation : tilupy.Notation
            Instance of the class Notation.
        _t : numpy.ndarray
            Time steps.
        _coords: numpy.ndarray
            Spatial coordinates.
        _coords_name: str
            Spatial coordinates name.
    
    Parameters:
    -----------
        name : str
            Name of the property.
        d : numpy.ndarray
            Values of the property. The last dimension correspond to time. 
            It can be a one dimensionnal Nt array, or a two dimensionnal [Nd, Nt] array,
            where Nt is the legnth of t, and Nd correspond to the number of scalar values 
            of interest (e.g. X and Y coordinates of the center of mass / front).
        t : numpy.ndarray
            Time steps, must match the last dimension of d (size Nt).
        coords: numpy.ndarray
            Spatial coordinates.
        coords_name: str
            Spatial coordinates name.
        notation : dict, optional
            Dictionnary of argument for creating an instance of the class Notation. 
            If None use the function get_notation. By default None.
    """
    def __init__(self, name: str, 
                 d: np.ndarray, 
                 t: np.ndarray, 
                 coords: np.ndarray=None, 
                 coords_name: str=None, 
                 notation: dict=None):
        super().__init__(name, d, t, notation=notation)
        # x and y arrays
        self._coords = coords
        self._coords_name = coords_name


    def plot(self, 
             coords=None, 
             **kwargs
             ) -> matplotlib.axes._axes.Axes:
        """Plot the temporal evolution of the 1D results.

        Parameters
        ----------
        coords : np.ndarray, optional
            Specified coordinates, if None uses the coordinates implemented when creating the instance (self._coords). 
            By default None.

        Returns
        -------
        matplotlib.axes._axes.Axes
            The created plot.

        Raises
        ------
        TypeError
            If missing coordinates. 
        """
        if coords is None:
            coords = self._coords

        if coords is None:
            raise TypeError("coords data missing")

        xlabel = notations.get_label(self._coords_name, with_unit=True)
        if "colorbar_kwargs" not in kwargs:
            kwargs["colorbar_kwargs"] = dict()
        if "label" not in kwargs["colorbar_kwargs"]:
            clabel = notations.get_label(self._notation)
            kwargs["colorbar_kwargs"]["label"] = clabel

        axe = plt_tlp.plot_shotgather(self._coords,
                                      self._t,
                                      self._d,
                                      xlabel=xlabel,
                                      ylabel=notations.get_label("t"),
                                      **kwargs
                                      )

        return axe


    def save(self):
        """Save the temporal 1D results.

        Raises
        ------
        NotImplementedError
            Not implemented yet.
        """
        raise NotImplementedError("Saving method for TemporalResults1D not implemented yet")


    def get_spatial_stat(self, stat, **kwargs) -> tilupy.read.TemporalResults0D:
        """Statistical analysis along spatial dimension for 1D results.

        Parameters
        ----------
        stat : str
            Statistical operator to apply. Must be implemented in NP_OPERATORS or in
            OTHER_OPERATORS.
            
        Returns
        -------
        tilupy.TemporalResults0D
            Instance of TemporalResults0D.
        """
        if stat in NP_OPERATORS:
            dnew = getattr(np, stat)(self._d, axis=0)
        elif stat == "int":
            dd = self._coords[1] - self._coords[0]
            dnew = np.sum(self._d, axis=0) * dd
        notation = notations.add_operator(self._notation, stat, axis=self._coords_name)
        
        return TemporalResults0D(self._name + "_" + stat, 
                                 dnew, 
                                 self._t, 
                                 notation=notation)

    
    @property
    def coords(self) -> np.ndarray:
        """Get spatial coordinates.

        Returns
        -------
        numpy.ndarray
            Attribute self._coords
        """
        return self._coords


    @property
    def coords_name(self) -> str:
        """Get spatial coordinates name.

        Returns
        -------
        str
            Attribute self._coords
        """
        return self._coords_name


class TemporalResults2D(TemporalResults):
    """
    Class for simulation results described by a two dimensional space and one dimension for time. 
    Inherits from TemporalResults.
    
    Attributes:
    -----------
        _name : str
            Name of the property.
        _d : numpy.ndarray
            Values of the property.
        _notation : tilupy.Notation
            Instance of the class Notation.
        _t : numpy.ndarray
            Time steps.
        _x : numpy.ndarray
            X coordinate values.
        _y : numpy.ndarray
            X coordinate values.
        _z : numpy.ndarray
            Elevation values of the surface.
    
    Parameters:
    -----------
        name : str
            Name of the property.
        d : numpy.ndarray
            Values of the property. The last dimension correspond to time. 
            It can be a one dimensionnal Nt array, or a two dimensionnal [Nd, Nt] array,
            where Nt is the legnth of t, and Nd correspond to the number of scalar values 
            of interest (e.g. X and Y coordinates of the center of mass / front).
        t : numpy.ndarray
            Time steps, must match the last dimension of d (size Nt).
        x : numpy.ndarray
            X coordinate values.
        y : numpy.ndarray
            X coordinate values.
        z : numpy.ndarray
            Elevation values of the surface.
        notation : dict, optional
            Dictionnary of argument for creating an instance of the class Notation. 
            If None use the function get_notation. By default None.
    """
    def __init__(self, 
                 name: str, 
                 d: np.ndarray, 
                 t: np.ndarray, 
                 x: np.ndarray=None, 
                 y: np.ndarray=None, 
                 z: np.ndarray=None, 
                 notation: dict=None):
        super().__init__(name, d, t, notation=notation)
        # x and y arrays
        self._x = x
        self._y = y
        # topography
        self._z = z


    def plot(self,
             x: np.ndarray=None,
             y: np.ndarray=None,
             z: np.ndarray=None,
             file_name: str=None,
             folder_out: str=None,
             figsize: tuple[float]=None,
             dpi: int=None,
             fmt: str="png",
             sup_plt_fn=None,
             sup_plt_fn_args=None,
             **kwargs
             ) -> None:
        """Plot the temporal evolution of the 2D results using pytopomap.plot_maps.

        Parameters
        ----------
        x : numpy.ndarray, optional
            X coordinate values, if None use self._x. By default None.
        y : numpy.ndarray, optional
            Y coordinate values, if None use self._y. By default None.
        z : numpy.ndarray, optional
            Elevation values, if None use self._z. By default None.
        file_name : str, optional
            Base name for the output image files, by default None.
        folder_out : str, optional
            Path to the output folder. If not provides, figures are not saved. By default None.
        figsize : tuple[float], optional
            Size of the figure, by default None.
        dpi : int, optional
            Resolution for saved figures. Only used if "folder_out" is set. By default None.
        fmt : str, optional
            File format for saving figures, by default "png".
        sup_plt_fn : callable, optional
            A custom function to apply additional plotting on the axes, by default None.
        sup_plt_fn_args : dict, optional
            Arguments to pass to "sup_plt_fn", by default None.

        Returns
        -------
        None

        Raises
        ------
        TypeError
            If no value for x, y.
        """
        if file_name is None:
            file_name = self._name

        if x is None:
            x = self._x
        if y is None:
            y = self._y
        if z is None:
            z = self._z

        if x is None or y is None:
            raise TypeError("x, y or z data missing")

        if z is None:
            warnings.warn("No topography given.")

        if "colorbar_kwargs" not in kwargs:
            kwargs["colorbar_kwargs"] = dict()
        if "label" not in kwargs["colorbar_kwargs"]:
            clabel = notations.get_label(self._notation)
            kwargs["colorbar_kwargs"]["label"] = clabel

        plt_fn.plot_maps(x,
                         y,
                         z,
                         self._d,
                         self._t,
                         file_name=file_name,
                         folder_out=folder_out,
                         figsize=figsize,
                         dpi=dpi,
                         fmt=fmt,
                         sup_plt_fn=sup_plt_fn,
                         sup_plt_fn_args=sup_plt_fn_args,
                         **kwargs
                         )

        return None


    def save(self,
             folder: str=None,
             file_name: str=None,
             fmt: str="asc",
             time: str | int=None,
             x: np.ndarray=None,
             y: np.ndarray=None,
             **kwargs
             ) -> None:
        """Save the temporal 2D results.

        Parameters
        ----------
        folder : str, optional
            Path to the output folder, if None create a folder with self._name. By default None.
        file_name : str, optional
            Base name for the output image files, if None use self._name. By default None.
        fmt : str, optional
            File format for saving result, by default "asc".
        time : str | int, optional
            Time instants to save the results. 
            If time is string, must be "initial" or "final".
            If time is int, used as index in self._t.
            If None use every instant in self._t.
            By default None.
        x : np.ndarray, optional
            X coordinate values, if None use self._x. By default None.
        y : np.ndarray, optional
            Y coordinate values, if None use self._y. By default None.

        Returns
        -------
        None
        
        Raises
        ------
        ValueError
            If no value for x, y.
        """
        if x is None:
            x = self._x
        if y is None:
            y = self._y
        if x is None or y is None:
            raise ValueError("x et y arrays must not be None")

        if file_name is None:
            file_name = self._name

        if folder is not None:
            file_name = os.path.join(folder, file_name)

        if time is not None:
            if isinstance(time, str):
                if time == "final":
                    inds = [self._d.shape[2] - 1]
                elif time == "initial":
                    inds = [0]
            else:
                inds = [np.argmin(time - np.abs(np.array(self._t) - time))]
        else:
            inds = range(len(self._t))

        for i in inds:
            file_out = file_name + "_{:04d}.".format(i) + fmt
            tilupy.raster.write_raster(x, y, self._d[:, :, i], file_out, fmt=fmt, **kwargs)


    def get_spatial_stat(self, 
                         stat: str, 
                         axis: str | int | tuple[int]=None
                         ) -> tilupy.read.TemporalResults0D | tilupy.read.TemporalResults1D:
        """Statistical analysis along spatial dimension for 2D results.

        Parameters
        ----------
        stat : str
            Statistical operator to apply. Must be implemented in NP_OPERATORS.
        axis : tuple[int]
            Axis where to do the analysis. 
            If axis is string, replace 'x' by 1, 'y' by 0 and 'xy' by (0, 1). 
            If axis is int, only use 0 or 1.
            If None use (0, 1). By default None.
            
        Returns
        -------
        tilupy.TemporalResults0D or tilupy.TemporalResults1D
            Instance of TemporalResults0D or TemporalResults1D.
        """
        if axis is None:
            axis = (0, 1)

        if isinstance(axis, str):
            axis_str = axis
            if axis == "x":
                axis = 1
            elif axis == "y":
                axis = 0
            elif axis == "xy":
                axis = (0, 1)
        else:
            if axis == 1:
                axis_str = "x"
            elif axis == 0:
                axis_str = "y"
            elif axis == (0, 1):
                axis_str = "xy"

        if stat in NP_OPERATORS:
            dnew = getattr(np, stat)(self._d, axis=axis)
        elif stat == "int":
            dnew = np.sum(self._d, axis=axis)
            if axis == 1:
                dd = self._x[1] - self._x[0]
            elif axis == 0:
                dd = self._y[1] - self._y[0]
            elif axis == (0, 1):
                dd = (self._x[1] - self._x[0]) * (self._y[1] - self._y[0])
            dnew = dnew * dd

        if axis == 1:
            # Needed to get correct orinetation as d[0, 0] is the upper corner
            # of the data, with coordinates x[0], y[-1]
            dnew = np.flip(dnew, axis=0)

        new_name = self._name + "_" + stat + "_" + axis_str
        notation = notations.add_operator(self._notation, stat, axis=axis_str)

        if axis == (0, 1):
            return TemporalResults0D(new_name, 
                                     dnew, 
                                     self._t, 
                                     notation=notation)
        else:
            if axis == 0:
                coords = self._x
                coords_name = "x"
            else:
                coords = self._x
                coords_name = "y"
            return TemporalResults1D(new_name,
                                     dnew,
                                     self._t,
                                     coords,
                                     coords_name=coords_name,
                                     notation=notation)
            
    
    @property
    def x(self) -> np.ndarray:
        """Get X coordinates.

        Returns
        -------
        numpy.ndarray
            Attribute self._x.
        """
        return self._x
    
    
    @property
    def y(self) -> np.ndarray:
        """Get Y coordinates.

        Returns
        -------
        numpy.ndarray
            Attribute self._y.
        """
        return self._y


    @property
    def z(self) -> np.ndarray:
        """Get elevations values.

        Returns
        -------
        numpy.ndarray
            Attribute self._z.
        """
        return self._z
    
    
class StaticResults(AbstractResults):
    """Abstract class for result of simulation without time dependence.

    Attributes:
    -----------
        _name : str
            Name of the property.
        _d : numpy.ndarray
            Values of the property.
        _notation : tilupy.Notation
            Instance of the class Notation.
            
    Parameters:
    -----------
        name : str
            Name of the property.
        d : numpy.ndarray
            Values of the property.
        notation : dict, optional
            Dictionnary of argument for creating an instance of the class Notation. 
            If None use the function get_notation. By default None.
    """
    def __init__(self, name: str, d: np.ndarray, notation: dict=None):
        super().__init__(name, d, notation=notation)
        # x and y arrays


    @abstractmethod
    def plot(self):
        """Abstract method to plot the results."""
        pass


    @abstractmethod
    def save(self):
        """Abstract method to save the results."""
        pass


class StaticResults0D(StaticResults):
    """
    Class for simulation results described where the data is one or multiple scalar. 
    Inherits from StaticResults.
    
    Attributes:
    -----------
        _name : str
            Name of the property.
        _d : numpy.ndarray
            Values of the property.
        _notation : tilupy.Notation
            Instance of the class Notation.
            
    Parameters:
    -----------
        name : str
            Name of the property.
        d : numpy.ndarray
            Values of the property.
        notation : dict, optional
            Dictionnary of argument for creating an instance of the class Notation. 
            If None use the function get_notation. By default None.
    """
    def __init__(self, name: str, d: np.ndarray, notation: dict=None):
        super().__init__(name, d, notation=notation)


class StaticResults1D(StaticResults):
    """
    Class for simulation results described by one dimension for space. 
    Inherits from StaticResults.
    
    Attributes:
    -----------
        _name : str
            Name of the property.
        _d : numpy.ndarray
            Values of the property.
        _notation : tilupy.Notation
            Instance of the class Notation.
        _coords: numpy.ndarray
            Spatial coordinates.
        _coords_name: str
            Spatial coordinates name.
    
    Parameters:
    -----------
        name : str
            Name of the property.
        d : numpy.ndarray
            Values of the property.
        coords: numpy.ndarray
            Spatial coordinates.
        coords_name: str
            Spatial coordinates name.
        notation : dict, optional
            Dictionnary of argument for creating an instance of the class Notation. 
            If None use the function get_notation. By default None.
    """
    def __init__(self, 
                 name: str, 
                 d: np.ndarray, 
                 coords: np.ndarray=None, 
                 coords_name: list[str]=None, 
                 notation: dict=None
                 ):
        super().__init__(name, d, notation=notation)
        # x and y arrays
        self._coords = coords
        self._coords_name = coords_name


    def plot(self, 
             axe: matplotlib.axes._axes.Axes=None, 
             figsize: tuple[float]=None, **kwargs
             ) -> matplotlib.axes._axes.Axes:
        """Plot the 1D results.

        Parameters
        ----------
        axe : matplotlib.axes._axes.Axes, optional
            Existing matplotlib window, if None create one. By default None.
        figsize : tuple[float], optional
            Size of the figure, by default None.

        Returns
        -------
        matplotlib.axes._axes.Axes
            The created plot.
        """
        if axe is None:
            fig, axe = plt.subplots(
                1, 1, figsize=figsize, layout="constrained"
            )

        if isinstance(self._d, np.ndarray):
            data = self._d.T
        else:
            data = self._d
        axe.plot(self._coords, data, label=self._name) # self._name replace self.scalar_names that wasnt implemented in this class
        axe.set_xlabel(notations.get_label(self._coords_name))
        axe.set_ylabel(notations.get_label(self._notation))

        return axe


    @property
    def coords(self) -> np.ndarray:
        """Get spatial coordinates.

        Returns
        -------
        numpy.ndarray
            Attribute self._coords
        """
        return self._coords


    @property
    def coords_name(self) -> str:
        """Get spatial coordinates name.

        Returns
        -------
        str
            Attribute self._coords
        """
        return self._coords_name


class StaticResults2D(StaticResults):
    """
    Class for simulation results described by a two dimensional space result. 
    Inherits from StaticResults.
    
    Attributes:
    -----------
        _name : str
            Name of the property.
        _d : numpy.ndarray
            Values of the property.
        _notation : tilupy.Notation
            Instance of the class Notation.
        _x : numpy.ndarray
            X coordinate values.
        _y : numpy.ndarray
            X coordinate values.
        _z : numpy.ndarray
            Elevation values of the surface.
    
    Parameters:
    -----------
        name : str
            Name of the property.
        d : numpy.ndarray
            Values of the property.
        x : numpy.ndarray
            X coordinate values.
        y : numpy.ndarray
            X coordinate values.
        z : numpy.ndarray
            Elevation values of the surface.
        notation : dict, optional
            Dictionnary of argument for creating an instance of the class Notation. 
            If None use the function get_notation. By default None.
    """
    def __init__(self, 
                 name: str, 
                 d: np.ndarray, 
                 x: np.ndarray=None, 
                 y: np.ndarray=None, 
                 z: np.ndarray=None, 
                 notation: dict=None
                 ):
        super().__init__(name, d, notation=notation)
        # x and y arrays
        self._x = x
        self._y = y
        # topography
        self._z = z

    def plot(self,
             axe=None,
             figsize=None,
             x=None,
             y=None,
             z=None,
             sup_plt_fn=None,
             sup_plt_fn_args=None,
             **kwargs
             ) -> matplotlib.axes._axes.Axes:
        """Plot the 2D results using pytopomap.plot_data_on_topo.

        Parameters
        ----------
        axe : matplotlib.axes._axes.Axes, optional
            Existing matplotlib window, if None create one. By default None
        figsize : tuple[float], optional
            Size of the figure, by default None
        x : numpy.ndarray, optional
            X coordinate values, if None use self._x. By default None.
        y : numpy.ndarray, optional
            Y coordinate values, if None use self._y. By default None.
        z : numpy.ndarray, optional
            Elevation values, if None use self._z. By default None.
        sup_plt_fn : callable, optional
            A custom function to apply additional plotting on the axes, by default None.
        sup_plt_fn_args : dict, optional
            Arguments to pass to "sup_plt_fn", by default None.
        kwargs
            Additional arguments to pass to "pytopomap.plot_data_on_topo".
            
        Returns
        -------
        matplotlib.axes._axes.Axes
            The created plot.

        Raises
        ------
        TypeError
            If no value for x, y.
        """
        if axe is None:
            fig, axe = plt.subplots(1, 1, figsize=figsize, layout="constrained")

        if x is None:
            x = self._x
        if y is None:
            y = self._y
        if z is None:
            z = self._z

        if x is None or y is None or z is None:
            raise TypeError("x, y or z data missing")

        if "colorbar_kwargs" not in kwargs:
            kwargs["colorbar_kwargs"] = dict()
        if "label" not in kwargs["colorbar_kwargs"]:
            clabel = notations.get_label(self._notation)
            kwargs["colorbar_kwargs"]["label"] = clabel

        axe = plt_fn.plot_data_on_topo(x, 
                                       y, 
                                       z, 
                                       self._d, 
                                       axe=axe, 
                                       figsize=figsize, 
                                       **kwargs)
        if sup_plt_fn is not None:
            if sup_plt_fn_args is None:
                sup_plt_fn_args = dict()
            sup_plt_fn(axe, **sup_plt_fn_args)

        return axe


    def save(self,
             folder: str=None,
             file_name: str=None,
             fmt: str="txt",
             x: np.ndarray=None,
             y: np.ndarray=None,
             **kwargs
             ) -> None:
        """Save the 2D results.

        Parameters
        ----------
        folder : str, optional
            Path to the output folder, if None create a folder with self._name. By default None.
        file_name : str, optional
            Base name for the output image files, if None use self._name. By default None.
        fmt : str, optional
            File format for saving result, by default "txt".
        x : np.ndarray, optional
            X coordinate values, if None use self._x. By default None.
        y : np.ndarray, optional
            Y coordinate values, if None use self._y. By default None.

        Raises
        ------
        ValueError
            If no value for x, y.
        """
        if x is None:
            x = self._x
        if y is None:
            y = self._y

        if x is None or y is None:
            raise ValueError("x et y arrays must not be None")

        if file_name is None:
            file_name = self._name + "." + fmt

        if folder is not None:
            file_name = os.path.join(folder, file_name)

        tilupy.raster.write_raster(x, y, self._d, file_name, fmt=fmt, **kwargs)


    def get_spatial_stat(self, 
                         stat: str, 
                         axis=None
                         ) -> tilupy.read.StaticResults0D | tilupy.read.StaticResults1D:
        """Statistical analysis along spatial dimension for 2D results.

        Parameters
        ----------
        stat : str
            Statistical operator to apply. Must be implemented in NP_OPERATORS.
        axis : tuple[int]
            Axis where to do the analysis. 
            If axis is string, replace 'x' by 1, 'y' by 0 and 'xy' by (0, 1). 
            If axis is int, only use 0 or 1.
            If None use (0, 1). By default None.

        Returns
        -------
        tilupy.StaticResults0D or tilupy.StaticResults1D
            Instance of StaticResults0D or StaticResults1D.

        """
        if axis is None:
            axis = (0, 1)

        if isinstance(axis, str):
            axis_str = axis
            if axis == "x":
                axis = 1
            elif axis == "y":
                axis = 0
            elif axis == "xy":
                axis = (0, 1)
        else:
            if axis == 1:
                axis_str = "x"
            elif axis == 0:
                axis_str = "y"
            elif axis == (0, 1):
                axis_str = "xy"

        if stat in NP_OPERATORS:
            dnew = getattr(np, stat)(self._d, axis=axis)
        elif stat == "int":
            dnew = np.sum(self._d, axis=axis)
            if axis == 1:
                dd = self._x[1] - self._x[0]
            elif axis == 0:
                dd = self._y[1] - self._y[0]
            elif axis == (0, 1):
                dd = (self._x[1] - self._x[0]) * (self._y[1] - self._y[0])
            dnew = dnew * dd

        if axis == 1:
            # Needed to get correct orinetation as d[0, 0] is the upper corner
            # of the data, with coordinates x[0], y[-1]
            dnew = np.flip(dnew, axis=0)

        new_name = self._name + "_" + stat + "_" + axis_str
        notation = notations.add_operator(self._notation, stat, axis=axis_str)

        if axis == (0, 1):
            return StaticResults0D(new_name, 
                                   dnew, 
                                   notation=notation)
        else:
            if axis == 0:
                coords = self._x
                coords_name = "x"
            else:
                coords = self._y
                coords_name = "y"
            return StaticResults1D(new_name,
                                   dnew,
                                   coords,
                                   coords_name=coords_name,
                                   notation=notation,)


    @property
    def x(self) -> np.ndarray:
        """Get X coordinates.

        Returns
        -------
        numpy.ndarray
            Attribute self._x.
        """
        return self._x
    
    
    @property
    def y(self) -> np.ndarray:
        """Get Y coordinates.

        Returns
        -------
        numpy.ndarray
            Attribute self._y.
        """
        return self._y


    @property
    def z(self) -> np.ndarray:
        """Get elevations values.

        Returns
        -------
        numpy.ndarray
            Attribute self._z.
        """
        return self._z
    
    
class Results:
    """Results of thin-layer model simulation

    This class is the parent class for all simulation results, whatever the
    kind of input data. Methods and functions for processing results are given
    here. Reading results from code specific outputs is done in inhereited
    classes.
    """

    def __init__(self, *args, **kwargs):
        """
        Call init function used for code in inherited classes.

        It is not recommended

        Parameters
        ----------
        code : string
            Name of code from which results must be read.
        """
        self._h_max = None
        self._h = None
        self._costh = None
        self._zinit = None
        self.folder_output = None
        self.tim = None

    @property
    def zinit(self):
        """Get initial topography"""
        return self._zinit

    @property
    def z(self):
        """Alias for zinit"""
        return self.zinit

    @property
    def h(self):
        """Get thickness"""
        if self._h is None:
            self._h = self.get_output("h").d
        return self._h

    @property
    def h_max(self):
        """Get maximum thickness"""
        if self._h_max is None:
            self._h_max = self.get_output("h_max").d
        return self._h_max

    def get_costh(self):
        """Get cos(slope) of topography"""
        [Fx, Fy] = np.gradient(self.zinit, np.flip(self.y), self.x)
        costh = 1 / np.sqrt(1 + Fx**2 + Fy**2)
        return costh

    @property
    def costh(self):
        """Compute or get cos(slope) of topography"""
        if self._costh is None:
            self._costh = self.get_costh()
        return self._costh

    def get_output(self, output_name, from_file=True, **kwargs):
        # Specific case of center of mass
        if output_name == "centermass":
            return self.get_center_of_mass(**kwargs)

        strs = output_name.split("_")
        n_strs = len(strs)

        # get topography
        if output_name in TOPO_DATA_2D:
            res = StaticResults2D(
                output_name,
                getattr(self, output_name),
                x=self.x,
                y=self.y,
                z=self.z,
                notation=None,
            )
            return res

        # If no operator is called, call directly _get_output
        if n_strs == 1:
            res = self._get_output(output_name, **kwargs)
            return res

        # Otherwise, get name, operator and axis (optional)
        name = strs[0]
        operator = strs[1]
        if n_strs == 3:
            axis = strs[2]
        else:
            axis = None

        res = None
        # If processed output is read directly from file, call the child method
        # _read_from_file.
        if from_file:
            res = self._read_from_file(name, operator, axis=axis, **kwargs)
            # res is None in case of function failure

        # If no results could be read from file, output must be
        # processed by tilupy
        if res is None:
            # Get output from name
            res = self._get_output(name, x=self.x, y=self.y, **kwargs)
            if axis is None:
                # If no axis is given, the operator operates over time by
                # default
                res = res.get_temporal_stat(operator)
            else:
                if axis == "t":
                    res = res.get_temporal_stat(operator)
                else:
                    res = res.get_spatial_stat(operator, axis=axis)

        return res

    def _get_output(self):
        raise NotImplementedError

    def _read_from_file(self):
        raise NotImplementedError

    def get_center_of_mass(self, h_thresh=None):
        """
        Compute center of mass coordinates depending on time
        :param h_thresh: DESCRIPTION, defaults to None
        :type h_thresh: TYPE, optional
        :return: DESCRIPTION
        :rtype: TYPE

        """
        dx = self.x[1] - self.x[0]
        dy = self.y[1] - self.y[0]
        # Make meshgrid
        X, Y = np.meshgrid(self.x, np.flip(self.y))

        # Weights for coordinates average (volume in cell / total volume)
        h2 = self.h.copy()
        if h_thresh is not None:
            h2[h2 < h_thresh] = 0
        w = h2 / self.costh[:, :, np.newaxis] * dx * dy
        vol = np.nansum(w, axis=(0, 1))
        w = w / vol[np.newaxis, np.newaxis, :]
        # Compute center of mass coordinates
        nt = h2.shape[2]
        coord = np.zeros((3, nt))
        tmp = X[:, :, np.newaxis] * w
        coord[0, :] = np.nansum(tmp, axis=(0, 1))
        tmp = Y[:, :, np.newaxis] * w
        coord[1, :] = np.nansum(tmp, axis=(0, 1))
        tmp = self.zinit[:, :, np.newaxis] * w
        coord[2, :] = np.nansum(tmp, axis=(0, 1))
        # Make TemporalResults
        res = TemporalResults0D(
            "centermass",
            coord,
            self.tim,
            scalar_names=["X", "Y", "z"],
            notation=None,
        )
        return res

    def get_volume(self, h_thresh=None):
        dx = self.x[1] - self.x[0]
        dy = self.y[1] - self.y[0]
        h2 = self.h.copy()
        if h_thresh is not None:
            h2[h2 < h_thresh] = 0
        w = h2 / self.costh[:, :, np.newaxis] * dx * dy
        vol = np.nansum(w, axis=(0, 1))
        res = TemporalResults0D(
            "volume",
            vol,
            self.tim,
            notation=None,
        )
        return res

    def plot(
        self,
        name,
        save=True,
        folder_out=None,
        dpi=150,
        fmt="png",
        file_suffix=None,
        file_prefix=None,
        h_thresh=None,
        from_file=True,
        display_plot=True,
        **kwargs
    ):
        """


        Parameters
        ----------
        name : TYPE
            DESCRIPTION.
        save : TYPE, optional
            DESCRIPTION. The default is True.
        folder_out : TYPE, optional
            DESCRIPTION. The default is None.
        dpi : TYPE, optional
            DESCRIPTION. The default is 150.
        fmt : TYPE, optional
            DESCRIPTION. The default is "png".
        h_thresh : TYPE, optional
            DESCRIPTION. The default is None.
        from_file : TYPE, optional
            DESCRIPTION. The default is False.
        display_plot : TYPE, optional
            DESCRIPTION. The default is True.
        **kwargs : TYPE
            DESCRIPTION.

        Returns
        -------
        axe : TYPE
            DESCRIPTION.
        fig : TYPE
            DESCRIPTION.

        """

        if not display_plot:
            backend = plt.get_backend()
            plt.close("all")
            plt.switch_backend("Agg")

        if name in ["z", "zinit", "z_init"]:
            topo_kwargs = dict()
            if "topo_kwargs" in kwargs:
                topo_kwargs = kwargs["topo_kwargs"]
            axe = plt_fn.plot_topo(self.z, self.x, self.y, **topo_kwargs)
            return axe

        data = self.get_output(name, from_file=from_file, h_thresh=h_thresh)

        # if name in TEMPORAL_DATA_2D + STATIC_DATA_2D:
        #     if "colorbar_kwargs" not in kwargs:
        #         kwargs["colorbar_kwargs"] = dict()
        #     if "label" not in kwargs["colorbar_kwargs"]:
        #         kwargs["colorbar_kwargs"]["label"] = notations.get_label(
        #             self.notation
        #         )

        # if "x" not in kwargs:
        #     kwargs["x"] = self.x
        # if "y" not in kwargs:
        #     kwargs["y"] = self.y
        # if "z" not in kwargs:
        #     kwargs["z"] = self.zinit

        if save:
            if folder_out is None:
                assert (
                    self.folder_output is not None
                ), "folder_output attribute must be set"
                folder_out = os.path.join(self.folder_output, "plots")
            os.makedirs(folder_out, exist_ok=True)

        if folder_out is not None and isinstance(data, TemporalResults2D):
            # If data is TemporalResults2D then saving is managed directly
            # by the associated plot method
            kwargs["folder_out"] = folder_out
            kwargs["dpi"] = dpi
            kwargs["fmt"] = fmt
            # kwargs["file_suffix"] = file_prefix
            # kwargs["file_prefix"] = file_prefix

        axe = data.plot(**kwargs)

        if folder_out is not None and not isinstance(data, TemporalResults2D):
            file_name = name
            if file_suffix is not None:
                file_name = file_name + "_" + file_suffix
            if file_prefix is not None:
                file_name = file_prefix + "_" + file_name
            file_out = os.path.join(folder_out, file_name + "." + fmt)
            # axe.figure.tight_layout(pad=0.1)
            axe.figure.savefig(
                file_out, dpi=dpi, bbox_inches="tight", pad_inches=0.05
            )

        if not display_plot:
            plt.close("all")
            plt.switch_backend(backend)

        return axe

    def save(
        self,
        name,
        folder_out=None,
        file_name=None,
        fmt="txt",
        from_file=True,
        **kwargs
    ):
        if folder_out is None:
            assert (
                self.folder_output is not None
            ), "folder_output attribute must be set"
            folder_out = os.path.join(self.folder_output, "processed")
            os.makedirs(folder_out, exist_ok=True)

        if name in DATA_NAMES:
            data = self.get_output(name, from_file=from_file)
            if data.d.ndim > 1:
                if "x" not in kwargs:
                    kwargs["x"] = self.x
                if "y" not in kwargs:
                    kwargs["y"] = self.y

            data.save(
                folder=folder_out, file_name=file_name, fmt=fmt, **kwargs
            )

        elif name in TOPO_DATA_2D:
            if file_name is None:
                file_name = name
            file_out = os.path.join(folder_out, file_name)
            tilupy.raster.write_raster(
                self.x,
                self.y,
                getattr(self, name),
                file_out,
                fmt=fmt,
                **kwargs
            )


def get_results(code, **kwargs):
    """
    Get simulation results for a given code. This function calls the
    appropriate module.

    Parameters
    ----------
    code : TYPE
        DESCRIPTION.
    **kwargs : TYPE
        DESCRIPTION.

    Returns
    -------
    None.

    """
    module = importlib.import_module("tilupy.models." + code + ".read")
    return module.Results(**kwargs)


def use_thickness_threshold(simu, array, h_thresh):
    thickness = simu.get_output("h")
    array[thickness.d < h_thresh] = 0
    return array
