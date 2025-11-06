#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations
from abc import abstractmethod
from typing import Callable

import matplotlib
import matplotlib.pyplot as plt
import numpy as np

import os
import importlib
import warnings

import pytopomap.plot as plt_fn
import tilupy.notations as notations
import tilupy.utils as utils
import tilupy.plot as plt_tlp
import tilupy.raster


ALLOWED_MODELS = ["shaltop",
                 "lave2D",
                 "saval2D"]
"""Allowed models for result reading."""


RAW_STATES = ["hvert", "h", "ux", "uy"]
"""Raw states at the output of a model.

Implemented states :

    - hvert : Fluid thickness taken vertically
    - h : Fluid thickness taken normal to topography
    - ux : X-component of fluid velocity
    - uy : Y-component of fluid velocity
"""

TEMPORAL_DATA_0D = ["ek", "volume"]
"""Time-varying 0D data.
   
Implemented 0D temporal data :

    - ek : kinetic energy
    - volume : Fluid volume
    
Also combine all the assembly possibilities between :data:`TEMPORAL_DATA_2D` and :data:`NP_OPERATORS` (or :data:`OTHER_OPERATORS`), at each point xy following this format:

`[TEMPORAL_DATA_2D]_[NP/OTHER_OPERATORS]_xy`

For instance with h :
    - h_max_xy: Maximum value of h over the entire surface for each time step.
    - h_min_xy: Minimal value of h over the entire surface for each time step.
    - h_mean_xy: Mean value of h over the entire surface for each time step.
    - h_std_xy: Standard deviation of h over the entire surface for each time step.
    - h_sum_xy: Sum of each value of h at each point of the surface for each time step.
    - h_int_xy: Integrated value of h at each point of the surface for each time step.
"""

TEMPORAL_DATA_1D = []
"""Time-varying 1D data.

Combine all the assembly possibilities between :data:`TEMPORAL_DATA_2D` and :data:`NP_OPERATORS` (or :data:`OTHER_OPERATORS`) and with an axis like this:

`[TEMPORAL_DATA_2D]_[NP/OTHER_OPERATORS]_[x/y]`

For instance with h :
    - h_max_x: For each Y coordinate, maximum value of h integrating the values of all points on the X axis (along the fixed Y axis) and integrating all time steps, giving hmax(y).
    - h_max_y: For each X coordinate, maximum value of h integrating the values of all points on the Y axis (along the fixed X axis) and integrating all time steps, giving hmax(x).
    - h_min_x: For each Y coordinate, minimum value of h integrating the values of all points on the X axis (along the fixed Y axis) and integrating all time steps, giving hmin(y).
    - h_min_y: For each X coordinate, minimum value of h integrating the values of all points on the Y axis (along the fixed X axis) and integrating all time steps, giving hmin(x).
    - h_mean_x: For each Y coordinate, mean value of h integrating the values of all points on the X axis (along the fixed Y axis) and integrating all time steps, giving hmean(y).
    - h_mean_y: For each X coordinate, mean value of h integrating the values of all points on the Y axis (along the fixed X axis) and integrating all time steps, giving hmean(x).
    - h_std_x: For each Y coordinate, standard deviation of h integrating the values of all points on the X axis (along the fixed Y axis) and integrating all time steps, giving hstd(y).
    - h_std_y: For each X coordinate, standard deviation of h integrating the values of all points on the Y axis (along the fixed X axis) and integrating all time steps, giving hstd(x).
    - h_sum_x: For each Y coordinate, sum of each value of h integrating the values of all points on the X axis (along the fixed Y axis) and integrating all time steps, giving hsum(y).
    - h_sum_y: For each X coordinate, sum of each value of h integrating the values of all points on the Y axis (along the fixed X axis) and integrating all time steps, giving hsum(x).
    - h_int_x: For each Y coordinate, integrate each value of h integrating the values of all points on the X axis (along the fixed Y axis) and integrating all time steps, giving hint(y).
    - h_int_y: For each X coordinate, integrate each value of h integrating the values of all points on the Y axis (along the fixed X axis) and integrating all time steps, giving hint(x).
"""

TEMPORAL_DATA_2D = ["hvert", "h", "u", "ux", "uy", "hu", "hu2"]
"""Time-varying 2D data.
   
Implemented 2D temporal data :

    - hvert : Fluid height taken vertically
    - h : Fluid height taken normal to topography
    - u : Fluid velocity
    - ux : X-component of fluid velocity
    - uy : Y-component of fluid velocity
    - hu : Momentum flux
    - hu2 : Convective momentum flux
"""

STATIC_DATA_0D = []
"""Static 0D data."""

STATIC_DATA_1D = []
"""Static 1D data."""

STATIC_DATA_2D = []
"""Static 2D data.

Combine all the assembly possibilities between :data:`TEMPORAL_DATA_2D` and :data:`NP_OPERATORS` (or :data:`OTHER_OPERATORS`) like this:

`[TEMPORAL_DATA_2D]_[NP/OTHER_OPERATORS]`

For instance with h :
    - h_max : Maximum value of h at each point on the map, integrating all the time steps.
    - h_min : Minimum value of h at each point on the map, integrating all the time steps.
    - h_mean : Mean value of h at each point on the map, integrating all the time steps.
    - h_std : Standard deviation of h at each point on the map, integrating all the time steps.
    - h_sum : Sum of h at each point on the map, integrating all the time steps.
    - h_final : Value of h at each point on the map, for the last time step.
    - h_init : Value of h at each point on the map, for the first time step.
    - h_int : Integrated value of h at each point on the map, integrating all the time steps.
"""

TOPO_DATA_2D = ["z", "zinit", "costh"]
"""Data related to topography.

Implemented topographic data :

    - z : Elevation value of topography
    - zinit : Initial elevation value of topography (same as z if the topography doesn't change during the flow)
    - costh : Cosine of the angle between the vertical and the normal to the relief. Factor to transform vertical height (hvert) into normal height (h).
"""

NP_OPERATORS = ["max", "min", "mean", "std", "sum"]
"""Statistical operators.
   
Implemented operators :

    - max : Maximum
    - min : Minimum
    - mean : Mean
    - std : Standard deviation
    - sum : Sum
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

for stat in NP_OPERATORS + ["int"]:
    for name in TEMPORAL_DATA_2D:
        TEMPORAL_DATA_0D.append(name + "_" + stat + "_xy")
        
for stat in NP_OPERATORS + ["int"]:
    for name in TEMPORAL_DATA_2D:
        for axis in ["x", "y"]:
            TEMPORAL_DATA_1D.append(name + "_" + stat + "_" + axis)
        

TEMPORAL_DATA = TEMPORAL_DATA_0D + TEMPORAL_DATA_1D + TEMPORAL_DATA_2D
"""Assembling all temporal data.

:data:`TEMPORAL_DATA_0D` + :data:`TEMPORAL_DATA_1D` + :data:`TEMPORAL_DATA_2D`
"""

STATIC_DATA = STATIC_DATA_0D + STATIC_DATA_1D + STATIC_DATA_2D
"""Assembling all static data.

:data:`STATIC_DATA_0D` + :data:`STATIC_DATA_1D` + :data:`STATIC_DATA_2D`
"""

DATA_NAMES = TEMPORAL_DATA_0D + TEMPORAL_DATA_1D + TEMPORAL_DATA_2D + STATIC_DATA_0D + STATIC_DATA_1D + STATIC_DATA_2D
"""Assembling all data.

:data:`TEMPORAL_DATA` + :data:`STATIC_DATA`
"""


class AbstractResults:
    """Abstract class for :class:`tilupy.read.TemporalResults` and :class:`tilupy.read.StaticResults`.

    Parameters
    ----------
        name : str
            Name of the property.
        d : numpy.ndarray
            Values of the property.
        notation : dict, optional
            Dictionnary of argument for creating an instance of the class :class:`tilupy.notations.Notation`. 
            If None use the function :func:`tilupy.notations.get_notation`. By default None.
        kwargs
        
    Attributes
    ----------
        _name : str
            Name of the property.
        _d : numpy.ndarray
            Values of the property.
        _notation : tilupy.notations.Notation
            Instance of the class :class:`tilupy.notations.Notation`.
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
            Attribute :attr:`_d`.
        """
        return self._d
        
    
    @property
    def name(self) -> str:
        """Get data name.

        Returns
        -------
        str
            Attribute :attr:`_name`.
        """
        return self._name
    
    
    @property
    def notation(self) -> tilupy.notations.Notation:
        """Get data notation.

        Returns
        -------
        tilupy.notations.Notation
            Attribute :attr:`_notation`.
        """
        return self._notation


class TemporalResults(AbstractResults):
    """Abstract class for time dependent result of simulation.

    Parameters
    ----------
        name : str
            Name of the property.
        d : numpy.ndarray
            Values of the property.
        t : numpy.ndarray
            Time steps, must match the last dimension of :data:`d`.
        notation : dict, optional
            Dictionnary of argument for creating an instance of the class :class:`tilupy.notations.Notation`. 
            If None use the function :func:`tilupy.notations.get_notation`. By default None.
    
    Attributes
    ----------
        _name : str
            Name of the property.
        _d : numpy.ndarray
            Values of the property.
        _notation : tilupy.notations.Notation
            Instance of the class :class:`tilupy.notations.Notation`.
        _t : numpy.ndarray
            Time steps.
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
            Statistical operator to apply. Must be implemented in :data:`NP_OPERATORS` or in
            :data:`OTHER_OPERATORS`.

        Returns
        -------
        tilupy.read.StaticResults2D or tilupy.read.StaticResults1D
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
            Statistical operator to apply. Must be implemented in :data:`NP_OPERATORS` or in
            :data:`OTHER_OPERATORS`.
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

    
    @abstractmethod
    def extract_from_time_step(*arg, **kwargs):
        """Abstract method to extract data from specific time steps of a TemporalResults."""
        pass

    
    @property
    def t(self) -> np.ndarray:
        """Get times.

        Returns
        -------
        numpy.ndarray
            Attribute :attr:`_t`
        """
        return self._t


class TemporalResults0D(TemporalResults):
    """
    Class for simulation results described where the data is one or multiple scalar functions of time.
    
    Parameters
    ----------
        name : str
            Name of the property.
        d : numpy.ndarray
            Values of the property. The last dimension correspond to time. 
            It can be a one dimensionnal Nt array, or a two dimensionnal [Nd, Nt] array,
            where Nt is the legnth of :data:`t`, and Nd correspond to the number of scalar values 
            of interest (e.g. X and Y coordinates of the center of mass / front).
        t : numpy.ndarray
            Time steps, must match the last dimension of :data:`d` (size Nt).
        scalar_names : list[str]
            List of length Nd containing the names of the scalar fields (one
            name per row of d)
        notation : dict, optional
            Dictionnary of argument for creating an instance of the class :class:`tilupy.notations.Notation`. 
            If None use the function :func:`tilupy.notations.get_notation`. By default None.
    
    Attributes
    ----------
        _name : str
            Name of the property.
        _d : numpy.ndarray
            Values of the property.
        _notation : tilupy.notations.Notation
            Instance of the class :class:`tilupy.notations.Notation`.
        _t : numpy.ndarray
            Time steps.
        _scalar_names : list[str]
            List of names of the scalar fields.
    """

    def __init__(self, name: str, 
                 d: np.ndarray, 
                 t: np.ndarray, 
                 scalar_names: list[str]=None, 
                 notation: dict=None):
        super().__init__(name, d, t, notation=notation)
        self._scalar_names = scalar_names


    def plot(self, 
             ax: matplotlib.axes._axes.Axes=None, 
             figsize: tuple[float]=None, 
             **kwargs
             ) -> matplotlib.axes._axes.Axes:
        """Plot the temporal evolution of the 0D results.

        Parameters
        ----------
        ax : matplotlib.axes._axes.Axes, optional
            Existing matplotlib window, if None create one. By default None.
        figsize : tuple[float], optional
            Size of the figure, by default None.

        Returns
        -------
        matplotlib.axes._axes.Axes
            The created plot.
        """
        if ax is None:
            fig, ax = plt.subplots(1, 1, figsize=figsize, layout="constrained")

        if isinstance(self._d, np.ndarray):
            data = self._d.T
        else:
            data = self._d
            
        if "color" not in kwargs and self._scalar_names is None:
            color = "black"
            kwargs["color"] = color
        
        ax.plot(self._t, data, label=self._scalar_names, **kwargs) # Remove label=self._scalar_names

        ax.grid(True, alpha=0.3)
        ax.set_xlim(left=min(self._t), right=max(self._t))
        ax.set_xlabel(notations.get_label("t"))
        ax.set_ylabel(notations.get_label(self._notation))

        return ax


    def save(self):
        """Save the temporal 0D results.

        Raises
        ------
        NotImplementedError
            Not implemented yet.
        """
        raise NotImplementedError("Saving method for :class:`tilupy.read.TemporalResults0D` not implemented yet")


    def get_spatial_stat(self, *arg, **kwargs):
        """Statistical analysis along spatial dimension for 0D results.

        Raises
        ------
        NotImplementedError
            Not implemented because irrelevant.
        """
        raise NotImplementedError("Spatial integration of :class:`tilupy.read.Spatialresults0D` is not implemented because non relevant")


    def extract_from_time_step(self,
                               time_steps: float | list[float], 
                               ) -> tilupy.read.StaticResults0D | tilupy.read.TemporalResults0D:
        """Extract data from specific time steps. 

        Parameters
        ----------
        time_steps : float | list[float]
            Value of time steps to extract data.

        Returns
        -------
        tilupy.read.StaticResults0D | tilupy.read.TemporalResults0D
            Extracted data, the type depends on the time step.
        """
        if isinstance(time_steps, float) or isinstance(time_steps, int):
            t_index = np.argmin(np.abs(self._t - time_steps))
            
            return StaticResults0D(name=self._name,
                                   d=self._d[t_index],
                                   notation=self._notation)
        
        elif isinstance(time_steps, list):
            time_steps = np.array(time_steps)
        
        if isinstance(time_steps, np.ndarray):
            if isinstance(self._t, list):
                self._t = np.array(self._t)
            t_distances = np.abs(self._t[None, :] - time_steps[:, None])
            t_indexes = np.argmin(t_distances, axis=1)
            
            return TemporalResults0D(name=self._name,
                                     d=self._d[t_indexes],
                                     t=self._t[t_indexes],
                                     notation=self._notation)
    
    
    @property
    def scalar_names(self) -> list[str]:
        """Get list of names of the scalar fields.

        Returns
        -------
        list[str]
            Attribute :attr:`_scalar_names`.
        """
        return self._scalar_names
    

class TemporalResults1D(TemporalResults):
    """
    Class for simulation results described by one dimension for space and one dimension for time. 
    
    Parameters
    ----------
        name : str
            Name of the property.
        d : numpy.ndarray
            Values of the property. The last dimension correspond to time. 
            It can be a one dimensionnal Nt array, or a two dimensionnal [Nd, Nt] array,
            where Nt is the legnth of :data:`t`, and Nd correspond to the number of scalar values 
            of interest (e.g. X and Y coordinates of the center of mass / front).
        t : numpy.ndarray
            Time steps, must match the last dimension of :data:`d` (size Nt).
        coords: numpy.ndarray
            Spatial coordinates.
        coords_name: str
            Spatial coordinates name.
        notation : dict, optional
            Dictionnary of argument for creating an instance of the class :class:`tilupy.notations.Notation`. 
            If None use the function :func:`tilupy.notations.get_notation`. By default None.
    
    Attributes
    ----------
        _name : str
            Name of the property.
        _d : numpy.ndarray
            Values of the property.
        _notation : tilupy.notations.Notation
            Instance of the class :class:`tilupy.notations.Notation`.
        _t : numpy.ndarray
            Time steps.
        _coords: numpy.ndarray
            Spatial coordinates.
        _coords_name: str
            Spatial coordinates name.
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
             coords = None,
             plot_type: str = "simple",
             figsize: tuple[float] = None,
             ax: matplotlib.axes._axes.Axes = None,
             linestyles: list[str] = None,
             cmap: str = 'viridis',
             highlighted_curve: bool = False,
             **kwargs
             ) -> matplotlib.axes._axes.Axes:
        """Plot the temporal evolution of the 1D results.

        Parameters
        ----------
        coords : numpy.ndarray, optional
            Specified coordinates, if None uses the coordinates implemented when creating the instance (:attr:`_coords`). 
            By default None.
        plot_type: str, optional
            Wanted plot :
            
                - "simple" : Every curve in the same graph
                - "multiples" : Every curve in separate graph
                - "'shotgather" : Shotgather graph
                
            By default "simple".
        ax : matplotlib.axes._axes.Axes, optional
            Existing matplotlib window, if None create one. By default None
        linestyles : list[str], optional
            Custom linestyles for each time step. If None, colors and styles are auto-assigned. 
            Used only for "simple". By default None.
        cmap : str, optional
            Color map for the ploted curves. Used only for "simple". By default "viridis".
        hightlighted_curved : bool, optional
            Option to display all time steps on each graph of the multiples and 
            highlight the curve corresponding to the time step of the subgraph. Used only for "multiples". 
            By default False.
        kwargs: dict, optional
            Additional arguments for plot functions.

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

        if ax is None and plot_type != "multiples":
            fig, ax = plt.subplots(1, 1, figsize=figsize, layout="constrained")
        
        if plot_type == "shotgather":
            xlabel = notations.get_label(self._coords_name, with_unit=True)
            
            if "colorbar_kwargs" not in kwargs:
                kwargs["colorbar_kwargs"] = dict()
            if "label" not in kwargs["colorbar_kwargs"]:
                clabel = notations.get_label(self._notation)
                kwargs["colorbar_kwargs"]["label"] = clabel

            ax = plt_tlp.plot_shotgather(self._coords,
                                          self._t,
                                          self._d,
                                          xlabel=xlabel,
                                          ylabel=notations.get_label("t"),
                                          **kwargs)
        
        if plot_type == "simple":
            if linestyles is None or len(linestyles)!=(len(self._t)):
                norm = plt.Normalize(vmin=0, vmax=len(self._t)-1)
                cmap = plt.get_cmap(cmap)
                
            for i in range(self._d.shape[1]):
                if linestyles is None or len(linestyles)!=(len(self._t)):
                    color = cmap(norm(i)) if self._t[i] != 0 else "red"
                    l_style = "-" if self._t[i] != 0 else (0, (1, 4))
                else:
                    color = "black" if self._t[i] != 0 else "red"
                    l_style = linestyles[i] if self._t[i] != 0 else (0, (1, 4))
                    
                ax.plot(self._coords, self._d[:, i], label=f"t={self._t[i]}s", color=color, linestyle=l_style, **kwargs)
            
            ax.grid(True, alpha=0.3)
            ax.set_xlim(left=min(self._coords), right=max(self._coords))
            
            ax.set_xlabel(notations.get_label(self._coords_name))
            ax.set_ylabel(notations.get_label(self._notation))
            
        if plot_type == "multiples":
            cols_nb = 3
            if len(self._t) < 3:
                cols_nb = len(self._t)
                
            row_nb = len(self._t) // 3
            if len(self._t) % 3 != 0:
                row_nb += 1
            
            fig, axes = plt.subplots(nrows=row_nb, 
                                     ncols=cols_nb, 
                                     figsize=figsize, 
                                     layout="constrained", 
                                     sharex=True, 
                                     sharey=True)
            axes = axes.flatten()
            
            for i in range(self._d.shape[1]):
                if highlighted_curve:
                    for j in range(self._d.shape[1]):
                        if i == j:
                            axes[i].plot(self._coords, self._d[:, j], color="black", linewidth=1.5, **kwargs)
                        else:
                            axes[i].plot(self._coords, self._d[:, j], color="gray", alpha=0.5, linewidth=0.5, **kwargs)
                else:
                    axes[i].plot(self._coords, self._d[:, i], color="black", **kwargs)
                
                axes[i].grid(True, alpha=0.3)
                axes[i].set_xlim(left=min(self._coords), right=max(self._coords))
                
                axes[i].set_xlabel(notations.get_label(self._coords_name))
                axes[i].set_ylabel(notations.get_label(self._notation))
                
                axes[i].set_title(f"t={self._t[i]}s", loc='left')

            for i in range(len(self._t), len(axes)):
                fig.delaxes(axes[i])
            
            return axes
            
        return ax


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
            Statistical operator to apply. Must be implemented in :data:`NP_OPERATORS` or in
            :data:`OTHER_OPERATORS`.
            
        Returns
        -------
        tilupy.read.TemporalResults0D
            Instance of :class:`tilupy.read.TemporalResults0D`.
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

    
    def extract_from_time_step(self,
                               time_steps: float | list[float], 
                               ) -> tilupy.read.StaticResults1D | tilupy.read.TemporalResults1D:
        """Extract data from specific time steps. 

        Parameters
        ----------
        time_steps : float | list[float]
            Value of time steps to extract data.

        Returns
        -------
        tilupy.read.StaticResults1D | tilupy.read.TemporalResults1D
            Extracted data, the type depends on the time step.
        """
        if isinstance(time_steps, float) or isinstance(time_steps, int):
            t_index = np.argmin(np.abs(self._t - time_steps))
            
            return StaticResults1D(name=self._name,
                                   d=self._d[:, t_index],
                                   coords=self._coords,
                                   coords_name=self._coords_name,
                                   notation=self._notation)
        
        elif isinstance(time_steps, list):
            time_steps = np.array(time_steps)
        
        if isinstance(time_steps, np.ndarray):
            if isinstance(self._t, list):
                self._t = np.array(self._t)
            t_distances = np.abs(self._t[None, :] - time_steps[:, None])
            t_indexes = np.argmin(t_distances, axis=1)
            
            return TemporalResults1D(name=self._name,
                                     d=self._d[:, t_indexes],
                                     t=self._t[t_indexes],
                                     coords=self._coords,
                                     coords_name=self._coords_name,
                                     notation=self._notation)
    
    
    @property
    def coords(self) -> np.ndarray:
        """Get spatial coordinates.

        Returns
        -------
        numpy.ndarray
            Attribute :attr:`_coords`
        """
        return self._coords


    @property
    def coords_name(self) -> str:
        """Get spatial coordinates name.

        Returns
        -------
        str
            Attribute :attr:`_coords_name`
        """
        return self._coords_name


class TemporalResults2D(TemporalResults):
    """
    Class for simulation results described by a two dimensional space and one dimension for time. 
    
    Parameters
    ----------
        name : str
            Name of the property.
        d : numpy.ndarray
            Values of the property. The last dimension correspond to time. 
            It can be a one dimensionnal Nt array, or a two dimensionnal [Nd, Nt] array,
            where Nt is the legnth of :data:`t`, and Nd correspond to the number of scalar values 
            of interest (e.g. X and Y coordinates of the center of mass / front).
        t : numpy.ndarray
            Time steps, must match the last dimension of :data:`d` (size Nt).
        x : numpy.ndarray
            X coordinate values.
        y : numpy.ndarray
            X coordinate values.
        z : numpy.ndarray
            Elevation values of the surface.
        notation : dict, optional
            Dictionnary of argument for creating an instance of the class :class:`tilupy.notation.Notation`. 
            If None use the function :func:`tilupy.notation.get_notation`. By default None.
    
    Attributes
    ----------
        _name : str
            Name of the property.
        _d : numpy.ndarray
            Values of the property.
        _notation : tilupy.notations.Notation
            Instance of the class :class:`tilupy.notations.Notation`.
        _t : numpy.ndarray
            Time steps.
        _x : numpy.ndarray
            X coordinate values.
        _y : numpy.ndarray
            X coordinate values.
        _z : numpy.ndarray
            Elevation values of the surface.
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
             x: np.ndarray = None,
             y: np.ndarray = None,
             z: np.ndarray = None,
             plot_multiples: bool = False,
             file_name: str = None,
             folder_out: str = None,
             figsize: tuple[float] = None,
             dpi: int = None,
             fmt: str = "png",
             sup_plt_fn = None,
             sup_plt_fn_args = None,
             **kwargs
             ) -> None:
        """Plot the temporal evolution of the 2D results using :func:`pytopomap.plot.plot_maps`.

        Parameters
        ----------
        x : numpy.ndarray, optional
            X coordinate values, if None use :attr:`_x`. By default None.
        y : numpy.ndarray, optional
            Y coordinate values, if None use :attr:`_y`. By default None.
        z : numpy.ndarray, optional
            Elevation values, if None use :attr:`_z`. By default None.
        file_name : str, optional
            Base name for the output image files, by default None.
        folder_out : str, optional
            Path to the output folder. If not provides, figures are not saved. By default None.
        figsize : tuple[float], optional
            Size of the figure, by default None.
        dpi : int, optional
            Resolution for saved figures. Only used if :data:`folder_out` is set. By default None.
        fmt : str, optional
            File format for saving figures, by default "png".
        sup_plt_fn : callable, optional
            A custom function to apply additional plotting on the axes, by default None.
        sup_plt_fn_args : dict, optional
            Arguments to pass to :data:`sup_plt_fn`, by default None.

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
               
        if plot_multiples:
            if "vmin" not in kwargs:
                kwargs["vmin"] = np.min(self._d)
            if "vmax" not in kwargs:
                kwargs["vmax"] = np.max(self._d)
            
            cols_nb = 3
            if len(self._t) < 3:
                cols_nb = len(self._t)
                
            row_nb = len(self._t) // 3
            if len(self._t) % 3 != 0:
                row_nb += 1
            
            fig, axes = plt.subplots(nrows=row_nb, 
                                     ncols=cols_nb, 
                                     figsize=figsize,
                                     layout="constrained",
                                     sharex=True, 
                                     sharey=True)
            axes = axes.flatten()
            
            for i in range(len(self._t)):
                plt_fn.plot_data_on_topo(x=x,
                                         y=y,
                                         z=z,
                                         data=self._d[:, :, i],
                                         axe=axes[i],
                                         plot_colorbar=False,
                                         **kwargs)

                axes[i].set_title(f"t={self._t[i]}s", loc='left')

            for i in range(len(self._t), len(axes)):
                fig.delaxes(axes[i])
            
            max_val, idx = 0, 0
            for i in range(len(self._t)):
                max_val_t = np.max(axes[i].images[1].get_array())
                if max_val_t > max_val:
                    max_val = max_val_t
                    idx = i
            mappable = axes[idx].images[1]
            fig.colorbar(mappable, ax=axes, orientation='vertical', **kwargs["colorbar_kwargs"])
            
            return axes
        
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
            Path to the output folder, if None create a folder with :attr:`_name`. By default None.
        file_name : str, optional
            Base name for the output image files, if None use :attr:`_name`. By default None.
        fmt : str, optional
            File format for saving result, by default "asc".
        time : str | int, optional
            Time instants to save the results. 
            
                - If time is string, must be "initial" or "final".
                - If time is int, used as index in :attr:`_t`.
                - If None use every instant in :attr:`_t`.
                
            By default None.
        x : np.ndarray, optional
            X coordinate values, if None use :attr:`_x`. By default None.
        y : np.ndarray, optional
            Y coordinate values, if None use :attr:`_y`. By default None.

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
            Statistical operator to apply. Must be implemented in :data:`NP_OPERATORS`.
        axis : tuple[int]
            Axis where to do the analysis:
            
                - If axis is string, replace 'x' by 1, 'y' by 0 and 'xy' by (0, 1). 
                - If axis is int, only use 0 or 1.
                - If None use (0, 1). By default None.
            
        Returns
        -------
        tilupy.read.TemporalResults0D or tilupy.read.TemporalResults1D
            Instance of :class:`tilupy.read.TemporalResults0D` or :class:`tilupy.read.TemporalResults1D`.
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
                coords = self._y
                coords_name = "y"
            return TemporalResults1D(new_name,
                                     dnew,
                                     self._t,
                                     coords,
                                     coords_name=coords_name,
                                     notation=notation)


    def extract_from_time_step(self,
                               time_steps: float | list[float], 
                               ) -> tilupy.read.StaticResults2D | tilupy.read.TemporalResults2D:
        """Extract data from specific time steps. 

        Parameters
        ----------
        time_steps : float | list[float]
            Value of time steps to extract data.

        Returns
        -------
        tilupy.read.StaticResults2D | tilupy.read.TemporalResults2D
            Extracted data, the type depends on the time step.
        """
        if isinstance(time_steps, float) or isinstance(time_steps, int):
            t_index = np.argmin(np.abs(self._t - time_steps))
            
            return StaticResults2D(name=self._name,
                                   d=self._d[:, :, t_index],
                                   x=self._x,
                                   y=self._y,
                                   z=self._z,
                                   notation=self._notation)
        
        elif isinstance(time_steps, list):
            time_steps = np.array(time_steps)

        if isinstance(time_steps, np.ndarray):
            if isinstance(self._t, list):
                self._t = np.array(self._t)
            t_distances = np.abs(self._t[None, :] - time_steps[:, None])
            t_indexes = np.argmin(t_distances, axis=1)
            
            return TemporalResults2D(name=self._name,
                                     d=self._d[:, :, t_indexes],
                                     t=self._t[t_indexes],
                                     x=self._x,
                                     y=self._y,
                                     z=self._z,
                                     notation=self._notation)


    @property
    def x(self) -> np.ndarray:
        """Get X coordinates.

        Returns
        -------
        numpy.ndarray
            Attribute :attr:`_x`.
        """
        return self._x
    
    
    @property
    def y(self) -> np.ndarray:
        """Get Y coordinates.

        Returns
        -------
        numpy.ndarray
            Attribute :attr:`_y`.
        """
        return self._y


    @property
    def z(self) -> np.ndarray:
        """Get elevations values.

        Returns
        -------
        numpy.ndarray
            Attribute :attr:`_z`.
        """
        return self._z
    
    
class StaticResults(AbstractResults):
    """Abstract class for result of simulation without time dependence.

    Parameters
    ----------
        name : str
            Name of the property.
        d : numpy.ndarray
            Values of the property.
        notation : dict, optional
            Dictionnary of argument for creating an instance of the class :class:`tilupy.notations.Notation`. 
            If None use the function :func:`tilupy.notations.get_notation`. By default None.
    
    Attributes
    ----------
        _name : str
            Name of the property.
        _d : numpy.ndarray
            Values of the property.
        _notation : tilupy.notations.Notation
            Instance of the class :class:`tilupy.notations.Notation`.        
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
    
    Parameters
    ----------
        name : str
            Name of the property.
        d : numpy.ndarray
            Values of the property.
        notation : dict, optional
            Dictionnary of argument for creating an instance of the class :class:`tilupy.notations.Notation`. 
            If None use the function :func:`tilupy.notations.get_notation`. By default None.
    
    Attributes
    ----------
        _name : str
            Name of the property.
        _d : numpy.ndarray
            Values of the property.
        _notation : tilupy.notations.Notation
            Instance of the class :class:`tilupy.notations.Notation`.        
    """
    def __init__(self, name: str, d: np.ndarray, notation: dict=None):
        super().__init__(name, d, notation=notation)


class StaticResults1D(StaticResults):
    """
    Class for simulation results described by one dimension for space. 
    
    Parameters
    ----------
        name : str
            Name of the property.
        d : numpy.ndarray
            Values of the property.
        coords: numpy.ndarray
            Spatial coordinates.
        coords_name: str
            Spatial coordinates name.
        notation : dict, optional
            Dictionnary of argument for creating an instance of the class :class:`tilupy.notations.Notation`. 
            If None use the function :func:`tilupy.notations.get_notation`. By default None.
    
    Attributes
    ----------
        _name : str
            Name of the property.
        _d : numpy.ndarray
            Values of the property.
        _notation : tilupy.notations.Notation
            Instance of the class :class:`tilupy.notations.Notation`.
        _coords: numpy.ndarray
            Spatial coordinates.
        _coords_name: str
            Spatial coordinates name.
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
             ax: matplotlib.axes._axes.Axes = None,
             **kwargs
             ) -> matplotlib.axes._axes.Axes:
        """Plot the 1D results.

        Parameters
        ----------
        ax : matplotlib.axes._axes.Axes, optional
            Existing matplotlib window, if None create one. By default None.

        Returns
        -------
        matplotlib.axes._axes.Axes
            The created plot.
        """
        if ax is None:
            fig, ax = plt.subplots(1, 1, layout="constrained")

        if isinstance(self._d, np.ndarray):
            data = self._d.T
        else:
            data = self._d
        
        if "color" not in kwargs:
            color = "black"
            kwargs["color"] = color
        
        ax.plot(self._coords, data, **kwargs) 
        
        ax.grid(True, alpha=0.3)
        ax.set_xlim(left=min(self._coords), right=max(self._coords))
        ax.set_xlabel(notations.get_label(self._coords_name))
        ax.set_ylabel(notations.get_label(self._notation))

        return ax


    @property
    def coords(self) -> np.ndarray:
        """Get spatial coordinates.

        Returns
        -------
        numpy.ndarray
            Attribute :attr:`_coords`
        """
        return self._coords


    @property
    def coords_name(self) -> str:
        """Get spatial coordinates name.

        Returns
        -------
        str
            Attribute :attr:`_coords_name`
        """
        return self._coords_name


class StaticResults2D(StaticResults):
    """
    Class for simulation results described by a two dimensional space result. 
    Inherits from StaticResults.
    
    Parameters
    ----------
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
            Dictionnary of argument for creating an instance of the class :class:`tilupy.notations.Notation`. 
            If None use the function :func:`tilupy.notations.get_notation`. By default None.
    
    Attributes
    ----------
        _name : str
            Name of the property.
        _d : numpy.ndarray
            Values of the property.
        _notation : tilupy.notations.Notation
            Instance of the class :class:`tilupy.notations.Notation`.
        _x : numpy.ndarray
            X coordinate values.
        _y : numpy.ndarray
            X coordinate values.
        _z : numpy.ndarray
            Elevation values of the surface.
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
             figsize: tuple[float] = None,
             x: np.ndarray = None,
             y: np.ndarray = None,
             z: np.ndarray = None,
             sup_plt_fn: Callable = None,
             sup_plt_fn_args: dict = None,
             ax: matplotlib.axes._axes.Axes = None,
             **kwargs
             ) -> matplotlib.axes._axes.Axes:
        """Plot the 2D results using :func:`pytopomap.plot.plot_data_on_topo`.

        Parameters
        ----------
        figsize : tuple[float], optional
            Size of the figure, by default None
        x : numpy.ndarray, optional
            X coordinate values, if None use :attr:`_x`. By default None.
        y : numpy.ndarray, optional
            Y coordinate values, if None use :attr:`_y`. By default None.
        z : numpy.ndarray, optional
            Elevation values, if None use :attr:`_z`. By default None.
        sup_plt_fn : callable, optional
            A custom function to apply additional plotting on the axes, by default None.
        sup_plt_fn_args : dict, optional
            Arguments to pass to :data:`sup_plt_fn`, by default None.
        ax : matplotlib.axes._axes.Axes, optional
            Existing matplotlib window, if None create one. By default None
        kwargs
            Additional arguments to pass to :func:`pytopomap.plot.plot_data_on_topo`.
            
        Returns
        -------
        matplotlib.axes._axes.Axes
            The created plot.

        Raises
        ------
        TypeError
            If no value for x, y.
        """
        if ax is None:
            fig, ax = plt.subplots(1, 1, figsize=figsize, layout="constrained")

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

        ax = plt_fn.plot_data_on_topo(x, 
                                      y, 
                                      z, 
                                      self._d, 
                                      axe=ax, 
                                      figsize=figsize, 
                                      **kwargs)
        if sup_plt_fn is not None:
            if sup_plt_fn_args is None:
                sup_plt_fn_args = dict()
            sup_plt_fn(ax, **sup_plt_fn_args)

        return ax


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
            Path to the output folder, if None create a folder with :attr:`_name`. By default None.
        file_name : str, optional
            Base name for the output image files, if None use :attr:`_name`. By default None.
        fmt : str, optional
            File format for saving result, by default "txt".
        x : np.ndarray, optional
            X coordinate values, if None use :attr:`_x`. By default None.
        y : np.ndarray, optional
            Y coordinate values, if None use :attr:`_y`. By default None.

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
            Statistical operator to apply. Must be implemented in :data:`NP_OPERATORS`.
        axis : tuple[int]
            Axis where to do the analysis:
            
                - If axis is string, replace 'x' by 1, 'y' by 0 and 'xy' by (0, 1). 
                - If axis is int, only use 0 or 1.
                - If None use (0, 1). By default None.

        Returns
        -------
        tilupy.read.StaticResults0D or tilupy.read.StaticResults1D
            Instance of :class:`tilupy.read.StaticResults0D` or :class:`tilupy.read.StaticResults1D`.

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
            Attribute :attr:`_x`.
        """
        return self._x
    
    
    @property
    def y(self) -> np.ndarray:
        """Get Y coordinates.

        Returns
        -------
        numpy.ndarray
            Attribute :attr:`_y`.
        """
        return self._y


    @property
    def z(self) -> np.ndarray:
        """Get elevations values.

        Returns
        -------
        numpy.ndarray
            Attribute :attr:`_z`.
        """
        return self._z
    
    
class Results:
    """Results of thin-layer model simulation

    This class is the parent class for all simulation results, whatever the
    kind of input data. Methods and functions for processing results are given
    here. Reading results from code specific outputs is done in inhereited
    classes.
    
    This class has global attributes used by all child classes and quick access 
    attributes calculated and stored for easier access to the main results of a 
    simulation. The quick attributes are only computed if needed and can be deleted
    to clean memory.
    
    Parameters
    ----------
    args and kwargs :
        Specific arguments for each models.
        
    Attributes
    ----------
        _code : str
            Name of the code that generated the result.
        _folder : str
            Path to find code files (like parameters).
        _folder_output :
            Path to find the results of the code.
        _zinit : numpy.ndarray
            Surface elevation of the simulation.
        _tim : list
            Lists of recorded time steps.
        _x : numpy.ndarray
            X-coordinates of the simulation.
        _y : numpy.ndarray
            Y-coordinates of the simulation.
        _dx : float
            Cell size along X-coordinates.
        _dy : float
            Cell size along Y-coordinates.
        _nx : float
            Number of cells along X-coordinates.
        _ny : float
            Number of cells along Y-coordinates.
            
        _h : tilupy.read.TemporalResults2D
            Quick access attributes for fluid height over time.
        _h_max : tilupy.read.TemporalResults0D
            Quick access attributes for max fluid hieght over time.
        _u : tilupy.read.TemporalResults2D
            Quick access attributes for norm of fluid velocity over time.
        _u_max : tilupy.read.TemporalResults0D
            Quick access attributes for max norm of fluid velocity over time.
        _costh : numpy.ndarray
            Quick access attributes for value of cos[theta] at any point on the surface. 
    """
    def __init__(self, *args, **kwargs):
        self._h = None
        self._h_max = None
        self._u = None
        self._u_max = None
        self._costh = None

        self._code = None
        self._folder = None
        self._folder_output = None
        self._z = None
        self._zinit = None
        self._tim = None
        self._x = None
        self._y = None
        self._dx = None
        self._dy = None
        self._nx = None
        self._ny = None


    def compute_costh(self) -> np.ndarray:
        """Get cos(slope) of topography.

        Returns
        -------
        numpy.ndarray
            Value of cos[theta] at any point on the surface.
        """
        [Fx, Fy] = np.gradient(self._zinit, np.flip(self._y), self._x)
        costh = 1 / np.sqrt(1 + Fx**2 + Fy**2)
        return costh


    def center_of_mass(self, h_thresh: float=None) -> tilupy.read.TemporalResults0D:
        """Compute center of mass coordinates depending on time.

        Parameters
        ----------
        h_thresh : float, optional
            Value of threshold for the flow height, by default None.

        Returns
        -------
        tilupy.read.TemporalResults0D
            Values of center of mass coordinates.
        """
        dx = self._x[1] - self._x[0]
        dy = self._y[1] - self._y[0]
        # Make meshgrid
        X, Y = np.meshgrid(self._x, np.flip(self._y))

        if self._h is None:
            self.h
        
        # Weights for coordinates average (volume in cell / total volume)
        h2 = self._h.copy()
        if h_thresh is not None:
            h2[h2 < h_thresh] = 0
        if self._costh is None:
            self._costh = self.compute_costh()
        w = h2 / self._costh[:, :, np.newaxis] * dx * dy
        vol = np.nansum(w, axis=(0, 1))
        w = w / vol[np.newaxis, np.newaxis, :]
        # Compute center of mass coordinates
        nt = h2.shape[2]
        coord = np.zeros((3, nt))
        tmp = X[:, :, np.newaxis] * w
        coord[0, :] = np.nansum(tmp, axis=(0, 1))
        tmp = Y[:, :, np.newaxis] * w
        coord[1, :] = np.nansum(tmp, axis=(0, 1))
        tmp = self._zinit[:, :, np.newaxis] * w
        coord[2, :] = np.nansum(tmp, axis=(0, 1))
        
        # Make TemporalResults
        res = TemporalResults0D("centermass",
                                coord,
                                self._tim,
                                scalar_names=["X", "Y", "z"],
                                notation=None)
        return res


    def volume(self, h_thresh: float=None) -> tilupy.read.TemporalResults0D:
        """Compute flow volume depending on time.

        Parameters
        ----------
        h_thresh : float, optional
            Value of threshold for the flow height, by default None.

        Returns
        -------
        tilupy.read.TemporalResults0D
            Values of flow volumes.
        """
        dx = self._x[1] - self._x[0]
        dy = self._y[1] - self._y[0]
        
        if self._h is None:
            self._h = self.get_output("h").d
        h2 = self._h.copy()
        if h_thresh is not None:
            h2[h2 < h_thresh] = 0
        if self._costh is None:
            self._costh = self.compute_costh()
        w = h2 / self._costh[:, :, np.newaxis] * dx * dy
        vol = np.nansum(w, axis=(0, 1))
        res = TemporalResults0D("volume",
                                vol,
                                self._tim,
                                notation=None)
        return res


    def get_output(self, 
                   output_name: str, 
                   from_file: bool=True, 
                   **kwargs
                   ) -> tilupy.read.TemporalResults0D | tilupy.read.StaticResults2D | tilupy.read.TemporalResults2D:
        """Get all the available outputs for a simulation :
            - Topographic outputs : "z", "zinit", "costh"
            - Temporal 2D outputs : "hvert", "h", "u", "ux", "uy", "hu", "hu2"
            - Other outputs : "centermass", "volume"
        
        It is possible to add operators to temporal 2D outputs : 
            - "max", "mean", "std", "sum", "min", "final", "init", "int"
        
        And it is possible to add axis (only if using operators) :
            - "x", "y", "xy"

        Parameters
        ----------
        output_name : str
            Name of the wanted output, composed of the output name and potentially 
            an operator and an axis: :data:`output_operator_axis`.
        from_file : bool, optional
            If True, find the output in a specific file. By default True.

        Returns
        -------
        tilupy.read.TemporalResults0D, tilupy.read.StaticResults2D or tilupy.read.TemporalResults2D
            Wanted output.
        """
        # Specific case of center of mass
        if output_name == "centermass":
            return self.center_of_mass(**kwargs)

        # Specific case of volume
        if output_name == "volume":
            return self.volume(**kwargs)
        
        strs = output_name.split("_")
        n_strs = len(strs)

        res = None
        
        # get topography
        if output_name in TOPO_DATA_2D:
            if output_name == "z":
                output_name = "zinit"
            output_name = '_' + output_name
            res = StaticResults2D(output_name,
                                  getattr(self, output_name),
                                  x=self._x,
                                  y=self._y,
                                  z=self._z,
                                  notation=None)
            return res

        # If no operator is called, call directly extract_output
        if n_strs == 1:
            res = self._extract_output(output_name, **kwargs)
            return res

        # Otherwise, get name, operator and axis (optional)
        name = strs[0]
        operator = strs[1]
        axis = None
        if n_strs == 3 :
            axis = strs[2]
        
        # If processed output is read directly from file, call the child method
        # read_from_file.
        if from_file:
            try:
                res = self._read_from_file(name, operator, axis=axis, **kwargs)
                if res is None:
                    raise UserWarning(f"{output_name} not found with _read_from_file for {self._code}, use get_spatial_stat")
                elif isinstance(res, str):
                    raise UserWarning(res)
            except UserWarning as w:
                print(f"[WARNING] {w}")
                res = None
            # res is None in case of function failure

        # If no results could be read from file, output must be
        # processed by tilupy
        if res is None:
            # Get output from name
            res = self._extract_output(name, x=self._x, y=self._y, **kwargs)
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


    def clear_quick_results(self) -> None:
        """Clear memory by erasing quick access attributes: :attr:`_h`, :attr:`_h_max`, :attr:`_u`, :attr:`_u_max`, :attr:`_costh`.
        """
        self._h = None
        self._h_max = None
        self._u = None
        self._u_max = None
        self._costh = None
    
    
    def get_profile(self,
                    output: str,
                    extraction_method: str = "axis",
                    **extraction_params
                    ) -> tuple[tilupy.read.TemporalResults1D | tilupy.read.StaticResults1D, np.ndarray]:
        """Extract a profile from a 2D data.

        Parameters
        ----------
        output : str
            Wanted data output.
        extraction_mode : str, optional
            Method to extract profiles:
            
                - "axis": Extracts a profile along an axis.
                - "coordinates": Extracts a profile along specified coordinates.
                - "shapefile": Extracts a profile along a shapefile (polylines).
            
            Be default "axis".
        extraction_params : dict, optional
            Different parameters to be entered depending on the extraction method chosen.
            See :meth:`tilupy.utils.get_profile`.

        Returns
        -------
        tuple[tilupy.read.TemporalResults1D | tilupy.read.StaticResults1D, np.ndarray]
            profile : tilupy.read.TemporalResults1D | tilupy.read.StaticResults1D
                Extracted profile.
            data : numpy.ndarray or float or tuple[nympy.ndarray, nympy.ndarray]
                Specific output depending on :data:`extraction_mode`:
                    
                    - If :data:`extraction_mode == "axis"`: float
                        Position of the profile.
                    - If :data:`extraction_mode == "coordinates"`: tuple[numpy.ndarray]
                        X coordinates, Y coordinates and distance values.
                    - If :data:`extraction_mode == "shapefile"`: numpy.ndarray
                        Distance values.

        Raises
        ------
        ValueError
            If :data:`output` doesn't generate a 2D data.
        """
        data = self.get_output(output)
            
        if not isinstance(data, tilupy.read.TemporalResults2D) and not isinstance(data, tilupy.read.StaticResults2D):
            raise ValueError("Can only extract profile from 2D data.")
        
        profile, data = utils.get_profile(data, extraction_method, **extraction_params)
        
        return profile, data
    

    def plot(self,
             output: str,
             from_file: bool =True, #get_output
             h_thresh: float=None, #get_output
             time_steps: float | list[float] = None,
             save: bool = False,
             folder_out: str = None,
             dpi: int = 150,
             fmt: str="png",
             file_suffix: str = None,
             file_prefix: str = None,
             display_plot: bool = True,
             **plot_kwargs
             ) -> matplotlib.axes._axes.Axes:
        """Plot output extracted from model's result.

        Parameters
        ----------
        output : str
            Wanted output to be plotted. Must be in :data:`DATA_NAMES`.
        from_file : bool, optional
            If True, find the output in a specific file. By default True.
        h_thresh : float, optional
            Threshold value to be taken into account when extracting output, by default None.
        time_steps : float or list[float], optional
            Time steps to show when plotting temporal data. If None shows every time
            steps recorded. By default None.
        save : bool, optional
            If True, save the plot as an image to the computer, by default False.
        folder_out : str, optional
            Path to the folder where to save the plot, by default None.
        dpi : int, optional
            Resolution for the saved plot, by default 150.
        fmt : str, optional
            Format of the saved plot, by default "png".
        file_suffix : str, optional
            Suffix to add to the file name when saving, by default None.
        file_prefix : str, optional
            Prefix to add to the file name when saving, by default None.
        display_plot : bool, optional
            If True, enables the display of the plot; otherwise, it disables the display to save memory. 
            By default True.

        Returns
        -------
        matplotlib.axes._axes.Axes
            Wanted plot.
        """
        if not display_plot:
            backend = plt.get_backend()
            plt.close("all")
            plt.switch_backend("Agg")
        
        if output in ["z", "zinit", "z_init"]:
            topo_kwargs = dict()
            if "topo_kwargs" in plot_kwargs:
                topo_kwargs = plot_kwargs["topo_kwargs"]
            axe = plt_fn.plot_topo(self._zinit, self._x, self._y, **topo_kwargs)
            return axe

        data = self.get_output(output, from_file=from_file, h_thresh=h_thresh)

        add_time_on_plot = False
        if (isinstance(time_steps, float) or isinstance(time_steps, int)) and isinstance(data, tilupy.read.TemporalResults):
            t_index = np.argmin(np.abs(self._tim - time_steps))
            add_time_on_plot = self._tim[t_index]
        
        if time_steps is not None and isinstance(data, tilupy.read.TemporalResults):
            data = data.extract_from_time_step(time_steps)

        if save:
            if folder_out is None:
                assert (self._folder_output is not None), "folder_output attribute must be set"
                folder_out = os.path.join(self._folder_output, "plots")
            os.makedirs(folder_out, exist_ok=True)

        # TODO
        # Edit Temporal/Static.plot() pour que la sauvegarde soit directement intégrer dans les méthodes plots
        # Dans les plots, modifier les fonctions pour appeler des méthodes de pytopomap selon le plot voulu
        # (shotgater, profil, surface, etc...) et créer une fonction globale qui appelle chaque sous méthode 
        # pour créer les graphes et gérer la sauvegarde 
        if folder_out is not None and isinstance(data, TemporalResults2D):
            # If data is TemporalResults2D then saving is managed directly
            # by the associated plot method
            plot_kwargs["folder_out"] = folder_out
            plot_kwargs["dpi"] = dpi
            plot_kwargs["fmt"] = fmt
            # kwargs["file_suffix"] = file_prefix
            # kwargs["file_prefix"] = file_prefix

        axe = data.plot(**plot_kwargs)
        
        if add_time_on_plot:
            axe.set_title(f"t={add_time_on_plot}s", loc="left")

        if folder_out is not None and not isinstance(data, TemporalResults2D):
            file_name = output
            if file_suffix is not None:
                file_name = file_name + "_" + file_suffix
            if file_prefix is not None:
                file_name = file_prefix + "_" + file_name
            file_out = os.path.join(folder_out, file_name + "." + fmt)
            # axe.figure.tight_layout(pad=0.1)
            axe.figure.savefig(file_out, dpi=dpi, bbox_inches="tight", pad_inches=0.05)
        
        if not display_plot:
            plt.close("all")
            plt.switch_backend(backend)

        return axe


    def plot_profile(self,
                     output: str,
                     from_file: bool=True,
                     extraction_method: str = "axis",
                     extraction_params: dict = None,
                     time_steps: float | list[float] = None,
                     save: bool = False,
                     folder_out: str = None,
                     display_plot: bool = True,
                     **plot_kwargs
                     ) -> matplotlib.axes._axes.Axes:
        """Plot a 1D output extracted from a 2D output.

        Parameters
        ----------
        output : str
            Wanted 2D output to extract the profile from. Must be in :data:`STATIC_DATA_2D`
            or in :data:`TEMPORAL_DATA_2D`.
        from_file : bool, optional
            If True, find the output in a specific file. By default True.
        extraction_mode : str, optional
            Method to extract profiles:
            
                - "axis": Extracts a profile along an axis.
                - "coordinates": Extracts a profile along specified coordinates.
                - "shapefile": Extracts a profile along a shapefile (polylines).
            Be default "axis".
        extraction_params : dict, optional
            Different parameters to be entered depending on the extraction method chosen.
            See :meth:`tilupy.utils.get_profile`.
        time_steps : float or list[float], optional
            Time steps to show when plotting temporal data. If None shows every time
            steps recorded. By default None.
        save : bool, optional
            If True, save the plot as an image to the computer, by default False.
        folder_out : str, optional
            Path to the folder where to save the plot, by default None.
        display_plot : bool, optional
            If True, enables the display of the plot; otherwise, it disables the display to save memory. 
            By default True.        
        
        Returns
        -------
        matplotlib.axes._axes.Axes
            Wanted plot.

        Raises
        ------
        ValueError
            If the :data:`output` is not a 2D output.
        """
        if not display_plot:
            backend = plt.get_backend()
            plt.close("all")
            plt.switch_backend("Agg")
        
        data = self.get_output(output, from_file=from_file)
        
        if not isinstance(data, tilupy.read.TemporalResults2D) and not isinstance(data, tilupy.read.StaticResults2D):
            raise ValueError("Can only extract profile from 2D data.")
        
        extraction_params = {} if extraction_params is None else extraction_params
        
        profile, _ = utils.get_profile(data, extraction_method, **extraction_params)
        closest_value = False
        
        if (isinstance(time_steps, float) or isinstance(time_steps, int)) and isinstance(data, tilupy.read.TemporalResults):
            t_index = np.argmin(np.abs(self._tim - time_steps))
            closest_value = self._tim[t_index]
        
        if time_steps is not None and isinstance(data, tilupy.read.TemporalResults):
            profile = profile.extract_from_time_step(time_steps)

        axe = profile.plot(**plot_kwargs)
        
        if closest_value:
            axe.set_title(f"t={closest_value}s", loc="left")
        
        if save:
            if folder_out is None:
                assert (self._folder_output is not None), "folder_output attribute must be set"
                folder_out = os.path.join(self._folder_output, "plots")
            os.makedirs(folder_out, exist_ok=True)
            
        # TODO
        # Same as plot() -> add save mode in plot functions in Temporal/StaticResults
        '''
        if folder_out is not None and isinstance(data, TemporalResults2D):
            # If data is TemporalResults2D then saving is managed directly
            # by the associated plot method
            kwargs["folder_out"] = folder_out
            kwargs["dpi"] = dpi
            kwargs["fmt"] = fmt
            # kwargs["file_suffix"] = file_prefix
            # kwargs["file_prefix"] = file_prefix

        

        if folder_out is not None and not isinstance(data, TemporalResults2D):
            file_name = output_name
            if file_suffix is not None:
                file_name = file_name + "_" + file_suffix
            if file_prefix is not None:
                file_name = file_prefix + "_" + file_name
            file_out = os.path.join(folder_out, file_name + "." + fmt)
            # axe.figure.tight_layout(pad=0.1)
            axe.figure.savefig(file_out, dpi=dpi, bbox_inches="tight", pad_inches=0.05)
        '''
        
        if not display_plot:
            plt.close("all")
            plt.switch_backend(backend)

        return axe
            
        
    def save(self,
             output_name: str,
             folder_out: str=None,
             file_name: str=None,
             fmt: str="txt",
             from_file: bool=True,
             **kwargs
             ) -> None:
        """Save simulation outputs (processed results or topographic data) to disk.

        Depending on the requested output_name, the method either:
            
            - Retrieves a result via :meth:`get_output` and calls its own :meth:`save` method,
            - Or, for static topography data, writes it directly to a raster file.

        Parameters
        ----------
        output_name : str
            Name of the variable or processed result to save 
            (e.g., "h", "u_mean_t", "centermass", or topographic data like "zinit").
        folder_out : str, optional
            Destination folder for saving files. If None, defaults to
            :data:`_folder_output/processed`. By default None.
        file_name : str, optional
            Base name of the output file (without extension). If None, uses
            :data:`output_name`. By default None.
        fmt : str, optional
            Output file format (e.g., "txt", "npy", "asc"), by default "txt".
        from_file : bool, optional
            If True, attempt to read precomputed results from file before computing,
            by default True.
        **kwargs : dict
            Extra arguments passed to the underlying save function. For raster data,
            forwarded to :func:`tilupy.raster.write_raster`.

        Raises
        ------
        AssertionError
            If neither :data:`folder_out` nor :attr:`_folder_output` is defined.
        """
    
        if folder_out is None:
            assert (self._folder_output is not None), "folder_output attribute must be set"
            folder_out = os.path.join(self._folder_output, "processed")
            os.makedirs(folder_out, exist_ok=True)

        if output_name in DATA_NAMES:
            data = self.get_output(output_name, from_file=from_file)
            if data.d.ndim > 1:
                if "x" not in kwargs:
                    kwargs["x"] = self.x
                if "y" not in kwargs:
                    kwargs["y"] = self.y

            data.save(folder=folder_out, file_name=file_name, fmt=fmt, **kwargs)

        elif output_name in TOPO_DATA_2D:
            if file_name is None:
                file_name = output_name
            name = "_" + output_name
            file_out = os.path.join(folder_out, file_name)
            tilupy.raster.write_raster(self._x,
                                       self._y,
                                       getattr(self, name),
                                       file_out,
                                       fmt=fmt,
                                       **kwargs)


    @abstractmethod
    def _extract_output(self):
        """Abstract method to extract output of simulation result files."""
        pass


    @abstractmethod
    def _read_from_file(self):
        """Abstract method for reading output from specific files."""
        pass


    @property
    def zinit(self):
        """Get initial topography.

        Returns
        -------
        numpy.ndarray
            Attribute :attr:`_zinit`
        """
        return self._zinit


    @property
    def z(self):
        """Get initial topography, alias for zinit.

        Returns
        -------
        numpy.ndarray
            Attribute :attr:`_zinit`
        """
        return self._zinit


    @property
    def x(self):
        """Get X-coordinates.
        
        Returns
        -------
        numpy.ndarray
            Attribute :attr:`_x`
        """
        return self._x


    @property
    def y(self):
        """Get Y-coordinates.
        
        Returns
        -------
        numpy.ndarray
            Attribute :attr:`_y`
        """
        return self._y
    
    
    @property
    def dx(self):
        """Get cell size along X.
        
        Returns
        -------
        numpy.ndarray
            Attribute :attr:`_dx`
        """
        return self._dx
    
    
    @property
    def dy(self):
        """Get cell size along Y.
        
        Returns
        -------
        numpy.ndarray
            Attribute :attr:`_dy`
        """
        return self._dy


    @property
    def nx(self):
        """Get number of cells along X.
        
        Returns
        -------
        numpy.ndarray
            Attribute :attr:`_nx`
        """
        return self._nx
    
    
    @property
    def ny(self):
        """Get number of cells along Y.
        
        Returns
        -------
        numpy.ndarray
            Attribute :attr:`_ny`
        """
        return self._ny
    
    
    @property
    def tim(self):
        """Get recorded time steps.
        
        Returns
        -------
        numpy.ndarray
            Attribute :attr:`_tim`
        """
        return self._tim
    
    
    @property
    def h(self):
        """Get flow thickness. Compute it if not stored.
        
        Returns
        -------
        tilupy.read.TemporalResults2D
            Attribute :attr:`_h`
        """
        if self._h is None:
            self._h = self.get_output("h").d
        return self._h


    @property
    def h_max(self):
        """Get maximum flow thickness. Compute it if not stored.
        
        Returns
        -------
        tilupy.read.TemporalResults0D
            Attribute :attr:`_h_max`
        """
        if self._h_max is None:
            self._h_max = self.get_output("h_max").d
        return self._h_max


    @property
    def u(self):
        """Get flow velocity. Compute it if not stored.
        
        Returns
        -------
        tilupy.read.TemporalResults2D
            Attribute :attr:`_u`
        """
        if self._u is None:
            self._u = self.get_output("u").d
        return self._u


    @property
    def u_max(self):
        """Get maximum flow velocity. Compute it if not stored.
        
        Returns
        -------
        tilupy.read.TemporalResults0D
            Attribute :attr:`_u_max`
        """
        if self._u_max is None:
            self._u_max = self.get_output("u_max").d
        return self._u_max
    
    
    @property
    def costh(self):
        """Get cos(slope) of topography. Compute it if not stored.
        
        Returns
        -------
        numpy.ndarray
            Attribute :attr:`_costh`
        """
        if self._costh is None:
            self._costh = self.compute_costh()
        return self._costh


def get_results(code, **kwargs) -> tilupy.read.Results:
    """Get simulation results for a given numerical model.

    Dynamically imports the corresponding reader module from
    `tilupy.models.<code>.read` and instantiates its :class:`tilupy.read.Results` class.

    Parameters
    ----------
    code : str
        Short name of the simulation model: must be in :data:`ALLOWED_MODELS`.
    **kwargs : dict
        Additional keyword arguments passed to the :class:`tilupy.read.Results` constructor
        of the imported module.

    Returns
    -------
    tilupy.read.Results
        Instance of the :class:`tilupy.read.Results` class containing the simulation outputs.

    Raises
    ------
    ModuleNotFoundError
        If the module `tilupy.models.<code>.read` cannot be imported.
    AttributeError
        If the module does not define a :class:`tilupy.read.Results` class.
    """
    module = importlib.import_module("tilupy.models." + code + ".read")
    return module.Results(**kwargs)


def use_thickness_threshold(simu: tilupy.read.Results, 
                            array: np.ndarray, 
                            h_thresh: float
                            ) -> np.ndarray:
    """Apply a flow thickness threshold to mask simulation results.

    Values of :data:`array` are set to zero wherever the flow thickness
    is below the given threshold.

    Parameters
    ----------
    simu : tilupy.read.Results
        Simulation result object providing access to thickness data :data:`h`.
    array : numpy.ndarray
        Array of values to be masked (must be consistent in shape with thickness).
    h_thresh : float
        Thickness threshold. Cells with thickness < :data:`h_thresh` are set to zero.

    Returns
    -------
    numpy.ndarray
        Thresholded array, with values set to zero where flow thickness is too low.
    """
    thickness = simu.get_output("h")
    array[thickness.d < h_thresh] = 0
    return array