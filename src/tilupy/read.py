#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun  1 14:20:52 2021

@author: peruzzetto
"""

import matplotlib.pyplot as plt
import numpy as np

import os
import importlib
import warnings

import tilupy.notations as notations
import tilupy.plot as plt_fn
import tilupy.raster

RAW_STATES = ["hvert", "h", "ux", "uy"]

TEMPORAL_DATA_0D = ["ek", "vol"]
TEMPORAL_DATA_1D = [""]
TEMPORAL_DATA_2D = ["hvert", "h", "u", "ux", "uy", "hu", "hu2"]
STATIC_DATA_0D = []
STATIC_DATA_1D = []
STATIC_DATA_2D = []

TOPO_DATA_2D = ["z", "zinit", "costh"]

NP_OPERATORS = ["max", "mean", "std", "sum", "min"]
OTHER_OPERATORS = ["final", "init", "int"]
TIME_OPERATORS = ["final", "init", "int"]

COMPUTED_STATIC_DATA_2D = []
for stat in NP_OPERATORS + OTHER_OPERATORS:
    for name in TEMPORAL_DATA_2D:
        COMPUTED_STATIC_DATA_2D.append(name + "_" + stat)
STATIC_DATA_2D += COMPUTED_STATIC_DATA_2D

COMPUTED_SPAT_1D_DATA = []
for stat in NP_OPERATORS + OTHER_OPERATORS:
    for name in TEMPORAL_DATA_2D:
        for axis in ["x", "y"]:
            COMPUTED_SPAT_1D_DATA.append(name + "_" + stat + "_" + axis)
TEMPORAL_DATA_1D += COMPUTED_SPAT_1D_DATA

COMPUTED_SPAT_0D_DATA = []
for stat in NP_OPERATORS + OTHER_OPERATORS:
    for name in TEMPORAL_DATA_2D:
        COMPUTED_SPAT_0D_DATA.append(name + "_" + stat + "_xy")
TEMPORAL_DATA_0D += COMPUTED_SPAT_0D_DATA

TEMPORAL_DATA = TEMPORAL_DATA_0D + TEMPORAL_DATA_1D + TEMPORAL_DATA_2D
STATIC_DATA = (
    STATIC_DATA_0D + STATIC_DATA_1D + STATIC_DATA_2D + COMPUTED_STATIC_DATA_2D
)

DATA_NAMES = TEMPORAL_DATA_0D + TEMPORAL_DATA_1D + TEMPORAL_DATA_2D
DATA_NAMES += STATIC_DATA_0D + STATIC_DATA_1D + STATIC_DATA_2D


class AbstractResults:
    """Abstract class for TemporalResults and StaticResults"""

    def __init__(self, name, d, notation=None, **kwargs):
        self.name = name
        self.d = d
        if isinstance(notation, dict):
            self.notation = notations.Notation(**notation)
        elif notation is None:
            self.notation = notations.get_notation(name)
        else:
            self.notation = notation
        self.__dict__.update(kwargs)


class TemporalResults(AbstractResults):
    """Time dependent result of simulation"""

    def __init__(
        self,
        name,
        d,
        t,
        notation=None,
    ):
        super().__init__(name, d, notation=notation)
        # 1d array with times, matching last dimension of self.d
        self.t = t

    def get_temporal_stat(self, stat):
        """Statistical analysis along temporal dimension"""
        if stat in NP_OPERATORS:
            dnew = getattr(np, stat)(self.d, axis=-1)
        elif stat == "final":
            dnew = self.d[..., -1]
        elif stat == "init":
            dnew = self.d[..., 0]
        elif stat == "int":
            dnew = np.trapz(self.d, x=self.t)

        notation = notations.add_operator(self.notation, stat, axis="t")

        if dnew.ndim == 2:
            return StaticResults2D(
                self.name + "_" + stat,
                dnew,
                notation=notation,
                x=self.x,
                y=self.y,
                z=self.z,
            )
        elif dnew.ndim == 1:
            return StaticResults1D(
                self.name + "_" + stat,
                dnew,
                notation=notation,
                coords=self.coords,
                coords_name=self.coords_name,
            )

    def get_spatial_stat(self, stat, axis):
        raise NotImplementedError()

    def plot(*arg, **kwargs):
        """Plot results as time dependent"""
        raise NotImplementedError()

    def save(*arg, **kwargs):
        raise NotImplementedError()


class TemporalResults0D(TemporalResults):
    """
    Class inheretied from TemporalResults where the data is one or multiple
    scalar functions of time
    of time.
    """

    def __init__(
        self,
        name,
        d,
        t,
        scalar_names=None,
        notation=None,
    ):
        """
        initiates TemporalResults0D instance

        Parameters
        ----------
        name : string
            Name of the data type.
        d : array
            array like data, with last dimension corresponding to time. It can
            be a one dimensionnal Nt array, or a two dimensionnal NdxNt array,
            where Nt is the legnth of t, and Nd correspond to the number of
            scalar values of interest (e.g. X and Y coordinates of the center
            of mass / front)
        t : array
            Array of time of length Nt.
        scalar_names : list of strings, optional
            List of length Nd containing the names of the scalar fields (one
            name per row of d)

        Returns
        -------
        None.

        """
        super().__init__(name, d, t, notation=notation)
        self.scalar_names = scalar_names

    def plot(self, axe=None, figsize=None, **kwargs):
        """Plot results.

        :param axe: DESCRIPTION, defaults to None
        :type axe: TYPE, optional
        :param figsize: DESCRIPTION, defaults to None
        :type figsize: TYPE, optional
        :param **kwargs: DESCRIPTION
        :type **kwargs: TYPE
        :return: DESCRIPTION
        :rtype: TYPE

        """

        if axe is None:
            fig, axe = plt.subplots(
                1, 1, figsize=figsize, layout="constrained"
            )

        if isinstance(self.d, np.ndarray):
            data = self.d.T
        else:
            data = self.d
        axe.plot(self.t, data, label=self.scalar_names)
        axe.set_xlabel(notations.get_label("t"))
        axe.set_ylabel(notations.get_label(self.notation))

        return axe

    def save(self):
        raise NotImplementedError(
            "Saving method for TemporalResults0D not implemented yet"
        )

    def get_spatial_stat(self, *arg, **kwargs):
        raise NotImplementedError(
            (
                "Spatial integration of Spatialresults0D"
                + " is not implemented because non relevant"
            )
        )


class TemporalResults1D(TemporalResults):
    """Class for simulation results described by one dimension for space
    and one dimension for time.

    :param name: Name of the data
    :type name: str
    :param d: data
    :type d: numpy.ndarray
    :param t: time array
    :type t: array like
    :param coords: 1D coordinate for spatial dimension
    :type coords: array like
    :param coords_name: name of the 1D coordinate (typically "X" or "Y")
    :type coords_name: str

    """

    def __init__(
        self, name, d, t, coords=None, coords_name=None, notation=None
    ):
        """Constructor method."""
        super().__init__(name, d, t, notation=notation)
        # x and y arrays
        self.coords = coords
        self.coords_name = coords_name

    def plot(self, coords=None, **kwargs):
        """Plot results.

        :param axe: DESCRIPTION, defaults to None
        :type axe: TYPE, optional
        :param figsize: DESCRIPTION, defaults to None
        :type figsize: TYPE, optional
        :param **kwargs: DESCRIPTION
        :type **kwargs: TYPE
        :return: DESCRIPTION
        :rtype: TYPE

        """

        if coords is None:
            coords = self.coords

        if coords is None:
            raise TypeError("coords data missing")

        xlabel = notations.get_label(self.coords_name, with_unit=True)
        if "colorbar_kwargs" not in kwargs:
            kwargs["colorbar_kwargs"] = dict()
        if "label" not in kwargs["colorbar_kwargs"]:
            clabel = notations.get_label(self.notation)
            kwargs["colorbar_kwargs"]["label"] = clabel

        axe = plt_fn.plot_shotgather(
            self.coords,
            self.t,
            self.d,
            xlabel=xlabel,
            ylabel=notations.get_label("t"),
            **kwargs
        )

        return axe

    def save(self):
        raise NotImplementedError(
            "Saving method for TemporalResults1D not implemented yet"
        )

    def get_spatial_stat(self, stat, **kwargs):
        """

        :param stat: DESCRIPTION
        :type stat: TYPE
        :return: DESCRIPTION
        :rtype: TYPE

        """

        if stat in NP_OPERATORS:
            dnew = getattr(np, stat)(self.d, axis=0)
        elif stat == "int":
            dd = self.coords[1] - self.coords[0]
            dnew = np.sum(self.d, axis=0) * dd
        notation = notations.add_operator(
            self.notation, stat, axis=self.coords_name
        )
        return TemporalResults0D(
            self.name + "_" + stat, dnew, self.t, notation=notation
        )


class TemporalResults2D(TemporalResults):
    def __init__(self, name, d, t, x=None, y=None, z=None, notation=None):
        """Initiate instance of TemporalResults2D.

        :param name: DESCRIPTION
        :type name: TYPE
        :param d: DESCRIPTION
        :type d: TYPE
        :param t: DESCRIPTION
        :type t: TYPE
        :param x: DESCRIPTION, defaults to None
        :type x: TYPE, optional
        :param y: DESCRIPTION, defaults to None
        :type y: TYPE, optional
        :param z: DESCRIPTION, defaults to None
        :type z: TYPE, optional
        :return: DESCRIPTION
        :rtype: TYPE

        """

        super().__init__(name, d, t, notation=notation)
        # x and y arrays
        self.x = x
        self.y = y
        # topography
        self.z = z

    def plot(
        self,
        x=None,
        y=None,
        z=None,
        file_name=None,
        folder_out=None,
        figsize=None,
        dpi=None,
        fmt="png",
        sup_plt_fn=None,
        sup_plt_fn_args=None,
        **kwargs
    ):
        """Plot results.

        :param **kwargs: DESCRIPTION
        :type **kwargs: TYPE
        :return: DESCRIPTION
        :rtype: TYPE

        """
        if file_name is None:
            file_name = self.name

        if x is None:
            x = self.x
        if y is None:
            y = self.y
        if z is None:
            z = self.z

        if x is None or y is None:
            raise TypeError("x, y or z data missing")

        if z is None:
            warnings.warn("No topography given.")

        if "colorbar_kwargs" not in kwargs:
            kwargs["colorbar_kwargs"] = dict()
        if "label" not in kwargs["colorbar_kwargs"]:
            clabel = notations.get_label(self.notation)
            kwargs["colorbar_kwargs"]["label"] = clabel

        plt_fn.plot_maps(
            x,
            y,
            z,
            self.d,
            self.t,
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

    def save(
        self,
        folder=None,
        file_name=None,
        fmt="asc",
        time=None,
        x=None,
        y=None,
        **kwargs
    ):
        if x is None:
            x = self.x
        if y is None:
            y = self.y
        if x is None or y is None:
            raise ValueError("x et y arrays must not be None")

        if file_name is None:
            file_name = self.name

        if folder is not None:
            file_name = os.path.join(folder, file_name)

        if time is not None:
            if isinstance(time, str):
                if time == "final":
                    inds = [self.d.shape[2] - 1]
                elif time == "initial":
                    inds = [0]
            else:
                inds = [np.argmin(time - np.abs(np.array(self.t) - time))]
        else:
            inds = range(len(self.t))

        for i in inds:
            file_out = file_name + "_{:04d}.".format(i) + fmt
            tilupy.raster.write_raster(
                x, y, self.d[:, :, i], file_out, fmt=fmt, **kwargs
            )

    def get_spatial_stat(self, stat, axis=None):
        """

        :param stat: DESCRIPTION
        :type stat: TYPE
        :param axis: DESCRIPTION, defaults to None
        :type axis: TYPE, optional
        :return: DESCRIPTION
        :rtype: TYPE

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
            dnew = getattr(np, stat)(self.d, axis=axis)
        elif stat == "int":
            dnew = np.sum(self.d, axis=axis)
            if axis == 1:
                dd = self.x[1] - self.x[0]
            elif axis == 0:
                dd = self.y[1] - self.y[0]
            elif axis == (0, 1):
                dd = (self.x[1] - self.x[0]) * (self.y[1] - self.y[0])
            dnew = dnew * dd

        if axis == 1:
            # Needed to get correct orinetation as d[0, 0] is the upper corner
            # of the data, with coordinates x[0], y[-1]
            dnew = np.flip(dnew, axis=0)

        new_name = self.name + "_" + stat + "_" + axis_str
        notation = notations.add_operator(self.notation, stat, axis=axis_str)

        if axis == (0, 1):
            return TemporalResults0D(new_name, dnew, self.t, notation=notation)
        else:
            if axis == 0:
                coords = self.x
                coords_name = "x"
            else:
                coords = self.x
                coords_name = "y"
            return TemporalResults1D(
                new_name,
                dnew,
                self.t,
                coords,
                coords_name=coords_name,
                notation=notation,
            )


class StaticResults(AbstractResults):
    """Result of simulation without time dependence"""

    def __init__(self, name, d, notation=None):
        super().__init__(name, d, notation=notation)
        # x and y arrays

    def plot(self):
        raise NotImplementedError()

    def save(self):
        raise NotImplementedError()


class StaticResults2D(StaticResults):
    def __init__(self, name, d, x=None, y=None, z=None, notation=None):
        super().__init__(name, d, notation=notation)
        # x and y arrays
        self.x = x
        self.y = y
        # topography
        self.z = z

    def plot(
        self,
        axe=None,
        figsize=None,
        x=None,
        y=None,
        z=None,
        sup_plt_fn=None,
        sup_plt_fn_args=None,
        **kwargs
    ):
        """Plot results as map"""

        if axe is None:
            fig, axe = plt.subplots(
                1, 1, figsize=figsize, layout="constrained"
            )

        if x is None:
            x = self.x
        if y is None:
            y = self.y
        if z is None:
            z = self.z

        if x is None or y is None or z is None:
            raise TypeError("x, y or z data missing")

        if "colorbar_kwargs" not in kwargs:
            kwargs["colorbar_kwargs"] = dict()
        if "label" not in kwargs["colorbar_kwargs"]:
            clabel = notations.get_label(self.notation)
            kwargs["colorbar_kwargs"]["label"] = clabel

        print(self.name)

        axe = plt_fn.plot_data_on_topo(
            x, y, z, self.d, axe=axe, figsize=figsize, **kwargs
        )
        if sup_plt_fn is not None:
            if sup_plt_fn_args is None:
                sup_plt_fn_args = dict()
            sup_plt_fn(axe, **sup_plt_fn_args)

        return axe

    def save(
        self,
        folder=None,
        file_name=None,
        fmt="txt",
        time=None,
        x=None,
        y=None,
        **kwargs
    ):
        if x is None:
            x = self.x
        if y is None:
            y = self.y

        if x is None or y is None:
            raise ValueError("x et y arrays must not be None")

        if file_name is None:
            file_name = self.name + "." + fmt

        if folder is not None:
            file_name = os.path.join(folder, file_name)

        tilupy.raster.write_raster(x, y, self.d, file_name, fmt=fmt, **kwargs)

    def get_spatial_stat(self, stat, axis=None):
        """

        :param stat: DESCRIPTION
        :type stat: TYPE
        :param axis: DESCRIPTION, defaults to None
        :type axis: TYPE, optional
        :return: DESCRIPTION
        :rtype: TYPE

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
            dnew = getattr(np, stat)(self.d, axis=axis)
        elif stat == "int":
            dnew = np.sum(self.d, axis=axis)
            if axis == 1:
                dd = self.x[1] - self.x[0]
            elif axis == 0:
                dd = self.y[1] - self.y[0]
            elif axis == (0, 1):
                dd = (self.x[1] - self.x[0]) * (self.y[1] - self.y[0])
            dnew = dnew * dd

        if axis == 1:
            # Needed to get correct orinetation as d[0, 0] is the upper corner
            # of the data, with coordinates x[0], y[-1]
            dnew = np.flip(dnew, axis=0)

        new_name = self.name + "_" + stat + "_" + axis_str
        notation = notations.add_operator(self.notation, stat, axis=axis_str)

        if axis == (0, 1):
            return StaticResults0D(new_name, dnew, notation=notation)
        else:
            if axis == 0:
                coords = self.x
                coords_name = "x"
            else:
                coords = self.y
                coords_name = "y"
            return StaticResults1D(
                new_name,
                dnew,
                coords,
                coords_name=coords_name,
                notation=notation,
            )


class StaticResults1D(StaticResults):
    def __init__(self, name, d, coords=None, coords_name=None, notation=None):
        """Constructor method."""
        super().__init__(name, d, notation=notation)
        # x and y arrays
        self.coords = coords
        self.coords_name = coords_name

    def plot(self, axe=None, figsize=None, **kwargs):
        """Plot results.

        :param axe: DESCRIPTION, defaults to None
        :type axe: TYPE, optional
        :param figsize: DESCRIPTION, defaults to None
        :type figsize: TYPE, optional
        :param **kwargs: DESCRIPTION
        :type **kwargs: TYPE
        :return: DESCRIPTION
        :rtype: TYPE

        """

        if axe is None:
            fig, axe = plt.subplots(
                1, 1, figsize=figsize, layout="constrained"
            )

        if isinstance(self.d, np.ndarray):
            data = self.d.T
        else:
            data = self.d
        axe.plot(self.coords, data, label=self.scalar_names)
        axe.set_xlabel(notations.get_label(self.coords_name))
        axe.set_ylabel(notations.get_label(self.notation))

        return axe


class StaticResults0D(StaticResults):
    def __init__(self, d, name, notation=None):
        super().__init__(name, d, notation=notation)


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
        strs = output_name.split("_")
        n_strs = len(strs)
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
            plt.switch_backend("Agg")

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
            plt.switch_backend(backend)

        return axe

    def save(
        self,
        name,
        folder=None,
        file_name=None,
        fmt="txt",
        from_file=True,
        **kwargs
    ):
        if folder is None:
            assert (
                self.folder_output is not None
            ), "folder_output attribute must be set"
            folder = os.path.join(self.folder_output, "processed")
            os.makedirs(folder, exist_ok=True)

        if name in DATA_NAMES:
            data = self.get_output(name, from_file=from_file)
            if data.d.ndim > 1:
                if "x" not in kwargs:
                    kwargs["x"] = self.x
                if "y" not in kwargs:
                    kwargs["y"] = self.y

            data.save(folder=folder, file_name=file_name, fmt=fmt, **kwargs)

        elif name in TOPO_DATA_2D:
            if file_name is None:
                file_name = name
            file_out = os.path.join(folder, file_name)
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
