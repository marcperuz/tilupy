#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
import pytopomap.plot as pyplt
import pandas as pd

import tilupy.download_data


def plot_shotgather(
    x: np.ndarray,
    t: np.ndarray,
    data: np.ndarray,
    xlabel: str = "X (m)",
    ylabel: str = "Time (s)",
    **kwargs,
) -> matplotlib.axes._axes.Axes:
    """Plot shotgather image.

    Plot shotgather like image, with vertical axis as time and horizontal axis
    and spatial dimension. This is a simple call to plot_shotgather, but
    input data is transposed because in tilupy the last axis is time by
    convention.

    Parameters
    ----------
    x : numpy.ndarray
        Spatial coordinates, size NX.
    t : numpy.ndarray
        Time array (assumed in seconds), size NT.
    data : numpy.ndarray
        NX*NT array of data to be plotted.
    xlabel : string, optional
        Label for x-axis, by default "X (m)".
    ylabel : string, optional
        Label for x-axis, by default "Time (s)".
    **kwargs : dict, optional
        parameters passed on to :func:`pytopomap.plot.plot_imshow`.

    Returns
    -------
    matplotlib.axes._axes.Axes
        Axes instance where data is plotted
    """
    if "aspect" not in kwargs:
        kwargs["aspect"] = "auto"
    axe = pyplt.plot_imshow(x, t[::-1], data.T, **kwargs)
    axe.set_adjustable("box")
    axe.set_ylabel(ylabel)
    axe.set_xlabel(xlabel)

    return axe


def plot_heatmaps(
    df,
    values,
    index,
    columns,
    axs=None,
    aggfunc="mean",
    figsize=None,
    ncols=3,
    heatmap_kws=None,
    notations=None,
    best_values=None,
    plot_best_value="point",
    text_kwargs=None,
) -> matplotlib.figure.Figure:
    """Plot one or several heatmaps from a pandas DataFrame.

    Each heatmap is created by pivoting the DataFrame with the given
    `index`, `columns`, and a variable from `values`.

    Parameters
    ----------
    df : pandas.DataFrame
        Input DataFrame containing the data.
    values : list[str]
        Column names in :data:`df` to plot as separate heatmaps.
    index : str
        Column name to use as rows of the pivot table.
    columns : str
        Column name to use as columns of the pivot table.
    axes : matplotlib.Axes or list(matplotlib.Axes), optional
        Axe or list of Axes where results will be plotted. If None,
        they are created iteratively. By default, None.
    aggfunc : str or callable, optional
        Aggregation function applied when multiple values exist for
        a given (index, column) pair. By default "mean".
    figsize : tuple of float, optional
        Size of the matplotlib figure, by default None.
    ncols : int, optional
        Maximum number of heatmaps per row, by default 3.
    heatmap_kws : dict or dict[dict], optional
        Keyword arguments passed to :data:`seaborn.heatmap`.
        If dict of dict, keys must match the values in :data:`values`.
    notations : dict, optional
        Mapping from variable names to readable labels
        (used for axis and colorbar labels).
    best_values : dict, optional
        Mapping from variable names to selection criterion:
        "min", "min_abs", or "max".
    plot_best_value : {"point", "text"}, optional
        How to highlight best values:
        - "point" : mark with circles
        - "text" : display numeric values
        By default "point".
    text_kwargs : dict, optional
        Keyword arguments passed to :data:`matplotlib.axes.Axes.text` when
        annotating best values. Only used if :data:`plot_best_value="text"`.

    Returns
    -------
    matplotlib.figure.Figure
        The matplotlib Figure containing the heatmaps.
    """
    nplots = len(values)
    add_axs = axs is None
    if add_axs:
        ncols = min(nplots, ncols)
        nrows = int(np.ceil(nplots / ncols))
        fig = plt.figure(figsize=figsize)
        axes = []
    else:
        axes = axs
        fig = axs.flat[0].figure

    for i in range(nplots):
        if add_axs:
            axe = fig.add_subplot(nrows, ncols, i + 1)
            axes.append(axe)
        else:
            axe = axes.flat[i]
        data = df.pivot_table(
            index=index, columns=columns, values=values[i], aggfunc=aggfunc
        ).astype(float)
        if heatmap_kws is None:
            kws = dict()
        elif isinstance(heatmap_kws, dict):
            if values[i] in heatmap_kws:
                kws = heatmap_kws[values[i]].copy()
            else:
                kws = heatmap_kws.copy()

        if "cmap" not in kws:
            minval = data.min().min()
            maxval = data.max().max()
            if minval * maxval < 0:
                val = max(np.abs(minval), maxval)
                kws["cmap"] = "seismic"
                kws["vmin"] = -val
                kws["vmax"] = val

        if "cbar_kws" not in kws:
            kws["cbar_kws"] = dict(pad=0.03)

        if notations is None:
            kws["cbar_kws"]["label"] = values[i]
        else:
            if values[i] in notations:
                kws["cbar_kws"]["label"] = notations[values[i]]
            else:
                kws["cbar_kws"]["label"] = values[i]

        sns.heatmap(data, ax=axe, **kws)

        if best_values is not None:
            best_value = best_values[values[i]]
            array = np.array(data)
            irow = np.arange(data.shape[0])

            if best_value == "min":
                ind = np.nanargmin(array, axis=1)
                i2 = np.nanargmin(array[irow, ind])
            if best_value == "min_abs":
                ind = np.nanargmin(np.abs(array), axis=1)
                i2 = np.nanargmin(np.abs(array[irow, ind]))
            elif best_value == "max":
                ind = np.nanargmax(array, axis=1)
                i2 = np.nanargmax(array[irow, ind])

            if plot_best_value == "point":
                axe.plot(
                    ind + 0.5,
                    irow + 0.5,
                    ls="",
                    marker="o",
                    mfc="w",
                    mec="k",
                    mew=0.5,
                    ms=5,
                )
                axe.plot(
                    ind[i2] + 0.5,
                    i2 + 0.5,
                    ls="",
                    marker="o",
                    mfc="w",
                    mec="k",
                    mew=0.8,
                    ms=9,
                )
            elif plot_best_value == "text":
                indx = list(ind)
                indx.pop(i2)
                indy = list(irow)
                indy.pop(i2)
                default_kwargs = dict(ha="center", va="center", fontsize=8)
                if text_kwargs is None:
                    text_kwargs = default_kwargs
                else:
                    text_kwargs = dict(default_kwargs, **text_kwargs)
                for i, j in zip(indx, indy):
                    axe.text(
                        i + 0.5, j + 0.5, "{:.2g}".format(array[j, i]), **text_kwargs
                    )
                text_kwargs2 = dict(text_kwargs, fontweight="bold")
                axe.text(
                    ind[i2] + 0.5,
                    i2 + 0.5,
                    "{:.2g}".format(array[i2, ind[i2]]),
                    **text_kwargs2,
                )

    if add_axs:
        axes = np.array(axes).reshape((nrows, ncols))
        for i in range(nrows):
            for j in range(1, ncols):
                axes[i, j].set_ylabel("")
                # axes[i, j].set_yticklabels([])

        for i in range(nrows - 1):
            for j in range(ncols):
                axes[i, j].set_xlabel("")
                # axes[i, j].set_xticklabels([])

    if notations is not None:
        if add_axs:
            for i in range(nrows):
                axes[i, 0].set_ylabel(notations[index])
            for j in range(ncols):
                axes[-1, j].set_xlabel(notations[columns])
        else:
            for ax in axes.flatten():
                ax.set_ylabel(notations[index])
                ax.set_xlabel(notations[columns])

    # fig.tight_layout()

    return fig, axes


def plot_shaltop_mus_calibrated(
    data: pd.DataFrame = None,
    publication_date: str = None,
    plot_lucas_law: bool = True,
    step_delta: float = 5,
    ax: matplotlib.axes._axes.Axes = None,
    figsize: tuple = None,
) -> matplotlib.axes._axes.Axes:
    """Plot friction coefficient calibrated with Shaltop

    Plot the friction coefficient calibrated with Shaltop as a function of landslide
    volume, with seaborn.scatterplot. The data used to create the plot is downloaded from zenodo if not given

    Parameters
    ----------
    data : pd.DataFrame
        pd.DataFrame containing the calibrated values. If not provided, the data is downloaded
        from zenodo.
    publication_date : str
        string YYYY-MM-DD with the date for the data version
    plot_lucas_law : bool
        Plot the lucas law mu=V**(-0.0774) with associated confidence interval
    step_delta : float
        Step between delta angles on the right axis. Default is 5.
    ax : matplotlib.axes._axes.Axes
        Axe instance where plot must be done. If None, an axe instance is created. Deafault is None.
    figsize : tuple
        If ax is None, size of the figure to be created in inches. Default is None.
    """

    if data is None:
        data, publication_date = tilupy.download_data.import_shaltop_mus_calibrated()
    # Default plot options, are updated with provided kwargs
    plot_kw = dict(data=data, x="volume (m3)", y="mus", hue="calibration data")

    if ax is None:
        _, ax = plt.subplots(figsize=figsize)
    ax.grid(axis="both", which="both", zorder=-1)
    sns.scatterplot(**plot_kw, ax=ax, zorder=1)
    ax.set_xscale("log")
    # ax.set_yscale('log')

    # Decorate
    ax.set_xlabel("Volume (m$^3$)")
    ax.set_ylabel("$\\mu_S$")

    if publication_date is not None:
        ax.set_title("10.5281/zenodo.18791118, " + publication_date)

    if plot_lucas_law:
        vmin, vmax = ax.get_xlim()
        v = np.linspace(np.log10(vmin), np.log10(vmax))
        v = 10**v
        mu_lucas = v ** (-0.0774)
        mu_inf = mu_lucas * 10 ** (0.0667 * 1.96)
        mu_sup = mu_lucas * 10 ** (-0.0667 * 1.96)
        ax.fill_between(
            v,
            mu_inf,
            mu_sup,
            edgecolor=[0, 0, 0, 0],
            facecolor=[0.5, 0.5, 0.5, 0.3],
            zorder=0.5,
        )
        ax.plot(v, mu_lucas, "-", color="grey", zorder=0.5)
        ax.set_xlim(vmin, vmax)

    # Add corresponding deltas values on the right y axis
    mu_min, mu_max = ax.get_ylim()
    delta_min = np.rad2deg(np.arctan(mu_min))
    delta_min = step_delta * np.ceil(delta_min / step_delta)
    delta_max = np.rad2deg(np.arctan(mu_max))
    delta_max = step_delta * np.floor(delta_max / step_delta)
    ax2 = ax.twinx()
    ax2.set_ylim([mu_min, mu_max])
    ax2.grid(False)
    deltas = np.arange(delta_min, delta_max + step_delta / 2, step_delta)
    mu_deltas = np.tan(np.deg2rad(deltas))
    ax2.set_yticks(mu_deltas)
    ax2.set_yticklabels(["{:.0f}".format(delta) for delta in deltas])
    ax2.set_ylabel("$\\delta$ (°)")

    return ax
