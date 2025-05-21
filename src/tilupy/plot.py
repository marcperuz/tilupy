#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun  2 16:16:39 2021

@author: peruzzetto
"""


import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pytopomap.plot as pyplt

def plot_shotgather(x, t, data, xlabel="X (m)", ylabel=None, **kwargs):
    """
    Plot shotgather like image, with vertical axis as time and horizontal axis
    and spatial dimension. This is a simple call to plot_shotgather, but
    input data is transposed because in tilupy the last axis is time by
    convention.

    Parameters
    ----------
    x : NX-array
        spatial coordinates
    t : NT-array
        time array (assumed in seconds)
    data : TYPE
        NX*NT array of data to be plotted
    spatial_label : string, optional
        label for y-axis. The default is "X (m)"
    **kwargs : dict, optional
        parameters passed on to plot_imshow

    Returns
    -------
    axe : Axes
        Axes instance where data is plotted

    """
    if "aspect" not in kwargs:
        kwargs["aspect"] = "auto"
    axe = pyplt.plot_imshow(x, t[::-1], data.T, **kwargs)
    axe.set_adjustable("box")
    if ylabel is None:
        ylabel = "Time (s)"
    axe.set_ylabel(ylabel)
    axe.set_xlabel(xlabel)

    return axe


def plot_heatmaps(
    df,
    values,
    index,
    columns,
    aggfunc="mean",
    figsize=None,
    ncols=3,
    heatmap_kws=None,
    notations=None,
    best_values=None,
    plot_best_value="point",
    text_kwargs=None,
):
    nplots = len(values)
    ncols = min(nplots, ncols)
    nrows = int(np.ceil(nplots / ncols))
    fig = plt.figure(figsize=figsize)
    axes = []

    for i in range(nplots):
        axe = fig.add_subplot(nrows, ncols, i + 1)
        axes.append(axe)
        data = df.pivot_table(
            index=index, columns=columns, values=values[i], aggfunc=aggfunc
        ).astype(float)
        if heatmap_kws is None:
            kws = dict()
        elif isinstance(heatmap_kws, dict):
            if values[i] in heatmap_kws:
                kws = heatmap_kws[values[i]]
            else:
                kws = heatmap_kws

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
                        i + 0.5,
                        j + 0.5,
                        "{:.2g}".format(array[j, i]),
                        **text_kwargs
                    )
                text_kwargs2 = dict(text_kwargs, fontweight="bold")
                axe.text(
                    ind[i2] + 0.5,
                    i2 + 0.5,
                    "{:.2g}".format(array[i2, ind[i2]]),
                    **text_kwargs2
                )

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
        for i in range(nrows):
            axes[i, 0].set_ylabel(notations[index])
        for j in range(ncols):
            axes[-1, j].set_xlabel(notations[columns])

    # fig.tight_layout()

    return fig
