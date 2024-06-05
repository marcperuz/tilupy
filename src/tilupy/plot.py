#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun  2 16:16:39 2021

@author: peruzzetto
"""

import matplotlib
import copy
import os

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import seaborn as sns

BOLD_CONTOURS_INTV = [
    0.1,
    0.2,
    0.5,
    1,
    2.0,
    5,
    10,
    20,
    50,
    100,
    200,
    500,
    1000,
]
NB_THIN_CONTOURS = 10
NB_BOLD_CONTOURS = 3


def centered_map(cmap, vmin, vmax, ncolors=256):
    """
    Create centered colormap

    Parameters
    ----------
    cmap : TYPE
        DESCRIPTION.
    vmin : TYPE
        DESCRIPTION.
    vmax : TYPE
        DESCRIPTION.
    ncolors : TYPE, optional
        DESCRIPTION. The default is 256.

    Returns
    -------
    new_map : TYPE
        DESCRIPTION.

    """
    p = vmax / (vmax - vmin)
    npos = int(ncolors * p)
    method = getattr(plt.cm, cmap)

    colors1 = method(np.linspace(0.0, 1, npos * 2))
    colors2 = method(np.linspace(0.0, 1, (ncolors - npos) * 2))
    colors = np.concatenate(
        (colors2[: ncolors - npos, :], colors1[npos:, :]), axis=0
    )
    # colors[ncolors-npos-1,:]=np.ones((1,4))
    # colors[ncolors-npos,:]=np.ones((1,4))
    new_map = mcolors.LinearSegmentedColormap.from_list("my_colormap", colors)

    return new_map


def get_contour_intervals(
    zmin, zmax, nb_bold_contours=None, nb_thin_contours=None
):
    if nb_thin_contours is None:
        nb_thin_contours = NB_THIN_CONTOURS
    if nb_bold_contours is None:
        nb_bold_contours = NB_BOLD_CONTOURS

    intv = (zmax - zmin) / nb_bold_contours
    i = np.argmin(np.abs(np.array(BOLD_CONTOURS_INTV) - intv))

    bold_intv = BOLD_CONTOURS_INTV[i]
    if BOLD_CONTOURS_INTV[i] != BOLD_CONTOURS_INTV[0]:
        if bold_intv - intv > 0:
            bold_intv = BOLD_CONTOURS_INTV[i - 1]

    if nb_thin_contours is None:
        thin_intv = bold_intv / NB_THIN_CONTOURS
        if (zmax - zmin) / bold_intv > 5:
            thin_intv = thin_intv * 2
    else:
        thin_intv = bold_intv / nb_thin_contours

    return bold_intv, thin_intv


def auto_uniform_grey(
    z, vert_exag, azdeg=315, altdeg=45, dx=1, dy=1, std_threshold=0.01
):
    """
    Detect if shading must be applied to topography or not (uniform grey). The
    criterion in colors.LightSource.hillshade is the difference between min
    and max illumination, and seems to restrictive.
    :param z: DESCRIPTION
    :type z: TYPE
    :param vert_exag: DESCRIPTION
    :type vert_exag: TYPE
    :param dx: DESCRIPTION, defaults to 1
    :type dx: TYPE, optional
    :param dy: DESCRIPTION, defaults to 1
    :type dy: TYPE, optional
    :return: DESCRIPTION
    :rtype: TYPE

    """

    # Get topography normal direction
    e_dy, e_dx = np.gradient(vert_exag * z, dy, dx)
    normal = np.empty(z.shape + (3,)).view(type(z))
    normal[..., 0] = -e_dx
    normal[..., 1] = -e_dy
    normal[..., 2] = 1
    sum_sq = 0
    for i in range(normal.shape[-1]):
        sum_sq += normal[..., i, np.newaxis] ** 2
    normal /= np.sqrt(sum_sq)

    # Light source direction
    az = np.radians(90 - azdeg)
    alt = np.radians(altdeg)
    direction = np.array(
        [np.cos(az) * np.cos(alt), np.sin(az) * np.cos(alt), np.sin(alt)]
    )

    # Compute intensity (equivalent to LightSource hillshade, whithour rescaling)
    intensity = normal.dot(direction)
    std = np.std(intensity)

    if std > std_threshold:
        return None
    else:
        return 0.5


def plot_topo(
    z,
    x,
    y,
    contour_step=None,
    nlevels=None,
    level_min=None,
    step_contour_bold="auto",
    contour_labels_properties=None,
    label_contour=True,
    contour_label_effect=None,
    axe=None,
    vert_exag=1,
    fraction=1,
    ndv=-9999,
    uniform_grey="auto",
    contours_prop=None,
    contours_bold_prop=None,
    figsize=None,
    interpolation=None,
    sea_level=0,
    sea_color=None,
    alpha=1,
    azdeg=315,
    altdeg=45,
    zmin=None,
    zmax=None,
):
    """
    Plot topography with hillshading.

    Parameters
    ----------
    z : TYPE
        DESCRIPTION.
    x : TYPE
        DESCRIPTION.
    y : TYPE
        DESCRIPTION.
    contour_step : TYPE, optional
        DESCRIPTION. The default is None.
    nlevels : TYPE, optional
        DESCRIPTION. The default is 25.
    contour_labels_properties : TYPE, optional
        DESCRIPTION. The default is None.
    axe : TYPE, optional
        DESCRIPTION. The default is None.
    vert_exag : TYPE, optional
        DESCRIPTION. The default is 1.
    contour_color : TYPE, optional
        DESCRIPTION. The default is 'k'.
    contour_alpha : TYPE, optional
        DESCRIPTION. The default is 1.
    figsize : TYPE, optional
        DESCRIPTION. The default is (10,10).
    interpolation : TYPE, optional
        DESCRIPTION. The default is None.
    sea_level : TYPE, optional
        DESCRIPTION. The default is 0.
    sea_color : TYPE, optional
        DESCRIPTION. The default is None.
    alpha : TYPE, optional
        DESCRIPTION. The default is 1.
    azdeg : TYPE, optional
        DESCRIPTION. The default is 315.
    altdeg : TYPE, optional
        DESCRIPTION. The default is 45.
    plot_terrain : TYPE, optional
        DESCRIPTION. The default is False.
    cmap_terrain : TYPE, optional
        DESCRIPTION. The default is 'gist_earth'.
    fraction : TYPE, optional
        DESCRIPTION. The default is 1.
    ndv : TYPE, optional
        DESCRIPTION. The default is -9999.

    Returns
    -------
    None.

    """
    dx = x[1] - x[0]
    dy = y[1] - y[0]
    im_extent = [x[0] - dx / 2, x[-1] + dx / 2, y[0] - dy / 2, y[-1] + dy / 2]
    ls = mcolors.LightSource(azdeg=azdeg, altdeg=altdeg)

    auto_bold_intv = None

    if nlevels is None and contour_step is None:
        auto_bold_intv, contour_step = get_contour_intervals(
            np.nanmin(z), np.nanmax(z)
        )

    if level_min is None:
        if contour_step is not None:
            level_min = np.ceil(np.nanmin(z) / contour_step) * contour_step
        else:
            level_min = np.nanmin(z)
    if contour_step is not None:
        levels = np.arange(level_min, np.nanmax(z), contour_step)
    else:
        levels = np.linspace(level_min, np.nanmax(z), nlevels)

    if axe is None:
        fig, axe = plt.subplots(1, 1, figsize=figsize, layout="constrained")

    axe.set_ylabel("Y (m)")
    axe.set_xlabel("X (m)")
    axe.set_aspect("equal")

    if uniform_grey == "auto":
        uniform_grey = auto_uniform_grey(
            z,
            vert_exag,
            azdeg=azdeg,
            altdeg=altdeg,
            dx=dx,
            dy=dx,
        )

    if uniform_grey is None:
        shaded_topo = ls.hillshade(
            z, vert_exag=vert_exag, dx=dx, dy=dy, fraction=1
        )
    else:
        shaded_topo = np.ones(z.shape) * uniform_grey
    shaded_topo[z == ndv] = np.nan
    axe.imshow(
        shaded_topo,
        cmap="gray",
        extent=im_extent,
        interpolation=interpolation,
        alpha=alpha,
        vmin=0,
        vmax=1,
    )

    if contours_prop is None:
        contours_prop = dict(alpha=0.5, colors="k", linewidths=0.5)
    axe.contour(
        x,
        y,
        np.flip(z, axis=0),
        extent=im_extent,
        levels=levels,
        **contours_prop
    )

    if contours_bold_prop is None:
        contours_bold_prop = dict(alpha=0.8, colors="k", linewidths=0.8)

    if step_contour_bold == "auto":
        if auto_bold_intv is None:
            auto_bold_intv, _ = get_contour_intervals(
                np.nanmin(z), np.nanmax(z)
            )
        step_contour_bold = auto_bold_intv

    if step_contour_bold > 0:
        lmin = np.ceil(np.nanmin(z) / step_contour_bold) * step_contour_bold
        if lmin < level_min:
            lmin = lmin + step_contour_bold
        levels = np.arange(lmin, np.nanmax(z), step_contour_bold)
        cs = axe.contour(
            x,
            y,
            np.flip(z, axis=0),
            extent=im_extent,
            levels=levels,
            **contours_bold_prop
        )
        if label_contour:
            if contour_labels_properties is None:
                contour_labels_properties = {}
            clbls = axe.clabel(cs, **contour_labels_properties)
            if contour_label_effect is not None:
                plt.setp(clbls, path_effects=contour_label_effect)

    if sea_color is not None:
        cmap_sea = mcolors.ListedColormap([sea_color])
        cmap_sea.set_under(color="w", alpha=0)
        mask_sea = (z <= sea_level) * 1
        if mask_sea.any():
            axe.imshow(
                mask_sea,
                extent=im_extent,
                cmap=cmap_sea,
                vmin=0.5,
                origin="lower",
                interpolation="none",
            )


def plot_imshow(
    x,
    y,
    data,
    axe=None,
    figsize=None,
    cmap=None,
    minval=None,
    maxval=None,
    vmin=None,
    vmax=None,
    alpha=1,
    minval_abs=None,
    cmap_intervals=None,
    extend_cc="max",
    plot_colorbar=True,
    axecc=None,
    colorbar_kwargs=None,
    aspect=None,
):
    """
    plt.imshow data with some pre-processing

    Parameters
    ----------
    x : TYPE
        DESCRIPTION.
    y : TYPE
        DESCRIPTION.
    data : TYPE
        DESCRIPTION.
    axe : TYPE, optional
        DESCRIPTION. The default is None.
    figsize : TYPE, optional
        DESCRIPTION. The default is None.
    cmap : TYPE, optional
        DESCRIPTION. The default is None.
    minval : TYPE, optional
        DESCRIPTION. The default is None.
    maxval : TYPE, optional
        DESCRIPTION. The default is None.
    vmin : TYPE, optional
        DESCRIPTION. The default is None.
    vmax : TYPE, optional
        DESCRIPTION. The default is None.
    minval_abs : TYPE, optional
        DESCRIPTION. The default is None.
    cmap_intervals : TYPE, optional
        DESCRIPTION. The default is None.
    extend_cc : TYPE, optional
        DESCRIPTION. The default is "max".

    Returns
    -------
    None.

    """
    if axe is None:
        _, axe = plt.subplots(1, 1, figsize=figsize, layout="constrained")

    f = copy.copy(data)

    # vmin and vmax are similar to minval and maxval
    # and supplent minval and maxval if used
    if vmin is not None:
        minval = vmin

    if vmax is not None:
        maxval = vmax

    # Remove values below and above minval and maxval, depending on whether
    # cmap_intervals are given with or without extend_cc
    if cmap_intervals is not None:
        norm = matplotlib.colors.BoundaryNorm(
            cmap_intervals, 256, extend=extend_cc
        )
        if extend_cc in ["neither", "max"]:
            minval = cmap_intervals[0]
            f[f < minval] = np.nan
        elif extend_cc in ["neither", "min"]:
            maxval = cmap_intervals[-1]
            f[f > maxval] = np.nan
    else:
        norm = None
        # if maxval is not None:
        #     f[f > maxval] = np.nan
        if minval is not None:
            f[f < minval] = np.nan

    # Get min and max values
    if maxval is None:
        maxval = np.nanmax(f)
    if minval is None:
        minval = np.nanmin(f)

    if minval_abs:
        f[np.abs(f) <= minval_abs] = np.nan
    else:
        f[f == 0] = np.nan

    # Define colormap type
    if cmap is None:
        if maxval * minval >= 0:
            cmap = "hot_r"
        else:
            cmap = "seismic"
    if (maxval * minval >= 0) or np.isnan(maxval * minval):
        color_map = matplotlib.colormaps[cmap]
    else:
        color_map = centered_map(cmap, minval, maxval)

    if cmap_intervals is not None:
        norm = matplotlib.colors.BoundaryNorm(
            cmap_intervals, 256, extend=extend_cc
        )
        maxval = None
        minval = None
    else:
        norm = None

    # get map_extent
    dx = x[1] - x[0]
    dy = y[1] - y[0]
    im_extent = [x[0] - dx / 2, x[-1] + dx / 2, y[0] - dy / 2, y[-1] + dy / 2]

    # Plot data

    fim = axe.imshow(
        f,
        extent=im_extent,
        cmap=color_map,
        vmin=minval,
        vmax=maxval,
        alpha=alpha,
        interpolation="none",
        norm=norm,
        zorder=4,
        aspect=aspect,
    )

    # Plot colorbar
    if plot_colorbar:
        colorbar_kwargs = {} if colorbar_kwargs is None else colorbar_kwargs
        if cmap_intervals is not None and extend_cc is not None:
            colorbar_kwargs["extend"] = extend_cc
        axe.figure.colorbar(fim, cax=axecc, **colorbar_kwargs)

    return axe


def plot_data_on_topo(
    x,
    y,
    z,
    data,
    axe=None,
    figsize=(15 / 2.54, 15 / 2.54),
    cmap=None,
    minval=None,
    maxval=None,
    vmin=None,
    vmax=None,
    minval_abs=None,
    cmap_intervals=None,
    extend_cc="max",
    topo_kwargs=None,
    sup_plot=None,
    alpha=1,
    plot_colorbar=True,
    axecc=None,
    colorbar_kwargs=None,
    mask=None,
    alpha_mask=None,
    color_mask="k",
    xlims=None,
    ylims=None,
):
    """
    Plot array data on topo.

    Returns
    -------
    None.

    """

    # Initialize figure properties
    dx = x[1] - x[0]
    dy = y[1] - y[0]
    im_extent = [x[0] - dx / 2, x[-1] + dx / 2, y[0] - dy / 2, y[-1] + dy / 2]
    if axe is None:
        fig, axe = plt.subplots(1, 1, figsize=figsize, layout="constrained")

    axe.set_ylabel("Y (m)")
    axe.set_xlabel("X (m)")
    axe.set_aspect("equal", adjustable="box")

    # Plot topo
    topo_kwargs = {} if topo_kwargs is None else topo_kwargs

    if z is not None:
        plot_topo(z, x, y, axe=axe, **topo_kwargs)

    # Plot mask
    if mask is not None:
        cmap_mask = mcolors.ListedColormap([color_mask])
        cmap_mask.set_under(color="w", alpha=0)
        axe.imshow(
            mask.transpose(),
            extent=im_extent,
            cmap=cmap_mask,
            vmin=0.5,
            origin="lower",
            interpolation="none",
            # zorder=3,
            alpha=alpha_mask,
        )

    # Plot data
    plot_imshow(
        x,
        y,
        data,
        axe=axe,
        cmap=cmap,
        minval=minval,
        maxval=maxval,
        vmin=vmin,
        vmax=vmax,
        minval_abs=minval_abs,
        cmap_intervals=cmap_intervals,
        extend_cc=extend_cc,
        plot_colorbar=plot_colorbar,
        axecc=axecc,
        colorbar_kwargs=colorbar_kwargs,
    )

    # Adjust axes limits
    if xlims is not None:
        axe.set_xlim(xlims[0], xlims[1])

    if ylims is not None:
        axe.set_ylim(ylims[0], ylims[1])

    return axe


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
    axe = plot_imshow(x, t[::-1], data.T, **kwargs)
    axe.set_adjustable("box")
    if ylabel is None:
        ylabel = "Time (s)"
    axe.set_ylabel(ylabel)
    axe.set_xlabel(xlabel)

    return axe


def plot_maps(
    x,
    y,
    z,
    data,
    t,
    file_name=None,
    folder_out=None,
    figsize=None,
    dpi=None,
    fmt="png",
    sup_plt_fn=None,
    sup_plt_fn_args=None,
    **kwargs
):
    """
    Plot and save maps of simulations outputs at successive time steps

    Parameters
    ----------
    x : TYPE
        DESCRIPTION.
    y : TYPE
        DESCRIPTION.
    data : TYPE
        DESCRIPTION.
    t : TYPE
        DESCRIPTION.
    file_name : TYPE
        DESCRIPTION.
    folder_out : TYPE
        DESCRIPTION.
    dpi : TYPE, optional
        DESCRIPTION. The default is None.
    fmt : TYPE, optional
        DESCRIPTION. The default is 'png'.

    Returns
    -------
    None.

    """

    nfigs = len(t)
    if nfigs != data.shape[2]:
        raise ValueError(
            "length of t must be similar to the last dimension of data"
        )
    if folder_out is not None:
        file_path = os.path.join(folder_out, file_name + "_{:04d}." + fmt)
    title_fmt = "t = {:.2f} s"

    for i in range(nfigs):
        axe = plot_data_on_topo(
            x, y, z, data[:, :, i], axe=None, figsize=figsize, **kwargs
        )
        axe.set_title(title_fmt.format(t[i]))
        if sup_plt_fn is not None:
            if sup_plt_fn_args is None:
                sup_plt_fn_args = dict()
            sup_plt_fn(axe, **sup_plt_fn_args)
        # axe.figure.tight_layout(pad=0.1)
        if folder_out is not None:
            axe.figure.savefig(
                file_path.format(i),
                dpi=dpi,
                bbox_inches="tight",
                pad_inches=0.05,
            )


def colorbar(
    mappable, ax=None, cax=None, size="5%", pad=0.1, position="right", **kwargs
):
    """
    Create nice colorbar matching height/width of axe.

    Parameters
    ----------
    mappable : TYPE
        DESCRIPTION.
    cax : TYPE, optional
        DESCRIPTION. The default is None.
    size : TYPE, optional
        DESCRIPTION. The default is "5%".
    pad : TYPE, optional
        DESCRIPTION. The default is 0.05.
    position : TYPE, optional
        DESCRIPTION. The default is 'right'.
    extend : TYPE, optional
        DESCRIPTION. The default is 'neither'.

    Returns
    -------
    TYPE
        DESCRIPTION.

    """
    if ax is None:
        ax = mappable.axes
    fig = ax.figure
    if position in ["left", "right"]:
        orientation = "vertical"
    else:
        orientation = "horizontal"

    # if cax is None:
    #     # divider = ax.get_axes_locator()
    #     # if divider is None:
    #     divider = make_axes_locatable(ax)
    #     cax = divider.append_axes(position, size=size, pad=pad)

    cc = fig.colorbar(mappable, cax=cax, orientation=orientation, **kwargs)

    if position == "top":
        cax.xaxis.tick_top()
        cax.xaxis.set_label_position("top")
    if position == "left":
        cax.yaxis.tick_left()
        cax.xaxis.set_label_position("left")
    return cc


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
