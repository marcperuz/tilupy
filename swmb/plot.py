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

from mpl_toolkits.axes_grid1 import make_axes_locatable


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
    p = vmax/(vmax-vmin)
    npos = int(ncolors*p)
    method = getattr(plt.cm, cmap)

    colors1 = method(np.linspace(0., 1, npos*2))
    colors2 = method(np.linspace(0., 1, (ncolors-npos)*2))
    colors = np.concatenate(
        (colors2[:ncolors-npos, :], colors1[npos:, :]), axis=0)
    # colors[ncolors-npos-1,:]=np.ones((1,4))
    # colors[ncolors-npos,:]=np.ones((1,4))
    new_map = mcolors.LinearSegmentedColormap.from_list(
        'my_colormap', colors)

    return new_map


def plot_topo(z, x, y, contour_step=None, nlevels=25, level_min=None,
              step_contour_bold=0, contour_labels_properties=None,
              label_contour=True, contour_label_effect=None,
              axe=None,
              vert_exag=1, fraction=1, ndv=0, uniform_grey=None,
              contours_prop=None, contours_bold_prop=None,
              figsize=(10, 10),
              interpolation=None,
              sea_level=0, sea_color=None, alpha=1, azdeg=315, altdeg=45,
              zmin=None, zmax=None):
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
        DESCRIPTION. The default is 0.

    Returns
    -------
    None.

    """
    dx = x[1] - x[0]
    dy = y[1] - y[0]
    im_extent = [x[0]-dx/2,
                 x[-1]+dx/2,
                 y[0]-dy/2,
                 y[-1]+dy/2]
    ls = mcolors.LightSource(azdeg=azdeg, altdeg=altdeg)

    if level_min is None:
        if contour_step is not None:
            level_min = np.ceil(np.nanmin(z)/contour_step)*contour_step
        else:
            level_min = np.nanmin(z)
    if contour_step is not None:
        levels = np.arange(level_min, np.nanmax(z), contour_step)
    else:
        levels = np.linspace(level_min, np.nanmax(z), nlevels)

    if axe is None:
        fig = plt.figure(figsize=figsize)
        axe = fig.gca()
    else:
        fig = axe.figure
    axe.set_ylabel('Y (m)')
    axe.set_xlabel('X (m)')
    axe.set_aspect('equal')

    if uniform_grey is None:
        shaded_topo = ls.hillshade(z, vert_exag=vert_exag, dx=dx, dy=dy,
                                   fraction=1)
    else:
        shaded_topo = np.ones(z.shape)*uniform_grey
    axe.imshow(shaded_topo, cmap='gray', origin='lower', extent=im_extent,
               interpolation=interpolation, alpha=alpha, vmin=0, vmax=1)

    if contours_prop is None:
        contours_prop = dict(alpha=0.5, colors='k',
                             linewidths=0.5)
    axe.contour(z, extent=im_extent,
                levels=levels,
                **contours_prop)

    if contours_bold_prop is None:
        contours_bold_prop = dict(alpha=0.8, colors='k',
                                  linewidths=0.8)

    if step_contour_bold > 0:
        lmin = np.ceil(np.nanmin(z)/step_contour_bold)*step_contour_bold
        levels = np.arange(lmin, np.nanmax(z), step_contour_bold)
        cs = axe.contour(z, extent=im_extent,
                         levels=levels,
                         **contours_bold_prop)
        if label_contour:
            if contour_labels_properties is None:
                contour_labels_properties = {}
            clbls = axe.clabel(cs, **contour_labels_properties)
            if contour_label_effect is not None:
                plt.setp(clbls, path_effects=contour_label_effect)

    if sea_color is not None:
        cmap_sea = mcolors.ListedColormap([sea_color])
        cmap_sea.set_under(color='w', alpha=0)
        mask_sea = (z <= sea_level)*1
        if mask_sea.any():
            axe.imshow(mask_sea, extent=im_extent, cmap=cmap_sea,
                       vmin=0.5, origin='lower', interpolation='none')


def plot_data_on_topo(x, y, z, data, axe=None, figsize=(10/2.54, 10/2.54),
                      cmap=None, minval=None, maxval=None, minval_abs=None,
                      cmap_intervals=None, extend_cc='max',
                      topo_kwargs=None, sup_plot=None,
                      plot_colorbar=True, axecc=None, colorbar_kwargs=None,
                      mask=None, alpha_mask=None, color_mask='k'
                      ):
    """
    Plot array data on topo.

    Returns
    -------
    None.

    """
    f = copy.copy(data)

    # Get min and max values
    if maxval is None:
        maxval = f.max()
    if minval is None:
        minval = f.min()

    f[f <= minval] = minval-1
    if minval_abs:
        f[np.abs(f) <= minval_abs] = minval-1
    else:
        f[f == 0] = minval-1

    # Define colormap type
    if cmap is None:
        if maxval*minval >= 0:
            cmap = 'hot_r'
        else:
            cmap = 'seismic'
    if maxval*minval >= 0:
        color_map = matplotlib.cm.get_cmap(cmap).copy()
    else:
        color_map = centered_map(cmap, minval, maxval)

    if cmap_intervals is not None:
        nbounds = len(cmap_intervals)
        cgen = [color_map(1.*i/(nbounds-1)) for i in range(nbounds)]
        if extend_cc == 'max':
            color_map = mcolors.ListedColormap(cgen[:-1])
            color_map.set_over(cgen[-1])
        elif extend_cc == 'min':
            color_map = mcolors.ListedColormap(cgen[1:])
            color_map.set_under(cgen[0])
        elif extend_cc == 'both':
            color_map = mcolors.ListedColormap(cgen[1:-1])
            color_map.set_under(cgen[0])
            color_map.set_over(cgen[-1])
        elif extend_cc == 'neither':
            color_map = mcolors.ListedColormap(cgen)
        norm = mcolors.BoundaryNorm(cmap_intervals, color_map.N)
        maxval = None
        minval = None
    else:
        norm = None
    color_map.set_under([1, 1, 1], alpha=0)

    # Initialize figure properties
    dx = x[1]-x[0]
    dy = y[1]-y[0]
    im_extent = [x[0]-dx/2, x[-1]+dx/2, y[0]-dy/2, y[-1]+dy/2]
    if axe is None:
        fig = plt.figure(figsize=figsize)
        axe = fig.gca()
    else:
        fig = axe.figure
    axe.set_ylabel('Y (m)')
    axe.set_xlabel('X (m)')
    axe.set_aspect('equal', adjustable='box')

    # Plot topo
    topo_kwargs = {} if topo_kwargs is None else topo_kwargs

    plot_topo(z, x, y, axe=axe, **topo_kwargs)

    # Plot mask
    if mask is not None:
        cmap_mask = mcolors.ListedColormap([color_mask])
        cmap_mask.set_under(color='w', alpha=0)
        axe.imshow(mask.transpose(), extent=im_extent, cmap=cmap_mask,
                   vmin=0.5, origin='lower', interpolation='none',
                   zorder=3, alpha=alpha_mask)

    # Plot data
    fim = axe.imshow(f, extent=im_extent, cmap=color_map,
                     vmin=minval, vmax=maxval, origin='lower',
                     interpolation='none', norm=norm, zorder=4)

    # Plot colorbar
    if plot_colorbar:
        colorbar_kwargs = {} if colorbar_kwargs is None else colorbar_kwargs
        cc = colorbar(fim, cax=axecc, **colorbar_kwargs)
        
    return axe
        

def plot_maps(x, y, z, data, t, file_name, folder_out, 
              figsize=None, dpi=None, fmt='png',
              **kwargs):
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
        raise ValueError('length of t must be similar to the last dimension of data')
    file_path = os.path.join(folder_out, file_name + '_{:04d}.' + fmt)
    title_fmt = 't = {:.2f} s'
    
    for i in range(nfigs):
        axe = plot_data_on_topo(x, y, z, data[:, :, i], axe=None,
                                figsize=figsize,
                                **kwargs)
        axe.set_title(title_fmt.format(t[i]))
        axe.figure.savefig(file_path.format(i), dpi=dpi)
        

def colorbar(mappable, ax=None,
             cax=None, size="5%", pad=0.05, position='right',
             **kwargs):
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
    if position in ['left', 'right']:
        orientation = 'vertical'
    else:
        orientation = 'horizontal'
    if cax is None:
        # divider = ax.get_axes_locator()
        # if divider is None:
        divider = make_axes_locatable(ax)
        cax = divider.append_axes(position, size=size, pad=pad)

    cc = fig.colorbar(mappable, cax=cax,
                      orientation=orientation, **kwargs)

    if position == 'top':
        cax.xaxis.tick_top()
        cax.xaxis.set_label_position('top')
    if position == 'left':
        cax.yaxis.tick_left()
        cax.xaxis.set_label_position('left')
    return cc
