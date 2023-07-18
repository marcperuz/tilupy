#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun  2 17:57:31 2021

@author: peruzzetto
"""

import importlib
import os
import string
import numpy as np
import matplotlib.pyplot as plt
import mpl_toolkits.axes_grid1 as mplt

import tilupy.dem
import tilupy.plot as plt_fn
import tilupy.notations
import tilupy

from tilupy.notations import LABELS
from tilupy.read import STATIC_DATA_2D


def compare_spatial_results(results, name, stat, import_kwargs=None,
                            x=None, y=None, zinit=None,
                            nrows_ncols=None, figsize=None,
                            folder_out=None, file_prefix=None,
                            fmt='png', dpi=150,
                            cbar_location='bottom', cbar_size="5%",
                            cmap_intervals=None, cmap='inferno',
                            topo_kwargs=None,
                            title_loc='top right', title_x_offset=0.02,
                            title_y_offset=0.02):
    """
    Compare spatiale results of different simulations

    Parameters
    ----------
    results : TYPE
        DESCRIPTION.
    name : TYPE
        DESCRIPTION.
    import_kwargs : TYPE, optional
        DESCRIPTION. The default is None.
    x : TYPE, optional
        DESCRIPTION. The default is None.
    y : TYPE, optional
        DESCRIPTION. The default is None.
    zinit : TYPE, optional
        DESCRIPTION. The default is None.
    figsize : TYPE, optional
        DESCRIPTION. The default is None.
    folder_out : TYPE, optional
        DESCRIPTION. The default is None.
    cbar_location : TYPE, optional
        DESCRIPTION. The default is 'bottom'.
    cbar_size : TYPE, optional
        DESCRIPTION. The default is "5%".
    cmap_intervals : TYPE, optional
        DESCRIPTION. The default is None.
    cmap : TYPE, optional
        DESCRIPTION. The default is 'inferno'.
    topo_kwargs : TYPE, optional
        DESCRIPTION. The default is None.

    Returns
    -------
    None.

    """
    if x is None or y is None:
        results[0].set_axes()
        x = results[0].x
        y = results[0].y

    if zinit is None:
        results[0].set_zinit()
        zinit = results[0].zinit

    topo_kwargs = {} if topo_kwargs is None else topo_kwargs
    import_kwargs = {} if import_kwargs is None else import_kwargs
    nplots = len(results)
    if nrows_ncols is None:
        nrows_ncols = (1, nplots)

    fig = plt.figure(figsize=figsize)
    axes = mplt.AxesGrid(fig, 111,
                         nrows_ncols=nrows_ncols,
                         axes_pad=0.1,
                         share_all=True,
                         aspect='True',
                         label_mode='L',
                         cbar_location=cbar_location,
                         cbar_mode="single",
                         cbar_pad="12%",
                         cbar_size=cbar_size)

    # Titles
    alignments = title_loc.split(' ')
    ha = alignments[1]
    va = alignments[0]
    if va == 'top':
        xt = 1-title_x_offset
    elif va == 'center':
        xt = 0.5
    elif va == 'bottom':
        xt = title_x_offset
    if ha == 'right':
        yt = 1-title_y_offset
    elif ha == 'center':
        yt = 0.5
    elif ha == 'bottom':
        yt = title_y_offset

    for i, result in enumerate(results):
        data = result.get_static_output(name, stat, **import_kwargs)
        plt_fn.plot_data_on_topo(x, y, zinit, data.d,
                                 axe=axes[i],
                                 plot_colorbar=False,
                                 colorbar_kwargs={'extend': 'max'},
                                 cmap_intervals=cmap_intervals,
                                 cmap=cmap,
                                 topo_kwargs=topo_kwargs)
        letter = '({:s})'.format(string.ascii_lowercase[i])
        axes[i].text(xt, yt, '{:s} {:s}'.format(letter, result.code),
                     ha=ha, va=va,
                     fontsize=11, zorder=100,
                     bbox=dict(boxstyle="round",
                               fc="w", ec="k", pad=0.2),
                     transform=axes[i].transAxes)
        axes[i].grid(False)

    # Colorbar
    axe_cc = axes.cbar_axes[0]
    if cbar_location in ['right', 'left']:
        orientation = 'vertical'
    else:
        orientation = 'horizontal'
    fig.colorbar(axes[0].images[-1],
                 cax=axe_cc,
                 ax=axes[0],
                 orientation=orientation,
                 extend='max')
    if orientation == 'vertical':
        axe_cc.set_ylabel(LABELS[name+stat])
    else:
        axe_cc.set_xlabel(LABELS[name+stat])

    fig.tight_layout()

    if folder_out is not None:
        if file_prefix is None:
            file = name + stat + '.' + fmt
        else:
            file = file_prefix + '_' + name + stat + '.' + fmt
        fig.savefig(os.path.join(folder_out, file),
                    dpi=dpi)


def compare_simus(codes, topo_name, law,
                  rheol_params, output_name, stat_name,
                  subfolder_topo='',
                  folder_benchmark=None, folder_out=None, **kwargs):
    """Compare simulations with different codes."""

    modules = dict()
    for c in codes:
        modules[c] = importlib.import_module('tilupy.models.'+c+'.read')

    if folder_benchmark is None:
        folder_benchmark = os.getcwd()

    folder_topo = os.path.join(folder_benchmark, topo_name)

    txt_params = tilupy.notations.make_rheol_string(rheol_params, law)

    if folder_out is None:
        folder_out = os.path.join(folder_topo, subfolder_topo,
                                  'comparison', law)
        os.makedirs(folder_out, exist_ok=True)

    file = np.os.path.join(folder_topo, 'topo.asc')
    x, y, zinit, dx = tilupy.dem.read_ascii(file)

    simus = []
    for code in codes:
        folder_simus = os.path.join(folder_topo, subfolder_topo, code, law)
        simus.append(modules[code].Results(txt_params, folder_simus))

    if output_name+stat_name in STATIC_DATA_2D:
        compare_spatial_results(simus, output_name, stat_name,
                                x=x, y=y, zinit=zinit,
                                folder_out=folder_out, file_prefix=txt_params,
                                **kwargs)
