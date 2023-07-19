#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun  3 13:04:31 2021

@author: peruzzetto
"""

import itertools
import os

import tilupy.compare

tmp = os.path.abspath(__file__)
folder_benchmark = os.path.join(tmp.split('bin')[0], 'data')

# %% Compare results for constant slope

cmap_intervals = dict(slope_10deg=dict(coulomb=dict()))
cmap_intervals['slope_10deg']['coulomb'][15] = dict(hmax=[0.1, 0.5, 1, 5,
                                                          10, 25, 40],
                                                    hfinal=[0.1, 0.2, 0.5, 1,
                                                            2, 3, 5])
cmap_intervals['slope_10deg']['coulomb'][20] = dict(hmax=[0.1, 0.5,  1, 5,
                                                          10, 25, 40],
                                                    hfinal=[0.1, 0.2, 0.5, 1,
                                                            2, 5, 10])
cmap_intervals['slope_10deg']['coulomb'][25] = dict(hmax=[0.1, 0.5, 1, 5,
                                                          10, 25, 40],
                                                    hfinal=[0.1, 0.5, 1,
                                                            2, 5, 10, 20])


deltas = [15, 20, 25]
stats = [('max', True), ('final', False)]
topo = 'slope_10deg'
law = 'coulomb'
state_name = 'h'
subfolders = ['h_min_1em3', 'h_min_1em15']
#stats = [('final', False)]

for delta, stat in itertools.product(deltas, stats):
    intervals = cmap_intervals[topo][law][delta][state_name+stat[0]]
    for subfolder in subfolders:
        tilupy.compare.compare_simus(['shaltop', 'ravaflow'],
                                   'slope_10deg', 'coulomb', {'delta1': delta},
                                   'h', stat[0],
                                   folder_benchmark=folder_benchmark,
                                   subfolder_topo=subfolder,
                                   folder_out=None,
                                   import_kwargs={'from_file': stat[1]},
                                   cmap_intervals=intervals,
                                   topo_kwargs={'contour_step': 10},
                                   figsize=(15/2.56, 8.5/2.56),
                                   title_x_offset=0.02,
                                   title_y_offset=0.04,
                                   fmt='png')
