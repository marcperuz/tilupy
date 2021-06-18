#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 26 14:41:54 2021

@author: peruzzetto
"""

import os
import importlib

import swmb.initdata
import swmb.models

folder_benchmark = '../simus'

# %% Make simus for simple plane

# Prepare folders
folder_topo = os.path.join(folder_benchmark, 'slope_10deg')

subfolders = [os.path.join(folder_topo, 'h_min_1em3'),
              os.path.join(folder_topo, 'h_min_1em15')]

# %% Make initial topograhies and initial mass
swmb.initdata.make_constant_slope(folder_topo)

# %% Prepare simulations for different codes
for code in ['shaltop', 'ravaflow']:
    for folder in subfolders:
        readme_file = os.path.join(folder, 'README.txt')
        folder_out = os.path.join(folder, code)
        os.makedirs(folder_out, exist_ok=True)
        module = importlib.import_module('swmb.models.'+code+'.initsimus')
        module.make_simus('coulomb', dict(delta1=[15, 20, 25]), folder_topo,
                          folder_out, readme_file)
