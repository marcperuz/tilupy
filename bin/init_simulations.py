#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 26 14:41:54 2021

@author: peruzzetto
"""

import os
import platform
import importlib

from swmb.init import init_benchmark
from swmb.init import init_shaltop
from swmb.init import init_ravaflow

folder_benchmark = '../simus'

folder_slope = os.path.join(folder_benchmark, 'slope_10deg')
os.makedirs(folder_slope, exist_ok=True)

# %% Make initial topograhies and initial mass
init_benchmark.make_constant_slope(folder_slope)

# %% Prepare simulations for different codes
for code in ['shaltop', 'ravaflow']:
    folder_out = os.path.join(folder_slope, code)
    os.makedirs(folder_out, exist_ok=True)
    module = importlib.import_module('swmb.init.init_'+code)
    module.make_simus('coulomb', dict(delta1=[15, 20, 25]), folder_slope,
                      folder_out)
