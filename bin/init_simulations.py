#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 26 14:41:54 2021

@author: peruzzetto
"""

import os
import platform

import init_benchmark
import init_shaltop
import init_ravaflow

if platform.system() == 'Linux':
    folder_base = '/media/peruzzetto/SSD'
elif platform.system() == 'Windows':
    folder_base = 'F:/'

folder_benchmark = os.path.join(folder_base, 'shaltop/benchmark')

folder_slope = os.path.join(folder_benchmark, 'slope_10deg')

#%% Make initial topograhies and initial mass
init_benchmark.make_constant_slope(folder_slope)

#%% Prepare simulations for different codes
init_shaltop.make_simus('coulomb', dict(delta1=[15, 20, 25]), folder_slope,
                        os.path.join(folder_slope, 'shaltop'))
# init_ravaflow.make_simus('coulomb', dict(delta1=[15, 20, 25]), folder_slope,
#                          os.path.join(folder_slope, 'ravaflow'))