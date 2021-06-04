#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun  3 13:04:31 2021

@author: peruzzetto
"""

import platform
import os

from comp import compare

if platform.system() == 'Linux':
    folder_base = '/media/peruzzetto/SSD'
elif platform.system() == 'Windows':
    folder_base = 'F:/'

folder_benchmark = os.path.join(folder_base, 'shaltop/benchmark')

# %% Compare results for constant slope

compare.compare_simus('slope_10deg', 'coulomb', {'delta1': 25},
                      'h', 'max',
                      ['shaltop', 'ravaflow'],
                      folder_benchmark=folder_benchmark,
                      folder_out=None,
                      import_kwargs={'from_file': True})
