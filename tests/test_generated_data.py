# -*- coding: utf-8 -*-
"""
Created on Fri Aug  4 12:45:15 2023

@author: peruzzetto
"""

import numpy as np

import tilupy.make_topo


def test_topo_gray():

    x, y, z = tilupy.make_topo.gray99()

    assert (np.abs(y[-1] - 0.6) < 1e-6) & (len(x) == 361)
