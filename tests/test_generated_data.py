# -*- coding: utf-8 -*-
"""
Created on Fri Aug  4 12:45:15 2023

@author: peruzzetto
"""

import tilupy.make_topo


def test_topo_gray():

    x, y, z = tilupy.make_topo.gray99()

    assert (y[-1] == 0.5) & (len(x) == 361)
