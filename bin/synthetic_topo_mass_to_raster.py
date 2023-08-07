# -*- coding: utf-8 -*-
"""
Created on Mon Aug  7 14:33:17 2023

@author: peruzzetto
"""

import inspect
import os

from tilupy.raster import write_ascii

import tilupy.initdata


def save_gray99_topo_mass(dx=0.1, dy=0.1, folder_out=None):

    if folder_out is None:
        folder = os.path.abspath(inspect.getsourcefile(lambda: 0))
        folder_out = os.path.join(os.path.dirname(folder),
                                  '../data/gray99/rasters')
        os.makedirs(folder_out, exist_ok=True)

    x, y, z, m = tilupy.initdata.gray99_topo_mass(dx=0.01, dy=0.01)
    write_ascii(x, y, z, os.path.join(folder_out, 'gray99_topography.asc'))
    write_ascii(x, y, m, os.path.join(folder_out, 'gray99_mass.asc'))


if __name__ == '__main__':

    save_gray99_topo_mass()
