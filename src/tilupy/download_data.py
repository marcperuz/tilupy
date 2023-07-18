# -*- coding: utf-8 -*-
"""
Created on Thu Jul 13 10:39:05 2023

@author: peruzzetto
"""

import requests

import os

def import_frankslide_dem(folder_out=None, file_out=None):
    
    if folder_out is None:
        folder_out = '.'
    if file_out is None:
        file_out = 'Frankslide_topography.asc'
    
    file_save = os.path.join(folder_out, file_out)
    
    url = ('https://raw.githubusercontent.com/marcperuz/tilupy/main/data/'+
           'frankslide/rasters/Frankslide_topography.asc')
    r = requests.get(url)
    open(file_save, 'w').write(r.text)
    
    return file_save

def import_frankslide_pile(folder_out=None, file_out=None):
    
    if folder_out is None:
        folder_out = '.'
    if file_out is None:
        file_out = 'Frankslide_pile.asc'
    
    file_save = os.path.join(folder_out, file_out)
    
    url = ('https://raw.githubusercontent.com/marcperuz/tilupy/main/data/'+
           'frankslide/rasters/Frankslide_pile.asc')
    r = requests.get(url)
    open(file_save, 'w').write(r.text)
    
    return file_save