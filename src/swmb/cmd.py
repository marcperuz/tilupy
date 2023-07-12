# -*- coding: utf-8 -*-
"""
Created on Wed Jun 14 18:16:45 2023

@author: peruzzetto
"""

import swmb.notations as notations
import swmb.plot as plt_fn
import swmb.dem
import swmb.read

import os
import glob
    

def process_results(fn_name, model, res_name, folder=None, param_files=None,
                    kwargs_read=None, **kwargs_fn):
    
    if folder is None:
        folder = os.getcwd()
        
    if param_files is None:
        param_files = '*.txt'
    
    param_files = glob.glob(param_files, root_dir=folder)
        
    if kwargs_read is None:
        kwargs_read = dict()
     
    kw_read = dict(folder_base=folder)   
    kw_read.update(kwargs_read)
     
    for param_file in param_files:
        kw_read['file_params'] = param_file
        res = swmb.read.get_results(model, **kw_read)
        getattr(res, fn_name)(res_name, **kwargs_fn)
        
def to_raster(model, res_name, param_files=None, folder=None, 
              kwargs_read=None, **kwargs):
    
    kw = dict(file_fmt='asc')
    kw.update(kwargs)
    
    process_results('save', model, res_name,
                    folder=folder, param_files=param_files,
                    kwargs_read=kwargs_read, **kw)
    
def plot_results(model, res_name, param_files=None, folder=None, 
                 kwargs_read=None, **kwargs):
    
    kw = dict(save=True)
    kw.update(kwargs)
    
    process_results('plot', model, res_name,
                    folder=folder, param_files=param_files,
                    kwargs_read=kwargs_read, **kw)
    
if __name__ == '__main__':
    
    folder = 'd:/Documents/peruzzetto/tmp/test_shaltop/7p30e04_m3/coulomb'
    plot_results('shaltop', 'h_max', '*18p00.txt', folder=folder)
    