# -*- coding: utf-8 -*-
"""
Created on Wed Jun 14 18:16:45 2023

@author: peruzzetto
"""

import swmb.notations as notations
import swmb.plot as plt_fn
import swmb.raster
import swmb.read

import os
import sys
import argparse
import glob
    

def process_results(fn_name, model, res_name, folder=None, param_files=None,
                    kwargs_read=None, **kwargs_fn):
    
    assert(model is not None)
    
    if folder is None:
        folder = os.getcwd()
        
    if param_files is None:
        param_files = '*.txt'
        
    param_files = glob.glob(param_files, root_dir=folder)
    
    if len(param_files) == 0:
        print('No parameter file matching param_files pattern was found')
        return
        
    if kwargs_read is None:
        kwargs_read = dict()
     
    kw_read = dict(folder_base=folder)   
    kw_read.update(kwargs_read)
     
    for param_file in param_files:
        print('Processing simulation {:s} .....'.format(param_file))
        kw_read['file_params'] = param_file
        res = swmb.read.get_results(model, **kw_read)
        getattr(res, fn_name)(res_name, **kwargs_fn)
        
def to_raster(model=None, res_name='h', param_files=None, folder=None, 
              kwargs_read=None, **kwargs):
    
    kw = dict(file_fmt='asc')
    kw.update(kwargs)
    
    process_results('save', model, res_name,
                    folder=folder, param_files=param_files,
                    kwargs_read=kwargs_read, **kw)
    
def plot_results(model=None, res_name='h', param_files=None, folder=None, 
                 kwargs_read=None, **kwargs):
    
    kw = dict(save=True)
    kw.update(kwargs)
    
    process_results('plot', model, res_name,
                    folder=folder, param_files=param_files,
                    kwargs_read=kwargs_read, **kw)
    
def _get_parser(prog, description):
    parser = argparse.ArgumentParser(prog=prog,
                                     description=description,
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('model', help="Model name",
                        type=str)
    parser.add_argument('-n', '--res_name', help="Name of output, only for maps",
                        default='h', type=str,
                        choices = swmb.read.TEMPORAL_DATA_2D + swmb.read.STATIC_DATA_2D)
    parser.add_argument('-p', '--param_files', help="Parameter file (globbing)",
                        default='.txt', type=str)
    parser.add_argument('-f', '--folder', help="Root folder, default is current folder",
                        default=None, type=str)
    return parser

def _swmb_plot():
    parser = _get_parser('swmb_plot', 'Plot thin-layer simulation results')
    parser.add_argument('--file_fmt', help="Plot output format (any accepted by matplotlib.savefig)",
                        default='png', type=str,
                        )
    args = parser.parse_args()
    plot_results(**vars(args))
    
def _swmb_to_raster():
    parser = _get_parser('swmb_to_raster', 'Convert simulation results to rasters')
    parser.add_argument('--file_fmt', help="File output format",
                        default='tif', type=str,
                        choices=['tif', 'tiff', 'txt', 'asc', 'ascii'])
    args = parser.parse_args()
    # plot_results(parser.model, parser.res_name)
    plot_results(**vars(args))
    
    
if __name__ == '__main__':
    
    # folder = 'd:/Documents/peruzzetto/tmp/test_shaltop/7p30e04_m3/coulomb'
    # plot_results('shaltop', 'h_max', '*18p00.txt', folder=folder)
    _swmb_plot()
    