# -*- coding: utf-8 -*-
"""
Created on Mon Oct 10 17:31:10 2022

@author: peruzzetto
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from swmb import read
from swmb import utils
from swmb import dem

def CSI(simu, observation=None, h_threshs=[1], state='h_final'):
    
    res=[]
    assert(observation is not None)
    
    if isinstance(observation, str):
        _, _, observation, _ = dem.read_raster(observation)
        
    strs = state.split('_')
    if state == 'h_max':
        d = simu.h_max
    else:
        d = simu.get_static_output(strs[0], strs[1]).d
    for h_thresh in h_threshs:
        array = 1*d>h_thresh
        res.append(utils.CSI(array, observation))
        
    return h_threshs, res
        
def diff_runout(simu, point=None, h_threshs=[1],
                section=None, orientation='W-E',
                get_contour_kws=None):
    
    res = []
    assert(point is not None)
    
    if get_contour_kws is None:
        get_contour_kws = dict()
        
    d=simu.h_max
    xc, yc = utils.get_contour(simu.x,simu.y,d,h_threshs, **get_contour_kws)
    
    for h in h_threshs:
        res.append(utils.diff_runout(xc[h], yc[h], point, section=section,
                                     orientation=orientation))
        
    return h_threshs, res
    
def eval_simus(simus, methods, methods_kws, code='shaltop',
               recorded_params=['delta1'], calib_parameter='h_thresh'):
    """
    Evaluate simulation results with different methods

    Parameters
    ----------
    simus : TYPE
        DESCRIPTION.
    methods : TYPE
        DESCRIPTION.

    Returns
    -------
    None.

    """
    if not isinstance(methods, list):
        methods = [methods]
    if not isinstance(methods_kws, list):
        methods_args = [methods]
            
    if isinstance(simus, dict):
        simus=pd.DataFrame(simus)
    if isinstance(simus, pd.DataFrame):
        simus_list = []
        for i in range(simus.shape[0]):
            simus_list.append(read.get_results(code, **simus.iloc[i, :].to_dict()))
        simus2 = simus.copy()
    else:
        simus_list = simus
        simus2 = pd.DataFrame()
    
    for param in recorded_params:
        simus2[param] = np.nan
    fn = dict()
    for method in methods:
        fn[method] = globals()[method]
        simus2[method] = np.nan
    simus2[calib_parameter] = np.nan
        
    res = pd.DataFrame(columns=simus2.columns)    
        
    for i, simu in enumerate(simus_list):
        for j, method in enumerate(methods):
            kws = methods_kws[j]
            calib_param, calib_res = fn[method](simu, **kws)
            for k in range(len(calib_res)):
                n = res.shape[0]
                res.loc[n] = np.nan
                # Initiate fiels
                res.loc[n, :] = simus2.loc[i, :].copy()
                # Specify simulation parameters
                for param in recorded_params:
                    res.loc[n, param] = simu.params[param]
                res.loc[n, calib_parameter] = calib_param[k]
                res.loc[n, method] = calib_res[k]
                
    return res


            
    
    
        
    
