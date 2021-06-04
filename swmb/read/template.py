#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun  1 15:27:36 2021

@author: peruzzetto
"""

import numpy as np
import results

# Dictionnary with results names lookup table, to match code output names
LOOKUP_NAMES = dict(h='h',
                    u='u', ux='ux', uy='uy',
                    hu='hu', ek='ek',
                    vol='vol', ekint='ekint')

def read_params(file):
    """
    Read simulation parameters from file

    Parameters
    ----------
    file : str
        Parameters file.

    Returns
    -------
    params : dict
        Dictionnary with parameters

    """
    params = dict()
    #####
    # Fill here
    #####
    return params


#####
# Add additional functions here
#####


class Results(results.Results):
    """ Results of shaltop simulations """

    def __init__(self, file_params, **varargs):
        """
        Init simulation results

        Parameters
        ----------
        file_params : str
            File where simulation parameters will be read
        folder_output : str
            Folder of simulation outputs

        """
        self.params = read_params(file)
        self.set_axes()

    def set_axes(self, x=None, y=None, **varargs):
        """ Set x and y axes """
        #####
        # Fill here
        #####
        self.x = x
        self.y = y
        self.nx = len(x)
        self.ny = len(y)

    def set_zinit(self, zinit=None):
        """ Set zinit, initial topography """
        #####
        # Fill here
        #####
        self.zinit = zinit

    def get_temporal_output(self, name, d=None, t=None, **varargs):
        """
        Read 2D time dependent simulation results

        Parameters
        ----------
        name : str
            Name of output.
        d : ndarray, optional
            Data to be read. If None, will be computed.
            The default is None.
        t : ndarray, optional
            Time of results snapshots (1D array), matching last dimension of d.
            If None, will be computed. The default is None.
        **varargs : TYPE
            DESCRIPTION.

        """
        #####
        # Fill here
        #####
        return results.TemporalResults(t, d, name)

    def get_static_output(self, name, d=None, dim='xy', **varargs):
        """
        Read 2D time dependent simulation results

        Parameters
        ----------
        name : str
            Name of output.
        d : ndarray, optional
            Data to be read. Last dimension is for time.
            If None, will be computed.
            The default is None.
        **varargs : TYPE
            DESCRIPTION.

        """
        #####
        # Fill here
        #####
        return results.StaticResults(name, d, dim)

