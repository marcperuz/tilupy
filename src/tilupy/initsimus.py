# -*- coding: utf-8 -*-

import importlib


def write(model_name: str,
          raster_topo: str,
          raster_mass: str,
          tmax : float,
          dt_im : float,
          rheology_type: str,
          rheology_params: dict = None,
          folder_out: str = None,
          **kwargs
          ):
    """
    Dynamically imports the corresponding initiation module from
    :data:`tilupy.models.<code>.initsimus` and use the corresponding :data:`write_simu()` function.

    Parameters
    ----------
    model_name : str
        Model to create simulation files.
    raster_topo : str, optional
        Path for an ASCII topography.
    raster_mass : str, optional
        Path for an ASCII initial mass.
    tmax : float
        Maximum simulation time.
    dt_im : float
        Output image interval (in time steps).
    rheology_type : str
        Rheology to use for the simulation. 
    rheology_params : dict
        Parameters specific to the selected rheology.
    folder_out : str, optional
        Output folder where simulation inputs will be saved.
    **kwargs
        Additional arguments for specific models.
    """
    module = importlib.import_module("tilupy.models." + model_name + ".initsimus")
        
    module.write_simu(raster_topo=raster_topo,
                      raster_mass=raster_mass,
                      tmax=tmax,
                      dt_im=dt_im,
                      rheology_type=rheology_type,
                      rheology_params=rheology_params,
                      folder_out=folder_out,
                      **kwargs)