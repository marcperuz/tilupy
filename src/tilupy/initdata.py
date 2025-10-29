#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import numpy as np

import tilupy.make_topo
import tilupy.make_mass
import tilupy.raster


def create_topo_constant_slope(folder_out: str, 
                               xmax: int = 30,
                               ymax: int = 25,
                               cell_size: float = 0.5,
                               theta: int = 5,
                               mass_type: str = 'r',
                               r_center: tuple = (7.5, 12.5),
                               r_radius: tuple = (3.75, 3.75),
                               s_vertex: list = [0, 5, 0, 25],
                               h_max: float = 3.75,
                               description: str = "No informations."
                               ) -> list[str, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Generates a synthetic topography and initial mass, saves them as ASCII files, 
    and stores all configuration parameters in a dedicated folder.

    Parameters
    ----------
    folder_out : str
        Output folder.
    xmax : int
        Domain length in the X direction (in meters).
    ymax : int
        Domain length in the Y direction (in meters).
    cell_size : float
        Spatial resolution of the grid (in meters).
    theta : int
        Slope angle of the topography (in degrees).
    mass_type : str
        Shape of the initial mass: 'r' for ellipse, 's' for rectangle.
    r_center : tuple
        Center (x, y) of the elliptical initial mass.
    r_radius : tuple
        Radii (rx, ry) of the elliptical initial mass.
    s_vertex : list
        Rectangle boundaries [xmin, xmax, ymin, ymax] for the rectangular mass.
    h_max : float
        Maximum height of the initial mass.
    description : str
        Text description of the configuration.

    Returns
    -------
    list[str, numpy.ndarray, numpy.ndarray, numpy.ndarray, numpy.ndarray]
        folder_path : str 
            Absolute path to the output directory.
        x, y : numpy.ndarray, numpy.ndarray 
            Meshgrid coordinates.
        z : numpy.ndarray
            Topography data.
        m : numpy.ndarray
            Initial mass distribution.
    """
    # Create mesh
    x = np.linspace(0, xmax, int(np.round(xmax / cell_size)) + 1)
    y = np.linspace(0, ymax, int(np.round(ymax / cell_size)) + 1)
    xmesh, ymesh = np.meshgrid(x, y)

    # Topography slope
    slope = np.tan(np.deg2rad(theta))

    # Topography array
    z = -slope * (xmesh - xmax)
    
    # Initial mass
    if mass_type != 'r' and mass_type != 's':
        mass_type = 'r'
        
    p = dict(
        hmax=h_max,
        mass_type=mass_type,
        r_center = r_center if mass_type=='r' else None,
        r_radius = r_radius if mass_type=='r' else None,
        s_vertex = s_vertex if mass_type=='s' else None,
    )

    if p["mass_type"] == 'r':
        m = (
            1
            - (xmesh - p["r_center"][0]) ** 2 / p["r_radius"][0] ** 2
            - (ymesh - p["r_center"][1]) ** 2 / p["r_radius"][1] ** 2
        )
        m = np.maximum(m * p["hmax"], 0)
        
    elif p["mass_type"] == 's':
        xmin_r, xmax_r, ymin_r, ymax_r = p["s_vertex"]
        in_x = np.logical_and(xmesh >= xmin_r, xmesh <= xmax_r)
        in_y = np.logical_and(ymesh >= ymin_r, ymesh <= ymax_r)
        mask = np.logical_and(in_x, in_y)
        m = np.zeros_like(z)
        m[mask] = p["hmax"]
    
    nbr_cell = np.count_nonzero(m > 0)
    
    p["nbr_cell"] = nbr_cell
    
    folder_name = f"x{xmax}_y{ymax}_" + "dx{:04.2f}_".format(cell_size).replace(".", "p") + "theta{:02.0f}_".format(theta) + mass_type + f"{nbr_cell}"
    folder_path = os.path.join(folder_out, folder_name)
    
    # Create folder
    if not os.path.isdir(folder_path):
        os.mkdir(folder_path)
        print(f"Create folder: {folder_name}")
    else:
        print(f"Existing topography: {folder_name}")
        
    # Create parameters file
    with open(os.path.join(folder_path, "parameters.txt"), "w") as file:
        file.write(f"xmax {xmax}\nymax {ymax}\ncell_size {cell_size}\ntheta {theta}\n\n")

        for param, value in p.items():
            file.write(f"{param} {value}\n")

        file.write("\n")
        file.write(description)
    
    # Save topography
    file_topo_out = os.path.join(folder_path, "topography.asc")
    file_mass_out = os.path.join(folder_path, "init_mass.asc")

    tilupy.raster.write_ascii(x, y, z, file_topo_out)
    tilupy.raster.write_ascii(x, y, m, file_mass_out)

    return folder_path, x, y, z, m


def gray99_topo_mass(dx: float = 0.1, 
                     dy: float = 0.1, 
                     res_type: str = "true_normal"
                     ) -> list[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Create Gray99 topographic and mass test.
    
    Create an initial spherical calotte above the topography, resulting as a mass with a 
    height of 0.22 m and a radius of 0.32 m (more precisely it is the length in the downslope 
    direction), following the indications in Gray et al 99 (p. 1859). 
    The correspondig radius of the sphere, and the offset from the topography in the topography 
    normal direction (norm_offset) are deduced from these parameters. 
    
    See also Gig 3 in Wieland, Gray and Hutter (1999)
    
    Parameters
    ----------
    dx : float, optional
        Cell size of the x axis, by default 0.1
    dy : float, optional
        Cell size of the y axis, by default 0.1
    res_type : str, optional
        Type of thickness output:
        
            - 'true_normal': Real thickness in the direction normal to the topography. 
            - 'vertical': Thickness in the vertical direction. 
            - 'projected_normal': Thickness normal to the topography is computed from the vertical 
              thickness projected on the axe normal to the topography. 
              
        The default is 'true_normal'.

    Returns
    -------
    list[numpy.ndarray, numpy.ndarray, numpy.ndarray, numpy.ndarray]
        X : numpy.ndarray
            Mesh of X coordinates in the cartesian frame (nx*ny).
        Y : numpy.ndarray
            Mesh of Y coordinates in the cartesian frame (nx*ny).
        Z : numpy.ndarray
            Mesh of Z coordinates in the cartesian frame (nx*ny).
        M : numpy.ndarray
            Array of mass height, in the direction normal to topography.
    """
    # Initiate topography
    X, Y, Z = tilupy.make_topo.gray99(dx=dx, dy=dy)

    x0 = 0.06 * np.cos(np.deg2rad(40))
    hmass = 0.22
    wmass = 0.32
    radius = (wmass**2 + hmass**2) / (2 * hmass)
    norm_offset = (wmass**2 - hmass**2) / (2 * hmass)
    # Z = -np.tile(X, [len(Y), 1])*np.tan(np.deg2rad(20))
    
    M = tilupy.make_mass.calotte(X, 
                                 Y, 
                                 Z, 
                                 x0, 
                                 0, 
                                 radius, 
                                 norm_offset=norm_offset, 
                                 res_type=res_type)

    return X, Y, Z, M


"""
if __name__ == "__main__":
    x, y, z, m = gray99_topo_mass(dx=0.01, dy=0.01)
    axe = pytopomap.plot.plot_data_on_topo(x, y, z, m, topo_kwargs=dict(level_min=0.1))
    plt.show()
"""