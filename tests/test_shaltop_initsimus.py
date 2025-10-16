# -*- coding: utf-8 -*-
"""
Created on Wed Jul 19 15:39:38 2023

@author: peruzzetto
"""

import os
import shutil

import tilupy.models.shaltop.initsimus as shinit
import tilupy.cmd


def test_shaltop_raster_to_input(folder_data):
    folder_simus = os.path.join(folder_data, "frankslide", "shaltop")

    raster_topo = os.path.join(
        folder_data, "frankslide", "rasters", "Frankslide_topography.asc"
    )

    file_topo_sh = os.path.join(folder_simus, "topography.d")

    if os.path.isfile(file_topo_sh):
        os.remove(file_topo_sh)

    shinit.raster_to_shaltop_txtfile(raster_topo, file_topo_sh)

    raster_pile = os.path.join(
        folder_data, "frankslide", "rasters", "Frankslide_pile.asc"
    )
    file_mass_sh = os.path.join(folder_simus, "init_mass.d")

    if os.path.isfile(file_mass_sh):
        os.remove(file_mass_sh)

    shinit.raster_to_shaltop_txtfile(raster_pile, file_mass_sh)

    assert os.path.isfile(file_topo_sh) & os.path.isfile(file_mass_sh)


def test_shaltop_make_read_param_file(folder_data):
    params = dict(
        nx=201,
        ny=201,
        per=201 * 20,
        pery=201 * 20,
        # Simulation maximum time in seconds (not comutation time)
        tmax=30,
        dt_im=10,  # Time interval (s) between snapshots recordings
        dt_force=15,
        file_z_init="topography.d",  # Name of topography input file
        file_m_init="init_mass.d",  # name of init mass input file
        initz=0,  # Topography is read from file
        ipr=0,  # Initial mass is read from file
        hinit_vert=1,  # Initial is given as vertical thicknesses and
        # must be converted to thicknesses normal to topography
        eps0=1e-13,  # Minimum value for thicknesses and velocities
        # choice of rheology (Coulomb with constant basal friction)
        icomp=1,
        # Min x value (used for plots after simulation is over)
        x0=1000,
        y0=2000,
    )  # Min y value (used for plots after simulation is over)

    deltas = [20, 25]
    folder_simus = os.path.join(folder_data, "frankslide", "shaltop")
    files_created = True

    for delta in deltas:
        params_txt = "delta_{:05.2f}".format(delta).replace(".", "p")
        param_file_path = os.path.join(folder_simus, params_txt + ".txt")
        if os.path.isfile(param_file_path):
            os.remove(param_file_path)
        # Specify folder where outputs are stored
        params["folder_output"] = params_txt
        params["delta1"] = delta  # Specify the friction coefficient
        # Write parameter file
        shinit.write_params_file(
            params, directory=folder_simus, file_name=params_txt + ".txt"
        )

        files_created = files_created & os.path.isfile(param_file_path)
        if not files_created:
            break

    assert files_created


def test_shaltop_plot_results(folder_data):
    folder_simus = os.path.join(folder_data, "frankslide", "shaltop")
    params_files = "delta_*p00.txt"
    folder_ress = [
        os.path.join(
            folder_data,
            "frankslide",
            "shaltop",
            "delta_{:d}p00".format(delta),
            "plots",
        )
        for delta in [20, 25]
    ]
    for folder in folder_ress:
        if os.path.isdir(folder):
            shutil.rmtree(folder)
    tilupy.cmd.plot_results(
        "shaltop",
        "h",
        params_files,
        folder_simus,
        save=True,
        display_plot=False,
        figsize=(10 / 2.54, 10 / 2.54),
        minval=0.1,
    )
    files_plots = ["h_0000.png", "h_0001.png", "h_0002.png", "h_0003.png"]

    all_files_created = True
    for folder in folder_ress:
        for file in files_plots:
            file_path = os.path.join(folder, file)
            all_files_created = all_files_created & os.path.isfile(file_path)
            if not all_files_created:
                break

    assert all_files_created
