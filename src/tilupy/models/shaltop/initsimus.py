#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import posixpath
import numpy as np

from tilupy.utils import format_path_linux

import tilupy.notations
import tilupy.raster


README_PARAM_MATCH = dict(tmax="tmax", CFL="cflhyp", h_min="eps0", dt_im_output="dt_im")
"""Dictionary of correspondence between parameters in read_me file and param file."""

SHALTOP_LAW_ID = dict(No_Friction=1, Herschel_Bulkley=61, Voellmy=8, Bingham=6, Coulomb=1, Coulomb_muI=7)
"""Dictionary of correspondence between the name of rheological laws and the corresponding ID in SHALTOP.

Correspondence :

    - No_Friction = 1
    - Coulomb = 1
    - Coulomb_muI = 7
    - Voellmy = 8
    - Bingham = 6
    - Herschel_Bulkley = 61
"""


def raster_to_shaltop_txtfile(file_in: str, 
                              file_out: str, 
                              folder_out: str=None
                              ) -> dict:
    """Convert a raster file to a Shaltop ASCII text format.

    Reads a raster (formats readable by :func:`tilupy.raster.read_raster`) and saves it as a 
    ASCII text file with values flattened column-wise and rows flipped vertically. 

    Parameters
    ----------
    file_in : str
        Path to the input raster file.
    file_out : str
        Name of the output ASCII text file.
    folder_out : str, optional
        Directory where the output file will be saved. If None, `file_out`
        is used as-is, by default None.

    Returns
    -------
    dict
        Dictionary containing grid metadata:
        
            - 'x0': X coordinate of the first column.
            - 'y0': Y coordinate of the first row.
            - 'dx': Grid spacing along X.
            - 'dy': Grid spacing along Y.
            - 'nx': Number of columns.
            - 'ny': Number of rows.
    """
    if folder_out is not None:
        file_out = os.path.join(folder_out, file_out)

    x, y, rast = tilupy.raster.read_raster(file_in)
    np.savetxt(file_out,
               np.reshape(np.flip(rast, axis=0), (rast.size, 1)),
               fmt="%.12G")

    res = dict(x0=x[0], 
               y0=y[0], 
               dx=x[1] - x[0], 
               dy=y[1] - y[0], 
               nx=len(x), 
               ny=len(y))

    return res


def write_params_file(params: dict, 
                      directory: str=None, 
                      file_name: str="params.txt"
                      ) -> None:
    """Write a dictionary of parameters to a text file.

    Each key-value pair in the dictionary is written on a separate line.

    Parameters
    ----------
    params : dict
        Dictionary of parameter names and values. Values can be int, float,
        or str.
    directory : str, optional
        Directory where the parameter file will be written. Default is the
        current working directory.
    file_name : str, optional
        Name of the parameter file. Default is "params.txt".
    """
    if directory is None:
        directory = os.getcwd()
        
    with open(os.path.join(directory, file_name), "w") as file_params:
        for name in params:
            val = params[name]
            if (isinstance(val, int)
                or isinstance(val, np.int64)
                or isinstance(val, np.int32)
                ):
                file_params.write("{:s} {:d}\n".format(name, val))
                
            if (isinstance(val, float)
                or isinstance(val, np.float64)
                or isinstance(val, np.float32)
                ):
                file_params.write("{:s} {:.8G}\n".format(name, val))
                
            if isinstance(val, str):
                file_params.write("{:s} {:s}\n".format(name, val))


def write_simu(raster_topo: str, 
               raster_mass: str,
               tmax : float,
               dt_im : float,
               rheology_type: str,
               rheology_params: dict,
               folder_out: str=None,
               ) -> None:
    """
    Prepares the input files required for a SHALTOP simulation and saves them in a dedicated folder.

    Parameters
    ----------
    raster_topo : str
        Name of the ASCII topography file.
    raster_mass : str
        Name of the ASCII initial mass file.
    tmax : float
        Maximum simulation time.
    dt_im : float
        Output image interval (in time steps).
    rheology_type : str
        Rheology to use for the simulation. 
    rheology_params : dict
        Necessary parameters for the rheology. For instance:
        
            - delta1
            - ksi
            - tau_density
            etc.
    folder_out : str, optional
        Output folder where simulation inputs will be saved.
    
    Raises
    ------
    ValueError
        If the rheology is wrong.
    """
    if folder_out is None:
        folder_out = "."
    
    # output_file = os.path.join(folder_out, "shaltop")
    
    os.makedirs(folder_out, exist_ok=True)

    x, y, z = tilupy.raster.read_raster(raster_topo)
    raster_to_shaltop_txtfile(raster_topo, os.path.join(folder_out, "z.d"))
    raster_to_shaltop_txtfile(raster_mass, os.path.join(folder_out, "m.d"))
    folder_output = "data2"
    os.makedirs(os.path.join(folder_out, folder_output), exist_ok=True)

    if rheology_type not in SHALTOP_LAW_ID:
        raise ValueError(f"Wrong law, choose in: {SHALTOP_LAW_ID}")
    
    params = dict(nx=len(x),
                  ny=len(y),
                  per=x[-1],
                  pery=y[-1],
                  tmax=tmax,
                  dt_im=dt_im,
                  initz=0,
                  file_z_init="z.d",
                  ipr=0,
                  file_m_init="m.d",
                  folder_output="data2",
                  icomp=SHALTOP_LAW_ID[rheology_type],
                  **rheology_params)
    write_params_file(params, directory=folder_out, file_name="params.txt")


def write_job_files(dirs: list[str],
                    param_files: list[str],
                    file_job: str,
                    job_name: str,
                    max_time_hours: int=24,
                    ncores_per_node: int=6,
                    partitions: str="cpuall,data,datanew",
                    shaltop_file: str="shaltop",
                    folder_conf_in_job: str=None,
                    replace_path: list=None,
                    number_conf_file: bool=True,
                    ) -> None:
    """
    Write job/conf files for slurm jobs. The conf contains all the commands
    needed to run each simulation (one command per simulation).

    Parameters
    ----------
    dirs : list[str]
        list of paths where simus will be run.
    param_files : list[str]
        list of shaltop parameter files.
    file_job : str
        name of job file called by sbatch.
    job_name : str
        name of conf file used by file_job.
    max_time_hours : int, optional
        Maximum job duration in hours before stop. The default is 24.
    ncores_per_node : int, optional
        Number of cores per nodes. Used to know the number of nodes required
        for the job. The default is 6.
    partitions : str, optional
        Names of partitions on which jobs can be launched.
        The default is "cpuall,data,datanew".
    shaltop_file : str, optional
        Bash command used to call shaltop. Can be a path.
        The default is "shaltop".
    folder_conf_in_job : str, optional
        Folder where the conf file is located. The default is the folder
        path of file_job.
    replace_path : list, optional
        replace replace_path[0] by replace_path[1] for every path in dir. This
        is used if simulations are prepared and run on two different machines
        (e.g. laptop and cluster).
        The default is None.
    number_conf_file : bool, optional
        If True, add a number in front of each line of the conf file. Required
        to identify slurm jobs.
        The default is True.
    """
    ntasks = len(dirs)
    nnodes = int(np.ceil(ntasks / ncores_per_node))

    if folder_conf_in_job is None:
        folder_conf_in_job = os.path.dirname(file_job)
        if folder_conf_in_job == "":
            folder_conf_in_job = "."

    with open(file_job + ".conf", "w", newline="\n") as conf_file:
        if number_conf_file:
            line = "{:d} {:s} {:s} {:s}\n"
        else:
            line = "{:s} {:s} {:s}\n"
        for i in range(ntasks):
            if replace_path is not None:
                folder = dirs[i].replace(replace_path[0], replace_path[1])
                param_file = param_files[i].replace(
                    replace_path[0], replace_path[1]
                )
            else:
                folder = dirs[i]
                param_file = param_files[i]
            folder = format_path_linux(folder)
            param_file = format_path_linux(param_file)
            if number_conf_file:
                line2 = line.format(i, shaltop_file, folder, param_file)
            else:
                line2 = line.format(shaltop_file, folder, param_file)
            conf_file.write(line2)

    n_hours = np.floor(max_time_hours)
    n_min = (max_time_hours - n_hours) * 60
    str_time = "{:02.0f}:{:02.0f}:00\n".format(n_hours, n_min)

    basename = os.path.basename(file_job)
    path_conf_in_job = posixpath.join(folder_conf_in_job, basename + ".conf")

    with open(file_job + ".job", "w", newline="\n") as job_file:
        job_file.write("#!/bin/sh\n")
        job_file.write("#SBATCH -J multijob\n")
        job_file.write("#SBATCH --job-name={:s}\n".format(job_name))
        job_file.write("#SBATCH --output={:s}%j.out\n".format(job_name))
        job_file.write("#SBATCH --partition " + partitions + "\n")
        job_file.write("#SBATCH --nodes={:d}".format(nnodes) + "\n")
        job_file.write("#SBATCH --ntasks={:d}".format(ntasks) + "\n")
        job_file.write("#SBATCH --time={:s}\n".format(str_time))
        job_file.write("\n")
        job_file.write("module purge\n")
        job_file.write("module load slurm\n")
        job_file.write("\n")
        line = "srun -n {:d} -l --multi-prog {:s}"
        job_file.write(line.format(ntasks, path_conf_in_job))


def make_simus(law: str, 
               rheol_params: dict, 
               folder_data: str, 
               folder_out: str, 
               readme_file: str
               )  -> None:
    """Write shaltop initial file for simple slope test case

    Reads topography and initial mass files in ASCII format,
    writes them in Shaltop-compatible format, prepares simulation parameters
    based on a README file and user-provided values,
    and generates a shell script to run the simulations.

    Parameters
    ----------
    law : str
        Name of the rheological law to use (must match a key in :data:`SHALTOP_LAW_ID`).
    rheol_params : dict of list
        Dictionary of rheology parameters. Each key corresponds to a parameter
        name and its value is a list of parameter values to simulate.
    folder_data : str
        Path to the folder containing input data files "topo.asc" and "mass.asc".
    folder_out : str
        Path to the folder where output simulation folders and Shaltop files
        will be created.
    readme_file : str
        Path to the README file containing simulation parameters and metadata.
    """
    # Get topography and initial mass, and write them in Shaltop format
    zfile = os.path.join(folder_data, "topo.asc")
    mfile = os.path.join(folder_data, "mass.asc")
    x, y, z, dx = tilupy.raster.read_ascii(zfile)
    _, _, m, _ = tilupy.raster.read_ascii(mfile)
    np.savetxt(os.path.join(folder_out, "z.d"), z.T.flatten())
    np.savetxt(os.path.join(folder_out, "m.d"), m.T.flatten())

    # Get simulation parameters from README.txt and raster .asc files
    params = tilupy.notations.readme_to_params(readme_file, README_PARAM_MATCH)
    params["nx"] = len(x)
    params["ny"] = len(y)
    params["per"] = dx * len(x)
    params["pery"] = dx * len(y)
    params["file_m_init"] = "../m.d"
    params["file_z_init"] = "../z.d"

    # Folder for rheological law, and set params accordingly
    folder_law = os.path.join(folder_out, law)
    params["icomp"] = SHALTOP_LAW_ID[law]

    param_names = [param for param in rheol_params]

    texts = tilupy.notations.make_rheol_string(rheol_params, law)

    # Run shaltop file
    run_shaltop_file = os.path.join(folder_law, "run_shaltop.sh")
    file_txt = ""

    for i in range(len(rheol_params[param_names[0]])):
        simu_text = texts[i]
        for param_name in param_names:
            params[param_name] = rheol_params[param_name][i]
        params["folder_output"] = simu_text
        folder_results = os.path.join(folder_law, simu_text)
        os.makedirs(folder_results, exist_ok=True)
        with open(os.path.join(folder_results, ".gitignore"), "w") as fid:
            fid.write("# Ignore everything in this directory")
            fid.write("*")
            fid.write("# Except this file")
            fid.write("!.gitignore")

        write_params_file(params, directory=folder_law, file_name=simu_text + ".txt")
        file_txt += "start_time=`date +%s`\n"
        file_txt += 'shaltop "" ' + simu_text + ".txt\n"
        file_txt += "end_time=`date +%s`\n"
        file_txt += "elapsed_time=$(($end_time - $start_time))\n"
        file_txt += ('string_time="${start_time} ' + simu_text + ' ${elapsed_time}"\n')
        file_txt += "echo ${string_time} >> simulation_duration.txt\n\n"

    with open(run_shaltop_file, "w") as fid:
        fid.write(file_txt)

