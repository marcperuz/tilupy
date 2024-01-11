#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 25 15:18:31 2021

@author: peruzzetto
"""

import os
import posixpath
import numpy as np

from tilupy.utils import format_path_linux

import tilupy.notations
import tilupy.raster


README_PARAM_MATCH = dict(
    tmax="tmax", CFL="cflhyp", h_min="eps0", dt_im_output="dt_im"
)

SHALTOP_LAW_ID = dict(coulomb=1, voellmy=8, bingham=6, muI=7)


def write_params_file(params, directory=None, file_name="params.txt"):
    """
    Write params file for shaltop simulations

    Parameters
    ----------
    params : TYPE
        DESCRIPTION.
    sup_data : TYPE, optional
        DESCRIPTION. The default is {}.
    directory : TYPE, optional
        DESCRIPTION. The default is None.
    file_name : TYPE, optional
        DESCRIPTION. The default is 'params.txt'.

    Returns
    -------
    None.

    """

    if directory is None:
        directory = os.getcwd()
    with open(os.path.join(directory, file_name), "w") as file_params:
        for name in params:
            val = params[name]
            if isinstance(val, int) or isinstance(val, np.int64):
                file_params.write("{:s} {:d}\n".format(name, val))
            if isinstance(val, float) or isinstance(val, np.float64):
                file_params.write("{:s} {:.8G}\n".format(name, val))
            if isinstance(val, str):
                file_params.write("{:s} {:s}\n".format(name, val))


def raster_to_shaltop_txtfile(file_in, file_out, folder_out=None):
    if folder_out is not None:
        file_out = os.path.join(folder_out, file_out)

    x, y, rast = tilupy.raster.read_raster(file_in)
    np.savetxt(
        file_out,
        np.reshape(np.flip(rast, axis=0), (rast.size, 1)),
        fmt="%.12G",
    )

    res = dict(
        x0=x[0], y0=y[0], dx=x[1] - x[0], dy=y[1] - y[0], nx=len(x), ny=len(y)
    )

    return res


def write_job_files(
    dirs,
    param_files,
    file_job,
    job_name,
    max_time_hours=24,
    ncores_per_node=6,
    partitions="cpuall,data,datanew",
    shaltop_file="shaltop",
    folder_conf_in_job=None,
    replace_path=None,
    number_conf_file=True,
):
    """
    Write job/conf files for slurm jobs. The conf contains all the commands
    needed to run each simulation (one command per simulation).

    Parameters
    ----------
    dirs : list of string
        list of paths where simus will be run.
    param_files : list string
        list of shaltop parameter files.
    file_job : string
        name of job file called by sbatch.
    job_name : string
        name of conf file used by file_job.
    max_time_hours : int, optional
        Maximum job duration in hours before stop. The default is 24.
    ncores_per_node : int, optional
        Number of cores per nodes. Used to know the number of nodes required
        for the job. The default is 6.
    partitions : string, optional
        Names of partitions on which jobs can be launched.
        The default is "cpuall,data,datanew".
    shaltop_file : string, optional
        Bash command used to call shaltop. Can be a path.
        The default is "shaltop".
    folder_conf_in_job : string, optional
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

    Returns
    -------
    None.

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


def make_simus(law, rheol_params, folder_data, folder_out, readme_file):
    """
    Write shaltop initial file for simple slope test case

    Parameters
    ----------
    deltas : TYPE
        DESCRIPTION.
    folder_in : TYPE
        DESCRIPTION.
    folder_out : TYPE
        DESCRIPTION.

    Returns
    -------
    None.

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

        write_params_file(
            params, directory=folder_law, file_name=simu_text + ".txt"
        )
        file_txt += "start_time=`date +%s`\n"
        file_txt += 'shaltop "" ' + simu_text + ".txt\n"
        file_txt += "end_time=`date +%s`\n"
        file_txt += "elapsed_time=$(($end_time - $start_time))\n"
        file_txt += (
            'string_time="${start_time} ' + simu_text + ' ${elapsed_time}"\n'
        )
        file_txt += "echo ${string_time} >> simulation_duration.txt\n\n"

    with open(run_shaltop_file, "w") as fid:
        fid.write(file_txt)
