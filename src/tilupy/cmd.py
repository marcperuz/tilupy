# -*- coding: utf-8 -*-
"""
Created on Wed Jun 14 18:16:45 2023

@author: peruzzetto
"""

import tilupy.raster
import tilupy.read

import os
import argparse
import glob


def process_results(
    fn_name,
    model,
    res_name,
    folder=None,
    param_files=None,
    kwargs_read=None,
    **kwargs_fn
):
    assert model is not None

    if folder is None:
        folder = os.getcwd()

    if param_files is None:
        param_files = "*.txt"

    print(folder, param_files)

    param_files = [
        os.path.basename(x)
        for x in glob.glob(os.path.join(folder, param_files))
    ]

    if len(param_files) == 0:
        print("No parameter file matching param_files pattern was found")
        return

    if kwargs_read is None:
        kwargs_read = dict()

    kw_read = dict(folder_base=folder)
    kw_read.update(kwargs_read)

    for param_file in param_files:
        print_str = "Processing simulation {:s}, {:s} {:s} ....."
        print(print_str.format(param_file, fn_name, res_name))
        kw_read["file_params"] = param_file
        res = tilupy.read.get_results(model, **kw_read)
        getattr(res, fn_name)(res_name, **kwargs_fn)


def to_raster(
    model=None,
    res_name="h",
    param_files=None,
    folder=None,
    kwargs_read=None,
    **kwargs
):
    kw = dict(fmt="asc")
    kw.update(kwargs)

    process_results(
        "save",
        model,
        res_name,
        folder=folder,
        param_files=param_files,
        kwargs_read=kwargs_read,
        **kw
    )


def plot_results(
    model=None,
    res_name="h",
    param_files=None,
    folder=None,
    kwargs_read=None,
    **kwargs
):
    kw = dict(save=True)
    kw.update(kwargs)

    process_results(
        "plot",
        model,
        res_name,
        folder=folder,
        param_files=param_files,
        kwargs_read=kwargs_read,
        **kwargs
    )


def _get_parser(prog, description):
    parser = argparse.ArgumentParser(
        prog=prog,
        description=description,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("model", help="Model name", type=str)
    parser.add_argument(
        "-n",
        "--res_name",
        help="Name of output, only for maps",
        default="h",
        type=str,
    )
    parser.add_argument(
        "-p",
        "--param_files",
        help="Parameter file (globbing)",
        default="*.txt",
        type=str,
    )
    parser.add_argument(
        "-f",
        "--folder",
        help="Root folder, default is current folder",
        default=None,
        type=str,
    )
    return parser


def _tilupy_plot():
    parser = _get_parser("tilupy_plot", "Plot thin-layer simulation results")
    parser.add_argument(
        "--fmt",
        help=("Plot output format " "(any accepted by matplotlib.savefig)"),
        default="png",
        type=str,
    )
    parser.add_argument(
        "--vmin",
        help=("Minimum plotted value, " "adapted to data by default"),
        default=None,
        type=float,
    )
    parser.add_argument(
        "--vmax",
        help=("Maximum plotted value, " "adapted to data by default"),
        default=None,
        type=float,
    )
    parser.add_argument(
        "--minval_abs",
        help=("Minimum plotted absolute value," " adapted to data by default"),
        default=None,
        type=float,
    )
    args = parser.parse_args()
    plot_results(**vars(args))


def _tilupy_to_raster():
    parser = _get_parser(
        "tilupy_to_raster", "Convert simulation results to rasters"
    )
    parser.add_argument(
        "--fmt",
        help=("File output format, " "tif/tiff requires rasterio"),
        default="asc",
        type=str,
        choices=["tif", "tiff", "txt", "asc", "ascii"],
    )
    args = parser.parse_args()
    # plot_results(parser.model, parser.res_name)
    to_raster(**vars(args))


if __name__ == "__main__":
    # folder = 'd:/Documents/peruzzetto/tmp/test_shaltop/7p30e04_m3/coulomb'
    # plot_results('shaltop', 'h_max', '*18p00.txt', folder=folder)
    _tilupy_plot()
