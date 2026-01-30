r"""
Process multiple simulations at once
===========================

We show here how multiple simulation results can be processed at once to generate plots
or rasters, here with shaltop.
"""

# %%
# Initial import:
import tilupy.cmd

# %%
# Specify where simulation results are located. You can dowload an example folder containing
# shaltop results by uncommenting the first two lines.

# import tilupy.download_data
# tilupy.download_data.import_shaltop_frankslide()
folder_simus = "./shaltop_frankslide"

# %%
# Multiple simulations can be processed at once from a given Unix style pattern
# for the parameter files, e.g.:
params_files = "delta_2*p00.txt"

# %%
# The :module:`tilupy.cmd` module can then be used to process all the corresponding simulations.
# In this example, we plot all the thicknesses at the different recorded time steps, and save results in
# the simulation result folder, in a new 'plots' folder. The plots can be displayed with ``display_plot=True``
state = "h"
tilupy.cmd.plot_results(
    "shaltop", state, params_files, folder_simus, save=True, display_plot=False
)

# %%
# ``cmd.plot_results`` calls the ``plot`` method of result instance corresponding to the specified ``state``.
# It can be parametrized accordingly, for instance:
topo_kwargs = dict(contour_step=10, step_contour_bold=100)
tilupy.cmd.plot_results(
    "shaltop",
    "h",
    params_files,
    folder_simus,
    save=True,
    display_plot=False,
    figsize=(15 / 2.54, 15 / 2.54),
    vmin=0.1,
    vmax=100,
    topo_kwargs=topo_kwargs,
)

# %%
# Similarly, you can plot and save the map of maximum thickness with:
tilupy.cmd.plot_results(
    "shaltop",
    "h_max",
    params_files,
    folder_simus,
    save=True,
    display_plot=False,
    figsize=(15 / 2.54, 15 / 2.54),
    cmap_intervals=[0.1, 5, 10, 25, 50, 100],
    topo_kwargs=topo_kwargs,
)

# %%
# Similarly, you can transform and save results to raster files in a new ``processed`` folder
# created in each folder where raw simulation results are stored.
tilupy.cmd.to_raster("shaltop", "h_max", params_files, folder_simus, fmt="tif")

# %%
# :mod:`tilupy.cmd` can be called directly from a command line for quick processing, although
# less parameters are allowed. Two commands are provided to plot and save results, respectively
# ``tilupy_plot`` and ``tilupy_to_raster``.  For instance in command line:
#
#    .. code-block:: console
#
#     tilupy_plot -n h -p *.txt --vmin 0.1 --vmax 100
#
# is equivalent to using in a python script:
#
#    .. code-block:: python
#
#      tilupy.cmd.plot_results("shaltop", "h", "*.txt", vmin=0.1, vmax=100)
#
# To get more information on ``tilupy_plot`` and ``tilupy_to_raster`` use the ``help``
# parameter:
#
#    .. code-block:: console
#
#      tilupy_plot -h
#      tilupy_to_raster -h
