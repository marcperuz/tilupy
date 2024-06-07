# tilupy

- [Description](#description)
- [Installation](#installation-pip)
- [Quick start](#quick-start)
  - [Prepare simulations](#prepare-simus)
  - [Get simulation results](#simu-results)
  - [Process simulation results in python script](#process-simus-python)
  - [Process simulation results in command line](#process-simus-cmd)
- [Models references](#models-refs)
  - [Shaltop](#ref-shaltop)
  - [r.avaflow](#ref-ravaflow) 
 

## Description <a name="description"></a>

`tilupy` (ThIn-Layer Unified Processing in pYthon) package is meant as a top-level tool for processing inputs and outputs of thin-layer geophysical
flow simulations on general topographies.
It contains one submodule per thin-layer model for writing and reading raw inputs and outputs of the model. 
Outputs are then easily compared between different simulations / models. The models themselves are not part of this package and must
be installed separately.

Note that `tilupy` is still under development, thus only minimal documentation is available at the moment, and testing is underway.
Contributions are feedback are most welcome. Reading and writing is available for the `SHALTOP` model (most commonly used by the author) and `r.avaflow`
(only partly maintained).

## Installation with pip <a name="installation-pip"></a>

To install `tilupy` from GitHub or PyPi, you'll need to have `pip` installed on your computer. 

It is strongly recommended to install `tilupy` in a virtual environnement dedicated to this package. This can be done with `virtualenv`
(see the documentation e.g. [here](https://packaging.python.org/en/latest/guides/installing-using-pip-and-virtual-environments/)).
Create the environnement with :

```bash
python -m venv /path/to/myenv
```

and activate it, on Linux:

```bash
source /path/to/myenv/bin/activate
```

and on Windows:

```cmd.exe
\path\to\myenv\Scripts\activate
```

Alternatively, if you are more used to Anaconda :

```bash
conda create -n tilupy pip
conda activate tilupy
```

or equivalently with Mamba :

```bash
mamba create -n tilupy pip
mamba activate tilupy
```

Before installing with `pip`, make sure `pip`, `steuptools` and `wheel` are up to date

```
python -m pip install --upgrade pip setuptools wheel
```

### Latest stable realease from PyPi <a name="pypi-install"></a>

```
python -m pip install tilupy
```

### Development version on from GitHub <a name="source-install"></a>

Download the GithHub repository [here](https://github.com/marcperuz/tilupy), or clone it with

```
git clone https://github.com/marcperuz/tilupy.git
```

Open a terminal in the created folder and type:

```
python -m pip install .
```

## Installation with from conda-forge

The latest stable version of `tilupy` on PyPi is also (supposedly) distributed on `conda-forge`. It can be intalled with Anaconda (or any equivalent) with

```
conda install conda-forge::tilupy
```

## Quick start <a name="quick-start"></a>

We give here a simple example to prepare simulations for SHALTOP, and process the results. The corresponding scripts can be found in `examples/frankslide`

### Prepare simulations <a name="prepare-simus"></a>

Import the different required modules :

```python
import os

# Read an write rasters
import tilupy.raster
# Functions to download examples of elevation and initial mass rasters
import tilupy.download_data
#Submodule used to prepare Shaltop simulations
import tilupy.models.shaltop.initsimus as shinit
```

Define the folder where input data will be downloaded and simulations carried out :

```python
FOLDER_BASE = '/path/to/myfolder'
```

Import data from GitHub, and create subfolder for simulation results

```python
folder_data = os.path.join(FOLDER_BASE, 'rasters')
os.makedirs(folder_data, exist_ok=True)
#raster_topo and raster_mass are the paths to the topography and initial mass rasters
raster_topo = tilupy.download_data.import_frankslide_dem(folder_out=folder_data)
raster_mass = tilupy.download_data.import_frankslide_pile(folder_out=folder_data)
# Create folder for shaltop simulations
folder_simus = os.path.join(FOLDER_BASE, 'shaltop')
os.makedirs(folder_simus, exist_ok=True)
```

Convert downloaded rasters to Shaltop input file type, and store the properties of the resulting grid

```python
shinit.raster_to_shaltop_txtfile(raster_topo,
                                 os.path.join(folder_simus, 'topography.d'))
axes_props = shinit.raster_to_shaltop_txtfile(raster_mass,
                                              os.path.join(folder_simus, 'init_mass.d'))
```

Initiate simulations parameters. See the SHALTOP documentation for details.

```python
params = dict(nx=axes_props['nx'], ny=axes_props['ny'],
              per=axes_props['nx']*axes_props['dx'],
              pery=axes_props['ny']*axes_props['dy'],
              tmax=100, # Simulation maximum time in seconds (not comutation time)
              dt_im=10, # Time interval (s) between snapshots recordings
              file_z_init = 'topography.d', # Name of topography input file
              file_m_init = 'init_mass.d',# name of init mass input file
              initz=0, # Topography is read from file
              ipr=0, # Initial mass is read from file
              hinit_vert=1, # Initial is given as vertical thicknesses and 
              # must be converted to thicknesses normal to topography
              eps0=1e-13, #Minimum value for thicknesses and velocities
              icomp=1, # choice of rheology (Coulomb with constant basal friction)
              x0=1000, # Min x value (used for plots after simulation is over)
              y0=2000) # Min y value (used for plots after simulation is over)
```

Finally, prepare simulations for a set of given rheological parameters (here, three basal friction coefficients)

```python
deltas = [15, 20, 25]
for delta in deltas:
    params_txt = 'delta_{:05.2f}'.format(delta).replace('.', 'p')
    params['folder_output'] = params_txt # Specify folder where outputs are stored
    params['delta1'] = delta # Specify the friction coefficient
    #Write parameter file
    shinit.write_params_file(params, directory=folder_simus,
                             file_name=params_txt + '.txt')
    #Create folder for results (not done by shlatop!)
    os.makedirs(os.path.join(folder_simus, params_txt), exist_ok=True)
```

You must then run the simulations (see Shaltop documentation)

### Get simulation results <a name="simu-results"></a>

Simulation results are read from a `Results` class. Main functions are defined in the parent class `tilupy.read.Results`, and each model has its own inheretied class `tilupy.models.[model_name].read.Results`. A class instance for a given model can be initiated with

```python
res = tilupy.read.get_results([model_name], **kwargs)
```

where `kwargs must be adapted to the considered model. For instance with Shaltop and the example above, the results of the simulation with a friction angle of 25 must be initiated as :

```python
res = tilupy.read.get_results('shaltop', folder_base=folder_simus, file_params='delta_25p00.txt)
```

The topography and axes can then directly be read from `res` :

```python
x, y, z = res.x, res.y, res.z
import matplotlib.pyplot as plt
plt.imshow(z)
```

`z[i, j]` is the altitude at `flip(y[i])` and `x[j]`. Thus `z[0, 0]` corresponds to the North-West corner of the topography. 

Specific simulation outputs can be extracted with 

```python
h_res = res.get_output(res_name)
```

where `res_name` must chosen among

- `h` : Flow thickness in the direction perpendicular to the topography
- `hvert` : Flow thickness in the vertical direction
- `ux` and `uy` : Flow velocity in the X and Y direction (check whether it is in the cartesian reference frame or not)
- `u` : Norm of the flow velocity
- `hu` : Momentum (thickness * flow velocity)
- `hu2` : Kinetic energy (thickness * square flow velocity)

It is also possible to extract 2D spatial static characteritics of the flow by using any of the previous states with `_[operation]`, where
`[operation]` is chosen among, `max`, `mean`, `std`, `sum`, `min`, `final`, `initial`. For instance `h_max` is a 2D array with the
maximum simulated thickness at each point of the grid. `h_final` is the simulated thickness at the end of the simulation.
`tilupy` will load these characteristics directly from the simulation output when possible (e.g. Shaltop records a maximum thickness array),
or compute then from the available data. For instance, if the simulation results contains a file whith the maximum thickness, `h_max` will be
read from this file. Otherwise, `h_max` is computed from the simulation recorded temporal snapshots (which is supposedly less precise).

For instance, to load the recorded thicknesses in the current example :

```python
res_name = 'h' 
h_res = res.get_output(res_name) # Thicknesses recorded at different times
# h_res.d is a 3D numpy array of dimension (len(x) x len(y) x len(h_res.t))
plt.imshow(h_res.d[:, :, -1]) # Plot final thickness.
t = h_res.t # Get times of simulation outputs.
```

And to load the maximum thickness array :

```python
res_name = 'h_max'
h_max_res = res.get_output(res_name) #h_max_res is read directly from simulation
# results when possible, and is deduced from res.get_output('h') otherwise
# h_max_res.d is a 2D numpy array of dimension (len(x) x len(y))
plt.imshow(h_max_res.d)
```

### Process simulation results in python script <a name="process-simus-python"></a>

`tilupy`can be used to plot the results as images, where the flow state (thickness, velocity, ...)
is shown with a colorscale on the underlying topography. Plots can be thouroughly customized (documentation not available yet).

For instance, the following code plots the flow thickness at each recorded time step, with a constant colorscale vraying from 0.1 to
100 m. The topography is represented as a shaded relief with thin contour lines every 10 m, and bold contour_lines every 100 m. Plots are saved
in a folder created in `folder_out`, but not displayed in the terminal (if you work in a developping environnement such as `spyder`).

```python
topo_kwargs = dict(contour_step=10, step_contour_bold=100) # give interval between thin and bold contour lines
params_files = 'delta_*.txt'
tilupy.cmd.plot_results('shaltop', 'h', params_files, folder_simus,
                        save=True, display_plot=False, figsize=(15/2.54, 15/2.54),
                        vmin=0.1, vmax=100, fmt='png',
                        topo_kwargs=topo_kwargs)
```

The following code acts similarly, but plots the maximum thickness instead, and used a segmented colormap given by `cmap_intervals`.

```python
tilupy.cmd.plot_results('shaltop', 'h_max', params_files, folder_simus,
                        save=True, display_plot=False, figsize=(15/2.54, 15/2.54),
                        cmap_intervals=[0.1, 5, 10, 25, 50, 100],
                        topo_kwargs=topo_kwargs)
```

It is also possible to save outputs back to rasters. The following code save all recorded thicknesses snapshots as tif files in a new folder
created in `folder_out`.

```python
tilupy.cmd.to_raster('shaltop', 'h_max', params_files,
                     folder_simus, fmt='tif')
```

### Process simulation results in command line <a name="process-simus-cmd"></a>

`tilupy` comes with command line scripts to allow for quick processing of results. They work similarly as the functions `tilupy.cmd.plot_results` and `tilupy.cmd.to_raster`, 
although there are less options. 

`tilupy_plot` will automatically plot and save results in a new folder `plot` located in the simulation output folder specified in the parameter file :

```
tilupy_plot [-h] [-n RES_NAME] [-p PARAM_FILES] [-f FOLDER] [--fmt FMT] [--vmin VMIN] [--vmax VMAX]
                   [--minval_abs MINVAL_ABS]
                   [model]
```

`RES_NAME` can be any of the strings listed in the previous section. For instance, to plot all thicknesses snaphsots from shaltop simulations in the current folder, type `tilupy_plot shaltop -n h`. If parameters files are located in `another/folder`, type, `tilupy shaltop -n h -f another/folder`. Similarly, to save  thicknesses snapshots as ascii rasters, use `tilupy_to_raster shaltop -n h --fmt asc`.

```
tilupy_to_raster [-h] [-n RES_NAME] [-p PARAM_FILES] [-f FOLDER] [--fmt {tif,tiff,txt,asc,ascii}] model
```

With both commands you can use the `-h` option do print help.

## Models references <a name="models-refs"></a>

We provide here a basic descriptions of models compatible with `tilupy`. The list of references is not exhaustive. 

### Shaltop <a name="ref-shaltop"></a>

`shaltop` models gravitational flow models over general topographies with small slope variation (small curvature) and friction. The equations are expressed in a horizontal/vertical reference frame and the shallow approximation is imposed in the direction normal to the slope. `shaltop` is not yet freely available. If you are interested, contact [m.peruzzetto@brgm.fr](mailto:m.peruzzetto@brgm.fr) or [mangeney@ipgp.fr](mailto:mangeney@ipgp.fr). 

- Bouchut, F., Mangeney-Castelnau, A., Perthame, B., Vilotte, J.-P., 2003. A new model of Saint Venant and Savage–Hutter type for gravity driven shallow water flows. Comptes Rendus Mathématique 336, 531–536. [https://doi.org/10.1016/S1631-073X(03)00117-1](https://doi.org/10.1016/S1631-073X(03)00117-1)
- Bouchut, F., Westdickenberg, M., 2004. Gravity driven shallow water models for arbitrary topography. Communications in Mathematical Sciences 2, 359–389. [https://doi.org/10.4310/CMS.2004.v2.n3.a2](https://doi.org/10.4310/CMS.2004.v2.n3.a2)
- Mangeney-Castelnau, A., Bouchut, F., Vilotte, J.P., Lajeunesse, E., Aubertin, A., Pirulli, M., 2005. On the use of Saint Venant equations to simulate the spreading of a granular mass: numerical simulation of granular spreading. Journal of Geophysical Research: Solid Earth 110, B09103. [https://doi.org/10.1029/2004JB003161](https://doi.org/10.1029/2004JB003161)
- Mangeney, A., Bouchut, F., Thomas, N., Vilotte, J.P., Bristeau, M.O., 2007. Numerical modeling of self-channeling granular flows and of their levee-channel deposits. Journal of Geophysical Research 112, F02017. [https://doi.org/10.1029/2006JF000469](https://doi.org/10.1029/2006JF000469)


### r.avaflow <a name="ref-ravaflow"></a>

`r.avaflow` is a GIS-supported open source software tool for the simulation of complex, cascading mass flows over arbitrary topography. It can be downloaded, along with the associated documentation, on the officiel [website](https://www.landslidemodels.org/r.avaflow/). Note that the integration of `r.avaflow` in `tilupy` is partial and potentially not adapted to new releases of `r.avaflow`. 

- Mergili, M., Fischer, J.-T., Krenn, J., Pudasaini, S.P., 2017. r.avaflow v1, an advanced open source computational framework for the propagation and interaction of two-phase mass flows. Geoscientific Model Development Discussions 10, 553–569. [https://doi.org/10.5194/gmd-10-553-2017](https://doi.org/10.5194/gmd-10-553-2017)
- Pudasaini, S.P., Mergili, M., 2019. A Multi-Phase Mass Flow Model. Journal of Geophysical Research: Earth Surface 124, 2920–2942. [https://doi.org/10.1029/2019JF005204](https://doi.org/10.1029/2019JF005204)

### Lave2D <a name="ref-Lave2D"></a>

`Lave2D` is a software developped by the INRAE for the modeling of debris flows with the Herschel-Bulkley rheology.

- Laigle, D., Hector, A.-F., Hübl, J., Rickenmann, D., 2006. Confrontation de la simulation numérique de l’étalement de laves torrentielles boueuses à des observations d’événements réels. La Houille Blanche 92, 105–112. https://doi.org/10.1051/lhb:2006108
- Rickenmann, D., Laigle, D., McArdell, B.W., Hübl, J., 2006. Comparison of 2D debris-flow simulation models with field events. Computational Geosciences 10, 241–264. https://doi.org/10.1007/s10596-005-9021-3
