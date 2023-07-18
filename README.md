# tilupy

## Description

`tilupy` (ThIn-Layer Unified Processing in pYthon) package is meant as a top-level tool for processing inputs and outputs of thin-layer geophysical
flow simulations on general topographies.
It contains one submodule per thin-layer model for writing and reading raw inputs and outputs of the model. 
Outputs are then easily compared between different simulations / models. The models themselves are not part of this package and must
be installed separately.

Note that `tilupy` is still under development, thus only minimal documentation is available at the moment and testing is underway.
Contributions are feedback are most welcome. Reading and writing is available for the `SHALTOP` model (most commonly used by the author) and `r.avaflow`
(only partly maintained).

## Installation

To install `tilupy` from GitHub or PyPi, you'll need to have `pip` installed on your computer. `tilupy`is not yet available on `conda-forge`. 

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

### Latest stable realease from PyPi

```
python -m pip install tilupy
```

### Development version on from GitHub

Download the GithHub repository [here](https://github.com/marcperuz/tilupy), or clone it with

```
git clone https://github.com/marcperuz/tilupy.git
```

Open a terminal in the created folder and type:

```
python -m pip install .
```

## Quick start

We give here a simple example to prepare simulations for SHALTOP, and process the results. The corresponding scripts can be found in `examples/frankslide`

### Prepare simulations

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

### Process simulations

Once simulations are over, `tilupy`can be used to plot the results as images, where the flow state (thickness, velocity, ...)
is shown with a colorscale on the underlying topography. Plots can be thouroughly customized (documentation not available yet).

For instance, the following code plots the flow thickness at each recorded time step, with a constant colorscale vraying from 0.1 to
100 m. The topography is represented as a shaded relief with thin contour lines every 10 m, and bold contour_lines every 100 m. Plots are saved
in a folder created in `folder_out`, but not displayed in the terminal (if you work in a developping environnement such as `spyder`).

```python
topo_kwargs = dict(contour_step=10, step_contour_bold=100)
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

In the previous examples, the outputs that are plotted or saved can be chosen among

- `h` : Flow thickness in the direction perpendicular to the topography
- `hvert` : Flow thickness in the vertical direction
- `ux`and `uy` : Flow velocity in the X and Y direction (check whether it is in the cartesian reference frame or not)
- `u` : Norm of the flow velocity
- `hu` : Momentum (thickness * flow velocity)
- `hu2` : Kinetic energy (thickness * square flow velocity)

It is also possible to extract 2D spatial static characteritics of the flow by using any of the previous states with `_[operation]`, where
`[operation]` is chosen among, `max`, `mean`, `std`, `sum`, `min`, `final`, `initial`. For instance `h_max` is a 2D array with the
maximum simulated thickness at each point of the grid. `h_final` is the simulated thickness at the end of the simulation.
`tilupy` will load these characteristics directly from the simulation output when possible (e.g. Shaltop records a maximum thickness array),
or compute then from the available data. For instance, if the simulation results contains a file whith the maximum thickness, `h_max` will be
read from this file. Otherwise, `h_max` is computed from the simulation recorded temporal snapshots (which is supposedly less precise).
