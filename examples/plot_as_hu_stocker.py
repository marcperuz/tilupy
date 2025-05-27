r"""
Dam-break solution on a wet domain without friction.
===========================

This example demonstrates a 1D analytical Stocker's solution of an ideal dam break on a wet domain.

The dam break is instantaneous, over an horizontal and flat surface without friction with a finite 
flow volume.
"""

import numpy as np
from tilupy.analytic_sol import Stocker_wet

# %%
# Initialisation:
A = Stocker_wet(x_0=0, l=10, h_l=0.5, h_r=0.1)
x = np.linspace(-5, 25, 100)

# %%
# Compute flow height for t = {0, 2, 4, 6, 8, 10}s:
A.compute_h(x, [0, 2, 4, 6, 8, 10])
A.show_res(show_h=True)


# %%
# Compute flow velocity for t = {0, 2, 4, 6, 8, 10}s:
A.compute_u(x, [0, 2, 4, 6, 8, 10])
A.show_res(show_u=True)