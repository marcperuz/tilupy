r"""
Dam-break solution on a dry domain with friction.
===========================

This example demonstrates a 1D analytical Dressler's solution of an ideal dam break on a dry domain.

The dam break is instantaneous, over an horizontal and flat surface with friction with a finite 
flow volume.
"""

import numpy as np
from tilupy.analytic_sol import Dressler_dry

# %%
# Initialisation:
A = Dressler_dry(x_0=0, h_l=0.5)
x = np.linspace(-5, 15, 100)

# %%
# Compute flow height for t = {0, 4, 6, 8, 10}s:
A.compute_h(x, [0, 4, 6, 8, 10])
A.show_res(show_h=True)


# %%
# Compute flow velocity for t = {0, 4, 6, 8, 10}s:
A.compute_u(x, [0, 4, 6, 8, 10])
A.show_res(show_u=True)