r"""
Dam-break solution on a dry domain with friction and inclined surface.
===========================

This example demonstrates a 1D analytical solution of an ideal dam break on a dry domain.

The dam break is instantaneous, over an inclined and flat surface with friction with a 
infinite flow volume.

Extract from fig 3 of Mangeney et al, 2000.
"""

import numpy as np
from tilupy.analytic_sol import Mangeney_dry


# %%
# Initialisation with no friction:
A = Mangeney_dry(theta=30, delta=0, h_0=20)
x = np.linspace(0, 1000, 1000)


# %%
# Compute flow height for t = {0, 5, 10, 15, 20}s:
A.compute_h(x, [0, 5, 10, 15, 20])
A.show_res(show_h=True)


# %%
# Compute flow velocity for t = {0, 2, 4, 6, 8, 10}s:
A.compute_u(x, [0, 2, 4, 6, 8, 10])
A.show_res(show_u=True)

# %%
# Adding friction:
B = Mangeney_dry(theta=30, delta=20, h_0=20)


# %%
# Compute flow height for t = {0, 5, 10, 15, 20}s:
B.compute_h(x, [0, 5, 10, 15, 20])
B.show_res(show_h=True)