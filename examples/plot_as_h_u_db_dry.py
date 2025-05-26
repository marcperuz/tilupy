r"""
Dam-break solution on a dry domain without friction.
===========================

This example demonstrates a 1D analytical Ritter's solution of an ideal dam break on a dry domain.

The dam break is instantaneous, over an horizontal and flat surface without friction with a finite 
flow volume.
"""

import numpy as np
from tilupy.analytic_sol import Dam_break_dry_domain

# %%
# Initialisation:
A = Dam_break_dry_domain(x_0=0, h_l=0.5)
x = np.linspace(-5, 25, 100)


# %%
# Compute flow height for t = {0, 2, 4, 6, 8, 10}s:
A.compute_h(x, [0, 2, 4, 6, 8, 10])
A.show_res(show_h=True)


# %%
# Compute flow velocity for t = {0, 2, 4, 6, 8, 10}s:
A.compute_u(x, [0, 2, 4, 6, 8, 10])
A.show_res(show_u=True)


# %%
# Specific case for hl = 0.005m, x0 = 5m, L = 10m and t = 6 s found in SWASHES (https://www.idpoisson.fr/swashes/)
B = Dam_break_dry_domain(x_0=5, h_l=0.005)
x = np.linspace(0, 10, 100)
B.compute_h(x, 6)
B.show_res(show_h=True)