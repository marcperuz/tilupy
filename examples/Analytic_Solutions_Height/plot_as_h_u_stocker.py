r"""
Stocker's solution of dam-break on a wet domain without friction.
===========================

This example demonstrates a 1D analytical Stocker's solution of an ideal dam break on a wet domain.

The dam break is instantaneous, over an horizontal and flat surface without friction with a finite 
flow volume.

The initial condition for this problem is:

.. math::
        h(x, t) = 
        \begin{cases}
            h_l > 0 & \text{for } 0 < x \leq x_0, \\\\
            0 < h_r < h_l & \text{for } x_0 < x,
        \end{cases}
        
.. math::
        u(x, t) = 0
with :math:`x_0` being the dam position.

The analytic solution is given by

    .. math::
            h(x, t) = 
            \begin{cases}
                h_l & \text{if } x \leq x_A(t), \\\\
                \frac{4}{9g} \left( \sqrt{g h_l} - \frac{x - x_0}{2t} \right)^2 & \text{if } x_A(t) < x \leq x_B(t), \\\\
                \frac{c_m^2}{g} & \text{if } x_B(t) < x \leq x_C(t), \\\\
                h_r & \text{if } x_C(t) < x,
            \end{cases}
            
    .. math::
            u(x,t) = 
            \begin{cases}
                0 & \text{if } x \leq x_A(t), \\\\
                \frac{2}{3} \left( \frac{x - x_0}{t} + \sqrt{g h_l} \right) & \text{if } x_A(t) < x \leq x_B(t), \\\\
                2 \left( \sqrt{g h_l} - c_m \right) & \text{if } x_B(t) < x \leq x_C(t), \\\\
                0 & \text{if } x_C(t) < x,
            \end{cases}

where

    .. math::
            \begin{cases}
                x_A(t) = x_0 - t \sqrt{g h_l}, \\\\
                x_B(t) = x_0 + t \left( 2 \sqrt{g h_l} - 3 c_m \right), \\\\
                x_C(t) = x_0 + t \frac{2 c_m^2 \left( \sqrt{g h_l} - c_m \right)}{c_m^2 - g h_r}
            \end{cases}

with :math:`c_m` being the solution of :math:`-8.g.hr.cm^{2}.(g.hl - cm^{2})^{2} + (cm^{2} - g.hr)^{2} . (cm^{2} + g.hr) = 0`
"""

# %%
# Initialisation with :math:`x_0 = 0m`, :math:`h_l = 0.5m` and :math:`h_r = 0.1m` :
import numpy as np
from tilupy.analytic_sol import Stocker_wet

A = Stocker_wet(x_0=0, h_l=0.5, h_r=0.1)
x = np.linspace(-5, 25, 100)

# %%
# Compute flow height for t = {0, 2, 4, 6, 8, 10}s:
A.compute_h(x, [0, 2, 4, 6, 8, 10])
A.show_res(show_h=True)


# %%
# Compute flow velocity for t = {0, 2, 4, 6, 8, 10}s:
A.compute_u(x, [0, 2, 4, 6, 8, 10])
A.show_res(show_u=True)