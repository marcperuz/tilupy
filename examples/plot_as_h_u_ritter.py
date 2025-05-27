r"""
Ritter's solution of dam-break on a dry domain without friction.
===========================

This example demonstrates a 1D analytical Ritter's solution of an ideal dam break on a dry domain.

The dam break is instantaneous, over an horizontal and flat surface without friction with a finite 
flow volume.

The initial condition for this problem is:

.. math::
        h(x, t) = 
        \begin{cases}
            h_l > 0 & \text{for } 0 < x \leq x_0, \\\\
            0 & \text{for } x_0 < x,
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
                0 & \text{if } x_B(t) < x,
            \end{cases}
            
    .. math::
            u(x,t) = 
            \begin{cases}
                0 & \text{if } x \leq x_A(t), \\\\
                \frac{2}{3} \left( \frac{x - x_0}{t} + \sqrt{g h_l} \right) & \text{if } x_A(t) < x \leq x_B(t), \\\\
                0 & \text{if } x_B(t) < x,
            \end{cases}

where 

    .. math::
            \begin{cases}
                x_A(t) = x_0 - t \sqrt{g h_l}, \\\\
                x_B(t) = x_0 + 2 t \sqrt{g h_l}
            \end{cases}
"""

# %%
# Initialisation with :math:`x_0 = 0m` and :math:`h_l = 0.5m` :
import numpy as np
from tilupy.analytic_sol import Ritter_dry

A = Ritter_dry(x_0=0, h_l=0.5)
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
# Specific case for :math:`x_0 = 5m`, :math:`h_l = 0.005m`, :math:`L = 10m` and :math:`t = 6s` found in SWASHES (https://www.idpoisson.fr/swashes/)
B = Ritter_dry(x_0=5, h_l=0.005)
x = np.linspace(0, 10, 100)
B.compute_h(x, 6)
B.show_res(show_h=True)