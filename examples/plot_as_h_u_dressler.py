r"""
Dressler's solution of dam-break on a dry domain with friction.
===========================

This example demonstrates a 1D analytical Dressler's solution of an ideal dam break on a dry domain.

The dam break is instantaneous, over an horizontal and flat surface with friction with a finite 
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
                \frac{1}{g} \left( \frac{2}{3} \sqrt{g h_l} - \frac{x - x_0}{3t} + \frac{g^{2}}{C^2} \alpha_1 t \right)^2 & \text{if } x_A(t) < x \leq x_B(t), \\\\
                0 & \text{if } x_B(t) < x,
            \end{cases}
            
    .. math::
            u(x,t) = 
            \begin{cases}
                0 & \text{if } x \leq x_A(t), \\\\
                \frac{2\sqrt{g h_l}}{3} + \frac{2(x - x_0)}{3t} + \frac{g^2}{C^2} \alpha_2 t & \text{if } x_A(t) < x \leq x_B(t), \\\\
                0 & \text{if } x_B(t) < x,
            \end{cases}

where 

    .. math::
            \begin{cases}
                x_A(t) = x_0 - t \sqrt{g h_l}, \\\\
                x_B(t) = x_0 + 2 t \sqrt{g h_l}
            \end{cases}

and

    .. math::
            \begin{cases}
                \alpha_1(\xi) = \frac{6}{5(2-\xi)} - \frac{2}{3} + \frac{4 \sqrt{3}}{135} (2-\xi)^{3/2}), \\\\
                \alpha_2(\xi) = \frac{12}{2-(2-\xi)} - \frac{8}{3} + \frac{8 \sqrt{3}}{189} (2-\xi)^{3/2}) - \frac{108}{7(2 - \xi)}, \\\\
                \xi = \frac{x-x_0}{t\sqrt{g h_l}}
            \end{cases}
"""

# %%
# Initialisation with :math:`x_0 = 0m` and :math:`h_l = 0.5m` :
import numpy as np
from tilupy.analytic_sol import Dressler_dry

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