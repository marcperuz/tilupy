r"""
Mangeney's solution of dam-break on a dry domain with friction.
===========================

This example demonstrates a 1D analytical solution of an ideal dam break on a dry domain.

The dam break is instantaneous, over an inclined and flat surface with friction with a 
infinite flow volume.

The initial condition for this problem is:

.. math::
        h(x, t) = 
        \begin{cases}
            h_0 > 0 & \text{for } x < 0, \\\\
            0 & \text{for } 0 < x,
        \end{cases}
        
.. math::
        u(x, t) = 0

The analytic solution is given by

    .. math::
            h(x, t) = 
            \begin{cases}
                0 & \text{if } x \leq x_A(t), \\\\
                \frac{1}{9g cos(\theta)} \left( \frac{x}{t} + 2 c_0 - \frac{1}{2} m t \right)^2 & \text{if } x_A(t) < x \leq x_B(t), \\\\
                h_0 & \text{if } x_B(t) < x,
            \end{cases}
            
    .. math::
            u(x,t) = 
            \begin{cases}
                0 & \text{if } x \leq x_A(t), \\\\
                \frac{2}{3} \left( \frac{x}{t} - c_0 + mt \right) & \text{if } x_A(t) < x \leq x_B(t), \\\\
                0 & \text{if } x_B(t) < x,
            \end{cases}
            
where 

    .. math::
            \begin{cases}
                x_A(t) = \frac{1}{2}mt - 2 c_0 t, \\\\
                x_B(t) = \frac{1}{2}mt + c_0 t
            \end{cases}
            
:math:`c_0` represent the initial wave propagation speed computed from :math:`\sqrt{g h_0 \cos{\theta}}` and 
:math:`m` the constant horizontal acceleration of the front computed from :math:`-g \sin{\theta} + g \cos{\theta} \tan{\delta}`.

"""

# %%
# Initialisation with no friction (:math:`\delta = 0°`), slope :math:`\theta = 30°` and :math:`h_0 = 20m`
# (this example is the same as figure 3 of Mangeney et al, 2000):
import numpy as np
from tilupy.analytic_sol import Mangeney_dry

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
# Now adding some friction (:math:`\delta = 20°`):
B = Mangeney_dry(theta=30, delta=20, h_0=20)

# %%
# Compute flow height for t = {0, 5, 10, 15, 20}s:
B.compute_h(x, [0, 5, 10, 15, 20])
B.show_res(show_h=True)