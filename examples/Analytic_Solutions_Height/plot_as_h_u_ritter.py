r"""
Analytical solution of a dam-break problem on a dry bed without friction (Ritter)
===========================================================================================

This example presents the classical one-dimensional analytical solution proposed by Ritter (1892)
for the dam-break problem on a dry, horizontal bed without friction.


Model Assumptions
-----------------

- Instantaneous dam break at position :math:`x = x_0` and at time :math:`t = 0`.
- The domain is initially dry for :math:`x > x_0`, with a finite volume of still water (with height :math:`h_0`) on the left of the dam (:math:`0 < x \leq x_0`).
- The bed is flat and horizontal (no slope).
- No bed friction is considered.
- The fluid is incompressible and inviscid, subject only to gravity.
- The solution is valid until the rarefaction wave reaches the left boundary of the water column.


Initial Conditions
------------------

    .. math::
        h(x, 0) = 
        \begin{cases}
            h_0 > 0 & \text{for } 0 < x \leq x_0, \\\\
            0 & \text{for } x_0 < x,
        \end{cases}

    .. math::
        u(x, 0) = 0

where :math:`x_0` is the initial dam location and :math:`h_0` is the height of the water column.


Analytical Solution
-------------------

The water height and velocity profiles at any time t are given by:

    .. math::
            h(x, t) = 
            \begin{cases}
                h_0 & \text{if } x \leq x_A(t), \\\\
                \frac{4}{9g} \left( \sqrt{g h_0} - \frac{x - x_0}{2t} \right)^2 & \text{if } x_A(t) < x \leq x_B(t), \\\\
                0 & \text{if } x_B(t) < x,
            \end{cases}
            
    .. math::
            u(x,t) = 
            \begin{cases}
                0 & \text{if } x \leq x_A(t), \\\\
                \frac{2}{3} \left( \frac{x - x_0}{t} + \sqrt{g h_0} \right) & \text{if } x_A(t) < x \leq x_B(t), \\\\
                0 & \text{if } x_B(t) < x,
            \end{cases}

where the positions of the rarefaction wave front and the dry front are:

    .. math::
            \begin{cases}
                x_A(t) = x_0 - t \sqrt{g h_0}, \\\\
                x_B(t) = x_0 + 2 t \sqrt{g h_0}
            \end{cases}


Implementation
--------------
"""
# %%
# First import required packages and define the spatial domain for visualization. 
# For following examples we will use a 1D space from -25 to 45 m.
import numpy as np
from tilupy.analytic_sol import Ritter_dry

x = np.linspace(-25, 45, 1000)

# %%
# 
# -------------------

# %%
# **Case 1**: Ritter's solution with dam at :math:`x_0 = 0 m` and initial height :math:`h_0 = 0.5 m`
case_1 = Ritter_dry(x_0=0, h_0=0.5)


# %%
# Compute and plot fluid height at times :math:`t = {0, 2, 4, 6, 8, 10} s`.
case_1.compute_h(x, [0, 2, 4, 6, 8])
case_1.show_res(show_h=True, linestyles=["", ":", "-.", "--", "-"])


# %%
# Compute and plot fluid velocity at times :math:`t = {0, 2, 4, 6, 8, 10} s`.
case_1.compute_u(x, [0, 2, 4, 6, 8])
case_1.show_res(show_u=True, linestyles=["", ":", "-.", "--", "-"])

# %%
# 
# -------------------

# %%
# **Case 2**: Specific example from SWASHES benchmark database
# Dam at :math:`x_0 = 5 m`, initial height :math:`h_0 = 0.005 m`, domain length :math:`L = 10 m`, solution at :math:`t = 6 s`.
x = np.linspace(0, 10, 1000)

case_2 = Ritter_dry(x_0=5, h_0=0.005)
case_2.compute_h(x, 6.0)
case_2.show_res(show_h=True, linestyles=["-"])

# %%
# 
# -------------------

# %%
# Original reference:
# 
# ID Poisson. Swashes. ID Poisson, [online]. Available at: https://www.idpoisson.fr/swashes/ ; accessed June 2025.
# 
# Ritter, A., 1892, Die Fortpflanzung der Wasserwellen, Zeitschrift des Vereines Deutscher Ingenieure, vol. 36(33), p. 947–954.
