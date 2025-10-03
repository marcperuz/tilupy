r"""
Analytical solution of a dam-break problem on a wet bed without friction (Stocker)
=============================================================================================

This example presents the one-dimensional analytical solution proposed by Stocker for the dam-break 
problem on a wet, horizontal bed without friction.


Model Assumptions
-----------------

- Instantaneous dam break at position :math:`x = x_0` and at time :math:`t = 0`.
- The bed is flat and horizontal (no slope).
- No bed friction is considered.
- A finite volume of still water of height :math:`h_0` is located to the left of the dam at :math:`0 < x \leq x_0`.
- A shallow layer of water of height :math:`h_r` (with :math:`h_r < h_0`) exists to the right of the dam (:math:`x_0 < x`).
- The fluid is incompressible and inviscid, subject only to gravity.


Initial Conditions
------------------

.. math::
        h(x, 0) = 
        \begin{cases}
            h_0 > 0 & \text{for } 0 < x \leq x_0, \\\\
            0 < h_r < h_0 & \text{for } x_0 < x,
        \end{cases}
        
.. math::
        u(x, 0) = 0

where :math:`x_0` is the initial dam location and :math:`h_0` is the height of the water column at the left of the dam and
:math:`h_r` is the height of the water column at the right of the dam.


Analytical Solution
-------------------

The water height and velocity profiles for :math:`t > 0` are given by:

    .. math::
            h(x, t) = 
            \begin{cases}
                h_0 & \text{if } x \leq x_A(t), \\\\
                \frac{\left( 2 \sqrt{g h_0} - \frac{x}{t} \right)^2}{9 g} & \text{if } x_A(t) < x \leq x_B(t), \\\\
                h_m = \frac{1}{2} h_r \left( \sqrt{1 + \frac{8 c_m^2}{g h_r}} - 1 \right) & \text{if } x_B(t) < x \leq x_C(t), \\\\
                h_r & \text{if } x_C(t) < x,
            \end{cases}
            
    .. math::
            u(x,t) = 
            \begin{cases}
                0 & \text{if } x \leq x_A(t), \\\\
                \frac{2}{3} \left( \frac{x}{t} + \sqrt{g h_0} \right) & \text{if } x_A(t) < x \leq x_B(t), \\\\
                2 \sqrt{g h_0} - 2 \sqrt{g h_m} & \text{if } x_B(t) < x \leq x_C(t), \\\\
                0 & \text{if } x_C(t) < x,
            \end{cases}

where the locations separating the flow regions evolve in time according to:

    .. math::
            \begin{cases}
                x_A(t) = - t \sqrt{g h_0}, \\\\
                x_B(t) = t \left( 2 \sqrt{g h_0} - 3 \sqrt{g h_m} \right), \\\\
                x_C(t) =  c_m t
            \end{cases}

with :math:`c_m` is the front shock wave speed, obtained as the solution of the nonlinear equation: 
    .. math::
        c_m h_r - h_r \left( \sqrt{1 + \frac{8 c_m^2}{g h_r}} - 1 \right) \left( \frac{c_m}{2} - \sqrt{g h_0} + \sqrt{\frac{g h_r}{2} \left( \sqrt{1 + \frac{8 c_m^2}{g h_r}} - 1 \right)} \right) = 0


Implementation
--------------
"""
# %%
# First import required packages and define the spatial domain for visualization.
# For following examples we will use a 1D space from -5.5 to 6 m.
import numpy as np
from tilupy.analytic_sol import Stoker_SARKHOSH_wet

x = np.linspace(-5.5, 6, 1000)

# %%
# 
# -------------------

# %%
# Case: Stocker's solution with dam at :math:`x_0 = 0 m`, initial fluid height :math:`h_0 = 0.5 m` and initial 
# domain height :math:`h_r = 0.05 m`
case = Stoker_SARKHOSH_wet(h_0=0.5, h_r=0.025)


# %%
# Compute and plot fluid height at times :math:`t = {0, 0.5, 1, 1.5, 2} s`.
case.compute_h(x, [0, 0.5, 1, 1.5, 2])
ax = case.plot(show_h=True,  linestyles=["", ":", "-.", "--", "-"])


# %%
# Compute and plot fluid velocity at times :math:`t = {0, 0.5, 1, 1.5, 2} s`.
case.compute_u(x, [0, 0.5, 1, 1.5, 2])
ax = case.plot(show_u=True,  linestyles=["", ":", "-.", "--", "-"])

# %%
# 
# -------------------

# %%
# Original reference:
# 
# Stoker, J.J., 1957, Water Waves: The Mathematical Theory with Applications, Pure and Applied Mathematics, 
# vol. 4, Interscience Publishers, New York, USA.


# %%
# Papier original (pas d'accès)
# https://www.researchgate.net/publication/230513364_Water_Waves_The_Mathematical_Theory_with_Applications
# 
# Code matlab:
# https://github.com/psarkhosh/Stoker_solution
# 
# Article utilisant les travaux
# https://www.mdpi.com/2073-4441/15/21/3841#
# 
# Article domaine humide sur pente inclinée (pas d'accès)
# https://ascelibrary.org/doi/10.1061/%28ASCE%29HY.1943-7900.0001683