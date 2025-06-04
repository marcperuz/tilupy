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
- A finite volume of still water of height :math:`h_l` is located to the left of the dam at :math:`0 < x \leq x_0`.
- A shallow layer of water of height :math:`h_r` (with :math:`h_r < h_l`) exists to the right of the dam (:math:`x_0 < x`).
- The fluid is incompressible and inviscid, subject only to gravity.


Initial Conditions
------------------

.. math::
        h(x, 0) = 
        \begin{cases}
            h_l > 0 & \text{for } 0 < x \leq x_0, \\\\
            0 < h_r < h_l & \text{for } x_0 < x,
        \end{cases}
        
.. math::
        u(x, 0) = 0

where :math:`x_0` is the initial dam location and :math:`h_l` is the height of the water column at the left of the dam and
:math:`h_r` is the height of the water column at the right of the dam.


Analytical Solution
-------------------

The water height and velocity profiles for :math:`t > 0` are given by:

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

where the locations separating the flow regions evolve in time according to:

    .. math::
            \begin{cases}
                x_A(t) = x_0 - t \sqrt{g h_l}, \\\\
                x_B(t) = x_0 + t \left( 2 \sqrt{g h_l} - 3 c_m \right), \\\\
                x_C(t) = x_0 + t \frac{2 c_m^2 \left( \sqrt{g h_l} - c_m \right)}{c_m^2 - g h_r}
            \end{cases}

with :math:`c_m` is the intermediate wave speed, obtained as the solution of the nonlinear equation: 
    .. math::
        -8.g.hr.cm^{2}.(g.hl - cm^{2})^{2} + (cm^{2} - g.hr)^{2} . (cm^{2} + g.hr) = 0


Implementation
--------------
"""
# %%
# First import required packages and define the spatial domain for visualization: 1D space from -5 to 10 m.
import numpy as np
from tilupy.analytic_sol import Stocker_wet

x = np.linspace(-5, 10, 100)

# %%
# 
# -------------------

# %%
# Case: Stocker's solution with dam at :math:`x_0 = 0 m`, initial fluid height :math:`h_l = 0.5 m` and initial 
# domain height :math:`h_r = 0.05 m`
case = Stocker_wet(x_0=0, h_l=0.5, h_r=0.05)


# %%
# Compute and plot fluid height at times :math:`t = {0, 0.5, 1, 1.5, 2} s`.
case.compute_h(x, [0, 0.5, 1, 1.5, 2])
case.show_res(show_h=True)


# %%
# Compute and plot fluid velocity at times :math:`t = {0, 0.5, 1, 1.5, 2} s`.
case.compute_u(x, [0, 0.5, 1, 1.5, 2])
case.show_res(show_u=True)

# %%
# 
# -------------------

# %%
# Original reference:
# 
# Stoker JJ. Water Waves: The Mathematical Theory with Applications, Pure and Applied Mathematics, 
# Vol. 4. Interscience Publishers: New York, USA, 1957.


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