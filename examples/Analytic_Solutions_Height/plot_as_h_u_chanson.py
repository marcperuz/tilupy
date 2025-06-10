r"""
Analytical solution of a dam-break problem on a dry slope with friction (Chanson)
=============================================================================================

This example presents the classical one-dimensional analytical solution proposed by Chanson (2005)
for the dam-break problem on a dry, horizontal bed with friction.


Model Assumptions
-----------------

- Instantaneous dam break at position :math:`x = x_0 = 0` and at time :math:`t = 0`.
- The domain is initially dry for :math:`x > x_0`, with a finite volume of still water (with height :math:`h_0`) on the left of the dam (:math:`0 < x \leq x_0`).
- The bed is flat and horizontal (no slope).
- Basal friction is modeled via Darcy friction factor :math:`f`.
- The fluid is incompressible, homogeneous, and only subject to gravity and friction.


Initial Conditions
------------------

.. math::
        h(x, 0) = 
        \begin{cases}
            h_0 > 0 & \text{for } 0 < x \leq x_0, \\\\
            0 & \text{for } x_0 < x,
        \end{cases}
        
where :math:`x_0` is the initial dam location and :math:`h_0` is the height of the water column.


Analytical Solution
-------------------

The water height and velocity profiles for :math:`t > 0` are given by:

    .. math::
            h(x, t) = 
            \begin{cases}
                h_0 & \text{if } x \leq x_A(t), \\\\
                \frac{4}{9g} \left( \sqrt{g h_0} - \frac{x - x_0}{2t} \right)^2 & \text{if } x_A(t) < x \leq x_B(t), \\\\
                \sqrt{\frac{f}{4} \frac{U(t)^2}{g h_0} \frac{x_C(t)-x}{h_0}} & \text{if } x_B(t) < x \leq x_C(t), \\\\
                0 & \text{if } x_C(t) < x
            \end{cases}

where the positions of the rarefaction wave front, the tip position and the dry front are:

    .. math::
            \begin{cases}
                x_A(t) = t \sqrt{g h_l}, \\\\
                x_B(t) = \left( \frac{3}{2} \frac{U(t)}{\sqrt{g h_0}} - 1 \right) t \sqrt{g h_0}, \\\\
                x_C(t) = \left( \frac{3}{2} \frac{U(t)}{\sqrt{g h_0}} - 1 \right) t \sqrt{\frac{g}{h_0}} + \frac{4}{f\frac{U(t)^2}{g h_0}} \left( 1 - \frac{U(t)}{2 \sqrt{g h_0}} \right)^4
            \end{cases}

with the celerity of the wave front :math:`U(t)` solution of:

    .. math::
            \left( \frac{U}{\sqrt{g h_0}}  \right)^3 - 8 \left( 0.75 - \frac{3 f t \sqrt{g}}{8 \sqrt{h_0}} \right) \left( \frac{U}{\sqrt{g h_0}}  \right)^2 + 12 \left( \frac{U}{\sqrt{g h_0}}  \right) - 8 = 0       


Implementation
--------------
"""
# %%
# First import required packages and define the spatial domain for visualization: 1D space from -5 to 15 m.
import numpy as np
from tilupy.analytic_sol import Chanson_dry

x = np.linspace(-600, 600, 1000)

# %%
# 
# -------------------

# %%
# Case: Chanson's solution with dam at :math:`x_0 = 0 m`, initial height :math:`h_l = 10 m` and friction coefficient
# :math:`f = 0.05`
case = Chanson_dry(h_0=10, f=0.05)


# %%
# Compute and plot fluid height at times :math:`t = {0, 1, 10, 30, 40, 60} s`.
case.compute_h(x, [0, 1, 10, 30, 40, 60])
case.show_res(show_h=True)


# %%
# 
# -------------------

# %%
# Original reference:
# 
# Chanson, Hubert. (2005). Analytical Solution of Dam Break Wave with Flow Resistance: Application to Tsunami Surges. 137. 