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


Basic equations
---------------
As a reminder, the general formula of the Saint-Venant equation system is:

.. math::
		\begin{cases}
			\partial_t h + \partial_x (hu) = 0 \\\\
			\partial_t (hu) + \partial_x (hu^2) + \frac{1}{2}g\cos{\theta} \partial_x (h^2) = gh\sin{\theta} - S
		\end{cases}

with:

 - :math:`h` the fluid depth
 - :math:`u` the fluid velocity
 - :math:`g` the gravitational acceleration
 - :math:`\theta` the surface slope
 - :math:`S` source term

Here is equation 1 from Chanson (2005) with the same notation and in 1D:

.. math::
		\begin{cases}
			\partial_t h + h \partial_x (u) + u \partial_x (h) = 0 \\\\
			\partial_t u + u \partial_x u + g \partial_x h + g S_f = 0
		\end{cases}

with :math:`S_f = \frac{f}{2} \frac{u^2}{g D_H}`, :math:`f` being Darcy friction factor and :math:`D_H` the hydraulic diameter. 

By transforming these equations, we find the Saint-Venant equations:

.. math::
		\begin{cases}
			\partial_t h + \partial_x (hu) = 0 \\\\
			h \partial_t u + hu \partial_x u + hg \partial_x h = - S
		\end{cases}

with :math:`S = \frac{h f u^2}{2 D_H}` the source term integrating the dissipative effects due to friction. 
In those conditions (approximate to a wide rectangular channel), the hydraulic diameter can be expressed :math:`D_H = 4 h` (see footnote number 5 of Chanson (2005)).

With this we have finally the source term :math:`S = \frac{f u^2}{8}`.

In fluid simulation, hydraulic models can be used to express the source term :math:`S`.
For example, we can cite an equation combining the Darcy-Weisbach and Manning laws:

.. math::
		S_g = g n^2 \frac{u^2}{h^{1/3}}

where :math:`n` is Manning coefficient (in :math:`s.m^{-1/3}`).

These forms diverge in that the fluid height :math:`h` is not present in the expression found for Chanson. However, it can be compared to Voellmy's expression by taking 
a friction coefficient :math:`\mu = 0` and with an empirical coefficient :math:`\xi = \frac{8}{f}`:

.. math::
		S = h \mu \left( g \cos{\theta} + \gamma u^2 \right) + \frac{u^2}{\xi}


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
                x_A(t) = x_0 - t \sqrt{g h_0}, \\\\
                x_B(t) = x_0 + \left( \frac{3}{2} \frac{U(t)}{\sqrt{g h_0}} - 1 \right) t \sqrt{g h_0}, \\\\
                x_C(t) = x_0 + \left( \frac{3}{2} \frac{U(t)}{\sqrt{g h_0}} - 1 \right) t \sqrt{\frac{g}{h_0}} + \frac{4}{f\frac{U(t)^2}{g h_0}} \left( 1 - \frac{U(t)}{2 \sqrt{g h_0}} \right)^4
            \end{cases}

with the celerity of the wave front :math:`U(t)` solution of:

    .. math::
            \left( \frac{U}{\sqrt{g h_0}}  \right)^3 - 8 \left( 0.75 - \frac{3 f t \sqrt{g}}{8 \sqrt{h_0}} \right) \left( \frac{U}{\sqrt{g h_0}}  \right)^2 + 12 \left( \frac{U}{\sqrt{g h_0}}  \right) - 8 = 0       


Implementation
--------------
"""

# %%
# First import required packages and define the spatial domain for visualization.
# For following examples we will use a 1D space from -600 to 600 m.
import numpy as np
from tilupy.analytic_sol import Chanson_dry

x = np.linspace(-600, 600, 1000)

# %%
#
# -------------------

# %%
# Case: Chanson's solution with dam at :math:`x_0 = 0 m`, initial height :math:`h_0 = 10 m` and friction coefficient
# :math:`f = 0.05`
case = Chanson_dry(h_0=10, x_0=0, f=0.05)


# %%
# Compute and plot fluid height at times :math:`t = {0, 1, 20, 30, 50} s`.
case.compute_h(x, [0, 1, 20, 30, 50])
ax = case.plot(show_h=True, linestyles=["", ":", "-.", "--", "-"])


# %%
#
# -------------------

# %%
# Original reference:
#
# Chanson, H., 2005, Applications of the Saint‑Venant Equations and Method of Characteristics to the Dam Break Wave Problem. https://espace.library.uq.edu.au/view/UQ:9438
