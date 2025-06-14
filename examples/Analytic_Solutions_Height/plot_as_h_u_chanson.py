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
			\delta_t h + \delta_x (hu) = 0 \\\\
			h \delta_t u + hu \delta_x u + hg\cos{\theta} \delta_x h = gh\sin{\theta} - S
		\end{cases}

It is possible to transform the equations visible in Chanson (2005) to find this form of the Saint-Venant equations, which gives us:

.. math::
		\begin{cases}
			\delta_t h + \delta_x (hu) = 0 \\\\
			h \delta_t u + hu \delta_x u + hg \delta_x h = - S
		\end{cases}

with :math:`S = \frac{f u^2}{6}` the source term integrating the dissipative effects due to friction. 

In fluid simulation, hydraulic models can be used to express the source term S. 
For example, we can cite an equation combining the Darcy-Weisbach and Manning laws:

.. math::
		S = g n^2 \frac{u^2}{h^{1/3}}

where :math:`n` is Manning coefficient (in :math:`s.m^{-1/3}`).

A more general way to write this expression is :math:`S = k \frac{u^2}{h^p}` with :math:`k` a constant containing the effects of friction :math:`p`
a coefficient to homogenize the equation.

Darcy friction factor being dimensionless, we have :math:`p = 0` and :math:`k = \frac{f}{6}` for this hydraulic source term model.


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
                x_A(t) = - t \sqrt{g h_0}, \\\\
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
case = Chanson_dry(h_0=10, f=0.05)


# %%
# Compute and plot fluid height at times :math:`t = {0, 1, 20, 30, 50} s`.
case.compute_h(x, [0, 1, 20, 30, 50])
case.show_res(show_h=True, linestyles=[":", "-.", "--", "-"])


# %%
# 
# -------------------

# %%
# Original reference:
# 
# Chanson, H., 2005, Applications of the Saint‑Venant Equations and Method of Characteristics to the Dam Break Wave Problem.
