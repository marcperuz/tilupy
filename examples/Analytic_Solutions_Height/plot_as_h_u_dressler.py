r"""
Analytical solution of a dam-break problem on a dry slope with friction (Dressler)
=============================================================================================

This example presents the classical one-dimensional analytical solution proposed by Dressler (1952)
for the dam-break problem on a dry, horizontal bed with friction.


Model Assumptions
-----------------

- Instantaneous dam break at position :math:`x = x_0` and at time :math:`t = 0`.
- The domain is initially dry for :math:`x > x_0`, with a finite volume of still water (with height :math:`h_l`) on the left of the dam (:math:`0 < x \leq x_0`).
- The bed is flat and horizontal (no slope).
- Basal friction is modeled via Chézy coefficient :math:`C`.
- The fluid is incompressible, homogeneous, and only subject to gravity and basal friction.


Basic equations
---------------
As a reminder, the general formula of the Saint-Venant equation system is:

.. math::
		\begin{cases}
			\delta_t h + \delta_x (hu) = 0 \\\\
			h \delta_t u + hu \delta_x u + hg\cos{\theta} \delta_x h = gh\sin{\theta} - S
		\end{cases}

with:

 - :math:`h` the fluid depth
 - :math:`u` the fluid velocity
 - :math:`g` the gravitational acceleration
 - :math:`\theta` the surface slope
 - :math:`S` source term

Here is equation 1 from Dressler (1952) with the same notation and in 1D:

.. math::
		\begin{cases}
			2 \delta_t c + c \delta_x (u) + 2u \delta_x (c) = 0 \\\\
			\delta_t u + u \delta_x u + 2c \delta_x c + R \left( \frac{u^2}{c^2} \right)^2 = 0
		\end{cases}

with :math:`c = \sqrt{gh}` and :math:`R` being roughness coefficient. 

By transforming these equations, we find the general expression of the Saint-Venant equations:

.. math::
		\begin{cases}
			\delta_t h + \delta_x (hu) = 0 \\\\
			h \delta_t u + hu \delta_x u + hg \delta_x h = - S
		\end{cases}

with :math:`S = R \frac{u^2}{g}` the source term integrating the dissipative effects due to friction.

In fluid simulation, hydraulic models can be used to express the source term :math:`S`. 
For example, we can cite an equation combining the Darcy-Weisbach and Manning laws:

.. math::
		S_g = g n^2 \frac{u^2}{h^{1/3}}

where :math:`n` is Manning coefficient (in :math:`s.m^{-1/3}`).

By replacing the Manning coefficient with Chezy coefficient with the relation: :math:`n = \frac{h^{1/6}}{C}`, we obtain :

.. math::
		S_g = \frac{g}{C^2} u^2
  
By identification, we can match the two expressions of the source term by applying :math:`R = \frac{g^2}{C^2}`. 


Initial Conditions
------------------

.. math::
        h(x, t) = 
        \begin{cases}
            h_l > 0 & \text{for } 0 < x \leq x_0, \\\\
            0 & \text{for } x_0 < x,
        \end{cases}
        
.. math::
        u(x, t) = 0
        
where :math:`x_0` is the initial dam location and :math:`h_l` is the height of the water column.


Analytical Solution
-------------------

The water height and velocity profiles for :math:`t > 0` are given by:

    .. math::
            h(x, t) = 
            \begin{cases}
                h_0 & \text{if } x \leq x_A(t), \\\\
                \frac{1}{g} \left( \frac{2}{3} \sqrt{g h_0} - \frac{x - x_0}{3t} + \frac{g^{2}}{C^2} \alpha_1 t \right)^2 & \text{if } x_A(t) < x \leq x_t(t), \\\\
                \frac{-b-\sqrt{b^2 - 4 a (c-x(t))}}{2 a}    & \text{if } x_t(t) < x \leq x_B(t), \\\\
                0 & \text{if } x_B(t) < x,
            \end{cases}
        
with :math:`r = \frac{dx}{dh}`, :math:`c = x_B(t)`, :math:`a = \frac{r h_t + c - x_t}{h_t^2}`, :math:`b = r - 2 a h_t`. :math:`x_t` and :math:`h_t` being the position
and the flow depth at the beginning of the tip area.

    .. math::
            u(x,t) = 
            \begin{cases}
                0 & \text{if } x \leq x_A(t), \\\\
                u_{co} = \frac{2\sqrt{g h_0}}{3} + \frac{2(x - x_0)}{3t} + \frac{g^2}{C^2} \alpha_2 t & \text{if } x_A(t) < x \leq x_t(t), \\\\
                \max_{x \in [x_A(t), x_t(t)]} u_{co}(x, t) & \text{if } x_t(t) < x \leq x_B(t), \\\\
                0 & \text{if } x_B(t) < x,
            \end{cases}

where the positions of the rarefaction wave front and the dry front are:

    .. math::
            \begin{cases}
                x_A(t) = x_0 - t \sqrt{g h_l}, \\\\
                x_B(t) = x_0 + 2 t \sqrt{g h_l}
            \end{cases}

The position of the tip area :math:`x_t(t)` is define by the position where the velocity :math:`u(x, t)` reaches the maximum. In the tip region, the velocity is uniform, 
but there is no information concerning the flow depth. In order to have an idea of the water height, a second order interpolation is used between :math:`x_t(t)` and :math:`x_B(t)`.

To take into account the Chézy coefficient in the flow simulation, corrective terms :math:`\alpha_1` and :math:`\alpha_2`are added to the equations:

    .. math::
            \begin{cases}
                \xi = \frac{x-x_0}{t\sqrt{g h_l}}
                \alpha_1(\xi) = \frac{6}{5(2-\xi)} - \frac{2}{3} + \frac{4 \sqrt{3}}{135} (2-\xi)^{3/2}), \\\\
                \alpha_2(\xi) = \frac{12}{2-(2-\xi)} - \frac{8}{3} + \frac{8 \sqrt{3}}{189} (2-\xi)^{3/2}) - \frac{108}{7(2 - \xi)}, \\\\
            \end{cases}
            

Implementation
--------------
"""
# %%
# First import required packages and define the spatial domain for visualization. 
# For following examples we will use a 1D space from -500 to 700 m.
import numpy as np
from tilupy.analytic_sol import Dressler_dry

x = np.linspace(-500, 700, 1000)

# %%
# 
# -------------------

# %%
# Case: Dressler's solution with dam at :math:`x_0 = 0 m`, initial height :math:`h_l = 6 m` and Chézy 
# coefficient :math:`C = 40`. 
case = Dressler_dry(x_0=0, h_0=6, C=40)


# %%
# Compute and plot fluid height at times :math:`t = {0, 10, 20, 30, 40} s`.
case.compute_h_u(x, T=[0, 10, 20, 30, 40])
case.show_res(show_h=True, linestyles=["", ":", "-.", "--", "-"])


# %%
# Plot fluid velocity at times :math:`t = {0, 10, 20, 30, 40} s`.
case.show_res(show_u=True, linestyles=["", ":", "-.", "--", "-"])


# %%
# Original reference:
# 
# Dressler, R.F., 1952, Hydraulic resistance effect upon the dam‑break functions, Journal of Research of the National Bureau of Standards, 
# vol. 49(3), p. 217–225.
