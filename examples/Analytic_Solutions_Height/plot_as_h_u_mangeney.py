r"""
Analytical solution of a dam-break problem on a dry slope with friction (Mangeney et al.)
=========================================================================================

This script presents a one-dimensional analytical solution of a dam-break problem,
as proposed by Mangeney et al. (2000), for gravity-driven flow on a dry, inclined plane,
including basal friction.


Model Assumptions
-----------------

- Instantaneous dam break at time :math:`t = 0`.
- The downstream domain (:math:`x > 0`) is initially dry, while the upstream side (:math:`x < 0`) contains a fluid column of height :math:`h_0`.
- The flow occurs on a constant slope inclined at an angle :math:`\theta`.
- Basal friction is modeled via a constant friction angle :math:`\delta`, leading to a constant horizontal acceleration.
- The fluid is incompressible, homogeneous, and only subject to gravity and basal friction.
- The initial fluid volume is considered infinite.


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
 
Here is equation 1 and 2 from Mangeney (2000) with the same notation and in 1D:

.. math::
		\begin{cases}
			\partial_t h + \partial_x (hu) = 0 \\\\
			h \partial_t u + hu \partial_x u + hg\cos{\theta} \partial_x h = gh\sin{\theta} - F
		\end{cases}

with :math:`F = g \cos{\theta} \tan{\delta}`, the source term integrating the dissipative effects due to friction for a 
Coulomb-type friction law.

The general form for an hydrostatic model with basal friction (Coulomb-type friction law) can be expressed:

.. math::
		S = h \mu \left( g \cos{\theta} + \gamma u^2 \right) 
  
with :math:`\gamma = \frac{1}{R}`, :math:`R` being the radius of curvature.

Since we are on a flat surface and with a flow restricted only by internal friction, we have :math:`\gamma = 0` and :math:`\mu = \tan{\delta}`,
allowing to obtain Mangeney's relation.


Initial Conditions
------------------

    .. math::
            h(x, 0) = 
            \begin{cases}
                h_0 > 0 & \text{for } x < x_0, \\\\
                0 & \text{for } x_0 < x,
            \end{cases}

    .. math::
            u(x, 0) = 0

where :math:`x_0` is the initial dam location and :math:`h_0` is the height of the water column.

Analytical Solution
-------------------

The analytical expressions for fluid height and velocity at time t are given by:

    .. math::
            h(x, t) = 
            \begin{cases}
                h_0 & \text{if } x \leq x_A(t), \\\\
                \frac{1}{9g cos(\theta)} \left(2 c_0 - \frac{x-x_0}{t}  + \frac{1}{2} m t \right)^2 & \text{if } x_A(t) < x \leq x_B(t), \\\\
                0 & \text{if } x_B(t) < x,
            \end{cases}
            
    .. math::
            u(x,t) = 
            \begin{cases}
                0 & \text{if } x \leq x_A(t), \\\\
                \frac{2}{3} \left( \frac{x-x_0}{t} + c_0 + mt \right) & \text{if } x_A(t) < x \leq x_B(t), \\\\
                0 & \text{if } x_B(t) < x,
            \end{cases}
            
where 

    .. math::
            \begin{cases}
                x_A(t) = x_0 + \frac{1}{2}mt^2 - c_0 t, \\\\
                x_B(t) = x_0 + \frac{1}{2}mt^2 + 2 c_0 t
            \end{cases}
            
with :math:`c_0` the initial wave propagation speed defined by: 

.. math::
    c_0 = \sqrt{g h_0 \cos{\theta}}

and :math:`m` the constant horizontal acceleration of the front defined by:

.. math::
    m = g \sin{\theta} - g \cos{\theta} \tan{\delta}


Implementation
--------------

The following example replicates the case shown in Figure 3 of Mangeney et al. (2000).
"""

# %%
# 
# -------------------

# %%
# First import required packages and define the spatial domain.
# For following examples we will use a 1D space from -100 to 1000 m.
import numpy as np
from tilupy.analytic_sol import Mangeney_dry

x = np.linspace(-100, 1000, 1000)

# %%
# 
# -------------------

# %%
# Case 1: No friction (:math:`\delta = 0°`), slope :math:`\theta = 30°`, initial height :math:`h_0 = 20 m`.
case_1 = Mangeney_dry(h_0=20, x_0= 0, theta=30, delta=0)


# %%
# Compute and plot the fluid height at times :math:`t = {0, 5, 10, 15} s`.
case_1.compute_h(x, [0, 5, 10, 15])
ax = case_1.plot(show_h=True, linestyles=["", ":", "--", "-"])


# %%
# Compute and plot the fluid velocity at times :math:`t = {0, 5, 10, 15} s`.
case_1.compute_u(x, [0, 5, 10, 15])
ax = case_1.plot(show_u=True, linestyles=["", ":", "--", "-"])

# %%
# 
# -------------------

# %%
# Case 2: Add basal friction (:math:`\delta = 20°`), same slope and initial height.
case_2 = Mangeney_dry(h_0=20, x_0= 0, theta=30, delta=20)


# %%
# Compute and plot the fluid height at times :math:`t = {0, 5, 10, 15} s`.
case_2.compute_h(x, [0, 5, 10, 15])
ax = case_2.plot(show_h=True, linestyles=["", ":", "--", "-"])


# %%
# Original reference:
#  
# Mangeney, A., Heinrich, P., & Roche, R., 2000, Analytical solution for testing debris avalanche numerical models, 
# Pure and Applied Geophysics, vol. 157, p. 1081-1096.