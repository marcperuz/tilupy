r"""
Coussot's shape solution of flow simulation without friction.
===========================

This example demonstrates the final shape of simulated flow of an ideal dam break on a dry domain.

The dam break is instantaneous, over an inclined and flat surface without friction with a finite 
flow volume.

The shape of the simulated flow at the final step is given by

    .. math::
            X = 
            \begin{cases}
                H - \ln(1 + H) & \text{if } 0 < X \leq X_0, \\\\
                H + L_0 + \ln(1 - H) & \text{if } X_0 < X \leq L_0
            \end{cases}
            
where 

    .. math::
            \begin{cases}
                \text{X : Normalized spatial coordinate}, \\\\
                \text{H : Normalized fluid depth}, \\\\
                \text{X_0 : Normalized spatial coordinate of the maximal fluid depth}, \\\\
                \text{L_0 : Normalized flow lenght}, \\\\
            \end{cases}

For :math:`0 < X \leq X_0` the equation represent the back front of the flow and :math:`X_0 < X \leq L_0` 
represents the front flow. 
"""

# %%
# Initialisation with :math:`l_0 = 10m`, :math:`h_{max} = 0.5m`, :math:`\rho = 1000kg/m^3`, :math:`\tau_c = 500Pa` and 
# :math:`\theta = 10°`:
import matplotlib.pyplot as plt

from tilupy.analytic_sol import Coussot_shape

A = Coussot_shape(l0=10, rho=1000, tau=500, theta=10, hmax=0.5)
A.compute_Xx_front()
A.show_res()

# %%
# An other way to compute the shape of the front flow is to compute this equation:
# .. math::
#       X = -H - \ln(1-H)
# With the same parameters, we obtain:

A = Coussot_shape(l0=10, rho=1000, tau=500, theta=10, hmax=0.5)
A.compute_Xx_front_remaitre()
A.show_res()

# %%
# The two solutions aren't exactly the same:
A = Coussot_shape(l0=10, rho=1000, tau=500, theta=10, hmax=0.5)
A.compute_Xx_front()

B = Coussot_shape(l0=10, rho=1000, tau=500, theta=10, hmax=0.5)
B.compute_Xx_front_remaitre()

fig, ax = plt.subplots()
ax.plot(A.x, A.h, label="First solution")
ax.plot(B.x, B.h, label="Second solution")
ax.legend()
ax.set_xlabel('x [m]')
ax.set_ylabel('h [m]')

ax.plot([A.x[0], A.x[-1]], [0, 0], color='black', linewidth=2)
fig.show()


# %%
# The result is different if we choose :math:`\theta = 0°`:
A = Coussot_shape(l0=10, rho=1000, tau=500, theta=0, hmax=0.5)
A.compute_Xx()
# A.show_res()

B = Coussot_shape(l0=10, rho=1000, tau=500, theta=0, hmax=0.5)
B.compute_Xx_front_remaitre()
# B.show_res()


fig, ax = plt.subplots()
ax.plot(A.x, A.h, label="First solution")
ax.plot(B.x, B.h, label="Second solution")
ax.legend()
ax.set_xlabel('x [m]')
ax.set_ylabel('h [m]')

ax.plot([A.x[0], A.x[-1]], [0, 0], color='black', linewidth=2)
fig.show()