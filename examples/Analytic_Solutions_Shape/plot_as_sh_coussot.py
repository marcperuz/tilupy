r"""
Coussot's model example
===========================

This example demonstrates the final shape of simulated flow using Coussot's equation.

The frontal lobe shape of the simulated flow at the final step is given by

    .. math::
            D = - H - \ln(1 - H)
            
where 
 - :math:`H`: normalized fluid depth.
 - :math:`D`: normalized distance of the front from the origin.
 
:math:`H` and :math:`D` are obtained with these expressions:

.. math::
	D = \frac{\rho g d (\sin(\theta))^2}{\tau_c \cos(\theta)} \text{  and  } H = \frac{\rho g h \sin(\theta)}{\tau_c}

with:
 - :math:`h`: fluid depth.
 - :math:`x`: distance of the front from the origin.
 - :math:`g`: gravitational acceleration.
 - :math:`\rho`: fluid density.
 - :math:`\tau_c`: threshold constraint.
 - :math:`\theta`: slope of the surface.

Implementation
---------------
"""
# %%
# First import required packages and define the context. For this example we will use a fluid with a density of :math:`\rho = 1000 kg/m^3`: 
# and :math:`\tau_c = 50 Pa`, with a slope of :math:`\theta = 10°`:
from tilupy.analytic_sol import Coussot_shape
import matplotlib.pyplot as plt

case_1 = Coussot_shape(rho=1000, tau=50, theta=10)
case_1.compute_rheological_test_front_morpho()
plt.plot(case_1.x, case_1.h, color="black")
plt.show()

# %%
# If :math:`\theta = 0°`, the equations are slightly different:
# 
# .. math::
# 	D^* = \frac{{H^*}^2}{2}
# 
# with:
# 
# .. math::
# 	D^* = \frac{\rho g d}{\tau_c} \text{  and  } H^* = \frac{\rho g h}{\tau_c}

case_2 = Coussot_shape(rho=1000, tau=50, theta=0)
case_2.compute_rheological_test_front_morpho(h_final=1)
plt.plot(case_2.x, case_2.h, color="black")
plt.show()

# %%
# Original reference:
#  
# Coussot, P., Proust, S., & Ancey, C., 1996, Rheological interpretation of deposits of yield stress fluids, 
# Journal of Non-Newtonian Fluid Mechanics, v. 66(1), p. 55-70, doi:10.1016/0377-0257(96)01474-7.