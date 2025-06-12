Analytic solutions for the morphology of the final state of the front.
======================

This section demonstrates how to use the Shape_result class which allows to compute the shape of the flow front at the final step of the flow.

Coussot's model
---------------

Coussot et al proposed a formula in 1996 to approximate the geometry of the frontal lobe of a flow at a final instant for rheological test on an inclined surface:

.. math::
	X = H + \ln{(1 - H)}
	
with :math:`X` and :math:`H` being the normalized version of the spatial dimension :math:`x` and the fluid depth :math:`h`, obtained by computing:

.. math::
	X = \frac{\rho g x (\sin(\theta))^2}{\tau_c \cos(\theta)} \text{  and  } H = \frac{\rho g h \sin(\theta)}{\tau_c}
	
where

 - :math:`h`: fluid depth.
 - :math:`x`: spatial dimension.
 - :math:`g`: gravitational acceleration.
 - :math:`\rho`: fluid density.
 - :math:`\tau_c`: threshold constraint.
 - :math:`\theta`: slope of the surface.

In a case where :math:`\theta = 0`, the equations are slightly different:

.. math::
	X^* = \frac{{H^*}^2}{2}

with:

.. math::
	X^* = \frac{\rho g x}{\tau_c} \text{  and  } H^* = \frac{\rho g h}{\tau_c}


References
----------
Coussot, P., Proust, S., & Ancey, C., 1996, Rheological interpretation of deposits of yield stress fluids, Journal of Non-Newtonian Fluid Mechanics, v. 66(1), p. 55–70, doi:10.1016/0377-0257(96)01474-7.


Examples
--------