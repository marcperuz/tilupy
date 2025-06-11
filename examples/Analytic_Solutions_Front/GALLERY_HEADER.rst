Front position example.
======================

Those examples demonstrate how to use the Front_result class, which takes the equations from the solutions 
presented in the flow height and velocity section of Analytic Solutions to calculate only the position of 
the flow front at a specified time.

Model equations
-----------------
Notations:
 - :math:`x_f`: front position.
 - :math:`h_0`: intial water depth.
 - :math:`h_r`: domain depth for wet dam-break problem.
 - :math:`\theta`: slope angle.
 - :math:`\delta`: friction angle.
 - :math:`g`: gravitational acceleration.
 - :math:`t`: time instant.
 - :math:`f`: Darcy friction coefficient.

All references are present in the corresponding section in the API or in the flow height and velocity section.

**Ritter's equation**: :math:`x_f(t) = 2 t \sqrt{g h_0}`

**Stocker's equation**: :math:`x_f(t) =t c_m`

with :math:`c_m` the front wave velocity solution of:

.. math::
    c_m h_r - h_r \left( \sqrt{1 + \frac{8 c_m^2}{g h_r}} - 1 \right) \left( \frac{c_m}{2} - \sqrt{g h_0} + \sqrt{\frac{g h_r}{2} \left( \sqrt{1 + \frac{8 c_m^2}{g h_r}} - 1 \right)} \right) = 0
          
**Mangeney's equation**: :math:`x_f(t) = \frac{1}{2}mt - 2 c_0 t`

with :math:`c_0` the initial wave propagation speed defined by: 

.. math::
    c_0 = \sqrt{g h_0 \cos{\theta}}

and :math:`m` the constant horizontal acceleration of the front defined by:

.. math::
    m = -g \sin{\theta} + g \cos{\theta} \tan{\delta}

**Dressler's equation**: :math:`x_f(t) = 2 t \sqrt{g h_0}`

**Chanson's equation**: :math:`x_f(t) = \left( \frac{3}{2} \frac{U(t)}{\sqrt{g h_0}} - 1 \right) t \sqrt{\frac{g}{h_0}} + \frac{4}{f\frac{U(t)^2}{g h_0}} \left( 1 - \frac{U(t)}{2 \sqrt{g h_0}} \right)^4`

with :math:`U(t)` the front wave velocity solution of:

.. math::
	\left( \frac{U}{\sqrt{g h_0}}  \right)^3 - 8 \left( 0.75 - \frac{3 f t \sqrt{g}}{8 \sqrt{h_0}} \right) \left( \frac{U}{\sqrt{g h_0}}  \right)^2 + 12 \left( \frac{U}{\sqrt{g h_0}}  \right) - 8 = 0       


Plot examples
-----------------