Analytic solutions for the front position.
======================

This section demonstrates how to use the Front_result class which allows to calculate the position of the flow front at a given time based on the models cited below.

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

-----------------

**Ritter's equation**: :math:`x_f(t) = 2 t \sqrt{g h_0}`

-----------------

**Stocker's equation (based on Sarkhosh)**: :math:`x_f(t) =c_m t`

with :math:`c_m` the front wave velocity, solution of:

.. math::
    c_m h_r - h_r \left( \sqrt{1 + \frac{8 c_m^2}{g h_r}} - 1 \right) \left( \frac{c_m}{2} - \sqrt{g h_0} + \sqrt{\frac{g h_r}{2} \left( \sqrt{1 + \frac{8 c_m^2}{g h_r}} - 1 \right)} \right) = 0
          
-----------------

**Mangeney's equation**: :math:`x_f(t) = \frac{1}{2}mt^2 + 2 c_0 t`

with :math:`c_0` the initial wave propagation speed defined by: 

.. math::
    c_0 = \sqrt{g h_0 \cos{\theta}}

and :math:`m` the constant horizontal acceleration of the front defined by:

.. math::
    m = g \sin{\theta} - g \cos{\theta} \tan{\delta}

-----------------

**Dressler's equation**: :math:`x_f(t) = 2 t \sqrt{g h_0}`

-----------------

**Chanson's equation**: :math:`x_f(t) = \left( \frac{3}{2} \frac{U(t)}{\sqrt{g h_0}} - 1 \right) t \sqrt{\frac{g}{h_0}} + \frac{4}{f\frac{U(t)^2}{g h_0}} \left( 1 - \frac{U(t)}{2 \sqrt{g h_0}} \right)^4`

with :math:`U(t)` the front wave velocity, solution of:

.. math::
	\left( \frac{U}{\sqrt{g h_0}}  \right)^3 - 8 \left( 0.75 - \frac{3 f t \sqrt{g}}{8 \sqrt{h_0}} \right) \left( \frac{U}{\sqrt{g h_0}}  \right)^2 + 12 \left( \frac{U}{\sqrt{g h_0}}  \right) - 8 = 0       


References
-----------------
Chanson, H., 2005, Applications of the Saint‑Venant Equations and Method of Characteristics to the Dam Break Wave Problem.

Delestre, O., Lucas, C., Ksinant, P.-A., Darboux, F., Laguerre, C., Vo, T.-N.-T., James, F. & Cordier, S., 2013, SWASHES: a compilation of shallow water analytic solutions for hydraulic and environmental studies, International Journal for Numerical Methods in Fluids, v. 72(3), p. 269-300, doi:10.1002/fld.3741.

Dressler, R.F., 1952, Hydraulic resistance effect upon the dam‑break functions, Journal of Research of the National Bureau of Standards, vol. 49(3), p. 217–225.

Mangeney, A., Heinrich, P., & Roche, R., 2000, Analytical solution for testing debris avalanche numerical models, Pure and Applied Geophysics, vol. 157, p. 1081–1096.

Ritter, A., 1892, Die Fortpflanzung der Wasserwellen, Zeitschrift des Vereines Deutscher Ingenieure, vol. 36(33), p. 947–954.

Sarkhosh, P., 2021, Stoker solution package, version 1.0.0, Zenodo. https://doi.org/10.5281/zenodo.5598374

Stoker, J.J., 1957, Water Waves: The Mathematical Theory with Applications, Pure and Applied Mathematics, vol. 4, Interscience Publishers, New York, USA.


Plot examples
-----------------