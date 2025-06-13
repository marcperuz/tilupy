Analytic solutions for height and velocity profile. 
======================

This section demonstrates how to use the Depth_result class which allows to calculate the depth and velocity profile of the flow at a given time.

To simplify the calculations, only 1D models will be taken into account. Under the assumptions of one-dimensional flow and an incompressible fluid, 
it is possible to model the flow using the Saint-Venant equations, which form a system of partial differential equations derived from the principles 
of mass conservation (continuity equation) and momentum conservation (momentum equation). In a case of a flat surface, the Saint-Venant's system can
be expressed as:

.. math::
		\begin{cases}
			\delta_t h + \delta_x (hu) = 0 \\\\
			\delta_t (hu) + \delta_x (\alpha hu^2) + \delta_x (\frac{1}{2}kgh^2 \cos{\theta}) = gh\sin{\theta} - S
		\end{cases}

where

 - :math:`h`: fluid depth.
 - :math:`u`: fluid velocity.
 - :math:`g`: gravitational acceleration.
 - :math:`\alpha`: form factor associated with nonlinear velocity profiles.
 - :math:`k`: coefficient introduced to take into account internal friction.
 - :math:`\delta_t`: partial derivative with respect to time.
 - :math:`\delta_x`: partial derivative with respect to space.
 - :math:`\theta`: slope of the surface.
 - :math:`S`: source term integrating the dissipative effects of energy which slows the flow.

It is possible to rewrite this system in a more general form. For example, by taking :math:`\alpha = 1` and :math:`k = 1` (a very common thing), 
and by developing the partial derivatives, we obtain this system which will serve as a basis for comparing the different models:

.. math::
		\begin{cases}
			\delta_t h + \delta_x (hu) = 0 \\\\
			h \delta_t u + hu \delta_x u + hg\cos{\theta} \delta_x h = gh\sin{\theta} - S
		\end{cases}

As already said, the source term :math:`S` contains all dissipative effects of energy which slows the flow (due to friction or viscosity). A large number of 
models exist to express this term as a function of flow conditions.

For example, we can cite an equation combining the Darcy-Weisbach and Manning laws:

.. math::
		S = g n^2 \frac{u^2}{h^{1/3}}

where :math:`n` is Manning coefficient (in :math:`s.m^{-1/3}`).

----------------

**References**

Peruzzetto, M., Grandjean, G., Mangeney, A., Levy, C., Thiery, Y., Vittecoq, B., Bouchut, F., Fontaine, 
F.R., & Komorowski, J.-C., 2023, Simulation des écoulements gravitaires avec les modèles d’écoulement en couche mince : 
état de l’art et exemple d’application aux coulées de débris de la Rivière du Prêcheur (Martinique, Petites Antilles), 
Revue Française de Géotechnique, vol. 176, p. 1, doi:10.1051/geotech/2023020.

Savage, S.B., & Hutter, K., 1991, The dynamics of avalanches of granular materials from initiation to runout. Part I: Analysis, Acta Mechanica, v. 86(1), p. 201–223.


Dam-break problem
-----------------

Below are grouped several analytical solutions for a dam-break problem under conditions varying from one model to another. The next table will 
summarize the main differences between the models:

.. list-table::
   :header-rows: 1
   :widths: 15 20 20 20 20 20

   * - **Feature**
     - **Ritter**
     - **Stocker**
     - **Mangeney**
     - **Dressler**
     - **Chanson**

   * - **Domain**
     - Dry bed
     - Wet bed
     - Dry bed
     - Dry bed
     - Dry bed

   * - **Friction Modeling**
     - Ignored
     - Ignored
     - With friction angle :math:`\delta`
     - With Chézy coefficient :math:`C`
     - With Darcy friction coefficient :math:`f`
	
   * - **Bed Slope**
     - Only horizontal
     - Only horizontal
     - Inclined
     - Only horizontal
     - Only horizontal
	 
   * - **Initial Fluid Height**
     - :math:`h_0 > 0`
     - :math:`h_0 > 0`
     - :math:`h_0 > 0`
     - :math:`h_0 > 0`
     - :math:`h_0 > 0`

   * - **Initial Domain Fluid Height**
     - :math:`0`
     - :math:`0 < h_r < h_0`
     - :math:`0`
     - :math:`0`
     - :math:`0`

   * - **Initial Velocity**
     - :math:`u(x,0) = 0`
     - :math:`u(x,0) = 0`
     - :math:`u(x,0) = 0`
     - :math:`u(x,0) = 0`
     - :math:`u(x,0) = 0`

   * - **Solution Zones**
     - 3 (reservoir, rarefaction, dry)
     - 4 (reservoir, rarefaction, shock, wet)
     - 3 (reservoir, rarefaction, dry)
     - 4 (reservoir, rarefaction, tip, dry)
     - 4 (reservoir, rarefaction, tip, dry)

   * - **Shock Wave**
     - None
     - Yes
     - None
     - None
     - None

   * - **Wave Speeds**
     - Closed-form
     - Via :math:`c_m`
     - Closed-form
     - Via :math:`c_m`
     - Via :math:`U(t)`
	 
   * - **Initial water volume**
     - Finite
     - Finite
     - Infinite
     - Finite
     - Finite

   * - **Reference**
     - Ritter (1892)
     - Stocker (1957)
     - Mangeney et al. (2000)
     - Dressler (1952)
     - Chanson (2005)


Below is a gallery of functions for each model type illustrating their specificity and how they work: