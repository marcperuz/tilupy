Flow height and velocity example.
======================

Comparison of dam-break analytical solutions:

.. list-table::
   :header-rows: 1
   :widths: 15 20 20 20 20

   * - **Feature**
     - **Ritter**
     - **Stocker**
     - **Mangeney**
     - **Dressler**

   * - **Domain**
     - Dry bed
     - Wet bed
     - Dry bed
     - Dry bed

   * - **Friction Modeling**
     - Ignored
     - Ignored
     - With friction angle :math:`\delta`
     - With Chézy coefficient :math:`C`
	
   * - **Bed Slope**
     - Only horizontal
     - Only horizontal
     - Inclined
     - Only horizontal
	 
   * - **Initial Fluid Height**
     - :math:`h_0 > 0`
     - :math:`h_0 > 0`
     - :math:`h_0 > 0`
     - :math:`h_0 > 0`

   * - **Initial Domain Fluid Height**
     - :math:`0`
     - :math:`0 < h_r < h_0`
     - :math:`0`
     - :math:`0`

   * - **Initial Velocity**
     - :math:`u(x,0) = 0`
     - :math:`u(x,0) = 0`
     - :math:`u(x,0) = 0`
     - :math:`u(x,0) = 0`

   * - **Solution Zones**
     - 3 (reservoir, rarefaction, dry)
     - 4 (reservoir, rarefaction, shock, wet)
     - 3 (reservoir, rarefaction, dry)
     - 4 (reservoir, rarefaction, tip, dry)

   * - **Shock Wave**
     - None
     - Yes
     - None
     - None

   * - **Wave Speeds**
     - Closed-form
     - Implicit via :math:`c_m`
     - Closed-form
     - Corrected rarefaction with friction terms

   * - **Reference**
     - Ritter (1892)
     - Stocker (1957)
     - Mangeney et al. (2000)
     - Dressler (1952)


Below is a gallery of example function of the package to show the class Depth_Result works: