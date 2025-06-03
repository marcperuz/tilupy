r"""
Front position of simulated flow.
===========================

This example demonstrates different equation to find the front position of a simulated flow at a 
specific time.

For all examples, we will have :math:`\theta = 30°`, :math:`h_0 = 20m` and :math:`t = 5s`:
"""
from tilupy.analytic_sol import Front_result

A = Front_result(theta=30, h0=20)
B = Front_result(theta=30, h0=20)
t = 5

# %%
# First example using Mangeney and Dressler's equations for a dam-break solution over an infinite 
# inclined dry domain with friction:
# 
# ---------------------------------
# 
# **Mangeney's equation**: :math:`x_f(t) = \frac{1}{2}mt - 2 c_0 t`
# 
# with :math:`c_0` the initial wave propagation speed defined by: 
# 
# .. math::
#     c_0 = \sqrt{g h_0 \cos{\theta}}
# 
# and :math:`m` the constant horizontal acceleration of the front defined by:
# 
# .. math::
#     m = -g \sin{\theta} + g \cos{\theta} \tan{\delta}
# 
# **Dressler's equation**: :math:`x_f(t) = 2 t \sqrt{g h_0}`
A.xf_mangeney(t, 25)
A.xf_dressler(t)

A.show_fronts_over_time()

# %%
# Second example using Ritter and Stocker's equations for a dam-break solution over an infinite 
# inclined dry and wet domain without friction:
# 
# ---------------------------------
# 
# **Ritter's equation**: :math:`x_f(t) = 2 t \sqrt{g h_0}`
# 
# **Stocker's equation**: :math:`x_f(t) =t \cdot \frac{2 c_m^2 \left( \sqrt{g h_l} - c_m \right)}{c_m^2 - g h_r}`
# 
# with :math:`c_m` the critical velocity defined by:
# 
# .. math::
#     -8.g.hr.cm^{2}.(g.hl - cm^{2})^{2} + (cm^{2} - g.hr)^{2} . (cm^{2} + g.hr) = 0
B.xf_ritter(t)
B.xf_stocker(t, hr=1)

B.show_fronts_over_time()