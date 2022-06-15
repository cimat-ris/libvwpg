# -*- coding: utf-8 -*-
"""
Created on Tue Dec 12 10:39:07 2017

@author: noe
"""

import sympy


phi_k = sympy.Symbol('phi')
theta_l = sympy.Symbol('theta')
phi_l = sympy.Symbol('psi')

cos_theta_l = sympy.cos(theta_l)
sin_theta_l = sympy.sin(theta_l)
cos_phi_k = sympy.cos(phi_k)
sin_phi_k = sympy.sin(phi_k)
cos_phi_l = sympy.cos(phi_l)
sin_phi_l = sympy.sin(phi_l)

cl_x_ml = sympy.Symbol('^{c_{l}}x_{m_l}')
cl_y_ml = sympy.Symbol('^{c_{l}}y_{m_l}')
cl_z_ml = sympy.Symbol('^{c_{l}}z_{m_l}') 

mk_x_ml = sympy.Symbol('^{m_{k}}x_{m_l}')
mk_y_ml = sympy.Symbol('^{m_{k}}y_{m_l}')
mk_z_ml = sympy.Symbol('^{m_{k}}z_{m_l}')

ck_x_r = sympy.Symbol('^{c_{k}}x_{r}')
ck_z_r = sympy.Symbol('^{c_{k}}z_{r}')
ck_y_r = sympy.Symbol('^{c_{k}}y_{v}')

# Transformation matrices
#cl_T_ml = sympy.Matrix([[0, -1, 0, cl_x_ml], [0, 0, -1, cl_y_ml], [1, 0, 0, cl_z_ml], [0, 0, 0, 1]])
cl_T_ml = sympy.Matrix([[0, -1, 0, 0], [0, 0, -1, cl_y_ml], [1, 0, 0, 0], [0, 0, 0, 1]])
mk_T_ml = sympy.Matrix([[cos_theta_l, -sin_theta_l, 0, mk_x_ml], [sin_theta_l, cos_theta_l, 0, mk_y_ml], [0, 0, 1, 0], [0, 0, 0, 1]])
mk_T_ml_inv = sympy.eye(4)
mk_T_ml_inv[0:3, 0:3] = mk_T_ml[0:3, 0:3].T
mk_T_ml_inv[0:3, 3] = -mk_T_ml[0:3, 0:3].T*mk_T_ml[0:3, 3]
ck_T_mk = cl_T_ml
ck_T_r = sympy.Matrix([[cos_phi_k, 0, sin_phi_k, ck_x_r], [0, 1, 0, ck_y_r], [-sin_phi_k, 0, cos_phi_k, ck_z_r], [0, 0, 0, 1]])


cl_T_r = cl_T_ml*mk_T_ml_inv*ck_T_mk.inv()*ck_T_r
cl_T_r_s = cl_T_r[0:3, 3].simplify
# print  sympy.latex(cl_T_r_s)


I_3x3 = sympy.Matrix([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
r_R_cl = cl_T_r[0:3, 0:3].T
cl_t_r = cl_T_r[0:3, 3]

# compute Esential
tx = sympy.Matrix([[0,-cl_t_r[2],cl_t_r[1]],[cl_t_r[2],0,-cl_t_r[0]],[-cl_t_r[1],cl_t_r[0],0]])
E  = tx*cl_T_r[0:3, 0:3]
# E  = E/E[0, 2]
#E  = E/E[0, 2]
# e11 = sympy.simplify(E[0, 0])
# e12 = sympy.simplify(E[0, 1])
# e13 = sympy.simplify(E[0, 2])
# e21 = sympy.simplify(E[1, 0])
# e22 = sympy.simplify(E[1, 1])
# e23 = sympy.simplify(E[1, 2])
# e31 = sympy.simplify(E[2, 0])
# e32 = sympy.simplify(E[2, 1])
# e33 = sympy.simplify(E[2, 2])

# print '\n\n'
# print 'e_{11} = ', sympy.latex(e11)
# print 'e_{12} = ', sympy.latex(e12)
# print 'e_{13} = ', sympy.latex(e13)
# print 'e_{21} = ', sympy.latex(e21)
# print 'e_{22} = ', sympy.latex(e22)
# print 'e_{23} = ', sympy.latex(e23)
# print 'e_{31} = ', sympy.latex(e31)
# print 'e_{32} = ', sympy.latex(e32)
# print 'e_{33} = ', sympy.latex(e33)

# Real Calibration matrix.
alpha_x_real = sympy.Symbol('rho')
alpha_y_real = sympy.Symbol('sigma')
u0_real = sympy.Symbol('u_r')
v0_real = sympy.Symbol('v_r')
K_real   = sympy.Matrix([[alpha_x_real,0,u0_real],[0,alpha_y_real,v0_real],[0,0,1]])
Kinv_real = K_real.inv()

# Est calibration matrix.
alpha_x_est = sympy.Symbol('tau')
alpha_y_est = sympy.Symbol('mu')
u0_est = sympy.Symbol('u_e')
v0_est = sympy.Symbol('v_e')
K_pert = sympy.Matrix([[alpha_x_est,0,u0_est],[0,alpha_y_est,v0_est],[0,0,1]])


E = K_pert.transpose()*Kinv_real.transpose()*E*Kinv_real*K_pert
E  = E/E[0, 2]
e11 = sympy.simplify(E[0, 0])
e12 = sympy.simplify(E[0, 1])
e13 = sympy.simplify(E[0, 2])
e21 = sympy.simplify(E[1, 0])
e22 = sympy.simplify(E[1, 1])
e23 = sympy.simplify(E[1, 2])
e31 = sympy.simplify(E[2, 0])
e32 = sympy.simplify(E[2, 1])
e33 = sympy.simplify(E[2, 2])

print '\n\n'
print 'e_{11} = ', sympy.latex(e11)
print 'e_{12} = ', sympy.latex(e12)
print 'e_{13} = ', sympy.latex(e13)
print 'e_{21} = ', sympy.latex(e21)
print 'e_{22} = ', sympy.latex(e22)
print 'e_{23} = ', sympy.latex(e23)
print 'e_{31} = ', sympy.latex(e31)
print 'e_{32} = ', sympy.latex(e32)
print 'e_{33} = ', sympy.latex(e33)