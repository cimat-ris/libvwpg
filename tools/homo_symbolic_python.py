# -*- coding: utf-8 -*-
"""
Created on Thu Aug 24 20:05:53 2017

@author: noe
"""

import sympy


phi_k = sympy.Symbol('phi')
theta_l = sympy.Symbol('theta')
phi_l = sympy.Symbol('psi')
omega_l = sympy.Symbol('omega')
omega_k = sympy.Symbol('mu')
ur = sympy.Symbol('u')
vr = sympy.Symbol('v')

cos_theta_l = sympy.cos(theta_l)
sin_theta_l = sympy.sin(theta_l)
cos_phi_k = sympy.cos(phi_k)
sin_phi_k = sympy.sin(phi_k)
cos_phi_l = sympy.cos(phi_l)
sin_phi_l = sympy.sin(phi_l)
cos_omega_l = sympy.cos(omega_l)
sin_omega_l = sympy.sin(omega_l)
cos_omega_k = sympy.cos(omega_k)
sin_omega_k = sympy.sin(omega_k)

h = sympy.Symbol('h')
mk_x_ml = sympy.Symbol('^{m_{k}}x_{m_l}')
mk_y_ml = sympy.Symbol('^{m_{k}}y_{m_l}')
mk_z_ml = sympy.Symbol('^{m_{k}}z_{m_l}')

ck_x_r = sympy.Symbol('^{c_{k}}x_{r}')
ck_z_r = sympy.Symbol('^{c_{k}}z_{r}')

# Transformation matrices
#cl_T_ml = sympy.Matrix([[0, -1, 0, 0], [0, 0, -1, h], [1, 0, 0, 0], [0, 0, 0, 1]])
ml_T_cl = sympy.Matrix([[sin_omega_l, 0, cos_omega_l, 0], [-cos_omega_l, 0, sin_omega_l, 0], [0, -1, 0, h], [0, 0, 0, 1]])
mk_T_ck = sympy.Matrix([[sin_omega_k, 0, cos_omega_k, 0], [-cos_omega_k, 0, sin_omega_k, 0], [0, -1, 0, h], [0, 0, 0, 1]])
mk_T_ml = sympy.Matrix([[cos_theta_l, -sin_theta_l, 0, mk_x_ml], [sin_theta_l, cos_theta_l, 0, mk_y_ml], [0, 0, 1, 0], [0, 0, 0, 1]])
mk_T_ml_inv = sympy.eye(4)
mk_T_ml_inv[0:3, 0:3] = mk_T_ml[0:3, 0:3].T
mk_T_ml_inv[0:3, 3] = -mk_T_ml[0:3, 0:3].T*mk_T_ml[0:3, 3]
cl_T_ml = ml_T_cl.inv()
ck_T_r = sympy.Matrix([[cos_phi_k, 0, sin_phi_k, ck_x_r], [0, 1, 0, 0], [-sin_phi_k, 0, cos_phi_k, ck_z_r], [0, 0, 0, 1]])


#cl_T_r = cl_T_ml*mk_T_ml_inv*mk_T_ck*ck_T_r
cl_T_r = mk_T_ck.inv()*mk_T_ml*ml_T_cl
#cl_T_r_s = cl_T_r[0:3, 3].simplify
#cl_T_r_t = cl_T_r[0:3, 3]
#cl_t_r_t_1 = sympy.simplify(cl_T_r_t[0])
#cl_t_r_t_2 = sympy.simplify(cl_T_r_t[1])
#cl_t_r_t_3 = sympy.simplify(cl_T_r_t[2])
#cl_T_r_R = cl_T_r[0:3, 0:3]
#cl_T_r_R_1 = sympy.simplify(cl_T_r_R[0,0])
#cl_T_r_R_2 = sympy.simplify(cl_T_r_R[0,1])
#cl_T_r_R_3 = sympy.simplify(cl_T_r_R[0,2])
#cl_T_r_R_4 = sympy.simplify(cl_T_r_R[1,0])
#cl_T_r_R_5 = sympy.simplify(cl_T_r_R[1,1])
#cl_T_r_R_6 = sympy.simplify(cl_T_r_R[1,2])
#cl_T_r_R_7 = sympy.simplify(cl_T_r_R[2,0])
#cl_T_r_R_8 = sympy.simplify(cl_T_r_R[2,1])
#cl_T_r_R_9 = sympy.simplify(cl_T_r_R[2,2])
#print '^{l}t_r1 = ', sympy.latex(cl_t_r_t_1)
#print '^{l}t_r2 = ', sympy.latex(cl_t_r_t_2)
#print '^{l}t_r3 = ', sympy.latex(cl_t_r_t_3)
#print '^{l}R_r_11 = ', sympy.latex(cl_T_r_R_1)
#print '^{l}R_r_12 = ', sympy.latex(cl_T_r_R_2)
#print '^{l}R_r_13 = ', sympy.latex(cl_T_r_R_3)
#print '^{l}R_r_21 = ', sympy.latex(cl_T_r_R_4)
#print '^{l}R_r_22 = ', sympy.latex(cl_T_r_R_5)
#print '^{l}R_r_23 = ', sympy.latex(cl_T_r_R_6)
#print '^{l}R_r_31 = ', sympy.latex(cl_T_r_R_7)
#print '^{l}R_r_32 = ', sympy.latex(cl_T_r_R_8)
#print '^{l}R_r_33 = ', sympy.latex(cl_T_r_R_9)

# camera
nx = sympy.Symbol('n_x')
ny = sympy.Symbol('n_y')
nz = sympy.Symbol('n_z')
d  = sympy.Symbol('d')

n      = sympy.Matrix([[nx], [ny], [nz]])
I_3x3  = sympy.Matrix([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
r_R_cl = sympy.simplify(cl_T_r[0:3, 0:3].T)
cl_t_r = sympy.simplify(cl_T_r[0:3, 3])

# compute homography
H = r_R_cl*(I_3x3 - (1/d) * cl_t_r * n.transpose())
h31 = sympy.simplify(H[2, 0])
h32 = sympy.simplify(H[2, 1])
h33 = sympy.simplify(H[2, 2])
h11 = sympy.simplify(H[0, 0])
h12 = sympy.simplify(H[0, 1])
h13 = sympy.simplify(H[0, 2])
h21 = sympy.simplify(H[1, 0])
h22 = sympy.simplify(H[1, 1])
h23 = sympy.simplify(H[1, 2])


print '\n\n'
# print 'u_num = ', sympy.latex(u_num)
# print 'u_den = ', sympy.latex(u_den)
print 'h_{31} = ', sympy.latex(h31)
print 'h_{32} = ', sympy.latex(h32)
print 'h_{33} = ', sympy.latex(h33)
print 'h_{11} = ', sympy.latex(h11)
print 'h_{12} = ', sympy.latex(h12)
print 'h_{13} = ', sympy.latex(h13)
print 'h_{21} = ', sympy.latex(h21)
print 'h_{22} = ', sympy.latex(h22)
print 'h_{23} = ', sympy.latex(h23)
