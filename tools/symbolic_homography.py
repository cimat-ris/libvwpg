import sympy
import sys

phi = sympy.Symbol('phi')
theta = sympy.Symbol('theta')
psi = sympy.Symbol('psi')

# scalars
cos_theta = sympy.cos(theta)
sin_theta = sympy.sin(theta)
cos_phi = sympy.cos(phi)
sin_phi = sympy.sin(phi)
cos_psi = sympy.cos(psi)
sin_psi = sympy.sin(psi)

camera_position = sympy.sin('camera_position')
mk_x_mk1 = sympy.Symbol('^{m_{k}}x_{m_{k+1}}')
mk_y_mk1 = sympy.Symbol('^{m_{k}}y_{m_{k+1}}')
mk_z_mk1 = sympy.Symbol('^{m_{k}}z_{m_{k+1}}')

ck_x_t = sympy.Symbol('^{c_{k}}x_{t}')
ck_z_t = sympy.Symbol('^{c_{k}}z_{t}')

# rotation matrices
ck_R_mk = sympy.Matrix([[0, -1, 0], [0, 0, -1], [1, 0, 0]])
ck1_R_mk1 = ck_R_mk
ck1_R_ck = sympy.Matrix([[cos_phi, 0, sin_phi], [0, 1, 0], [-sin_phi, 0, cos_phi]])
ck1_R_ck_target = sympy.Matrix([[cos_psi, 0, sin_psi], [0, 1, 0], [-sin_psi, 0, cos_psi]])
mk1_R_mk = sympy.Matrix([[cos_theta, -sin_theta, 0], [sin_theta, cos_theta, 0], [0, 0, 1]])

ck1_R_mk = ck1_R_mk1 * mk1_R_mk

# vectors
ck1_p_mk1 = sympy.Matrix([[0], [camera_position], [0]])
mk_p_mk1 = sympy.Matrix([[mk_x_mk1], [mk_y_mk1], [mk_z_mk1]]) 
mk_p_ck = sympy.Matrix([[0], [0], [camera_position]])
ck_p_t = sympy.Matrix([[ck_x_t], [0], [ck_z_t]])

ck1_p_t = ck1_p_mk1 - ck1_R_mk * mk_p_mk1 + ck1_R_mk1 * mk_p_ck + ck1_R_ck * ck_p_t

# camera
alpha_x = sympy.Symbol('alpha_x')
alpha_y = sympy.Symbol('alpha_y')
nx = sympy.Symbol('n_x')
ny = sympy.Symbol('n_y')
nz = sympy.Symbol('n_z')
d = sympy.Symbol('d')

n = sympy.Matrix([[nx], [ny], [nz]])
I_3x3 = sympy.Matrix([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
K = sympy.Matrix([[alpha_x, 0, 0], [0, alpha_y, 0], [0, 0, 1]])


H = K * (I_3x3 - (1/d) * ck1_p_t * n.transpose() ) * K.inv()

H1 = K * (ck1_R_ck + (1/d) * ck1_p_t * n.transpose() ) * K.inv()
H2 = K * (ck1_R_ck_target + (1/d) * ck1_p_t * n.transpose() ) * K.inv()

print '\n\n'
print 'noe'
print 'h_{11} = ', sympy.latex(H[0, 0].expand())
print 'h_{13} = ', sympy.latex(H[0, 2].expand())
print 'h_{33} = ', sympy.latex(H[2, 2].expand())

print '\n\n'
print 'mine'
print 'h_{11} = ', sympy.latex(H2[0, 0].expand())
print 'h_{13} = ', sympy.latex(H2[0, 2].expand())
print 'h_{33} = ', sympy.latex(H2[2, 2].expand())
