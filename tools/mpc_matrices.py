import numpy as np
import math

#np.set_printoptions(precision=3)

N = 15
m = 2
DSL = 2
SSL = 7
T = 0.1
g = 9.81
h = 0.814
ANGLES = [1.57]*N
X_SPEED = [0.3]*N
Y_SPEED = [0.0]*N
Xc = 1.0
Yc = 0.0
BETA = 1.0
GAMMA = 1e-3
ALPHA = 1e-3

def pretty_print_vector(vector):
    to_print = []
    for v in vector:
        to_print.append('{0:.5f}, '.format(float(v)))
    print ''.join(to_print)


def pretty_print_matrix(matrix):
    for row in matrix:
        row_to_print = []
        for i in row:
            row_to_print.append('{0:.5f}, '.format(float(i)))
        print ''.join(row_to_print)

def Pps():
    c1 = [1.0 for i in range(N)]
    c2 = [float(i)*T for i in range(N)]
    c3 = [((i*T)**2)/2.0 for i in range(N)]

    Pps = np.array(zip(*[c1, c2, c3]))
    return Pps

def Pvs():
    c1 = [0.0 for i in range(N)]
    c2 = [1.0 for i in range(N)]
    c3 = [float(i)*T for i in range(N)]

    Pvs = np.array(zip(*[c1, c2, c3]))
    return Pvs

def Pzs():
    c1 = [1.0 for i in range(N)]
    c2 = [float(i)*T for i in range(N)]
    c3 = [(((i*T)**2)/2.0) -(h/g) for i in range(N)]

    Pzs = np.array(zip(*[c1, c2, c3]))
    return Pzs

def Ppu():
    rows = []
    for i in range(N+1):
        k = (T**3)/6.0
        rows.append( [k*(1 + 3*j + 3*j**2) for j in reversed(range(i))] + [0]*(N-i) )

    Ppu = np.array(rows[1:])
    return Ppu

def Pvu():
    rows = []
    for i in range(N+1):
        k = (T**2)/2.0
        rows.append( [k*(1 + 2*j) for j in reversed(range(i))] + [0]*(N-i) )

    Pvu = np.array(rows[1:])
    return Pvu


def Pzu():
    rows = []
    for i in range(N+1):
        k = (T**3)/6.0
        c = T*h/g
        rows.append( [k*(1 + 3*j + 3*j**2)-c for j in reversed(range(i))] + [0]*(N-i) )

    Pzu = np.array(rows[1:])
    return Pzu

x_jerks = np.array([[i,] for i in [2.78937446,  2.66571868,  0.24137145,  5.51374576,  5.67631765, 4.69410839,  5.9106198 ,  3.41090947,  3.50717172,  4.79457518]])
y_jerks = np.array([[i,] for i in [0.31207815,  2.77970676,  0.6855483 ,  0.15505109,  1.91257035, 0.99694659,  1.84712156,  1.869668  ,  0.53948499,  1.2209777]])

x_state = np.array([[i,] for i in [1.92851167,  1.2196986 ,  0.50006897]])
y_state = np.array([[i,] for i in [0.44633795,  0.79862202,  0.88712761]])


#print np.dot(Pps, x_state) + np.dot(Ppu, x_jerks)
#pretty_print( np.dot(Pvs, x_state) + np.dot(Pvu, x_jerks) )

#pretty_print( np.dot(Pps, y_state) + np.dot(Ppu, y_jerks) )
#pretty_print( np.dot(Pvs, y_state) + np.dot(Pvu, y_jerks) )

#pretty_print( np.dot(Pzs, x_state) + np.dot(Pzu, x_jerks) )
#pretty_print( np.dot(Pzs, y_state) + np.dot(Pzu, y_jerks) )

def U_current():
    u = [1]*(DSL) + [0]*(N-DSL)
    return np.array([u]).transpose()

def U_future():
    u0 = [0]*DSL + [1]*SSL + [0]*(N-DSL-SSL)
    u1 = [0]*DSL + [0]*SSL + [1]*(N-DSL-SSL)
    return np.array([u0, u1]).transpose()

def D():

    x_restrictions = [[0.0]*N for i in range(2*N)]
    y_restrictions = [[0.0]*N for i in range(2*N)]

    row = 0
    for col in range(N):
        x_restrictions[row][col] = math.cos(ANGLES[col])
        x_restrictions[row+1][col] = -math.sin(ANGLES[col])
        row += 2

    row = 0
    for col in range(N):
        y_restrictions[row][col] = math.sin(ANGLES[col])
        y_restrictions[row+1][col] = math.cos(ANGLES[col])
        row += 2

    restrictions = [x+y for x,y in zip(x_restrictions, y_restrictions)]
    return np.array(restrictions)

def zmp_restiction_matrix():
    matrix_Pzu = Pzu()
    matrix_U_future = U_future()

    first_row = np.concatenate((matrix_Pzu, -matrix_U_future, np.zeros(matrix_Pzu.shape), np.zeros(matrix_U_future.shape)), axis=1)
    second_row = np.concatenate((np.zeros(matrix_Pzu.shape), np.zeros(matrix_U_future.shape), matrix_Pzu, -matrix_U_future), axis=1)
    combined = np.concatenate((first_row, second_row), axis=0)

    return np.dot(D(), combined)

#pretty_print_matrix(zmp_restiction_matrix())

def Q_prime():
    q_prime_00 = ALPHA*np.eye(N) + BETA*np.dot(Pvu().transpose(), Pvu()) + GAMMA*np.dot(Pzu().transpose(), Pzu())
    q_prime_01 = -GAMMA*np.dot(Pzu().transpose(), U_future())
    q_prime_10 = -GAMMA*np.dot(U_future().transpose(), Pzu())
    q_prime_11 = GAMMA*np.dot(U_future().transpose(), U_future())

    q_prime_first_row = np.concatenate((q_prime_00, q_prime_01), axis=1)
    q_prime_second_row = np.concatenate((q_prime_10, q_prime_11), axis=1)

    return np.concatenate((q_prime_first_row, q_prime_second_row), axis=0)

def Q():
    q_prime = Q_prime()
    shape = (N+m, N+m)
    first_row = np.concatenate((q_prime, np.zeros(shape)), axis=1)
    second_row = np.concatenate((np.zeros(shape), q_prime), axis=1)

    return np.concatenate((first_row, second_row), axis=0)

#pretty_print_matrix(Q())


def b():
    x_ref_speed = np.array([X_SPEED]).transpose()
    x_state = np.array([0.15, 0.0, 0.0]).transpose()
    xc = 0.15

    first_row = ( BETA*np.dot(Pvu().transpose(), np.dot(Pvs(), x_state) - x_ref_speed) +
                  GAMMA*np.dot(Pzu().transpose(), np.dot(Pzs(), x_state) - xc*U_current()) )
#    second_row =
#    third_row =
#    forth_row =

#    return np.concatenate((first_row, second_row, third_row, forth_row), axis=0)
    import pdb; pdb.set_trace()
    return first_row

pretty_print_matrix(b())


def zmp_restiction_vector():
    #import pdb; pdb.set_trace()
    x_state = np.array([[0.0, 0.0, 0.0]]).transpose()
    y_state = np.array([[0.0, 0.0, 0.0]]).transpose()
    
    first_row = U_current()*Xc - np.dot(Pzs(), x_state)
    second_row = U_current()*Yc - np.dot(Pzs(), y_state)

    return np.dot(D(), np.concatenate((first_row, second_row), axis=0))

#pretty_print_vector(zmp_restiction_vector())

def foo():
    xc = 1.0
    yc = 0.0
    x_state = np.array([xc, 0.0, 0.0]).transpose()
    y_state = np.array([yc, 0.0, 0.0]).transpose()

    first_row = U_current()*xc - np.dot(Pzs(), x_state)
    second_row = U_current()*yc - np.dot(Pzs(), y_state)
    result = np.dot(D(), np.concatenate((first_row, second_row), axis=0))
    pretty_print_matrix(result)

#foo()
