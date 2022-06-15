import math
import numpy


def prettify_vector(vector):
    elemnts_as_string = ['{:.3f}'.format(x) for x in vector]
    return','.join(elemnts_as_string)

R = 1.5
SPEED = 0.118
TIMES = 500

derivative = lambda x : -x / math.sqrt(R**2 - x**2)
points = numpy.linspace(-R, R, TIMES)
angles = [math.atan(derivative(x)) for x in points]

x_speed = [SPEED*math.cos(theta) for theta in angles]
y_speed = [SPEED*math.sin(theta) for theta in angles]


print '[simulation]'
print 'formulation=herdt'
print 'N = 22'
print 'm = 3'
print 'T = 0.1'
print 'double_support_lenght = 2'
print 'single_support_lenght = 7'
print 'iterations = {}'.format(TIMES)

print '[qp]'
print 'alpha = 1e-6'
print 'beta = 1e-3'
print 'gamma = 1e-6'

print '[reference]'
print 'x_com_speed = {0}'.format(prettify_vector(x_speed))
print 'y_com_speed = {0}'.format(prettify_vector(y_speed))
print 'orientation = {0}'.format(prettify_vector(angles))

print '[initial_values]'
'foot_x_position = {}'.format(R)
'foot_y_position = 0.0'

