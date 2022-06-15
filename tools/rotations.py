import numpy
from math import sin, cos

def rotation_around_z_axis(theta):
    x = [cos(theta), 0.0, sin(theta)]
    y = [0.0, 1.0, 0.0]
    z = [-sin(theta), 0.0, cos(theta)]
    return numpy.array([x, y, z])

few_deegrees_around_z_axisi = rotation_around_z_axis(0.1);
com_to_c = numpy.array([[0, -1., 0], [0, 0, -1,], [1., 0, 0]])


import pdb; pdb.set_trace()
print "done"
