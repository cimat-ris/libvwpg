import numpy
import math

import cv2

K = numpy.array([[391.5937, 0.0, 0.0000], [0.0, 391.5937, 0.0000], [0.0, 0.0, 1.0000]])
R = numpy.array([[0., -1., 0.], [0., 0., -1.], [1., 0., 0.]])
T = numpy.concatenate((R, numpy.array([[0.0], [0.75], [0.0]])), axis=1)
camera = numpy.dot(K, T)

def plane_points(offset=0.0):
    a = math.cos(math.radians(-15))
    b = math.sin(math.radians(-15))
    c = 0.0
    x0, y0, z0 = (11.5, 0.0, 0.75)
    d = -a*x0 -b*y0 - c*z0

    foo = lambda y: (-d - a*y) / b
    
    p1 = numpy.array([[12.07 - offset], [foo(12.07)], [0.90], [1.0]])
    p2 = numpy.array([[12.07 - offset], [foo(12.07)], [0.60], [1.0]])
    p3 = numpy.array([[10.93 - offset], [foo(10.93)], [0.90], [1.0]])
    p4 = numpy.array([[10.93 - offset], [foo(10.93)], [0.60], [1.0]])
    return numpy.concatenate((p1, p2, p3, p4), axis=1)


def take_picture(camera_world_position, world_points):
    t = -numpy.dot(R, camera_world_position)
    T0 = numpy.concatenate((R, t), axis=1)
    #return numpy.dot(T0, world_points)
    return numpy.dot(K, numpy.dot(T0, world_points))


def normalize_homogeneous(matrix):
    for i in range(4):
        matrix[:, i] = matrix[:, i] / matrix[2, i]
    return (matrix[0:2, :]).transpose()

picture1 = take_picture(numpy.array([[0.0], [0.0], [0.75]]), plane_points(3.0))
picture2 = take_picture(numpy.array([[0.0], [0.0], [0.75]]), plane_points(5.0))
picture3 = take_picture(numpy.array([[0.0], [0.0], [0.75]]), plane_points(7))

picture1_normalized = normalize_homogeneous(picture1)
picture2_normalized = normalize_homogeneous(picture2)
picture3_normalized = normalize_homogeneous(picture3)
print picture3_normalized
raise jlk

numpy.set_printoptions(suppress=True)
h_matrix, status_h = cv2.findHomography(picture1_normalized, picture2_normalized)

#import pdb; pdb.set_trace()

print h_matrix
