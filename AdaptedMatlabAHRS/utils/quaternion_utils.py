# Library implemented to carry out quaternion functions in the same convention as Matlab

import numpy as np


def quaternion_rotation_matrix(Q):
    """
    Covert a quaternion into a full three-dimensional rotation matrix.

    Input
    :param Q: A 4 element array representing the quaternion (q0,q1,q2,q3)

    Output
    :return: A 3x3 element matrix representing the full 3D rotation matrix.
             This rotation matrix converts a point in the local reference
             frame to a point in the global reference frame.
    """
    # normalise first
    Q = quaternion_normalise(Q)

    # Extract the values from Q
    a = Q[3]
    b = Q[0]
    c = Q[1]
    d = Q[2]

    r00 = 2 * a ** 2 - 1 + 2 * b ** 2
    r01 = 2 * b * c + 2 * a * d
    r02 = 2 * b * d - 2 * a * c
    # Second row
    r10 = 2 * b * c - 2 * a * d
    r11 = 2 * a ** 2 - 1 + 2 * c ** 2
    r12 = 2 * c * d + 2 * a * b
    # Bottom row
    r20 = 2 * b * d + 2 * a * c
    r21 = 2 * c * d - 2 * a * b
    r22 = 2 * a ** 2 - 1 + 2 * d ** 2

    # 3x3 rotation matrix
    rot_matrix = np.array([[r00, r01, r02],
                           [r10, r11, r12],
                           [r20, r21, r22]])

    return rot_matrix

def quaternion_multiply(Q1, Q2):
    # 0, 1, 2, 3
    # 3, 0, 1, 2

    w = Q1[3] * Q2[3] - Q1[0] * Q2[0] - Q1[1] * Q2[1] - Q1[2] * Q2[2]
    x = Q1[3] * Q2[0] + Q1[0] * Q2[3] + Q1[1] * Q2[2] - Q1[2] * Q2[1]
    y = Q1[3] * Q2[1] - Q1[0] * Q2[2] + Q1[1] * Q2[3] + Q1[2] * Q2[0]
    z = Q1[3] * Q2[2] + Q1[0] * Q2[1] - Q1[1] * Q2[0] + Q1[2] * Q2[3]
    return np.array([x, y, z, w])


def quaternion_normalise(q):
    n = np.linalg.norm(q)
    q /= n
    return q


def quaternion_conj(Q):
    """
    Returns the conjugate of the quaternion
    :rtype : Quaternion
    :return: the conjugate of the quaternion
    """
    return np.array([-Q[0], -Q[1], -Q[2], Q[3]])
