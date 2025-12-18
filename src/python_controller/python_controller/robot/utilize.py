#!/usr/bin/env python3
import numpy as np
import casadi as ca

def a_hat(a, type=None):
    """
    Compute the skew-symmetric matrix (hat operator) for a 3D vector.

    Parameters:
        a: 3x1 vector (CasADi SX/NumPy array)
        type: 'casadi' or 'numpy'

    Returns:
        3x3 skew-symmetric matrix
    """
    if type == 'casadi':
        return ca.skew(a)

    elif type == 'numpy':
        a = np.asarray(a).reshape(3,)
        return np.array([
            [0,        -a[2],  a[1]],
            [a[2],  0,        -a[0]],
            [-a[1], a[0],  0]
        ])
    else:
        raise ValueError("Type must be 'casadi' or 'numpy'")
def RotZ(psi, type=None): # rotation about z axis
    if type == 'casadi':
        c = ca.cos
        s = ca.sin
        Rz = ca.DM([
            [c(psi), -s(psi), 0],
            [s(psi),  c(psi), 0],
            [0,        0,     1]
        ])
    elif type == 'numpy':
        c = np.cos
        s = np.sin
        Rz = np.array([
            [c(psi), -s(psi), 0],
            [s(psi),  c(psi), 0],
            [0,        0,     1]
        ])
    return Rz
def RotationMatrix(phi, theta, psi, type=None): # Euler angles roll, pitch, yaw on x, y, z axes
    if type == 'casadi':
        c = ca.cos
        s = ca.sin
        Rz = ca.DM([
            [c(psi), -s(psi), 0],
            [s(psi),  c(psi), 0],
            [0,        0,     1]
        ])
        Ry = ca.DM([
            [c(theta), 0, s(theta)],
            [0,        1, 0],
            [-s(theta), 0, c(theta)]
        ])
        Rx = ca.DM([
            [1, 0,       0],
            [0, c(phi), -s(phi)],
            [0, s(phi),  c(phi)]
        ])
        R = ca.mtimes(Rz, ca.mtimes(Ry, Rx))  # Rz * Ry * Rx
    elif type == 'numpy':
        c = np.cos
        s = np.sin
        Rz = np.array([
            [c(psi), -s(psi), 0],
            [s(psi),  c(psi), 0],
            [0,        0,     1]
        ])
        Ry = np.array([
            [c(theta), 0, s(theta)],
            [0,        1, 0],
            [-s(theta), 0, c(theta)]
        ])
        Rx = np.array([
            [1, 0,       0],
            [0, c(phi), -s(phi)],
            [0, s(phi),  c(phi)]
        ])
        R = np.dot(Rz, np.dot(Ry, Rx))
    return R
def RotationMatrix_inv(R, type=None):
    """
    Compute the inverse of a rotation matrix.

    Parameters:
        R: 3x3 rotation matrix (CasADi SX/NumPy array)
        type: 'casadi' or 'numpy'

    Returns:
        3x3 inverse rotation matrix
    """
    if type == 'casadi':
        return ca.inv(R)

    elif type == 'numpy':
        return np.linalg.inv(R)