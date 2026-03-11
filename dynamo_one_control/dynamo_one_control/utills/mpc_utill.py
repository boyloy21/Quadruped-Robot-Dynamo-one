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
        c = ca.cos(psi)
        s = ca.sin(psi)
        Rz = ca.DM([
            [c, -s, 0],
            [s,  c, 0],
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

def skew_symmetric(v):
    """Returns skew-symmetric matrix of vector v"""
    return np.array([[0, -v[2], v[1]],
                    [v[2], 0, -v[0]],
                    [-v[1], v[0], 0]])

def exp_rotation(omega, dt):
    """Exponential map for rotation matrix update"""
    angle = np.linalg.norm(omega)
    if angle < 1e-6:
        return np.eye(3)
    axis = omega / angle
    skew = skew_symmetric(axis)
    return np.eye(3) + np.sin(angle*dt)*skew + (1-np.cos(angle*dt))*(skew @ skew)

def rot_to_rpy(R):
    """Convert rotation matrix to roll, pitch, yaw angles"""
    pitch = np.arctan2(-R[2,0], np.sqrt(R[0,0]**2 + R[1,0]**2))
    roll = np.arctan2(R[2,1]/np.cos(pitch), R[2,2]/np.cos(pitch))
    yaw = np.arctan2(R[1,0]/np.cos(pitch), R[0,0]/np.cos(pitch))
    return np.array([roll, pitch, yaw])

def convert_rpy_to_rot(rpy):
    """Convert roll, pitch, yaw angles to rotation matrix."""
    roll, pitch, yaw = rpy
    rot_x = np.array([[1, 0, 0],
                      [0, np.cos(roll), -np.sin(roll)],
                      [0, np.sin(roll), np.cos(roll)]])

    rot_y = np.array([[np.cos(pitch), 0, np.sin(pitch)],
                      [0, 1, 0],
                      [-np.sin(pitch), 0, np.cos(pitch)]])

    rot_z = np.array([[np.cos(yaw), -np.sin(yaw), 0],
                      [np.sin(yaw), np.cos(yaw), 0],
                      [0, 0, 1]])

    return rot_z @ rot_y @ rot_x