#!/usr/bin/env python3
import numpy as np
from math import *
from .Kinematic_Model import KinematicQuadruped

def rotMatrix3D(rotation=[0, 0, 0], is_radians=True, order='xyz'):
    roll, pitch, yaw = rotation

    # Convert to radians if not already
    if not is_radians:
        roll = radians(roll)
        pitch = radians(pitch)
        yaw = radians(yaw)

    # 3x3 rotation matrices
    rotX = np.array([
        [1, 0, 0],
        [0, cos(roll), -sin(roll)],
        [0, sin(roll), cos(roll)]
    ])

    rotY = np.array([
        [cos(pitch), 0, sin(pitch)],
        [0, 1, 0],
        [-sin(pitch), 0, cos(pitch)]
    ])

    rotZ = np.array([
        [cos(yaw), -sin(yaw), 0],
        [sin(yaw), cos(yaw), 0],
        [0, 0, 1]
    ])

    # Apply rotation order
    if order == 'xyz':
        R = rotZ @ rotY @ rotX
    elif order == 'xzy':
        R = rotY @ rotZ @ rotX
    elif order == 'yxz':
        R = rotZ @ rotX @ rotY
    elif order == 'yzx':
        R = rotX @ rotZ @ rotY
    elif order == 'zxy':
        R = rotY @ rotX @ rotZ
    elif order == 'zyx':
        R = rotX @ rotY @ rotZ
    else:
        raise ValueError(f"Invalid rotation order: {order}")

    # # Create 4x4 homogeneous rotation matrix
    # R_4x4 = np.eye(4)
    # R_4x4[:3, :3] = R  # embed 3x3 rotation into top-left
    return R

class InverseDynamic:
    def __init__(self, lb, wb, l1, l2, l3, m1, m2, m3, I1, I2, I3, Kp, Kd, dt, omega):
        self.l1 = l1    # Hip Length
        self.l2 = l2    # Thigh length
        self.l3 = l3    # calf length
        self.lh = l1/2    # Hip COM distance from hip
        self.lt = l2/2    # Thigh COM distance from hip
        self.lc = l3/2    # Shank COM distance from knee
        self.m1 = m1    # Hip mass
        self.m2 = m2    # Thigh mass
        self.m3 = m3    # Calf mass
        self.I1 = I1    # Hip inertia
        self.I2 = I2    # Thigh inertia
        self.I3 = I3    # Calf inertia
        self.g = 9.81   # Gravity
        self.dt = dt    # Time step
        self.model = KinematicQuadruped(lb, wb, l1, l2, l3)
        self.Kp = Kp    # Proportional gain for feedback control
        self.Kd = Kd    # Derivative gain for feedback control
        self.omega = omega    # Natural frequency
        self.damping = 0.7

        FL = 0
        FR = 1
        RR = 2
        RL = 3
        self.right_leg = [FR, RR]
        self.left_leg = [FL, RL]
        
    def theta2dot(self, theta):
        theta2dot = (theta[-1] - theta[-2])/self.dt
        return theta2dot
    
    def theta3dot(self, theta):
        theta3dot = (theta[-1] - 2*theta[-2] + theta[-3])/self.dt**2
        return theta3dot
    
    def MassMatrix(self, theta2):
        """Compute the 2x2 mass matrix M(q)."""
        M11 = self.m1*self.lh**2 + self.I1 + self.m2*(self.l1**2 + self.lt**2 + 2*self.l1*self.lt*np.cos(theta2)) + self.I2 + self.m3*(self.l1**2 + self.l2**2 + self.lc**2) + self.I3
        M12 = self.m2*self.lt*(self.lt + self.l1*np.cos(theta2)) + self.I2 + self.m3*(self.l2**2 + self.lc**2) + self.I3
        M21 = M12  # Symmetric
        M13 = self.m3*self.lc**2 + self.I3
        M31 = M13  # Symmetric
        M22 = self.m3*self.lc**2 + self.I3 + self.I2 + self.m3*(self.l2**2 + self.lc**2)
        M23 = self.m3*self.lc**2 + self.I3
        M32 = M23 
        M33 = self.m3*self.lc**2 + self.I3
        M = np.array([
            [M11, M12, M13],
            [M21, M22, M23],
            [M31, M32, M33]
            ])
        
        return M

    def CoriolisMatrix(self, theta2, theta1dot, theta2dot):
        """Compute the 2x2 Coriolis matrix C(q, qdot)."""
        h = -self.m2 * self.l1 * self.lt * np.sin(theta2)
        C11 = h * theta2dot
        C12 = h * (theta1dot + theta2dot)
        C21 = -h * theta1dot
        C13, C22, C23, C31, C32, C33 = 0, 0, 0, 0, 0, 0
        C = np.array([
            [C11, C12, C13], 
            [C21, C22, C23],
            [C31, C32, C33]
        ])
        return C

    def GravityVector(self, theta1, theta2, theta3):
        """Compute the 2x1 gravity vector G(q)."""
        G1 = self.g*(self.m1*self.lh + self.m2*(self.l1*np.cos(theta1 + theta2)) + self.m3*(self.l1*np.cos(theta1) + self.l2*np.cos(theta1 + theta2) + self.lc*np.cos(theta1 + theta2 + theta3)))
        G2 = self.g*(self.m2*self.lt*np.cos(theta1 + theta2) + self.m3*(self.l3*np.cos(theta1 + theta2) + self.lc*np.cos(theta1 + theta2 + theta3)))
        G3 = self.g*(self.m3*self.lc*np.cos(theta1 + theta2 + theta3))
        return np.array([
            [G1],
            [G2],
            [G3]
        ])

    def compute_torques(self, theta, thetadot , thetadotdot):
        """Compute inverse dynamics torques: M(q)qddot + C(q,qdot)qdot + G(q)."""
        theta1, theta2 , theta3 = theta
        theta1dot, theta2dot, theta3dot = thetadot
        theta1dotdot, theta2dotdot, theta3dotdot = thetadotdot
        M = self.MassMatrix(theta2)
        C = self.CoriolisMatrix(theta2, theta1dot, theta2dot)
        G = self.GravityVector(theta1, theta2, theta3)
        
        qddot = np.array([[theta1dotdot], [theta2dotdot], [theta3dotdot]])
        qdot = np.array([[theta1dot], [theta2dot], [theta3dot]])
        
        tau = M @ qddot + C @ qdot + G
        return tau
    
    def torque_feedforward(self, q, qdot, a_des, legID=0):
        # Compute inverse dynamics torques: M(q)qddot + C(q,qdot)qdot + G(q).
        M = self.MassMatrix(q[1])
        C = self.CoriolisMatrix(q[1], qdot[0], qdot[1])
        G = self.GravityVector(q[0], q[1], q[2])
        J = self.model.jacobian(q, legID=legID)
        J_dot = self.model.jacobian_derivative(q, qdot, legID=legID)
        
        # Regularization to prevent singular matrix
        # Task-space inertia
        Lambda = np.linalg.pinv(J @ np.linalg.pinv(M) @ J.T)
        # Update Kp and Kd
        self.Kp = np.diag([self.omega[0]**2 * Lambda[0,0], self.omega[1]**2 * Lambda[1,1], self.omega[2]**2 * Lambda[2,2]])
        self.Kd = np.diag([2 * self.damping * self.omega[0]*Lambda[0,0], 2 * self.damping * self.omega[1]*Lambda[1,1], 2 * self.damping * self.omega[2]*Lambda[2,2]])
        tau_ff = J.T @ Lambda @ (a_des.reshape(3,1) - J_dot @ qdot.reshape(3,1)) + C @ qdot.reshape(3,1) + G
        
        return tau_ff
    
    def TorqueSwing(self, footpos_des, footvel_des, footacc_des, q, qdot, legID=0):
        if legID in self.right_leg:
            q[0] = -q[0]
        else:
            q[0] = q[0]
        self.footpos_cur = self.model.ForwardKinematic(q, legID=legID)
        self.footvel_cur = self.model.jacobian(q, legID=legID) @ qdot.reshape(3,1)
        errorpos = (np.array(footpos_des).flatten() - np.array(self.footpos_cur).flatten())
        errorvel = (np.array(footvel_des).flatten() - np.array(self.footvel_cur).flatten())
        J = self.model.jacobian(q, legID=legID)
        
        # Calculate Torque Feed Forwared
        tau_ff = self.torque_feedforward(q, qdot, footacc_des, legID=legID)
        # Calculate Torque Feedback
        torque_cartian = J.T @ (self.Kp @ errorpos.reshape(3,1) + self.Kd @ errorvel.reshape(3,1))
        
        return torque_cartian + tau_ff
    
    def TorqueStance(self, force, q, rot=[0.0, 0.0, 0.0], legID=0):
        if legID in self.right_leg:
            q[0] = -q[0]
        else:
            q[0] = q[0]
        Rot = rotMatrix3D(rot, is_radians=True)
        J = self.model.jacobian(q, legID=legID)
        
        f_leg = Rot.T @ np.array(force).reshape(3,1)
        torque = J.T @ f_leg 
        return torque
    
    def rotate_world_to_leg(self, F_world, leg_angles, legID):
        """
        Convert force from world frame to leg frame
        Args:
            F_world: [Fx, Fy, Fz] in world frame
            leg_angles: [q1, q2, q3] current joint angles (rad)
            legID: Leg index (0-3)
        Returns:
            F_leg: Force vector in leg frame
        """
        q1, q2, q3 = leg_angles  # Hip angle
        right_legs = [2, 3]
        # 1. Get hip rotation matrix (world-to-hip)
        if legID in right_legs:  # Right legs
            R_hip = np.array([
                [np.cos(q1 + np.pi), -np.sin(q1 + np.pi), 0],
                [np.sin(q1 + np.pi),  np.cos(q1 + np.pi), 0],
                [0, 0, 1]
            ])
        else:  # Left legs
            R_hip = np.array([
                [np.cos(q1), -np.sin(q1), 0],
                [np.sin(q1),  np.cos(q1), 0],
                [0, 0, 1]
            ])

        # 2. Thigh/knee rotation (hip-to-leg)
        R_thigh = np.array([
            [np.cos(q2), 0, np.sin(q2)],
            [0, 1, 0],
            [-np.sin(q2), 0, np.cos(q2)]
        ])

        # 3. Combined rotation
        R_world_to_leg = R_thigh @ R_hip

        # 4. Rotate force
        F_leg = R_world_to_leg.T @ F_world.reshape(3,1)

        return F_leg

