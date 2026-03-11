#!/usr/bin/env python3
import numpy as np
from math import *
from .Kinematic_Model import KinematicQuadruped

class InverseDynamic:
    def __init__(self, lb, wb, l1, l2, l3, m1, m2, I1zz, I2zz, Kp, Kd, dt):
        self.l1 = l1    # Hip Length
        self.l2 = l2    # Thigh length
        self.l2 = l3    # calf length
        self.lt = l2/2    # Thigh COM distance from hip
        self.lc = l3/2    # Shank COM distance from knee
        self.m1 = m1    # Thigh mass
        self.m2 = m2    # Shank mass
        self.I1zz = I1zz  # Thigh inertia
        self.I2zz = I2zz  # Shank inertia
        self.g = 9.81   # Gravity
        self.dt = dt    # Time step
        self.model = KinematicQuadruped(lb, wb, l1, l2, l3)
        self.Kp = Kp    # Proportional gain for feedback control
        self.Kd = Kd    # Derivative gain for feedback control
        
    def theta2dot(self, theta):
        theta2dot = (theta[-1] - theta[-2])/self.dt
        return theta2dot
    
    def theta3dot(self, theta):
        theta3dot = (theta[-1] - 2*theta[-2] + theta[-3])/self.dt**2
        return theta3dot
    
    def MassMatrix(self, theta3):
        """Compute the 2x2 mass matrix M(q)."""
        M11 = (self.m1 * self.lt**2 + self.m2 * (self.l2**2 + self.lc**2 + 
               2 * self.l2 * self.lc * np.cos(theta3)) + self.I1zz + self.I2zz)
        M12 = self.m2 * (self.lc**2 + self.l2 * self.lc * np.cos(theta3)) + self.I2zz
        M21 = M12  # Symmetric
        M22 = self.m2 * self.lc**2 + self.I2zz
        M = np.array([[M11, M12], [M21, M22]])
        return M

    def CoriolisMatrix(self, theta3, theta2dot, theta3dot):
        """Compute the 2x2 Coriolis matrix C(q, qdot)."""
        h = -self.m2 * self.l2 * self.lc * np.sin(theta3)
        C11 = h * theta3dot
        C12 = h * (theta2dot + theta3dot)
        C21 = -h * theta2dot
        C22 = 0
        C = np.array([[C11, C12], [C21, C22]])
        return C

    def GravityVector(self, theta2, theta3):
        """Compute the 2x1 gravity vector G(q)."""
        G1 = (self.m1 * self.g * self.lt * np.sin(theta2) + 
              self.m2 * self.g * (self.l2 * np.sin(theta2) + 
              self.lc * np.sin(theta2 + theta3)))
        G2 = self.m2 * self.g * self.lc * np.sin(theta2 + theta3)
        return np.array([G1, G2])  # Shape (2,)

    def compute_torques(self, theta2, theta3, theta2dot, theta3dot, theta2dotdot, theta3dotdot):
        """Compute inverse dynamics torques: M(q)qddot + C(q,qdot)qdot + G(q)."""
        M = self.MassMatrix(theta3)
        C = self.CoriolisMatrix(theta3, theta2dot, theta3dot)
        G = self.GravityVector(theta2, theta3)
        
        qddot = np.array([theta2dotdot, theta3dotdot])
        qdot = np.array([theta2dot, theta3dot])
        
        tau = M @ qddot + C @ qdot + G
        return tau
    
    def torque_feedforward(self, q, qdot, a_des, legID=0):
        
        M = np.zeros((3,3))
        M[1:3, 1:3] = self.MassMatrix(q[2])
        C = np.zeros((3,3))
        C[1:3, 1:3] = self.CoriolisMatrix(q[2], qdot[1], qdot[2])
        G = np.zeros((3,1))
        G[1:3, 0] = self.GravityVector(q[1], q[2])
        J = self.model.jacobian(q, legID=legID)
        J_dot = self.model.jacobian_derivative(q, qdot, legID=legID)
        
        # Regularization to prevent singular matrix
        # Task-space inertia
        Lambda = np.linalg.pinv(J @ np.linalg.pinv(M) @ J.T)
        
        tau_ff = J.T @ Lambda @ (a_des.reshape(3,1) - J_dot @ qdot.reshape(3,1)) + C @ qdot.reshape(3,1) + G
        
        return tau_ff
    
    def TorqueSwing(self, footpos_des, footvel_des, footacc_des, q, qdot, legID=0):
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
    
    def TorqueStance(self, force, q, legID=0):
        F_leg = self.rotate_world_to_leg(force, q, legID=legID)
        J = self.model.jacobian(q, legID=legID)
        torque = J.T @ F_leg
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

