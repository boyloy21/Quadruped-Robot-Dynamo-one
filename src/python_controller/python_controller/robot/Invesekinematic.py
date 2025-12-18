#!/usr/bin/env python3
import numpy as np
from math import *
from .utils import point_to_rad , RotMatrix3D

# Note : 
# _ FL(Front Left) = Leg1 , FL(Front Right)= Leg2 , RL(Rear Left) = Leg3 , RR(Rear Right) = Leg4
# _ Xf is axis for move forward and backward
# _ Yf is axis for move left and right
# _ Zf is axis for move up and down
# _ theta1 is hip angle rotate in X axis of base frame
# _ theta2 is knee angle rotate in Y axis of base frame
# _ theta3 is ankle angle rotate in Y axis of base frame

class InverseKinematic():
    def __init__(self, L1, L2, L3):
        self.L1 = L1
        self.L2 = L2
        self.L3 = L3
        self.phi = np.pi/2
    
    def InverseKinematic1(self, Xf, Yf, Zf, right=True):
        F = np.sqrt(Yf**2 + Zf**2 - self.L1**2)
        G = F 
        H = np.sqrt(G**2 + Xf**2)

        # Hip Angle
        if (right):
            theta1 = -atan2(Zf, Yf) - atan2(F, -self.L2)
        else :
            theta1 = atan2(Zf, Yf) + atan2(F, self.L2)
        # Knee Angle
        D = (H**2 - self.L2**2 - self.L3**2) / (2 * self.L2 * self.L3)
        D = max(-1, min(1, D))  # Ensure valid range for acos

        theta3 = acos(-D) - pi

        # Ankle Angle
        theta2 = atan2(Xf, G) - atan2(self.L3 * sin(theta3), self.L3 + self.L3 * cos(theta3))

        return theta1, theta2, theta3

    def InverseKinematic2(self, Xf, Yf, Zf, right = True):
       
        D = np.sqrt(Xf**2 + Yf**2 - self.L1**2 + Zf**2 - self.L2**2 - self.L3**2)/(2*self.L2*self.L3)
        theta1 = -np.arctan2(-Zf, Yf) - np.arctan2(np.sqrt(Zf**2 + Yf**2 - self.L1**2), -self.L1)
        if (right):
            theta3 = np.arctan2(np.sqrt(1 - D**2), D)
        else:
            theta3 = np.arctan2(-np.sqrt(1 - D**2), D)
        theta2 = np.arctan2(Zf, np.sqrt(Xf**2 + Yf**2 - self.L1**2)) - np.arctan2(self.L3 * np.sin(theta3), self.L2 + self.L3 * np.cos(theta3))

        return theta1, theta2, theta3
    def InverseKinematic3(self, Xf, Yf, Zf):
        D = np.sqrt(Xf**2 + Yf**2 - self.L1**2)
        E = (self.L2**2 + self.L3**2 - Zf**2 - D**2)/(2 * self.L2 * self.L3)
        E = np.clip(E, -1, 1)
        theta1 = np.arctan2(D, self.L1) - np.arctan2(-Yf, Xf)
        theta3 = np.pi - np.arccos(E)
        theta2 = -np.arctan2(self.L3*np.sin(theta3), self.L2 + self.L3*np.cos(theta3)) - np.arctan2(-Zf, D)

        return theta1, theta2, theta3
    
    def ForwardKinematic4(self, theta1, theta2, theta3, right=False):
        """
        Calculate the end-effector position (Xf, Yf, Zf) given joint angles (theta1, theta2, theta3).
        :param theta1: Hip joint angle (rotation around the Y-Z plane)
        :param theta2: Thigh joint angle (rotation in the X-Z plane)
        :param theta3: Calf joint angle (rotation in the X-Z plane)
        :param right: Boolean indicating whether the leg is on the right side
        :return: (Xf, Yf, Zf) - End-effector position in 3D space
        """
        # Step 1: Calculate the position of joint J2 (after the hip joint)
        if right:
            theta1 = theta1 - self.phi
        else:
            theta1 = theta1 + self.phi

        J2_x = 0
        J2_y = self.L1 * cos(theta1)
        J2_z = self.L1 * sin(theta1)

        # Step 2: Calculate the position of joint J3 (after the thigh joint)
        theta2_global = theta2 + pi / 2  # Adjust for the robot's configuration
        J3_x = J2_x + self.L2 * cos(theta2_global)
        J3_y = J2_y
        J3_z = J2_z + self.L2 * sin(theta2_global)

        # Step 3: Calculate the position of the end-effector (foot, J4)
        theta3_global = theta2_global + theta3
        Xf = J3_x + self.L3 * cos(theta3_global)
        Yf = J3_y
        Zf = J3_z + self.L3 * sin(theta3_global)

        return Xf, Yf, Zf
    
    def InverseKinematic4(self, Xf, Yf, Zf, right = False):
        len_A = np.linalg.norm([Yf, Zf])
        if len_A < 1e-6:
            len_A = 1e-6
        a_1 = point_to_rad(Yf, Zf)
        # a_2 = asin(sin(self.phi) * self.L1 / len_A)
        a_2 = asin(np.clip(sin(self.phi) * self.L1 / len_A, -1, 1))

        a_3 = pi - a_2 - self.phi
        # or 
        # a_3 = atan2(np.sqrt(Yf**2 + Zf**2 - self.L1**2), self.L1)
        
        if (right):
            theta1 = a_1 - a_3
        else: 
            theta1 = a_1 + a_3 
            if theta1 >= 2*pi: theta1 -= 2*pi
        
        # calculate Joint1, J1, 
        j1 = np.array([0.0, 0.0, 0.0])
        j2 = np.array([0, self.L1*cos(theta1), self.L1*sin(theta1)])
        j4 = np.array([Xf, Yf, Zf])
        j4_2_vec = j4 - j2  # vector from j2 to j4

        if right : R = theta1 - self.phi - pi/2
        else: R = theta1 + self.phi - pi/2

        # Create rotation matix to work on new 2D plane (XZ_)
        rot_mtx = RotMatrix3D([-R, 0, 0], is_radians=True)
        j4_2_vec_ = rot_mtx * (np.reshape(j4_2_vec, [3,1]))

        # xyz in the rotated coordinate system + offset due to link_1 removed
        x_, y_, z_ = j4_2_vec_[0], j4_2_vec_[1], j4_2_vec_[2]

        len_B = np.linalg.norm([x_, z_])
        if (len_B >= (self.L2 + self.L3)):
            len_B = (self.L2 + self.L3) * 0.99999
        
        b_1 = point_to_rad(x_, z_)
        b_2 = acos((self.L2**2 + len_B**2 -self.L3**2) / (2 * self.L2 * len_B))
        b_3 = acos((self.L2**2 + self.L3**2 - len_B**2) / (2 * self.L2 * self.L3))

        # assuming theta_2 = 0 when the leg is pointing down (i.e., 270 degrees offset from the +ve x-axis)
        theta2 = b_1 - b_2
        theta3 = pi - b_3

        # Caculate joint3
        j3_ = np.reshape(np.array([self.L2*cos(theta2), 0, self.L2*sin(theta2)]), [3,1])
        j3 = np.asarray(j2 + np.reshape(np.linalg.inv(rot_mtx)*j3_, [1,3])).flatten()

        # Calculate Joint 4
        j4_ = j3_ + np.reshape(np.array([self.L3*cos(theta2+theta3),0, self.L3*sin(theta2+theta3)]), [3,1])
        j4 = np.asarray(j2 + np.reshape(np.linalg.inv(rot_mtx)*j4_, [1,3])).flatten()

        # Modify angles to match robot's configuration (ie., adding offsets)
        angles = self.angle_corrector(angles=[theta1, theta2, theta3], is_right=right)

        return [angles[0], angles[1], angles[2]]
        # return [angles[0], angles[1], angles[2], j1, j2, j3, j4]

    def InverseKinematic5(self, Xf, Yf, Zf):
        
        # calculate Joint1 is q1 angle
        alpha = acos(abs(Yf)/np.sqrt(Yf**2 + Zf**2))
        E = np.sqrt(Yf**2 + Zf**2)
        E = np.clip(E, -1, 1)
        beta = acos(self.L1/E)
        if (Zf <= 0 and Yf>=0):
            q1 = alpha - beta
        elif (Zf < 0 and Yf<0):
            q1 = pi - alpha - beta
        
        # calculate q1 and q2
        # calculate z'
        z_prim = -1*np.sqrt(Zf**2 + Yf**2 - self.L1**2)

        # calculate phi
        phi = acos((abs(Xf))/(np.sqrt(Xf**2 + z_prim**2)))
        psi = acos((self.L2**2 + Xf**2 + z_prim**2 - self.L3**2)/(2*self.L2*np.sqrt(Xf**2 + z_prim**2)))
        
        q3 = acos((self.L2**2 + self.L3**2 - Xf**2 - z_prim**2)/(2*self.L2*self.L3))
        
        # calculate q2
        if (q3 >= 0):
            if (Xf>=0 and z_prim<=0):
                q2 = pi/2 - phi - psi
            elif (Xf<0 and z_prim<=0):
                q2 = -pi/2 - phi - psi
        elif (q3 < 0):
            if (Xf>=0 and z_prim<=0):
                q2 = pi/2 + phi - psi
            elif (Xf<0 and z_prim<=0):
                q2 = -pi/2 + phi + psi

        # Modify angles to match robot's configuration (ie., adding offsets)
        # angles = self.angle_corrector(angles=[q1, q2, q3], )

        # return [angles[0], angles[1], angles[2]]
        return [q1, q2, q3]

    def angle_corrector(self, angles=[0,0,0], is_right=True):
        angles[1] -= 1.5*pi; # add offset 
        
        if is_right:
            theta_1 = angles[0] - pi
             # 45 degrees initial offset
        else: 
            if angles[0] > pi:  
                theta_1 = angles[0] - 2*pi
            else: theta_1 = angles[0]
            
            # theta_2 = -angles[1] - 45*pi/180
        theta_2 = angles[1] + 45*pi/180
        theta_3 = -angles[2] + 45*pi/180
        return [theta_1, -theta_2, theta_3]

