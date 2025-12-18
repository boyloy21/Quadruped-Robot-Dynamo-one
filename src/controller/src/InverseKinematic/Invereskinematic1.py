#!/usr/bin/env python3
import numpy as np
from math import *
from RoboticsUtilities.utils import point_to_rad, RotMatrix3D

class QuadrupedModel:
    def __init__(self, L, W, L1, L2, L3):

        # note: leg IDs
        left_front = 0
        left_back  = 1
        right_front= 2
        right_back = 3
        
        self.right_legs = [right_front, right_back]
        self.L = L      #  Length
        self.W = W      #  Width
        self.L1 = L1    #  Hip to Thigh
        self.L2 = L2    #  Thigh to Shin
        self.L3 = L3    #  Shin to Foot
        self.phi = np.pi/2      #  Hip angle

        # leg origins (left_f, left_b, right_b, right_f), i.e., the coordinate of j1
        self.leg_origins = np.matrix([[self.L/2, self.W/2, 0],
                          [-self.L/2, self.W/2, 0],
                          [-self.L/2, -self.W/2, 0],
                          [self.L/2, -self.W/2, 0],
                          [self.L/2, self.W/2, 0]])
    def leg_IK(self, xyz, rot = [0,0,0], legID=0, is_radians=True, center_offset=[0,0,0]):
        
        # check is the leg is from the right side 
        is_right = (legID in self.right_legs)
        
        # add offset of each leg from the axis of rotation
        XYZ = np.asarray((np.linalg.inv(RotMatrix3D(rot,is_radians)) * \
            ((np.array(xyz) + self.leg_origins[legID,:] - np.array(center_offset)).transpose())).transpose())
       
        # subtract the offset between the leg and the center of rotation 
        # so that the resultant coordiante is relative to the origin (j1) of the leg
        xyz_ = np.asarray(XYZ - self.leg_origins[legID,:] + np.array(center_offset)).flatten()

        # calculate the angles and coordinates of the leg relative to the origin of the leg
        return self.InverseKinematic(xyz_[0], xyz_[1], xyz_[2], is_right)

    def InverseKinematic(self, Xf, Yf, Zf, right = False):

        # Calculate Hypotenuse of A
        len_A = np.linalg.norm([Yf, Zf])
        if len_A < 1e-6:
            len_A = 1e-6

        # Calculate angles between OY and OD
        a_1 = point_to_rad(Yf, Zf)
        a_2 = asin(np.clip(sin(self.phi) * self.L1 / len_A, -1, 1))

        a_3 = pi - a_2 - self.phi

        # Calculate theta1 Differently for each leg
        if (right):
            theta1 = a_1 - a_3
        else: 
            theta1 = a_1 + a_3 
            if theta1 >= 2*pi: theta1 -= 2*pi
        
        # calculate Joint1, J1, 
        j1 = np.array([0.0, 0.0, 0.0])

        # calculate Joint2, J2 for offset of link_1
        j2 = np.array([0, self.L1*cos(theta1), self.L2*sin(theta1)])
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

    def angle_corrector(self, angles=[0,0,0], is_right=True):
        angles[1] -= 1.5*pi; # add offset 
        
        if is_right:
            theta_1 = angles[0] - pi
            theta_2 = angles[1] + 45*pi/180 # 45 degrees initial offset
        else: 
            if angles[0] > pi:  
                theta_1 = angles[0] - 2*pi
            else: theta_1 = angles[0]
            
            theta_2 = -angles[1] - 45*pi/180
        
        theta_3 = -angles[2] + 45*pi/180
        return [theta_1, -theta_2, theta_3]
        
    