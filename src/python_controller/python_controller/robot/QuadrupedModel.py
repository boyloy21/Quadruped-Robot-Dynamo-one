#!/usr/bin/env python3
import numpy as np
from math import *
from .utils import point_to_rad, RotMatrix3D

import numpy as np
from math import sin, cos, radians

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

    # Create 4x4 homogeneous rotation matrix
    R_4x4 = np.eye(4)
    R_4x4[:3, :3] = R  # embed 3x3 rotation into top-left
    return R_4x4

class RobotModel():
    def __init__(self, L, W, L1, L2, L3):

        # note: leg IDs
        left_front = 0
        left_back  = 3
        right_front= 1
        right_back = 2
        
        self.right_legs = [right_front, right_back]
        self.L = L      #  Length
        self.W = W      #  Width
        self.L1 = L1    #  Hip to Thigh
        self.L2 = L2    #  Thigh to Shin
        self.L3 = L3    #  Shin to Foot
        self.link_1 = L1   # Hip offset
        self.link_2 = L2  # Upper leg length
        self.link_3 = L3  # Lower leg length
        self.phi = np.pi/2      #  Hip angle
        
        ## Leg origin coordinates (FR, FL, RR, RL)
        # self.leg_origins = np.matrix([[self.L/2, self.W/2, 0],
        #                   [self.L/2, -self.W/2, 0],
        #                   [-self.L/2, -self.W/2, 0],
        #                   [-self.L/2, self.W/2, 0]])
        self.leg_origins = np.array([
            [ self.L / 2,  self.W / 2, 0],
            [ self.L / 2, -self.W / 2, 0],
            [-self.L / 2, -self.W / 2, 0],
            [-self.L / 2,  self.W / 2, 0]
        ])

        self.joint_angles = np.zeros((4,3))
        self.leg_signs = [
            (1.0, 1.0),   # Front Left
            (1.0, -1.0),  # Front Right
            (-1.0, -1.0),  # Back Right
            (-1.0, 1.0)    # Back Left
        ]
        self.y_offset = np.array([[0, self.L1, 0],
                                  [0, -self.L1, 0],
                                  [0, -self.L1, 0],
                                  [0, self.L1, 0]])
        # Joint limits
        self.joint_limits = {
            "hip": (-np.pi / 2, np.pi / 2),
            "thigh": (-np.pi, np.pi),
            "knee": (-np.pi, np.pi)
        }
        self.error = np.zeros((4, 3))
        self.joint_angles_back = np.zeros((4, 3))
        
    def compute_leg_frame_foot_position(self, p_B_world, R_B, f_i_world, O_i_body):
        """Compute foot position in local leg frame."""
        # Use only 3x3 rotation from 4x4 matrix
        R3 = R_B[:3, :3]

        O_i_world = p_B_world + R3 @ O_i_body
        f_i_relative = f_i_world - O_i_world
        f_i_leg = R3.T @ f_i_relative  # Transform to leg frame

        return f_i_leg

    def Calculate_IK(self, pB, foot_world, rot=[0.0, 0.0, 0.0], is_radians=True):
        Rb = rotMatrix3D(rot, is_radians=is_radians)  # 4x4 matrix
        for i in range(4):
            f_leg = self.compute_leg_frame_foot_position(
                p_B_world=pB,
                R_B=Rb,
                f_i_world=foot_world[i, :],
                O_i_body=self.leg_origins[i, :]
            )
            self.joint_angles[i, :] = self.InverseKinematic(
                f_leg[0], f_leg[1], f_leg[2],
                right=(i in self.right_legs)
            )
            print(f_leg)
        FL_Joint = self.joint_angles[0, :]
        FR_Joint = self.joint_angles[1, :]
        RR_Joint = self.joint_angles[2, :]
        RL_Joint = self.joint_angles[3, :]

        return FL_Joint, FR_Joint, RR_Joint, RL_Joint

    def leg_IK(self, xyz, rot = [0,0,0], is_radians=True, center_offset=[0,0,0]):
        rot_matrix = np.linalg.inv(RotMatrix3D(rot, is_radians=is_radians))
        for i in range(4):
            leg_pos = xyz[i, :] + self.leg_origins[i, :] - np.array(center_offset)
            transformed_xyz = (rot_matrix @ leg_pos.T).T
            xyz_tf = np.array(transformed_xyz - self.leg_origins[i, :] + np.array(center_offset)).flatten()
            
            ### Compute to Inverskenematic of each leg
            #1. By Yun
            angles_and_joints = self.InverseKinematic(xyz_tf[0], xyz_tf[1], xyz_tf[2], right=(i in self.right_legs))

            #2. By B.Phanny
            # angles_and_joints = self.compute_leg_ik([xyz_tf[0], xyz_tf[1], xyz_tf[2]], leg_id=i)

            self.joint_angles[i, :] = angles_and_joints

        FL_Joint = self.joint_angles[0, :]
        FR_Joint = self.joint_angles[1, :]
        RR_Joint = self.joint_angles[2, :]
        RL_Joint = self.joint_angles[3, :]
        # calculate the angles and coordinates of the leg relative to the origin of the leg
        return FL_Joint, FR_Joint, RR_Joint, RL_Joint
    
    #####===== By Mr.Yun ====#####
    def InverseKinematic(self, Xf, Yf, Zf, right = False):
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
        # In your QuadrupedModel.py, line 119
        # len_B = np.linalg.norm([x_, z_])  # Ensure this isn't zero before division
        # if len_B == 0 or (self.L2**2 + len_B**2 - self.L3**2) / (2 * self.L2 * len_B) > 1:
        #     # Handle degenerate case
        #     b_2 = 0.0  # Or some safe default
        # else:
        #     b_2 = np.arccos(np.clip((self.L2**2 + len_B**2 - self.L3**2) / (2 * self.L2 * len_B), -1, 1))
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
    def clamp_angle(self, angle, min_angle, max_angle):
        return max(min_angle, min(max_angle, angle))
    
    #####==== By Mr.phanny ====#####
    def compute_leg_ik(self, target_position, leg_id):
        x, y, z = target_position
        x_sign, y_sign = self.leg_signs[leg_id]

        yz_d = np.sqrt(y**2 + z**2 - self.link_1**2)
        theta1 = np.arctan2(y, -z) + np.arctan2(-self.link_1 * y_sign, yz_d)
        theta1 = self.clamp_angle(theta1, *self.joint_limits["hip"])

        xyz_d = yz_d**2 + x**2
        theta3 = -np.pi + np.arccos((xyz_d - self.link_2**2 - self.link_3**2) / (-2 * self.link_2 * self.link_3))  
        theta3 = self.clamp_angle(theta3, *self.joint_limits["knee"])

        theta2 = np.arctan2(-x, yz_d) - np.arcsin(self.link_3 * np.sin(theta3) / np.sqrt(xyz_d)) 
        theta2 = self.clamp_angle(theta2, *self.joint_limits["thigh"]) 
        
        return [theta1, theta2, theta3]
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
        theta_2 = angles[1] + 55*pi/180
        theta_3 = -angles[2] 
        return [theta_1, -theta_2, theta_3]
    
    def ForwardKinematic(self, angles, legID=0, center_offset=[0,0,0]):
        theta1, theta2, theta3 = angles
        is_right = (legID in self.right_legs)

        # Correct angles to match robot configuration (reverse of angle_corrector)
        if is_right:
            theta1 = theta1 + pi
            # theta2 = theta2 - 45*pi/180
        else:
            if theta1 < 0:
                theta1 = theta1 + 2*pi
            # theta2 = -theta2 - 45*pi/180
        theta2 = -theta2 - 55*pi/180
        theta3 = -theta3 

        # Add 270 degrees (1.5*pi) to theta2 because it was subtracted in angle_corrector
        theta2 += 1.5*pi

        # Forward Kinematics calculation
        # Step 1: calculate j2
        j1 = np.array([0.0, 0.0, 0.0])
        j2 = np.array([0, self.L1*cos(theta1), self.L1*sin(theta1)])

        # Step 2: calculate rotation
        if is_right:
            R = theta1 - self.phi - pi/2
        else:
            R = theta1 + self.phi - pi/2

        rot_mtx = RotMatrix3D([-R, 0, 0], is_radians=True)

        # Step 3: calculate j3 and j4 in rotated frame
        j3_ = np.array([
            self.L2 * cos(theta2),
            0,
            self.L2 * sin(theta2)
        ])
        j3 = j2 + np.reshape(np.linalg.inv(rot_mtx) @ np.reshape(j3_, [3,1]), [3])

        j4_ = j3_ + np.array([
            self.L3 * cos(theta2 + theta3),
            0,
            self.L3 * sin(theta2 + theta3)
        ])
        j4 = j2 + np.reshape(np.linalg.inv(rot_mtx) @ np.reshape(j4_, [3,1]), [3])

        # Step 4: move to world coordinate
        # Pfoot_world = j4 + np.squeeze(np.asarray(self.leg_origins[legID])) - np.array(center_offset)
        # Pfoot_world = j4
        return j4
    
    def Calculate_error(self, xyz, angles, rot=[0, 0, 0], center_offset=[0,0,0], is_radians=False):
        self.joint_angles_back = angles
        rot_matrix = np.linalg.inv(RotMatrix3D(rot, is_radians=is_radians))
        for i in range(3):
            leg_pos = xyz[i, :] + self.leg_origins[i, :] - np.array(center_offset)
            transformed_xyz = (rot_matrix @ leg_pos.T).T
            xyz_tf = np.array(transformed_xyz - self.leg_origins[i, :] + np.array(center_offset)).flatten()

            p_foot = self.ForwardKinematic(np.array(self.joint_angles[i, :]), i, center_offset)
            Foot_back_base = p_foot + np.squeeze(np.asarray(self.leg_origins[i])) - np.array(center_offset)

            # Then rotate it to the world frame
            Foot_back_world = RotMatrix3D(rot, is_radians) @ Foot_back_base.reshape(3,1)
            Foot_back_world = np.squeeze(np.asarray(Foot_back_world)) - self.leg_origins[i]
            self.error[i, :] = xyz_tf - Foot_back_world

        return self.error
            
           