#!/usr/bin/env python3
import numpy as np

class PD_Control_torque():
    def __init__(self, hip_gains, thigh_gains, calf_gains, limit_toque=30):
        # self.kp = kp
        # self.kd = kd
        """
            Initialize PD controller with separate gains for each joint type.
            
            :param hip_gains: Dictionary {'kp': float, 'kd': float}
            :param thigh_gains: Dictionary {'kp': float, 'kd': float}
            :param calf_gains: Dictionary {'kp': float, 'kd': float}
            :param max_torque: Maximum allowable torque (Nm)
        """
        # Joint-specific gains
        self.gains = {
            'hip': hip_gains,
            'thigh': thigh_gains,
            'calf': calf_gains
        }
        self.limit_toque = limit_toque
        self.torque_command = np.zeros(12)
        self.alpha = 0.3
        self.velocity_filter = np.zeros(12)
    
    def get_joint_type(self, index):
        """Determine joint type based on index (0=hip, 1=thigh, 2=calf)"""
        joint_types = ['hip', 'thigh', 'calf']
        return joint_types[index % 3]
    
    def compute_torque(self, joint_pos_desired, joint_vel_desired, joint_pos_feedback, joint_vel_feedback):
        """
        Compute the torque command using PD control.

        :param joint_pos_desired: Desired joint positions (numpy array)
        :param joint_vel_desired: Desired joint velocities (numpy array)
        :param joint_pos_feedback: Current joint positions (numpy array)
        :param joint_vel_feedback: Current joint velocities (numpy array)
        :return: Torque command (numpy array)
        """
        position_error = joint_pos_desired - joint_pos_feedback
        velocity_error = joint_vel_desired - joint_vel_feedback

        self.velocity_filter = self.alpha * self.velocity_filter + (1 - self.alpha) * velocity_error
        # Limit the torque command to a specified range
        for i in range(12):
            joint_type = self.get_joint_type(i)
            self.torque_command[i] = self.gains[joint_type]['kp'] * position_error[i] + self.gains[joint_type]['kd'] * velocity_error[i]
            if self.torque_command[i] >= self.limit_toque:
                self.torque_command[i] = self.limit_toque
            elif self.torque_command[i] <= -self.limit_toque:
                self.torque_command[i] = -self.limit_toque
            else:
                self.torque_command[i] = self.torque_command[i]
    
        return self.torque_command
    def reset(self):
        """
        Reset the controller state if needed.
        """
        # In this case, there is no state to reset, but this method can be overridden if needed.
        self.joint_pos_desired = np.array([
            0.0, 0.0, -2.4,    # FL leg
            0.0, 0.0, -2.4,    # FR leg
            0.0, 0.0, -2.4,     # RR leg
            0.0, 0.0, -2.4,    # RL leg
        ])
        self.joint_vel_desired = np.zeros(12)
        self.joint_pos_feedback = np.zeros(12)
        self.joint_vel_feedback = np.zeros(12)
        self.gravity_compensation = 1.0
