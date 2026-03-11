#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist
from std_msgs.msg import Float64
import numpy as np
from math import pi, sin, cos
from python_controller.robot.utils import RotMatrix3D


class RaibertHeuristic(Node):
    def __init__(self):
        super().__init__('raibert_heuristic')
        self.Foot_desired = self.create_publisher(Twist, 'dynamo_one/foot_desired', 10)

        self.P_footdesired = np.array([[0.0, 0.0, 0.0, 0.0],    # FL
                                      [0.0, 0.0, 0.0, 0.0],     # FR
                                      [0.0, 0.0, 0.0, 0.0],     # RL
                                      [0.0, 0.0, 0.0, 0.0]])    # RR
        self.P_com = np.array([0.0, 0.0, 0.0])
        self.P_norm_stance_FL = np.array([0.0, 0.0, 0.0])
        self.P_norm_stance_FR = np.array([0.0, 0.0, 0.0])
        self.P_norm_stance_RL = np.array([0.0, 0.0, 0.0])
        self.P_norm_stance_RR = np.array([0.0, 0.0, 0.0])
        self.Vx_command = 0.0
        self.Vy_command = 0.0
        self.a_command = 0.0
        self.Vx_desired = 0.0
        self.Vy_desired = 0.0
        self.a_desired = 0.0
        
        self.V_command = np.array([self.Vx_command, self.Vy_command, self.a_command])
        self.V_desired = np.array([self.Vx_desired, self.Vy_desired, self.a_desired])

        self.Tstep = 0.0 # Swing time step
        self.kx = 0.02  # Range(0.1 - 0.3)
        self.ky = 0.02  # Range(0.1 - 0.3)
        self.ka = 0.15 # Range(0.05 - 0.3)
        self.Kv = np.diagonal(np.array([self.kx, self.ky, self.ka])) # Velocity gain

    def Calculate(self, Tstep, roll, pitch, yaw):
        """
        Calculate the desired foot position based on the current state and desired velocity.
        :param Tstep: Time step for the swing phase
        :param roll: Current roll angle
        :param pitch: Current pitch angle
        :param yaw: Current yaw angle
        """
        self.Tstep = Tstep

        self.P_footdesired = self.P_com + (1/2)*Tstep*self.V_command + self.Kv*(self.V_command - self.V_desired) + np.array([[RotMatrix3D(roll,pitch,yaw)@self.P_norm_stance_FL]
                                                                                                                             [RotMatrix3D(roll,pitch,yaw)@self.P_norm_stance_FR],
                                                                                                                             [RotMatrix3D(roll,pitch,yaw)@self.P_norm_stance_RL],
                                                                                                                             [RotMatrix3D(roll,pitch,yaw)@self.P_norm_stance_RR]])
        