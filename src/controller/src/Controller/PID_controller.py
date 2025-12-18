#!/usr/bin/env python3

import rclpy
import numpy as np
from rclpy.time import Time


class PID_controller(object):
    def __init__(self, kp, ki, kd):
        self.kp = kp
        self.ki = ki
        self.kd = kd

        # desired roll and pitch angles
        # (we don't really care about yaw)
        self.desired_roll_pitch = np.array([0.0,0.0])

        self.I_term = np.array([0.0,0.0])
        self.D_term = np.array([0.0,0.0])

        # TODO: Tune max_I
        self.max_I = 0.2
        self.last_error = np.array([0.0,0.0])
         # Initialize last_time
        self.last_time = Time().to_msg().sec  # Get the current ROS2 time

    def run(self, roll, pitch):
        # determine error
        error = self.desired_roll_pitch - np.array([roll, pitch])

        # determine time step
        t_now = Time().to_msg().sec  # Get current time
        step = max(t_now - self.last_time, 1e-6)  # Avoid division by zero


        # I term update
        self.I_term = self.I_term + error * step

        # anti-windup
        for i in range(2):
            if(self.I_term[i] < -self.max_I):
                self.I_term[i] = -self.max_I
            elif(self.I_term[i] > self.max_I):
                self.I_term[i] = self.max_I

        # approximate first derivate
        self.D_term = (error - self.last_error) / step

        # update last values 
        self.last_time = t_now
        self.last_error = error

        # compute return values
        P_ret = self.kp * error
        I_ret = self.I_term * self.ki
        D_ret = self.D_term * self.kd

        return P_ret + I_ret + D_ret

    def reset(self):
        self.last_time = Time().to_msg().sec
        self.I_term = np.array([0.0,0.0])
        self.D_term = np.array([0.0,0.0])
        self.last_error = np.array([0.0,0.0])

    def desired_RP_angles(self,des_roll, des_pitch):
        # set desired roll and pitch angles
        self.desired_roll_pitch = np.array([des_roll, des_pitch])