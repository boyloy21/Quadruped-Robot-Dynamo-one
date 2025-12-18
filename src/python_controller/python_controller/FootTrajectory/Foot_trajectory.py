#!/usr/bin/env python3
import numpy as np
import matplotlib.pyplot as plt
import scipy.linalg
import scipy.special
from math import *

class FootTrajectoryGenerator:
    def __init__(self, Ts=1.0, step_length=0.1, step_height=0.03, duty_factor=0.75, z=-0.25, type='bezier'):
        self.Ts = Ts
        self.step_length = step_length
        self.step_height = step_height
        self.duty_factor = duty_factor
        self.z_initial = z  # Initial Z position for stance phase
        self.z_final = z  # Final Z position for stance phase
        self.x_initial = 0.0  # Initial X position for stance phase
        self.x_final =-0.08  # Final X position for stance phase
        self.data_points_front = np.array([
            [0, self.z_initial],            # P0: Start of swing phase
            [step_length * 0.2, step_height * 0.7 + self.z_initial],  # P1
            [step_length * 0.4, step_height + self.z_initial],  # P2: Peak height
            [step_length * 0.6, step_height + self.z_initial],  # P3: Peak height
            [step_length * 0.8, step_height * 0.7 + self.z_initial],  # P4
            [step_length, self.z_initial]  # P5: End of swing phase
        ])
        self.data_points_back = np.array([
            [0, self.z_initial],
            [-step_length * 0.2, step_height * 0.7 + self.z_initial],
            [-step_length * 0.4, step_height + self.z_initial],
            [-step_length * 0.6, step_height + self.z_initial],
            [-step_length * 0.8, step_height * 0.7 + self.z_initial],
            [-step_length, self.z_initial]
        ])
        diag_step = self.step_length / np.sqrt(2)

        self.data_points_diag_front = np.array([
            [0, self.z_initial],
            [diag_step * 0.2, step_height * 0.7 + self.z_initial],
            [diag_step * 0.4, step_height + self.z_initial],
            [diag_step * 0.6, step_height + self.z_initial],
            [diag_step * 0.8, step_height * 0.7 + self.z_initial],
            [diag_step, self.z_initial]
        ])
        self.data_points_diag_back = -self.data_points_diag_front
        ## Move Type : Forward , Backward, Left, Right
        self.type= type
        
    def bezier_control_points_swing(self, move):
        if move in ['forward', 'left']:
            self.data_points = self.data_points_front
        elif move in ['backward', 'right']:
            self.data_points = self.data_points_back
        elif move in ['forward_left', 'forward_right']:
            self.data_points = self.data_points_diag_front
        elif move in ['backward_left', 'backward_right']:
            self.data_points = self.data_points_diag_back
        else:
            self.data_points = self.data_points_front

        self.degree = len(self.data_points) - 1
        self.time_samples = np.linspace(0, 1, len(self.data_points))
        n = self.degree
        m = len(self.time_samples)

        B = np.zeros((m, n + 1))
        for j, t in enumerate(self.time_samples):
            for i in range(n + 1):
                B[j, i] = scipy.special.comb(n, i) * (1 - t)**(n - i) * t**i

        P = scipy.linalg.lstsq(B, self.data_points)[0]
        return P
    # @staticmethod
    def bezier_curve(self, P, t):
        n = len(P) - 1
        point = np.zeros(2)
        for i in range(n + 1):
            B_i = scipy.special.comb(n, i) * (1 - t)**(n - i) * t**i
            point += B_i * P[i]
        return point
    
    def Swing_Trajectory(self, t, swing_time, move):
        swing_t = t / swing_time
        if self.type == 'cycloid':
            # Cycloid motion
            offset = swing_t * self.step_length - self.step_length / 2
            x, y = 0, 0
            if move == 'forward':
                x = offset
            elif move == 'backward':
                x = -offset
            elif move == 'left':
                y = offset
            elif move == 'right':
                y = -offset
            elif move == 'forward_left':
                x = offset / np.sqrt(2)
                y = offset / np.sqrt(2)
            elif move == 'forward_right':
                x = offset / np.sqrt(2)
                y = -offset / np.sqrt(2)
            elif move == 'backward_left':
                x = -offset / np.sqrt(2)
                y = offset / np.sqrt(2)
            elif move == 'backward_right':
                x = -offset / np.sqrt(2)
                y = -offset / np.sqrt(2)

            z = self.step_height * np.sin(np.pi * swing_t) + self.z_initial
        else:
            # # Bézier trajectory
            P = self.bezier_control_points_swing(move)
            pos = self.bezier_curve(P, swing_t)
            x = y = 0
            delta = self.step_length / 2
            if move == 'forward':
                x = pos[0] - delta
            elif move == 'backward':
                x = pos[0] + delta
            elif move == 'left':
                y = pos[0] - delta
            elif move == 'right':
                y = pos[0] + delta
            elif move == 'forward_left':
                x = (pos[0] - delta) / np.sqrt(2)
                y = (pos[0] - delta) / np.sqrt(2)
            elif move == 'forward_right':
                x = (pos[0] - delta) / np.sqrt(2)
                y = -(pos[0] - delta) / np.sqrt(2)
            elif move == 'backward_left':
                x = (pos[0] + delta) / np.sqrt(2)
                y = (pos[0] - delta) / np.sqrt(2)
            elif move == 'backward_right':
                x = (pos[0] + delta) / np.sqrt(2)
                y = -(pos[0] - delta) / np.sqrt(2)
            z = pos[1]
        return np.array([x, y, z])

    def Stance_Trajectory(self, t, swing_time, stance_time, move):
        stance_t = (t - swing_time) / stance_time
        offset = (1 - stance_t) * self.step_length - self.step_length / 2
        x, y = 0, 0
        if move == 'forward':
            x = offset
        elif move == 'backward':
            x = -offset
        elif move == 'left':
            y = offset
        elif move == 'right':
            y = -offset
        elif move == 'forward_left':
            x = offset / np.sqrt(2)
            y = offset / np.sqrt(2)
        elif move == 'forward_right':
            x = offset / np.sqrt(2)
            y = -offset / np.sqrt(2)
        elif move == 'backward_left':
            x = -offset / np.sqrt(2)
            y = offset / np.sqrt(2)
        elif move == 'backward_right':
            x = -offset / np.sqrt(2)
            y = -offset / np.sqrt(2)

        z = self.z_final
        return np.array([x, y, z])

    
    
    def generate(self, t, move_direction='backward'):
        t %= self.Ts  # Wrap time to [0, T]
        swing_time = self.Ts * (1 - self.duty_factor)
        stance_time = self.Ts * self.duty_factor
        
        if t < swing_time:
            # SWING PHASE FIRST
            x, y, z = self.Swing_Trajectory(t, swing_time, move_direction)
        else:
            # STANCE PHASE
            x, y, z = self.Stance_Trajectory(t, swing_time, stance_time, move_direction)

        return np.array([x, y, z])
    