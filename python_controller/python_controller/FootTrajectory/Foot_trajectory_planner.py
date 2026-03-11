#!/usr/bin/env python3
import numpy as np
from math import *
import matplotlib.pyplot as plt
from scipy.special import comb
import scipy.linalg
import scipy.special

class FootTrajectory2:
    def __init__(self, Ts=1.0, step_length=0.1, step_height=0.03, duty_factor=0.75, z=-0.25, type='cycloid'):
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
        # Compute Bézier control points
        
    # def bezier_control_points_swing(self, move):
    #     """
    #     Solves for Bézier control points P_i given known trajectory data points (Swing Phase).
        
    #     degree: int -> Degree of the Bézier curve (n)
    #     time_samples: list -> t values in [0,1] where points are known
    #     data_points: list -> Corresponding (x, z) positions

    #     Returns:
    #         P (array): Solved control points
    #     """
    #     if move == 'forward' or move == 'left':
    #         self.data_points = self.data_points_front
    #     else:
    #         self.data_points = self.data_points_back

    #     self.degree = len(self.data_points) - 1
    #     self.time_samples = np.linspace(0, 1, len(self.data_points))
    #     n = self.degree
    #     m = len(self.time_samples)  # Number of known data points
        
    #     # Construct Bernstein basis matrix
    #     B = np.zeros((m, n+1))
    #     for j, t in enumerate(self.time_samples):
    #         for i in range(n+1):
    #             B[j, i] = scipy.special.comb(n, i) * (1 - t)**(n-i) * t**i

    #     # Solve linear system B * P = Q
    #     P = scipy.linalg.lstsq(B, self.data_points)[0]  # Least squares solution
        
    #     return P
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
            # x, y = 0, 0
            # if move == 'forward':
            #     x = offset
            # elif move == 'backward':
            #     x = -offset
            # elif move == 'left':
            #     y = offset
            # elif move == 'right':
            #     y = -offset
            # else:
            #     x = 0
            #     y = 0

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
            # P = self.bezier_control_points_swing(move)
            # pos = self.bezier_curve(P, swing_t)
            # x, y = 0, 0
            # if move == 'forward':
            #     x = pos[0] - self.step_length / 2
            # elif move == 'backward':
            #     x = pos[0] + self.step_length / 2
            # elif move == 'left':
            #     y = pos[0] - self.step_length / 2
            # elif move == 'right':
            #     y = pos[0] + self.step_length / 2
            # else:
            #     x = 0
            #     y = 0
            # z = pos[1]
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

    # def Stance_Trajectory(self, t, swing_time, stance_time, move):
    #     stance_t = (t - swing_time) / stance_time
    #     offset = (1 - stance_t) * self.step_length - self.step_length / 2
    #     x, y = 0, 0
    #     if move == 'forward':
    #         x = offset
    #     elif move == 'backward':
    #         x = -offset
    #     elif move == 'left':
    #         y = offset
    #     elif move == 'right':
    #         y = -offset
    #     else:
    #         x = 0
    #         y = 0
    #     z = self.z_final
    #     return np.array([x, y, z])
    
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
    
class Foot_trajectory_planner():
    def __init__(self, control_points, num_points):
        self.control_points = control_points
        self.num_points = num_points
        
    def bernstein(self, n, i, t):
        return comb(n, i) * (t ** i) * ((1 - t) ** (n - i))

    # Function to compute a 12-degree Bézier curve
    def bezier_curve(self):
        n = len(self.control_points) - 1  # Degree of the curve
        t_values = np.linspace(0, 1, self.num_points)
        curve = np.zeros((self.num_points, 2))  # 3D curve

        for i in range(n + 1):
            curve += np.outer(self.bernstein(n, i, t_values), self.control_points[i])

        return curve

    def stance_trajectory(self):
        """Generate a stance trajectory using a straight line."""
        t_values = np.linspace(0, 1, self.num_points)
        trajectory = self.control_points[-1] + t_values[:, np.newaxis] * (self.control_points[0] - self.control_points[-1])
        return trajectory


class FootTrajectory:
    def __init__(self, Ts=1.0, step_length=0.1, step_height=0.05, duty_factor=0.6, z=-0.25):
        self.Ts = Ts
        self.step_length = step_length
        self.step_height = step_height
        self.duty_factor = duty_factor
        self.z_initial = z  # Initial Z position for stance phase
        self.z_final = z   # Final Z position for stance phase

    def generate(self, t, move_direction='forward'):
        t %= self.Ts  # Wrap time to [0, T]
        swing_time = self.Ts * (1 - self.duty_factor)
        stance_time = self.Ts * self.duty_factor

        if move_direction == 'forward':
            if t < swing_time:
                # SWING PHASE FIRST
                swing_t = t / swing_time
                x = swing_t * self.step_length - self.step_length / 2
                y = 0
                z = self.step_height * np.sin(np.pi * swing_t) + self.z_initial
            else:
                # STANCE PHASE
                stance_t = (t - swing_time) / stance_time
                x = (1 - stance_t) * self.step_length - self.step_length / 2
                y = 0
                z = self.z_final
        elif move_direction == 'backward':
            if t < swing_time:
                # SWING PHASE FIRST
                swing_t = t / swing_time
                x = -(swing_t * self.step_length - self.step_length / 2) 
                y = 0
                z = self.step_height * np.sin(np.pi * swing_t) + self.z_initial
            else:
                # STANCE PHASE
                stance_t = (t - swing_time) / stance_time
                x = -((1 - stance_t) * self.step_length - self.step_length / 2)
                y = 0
                z = self.z_final

        elif move_direction == 'stop':
            x = 0
            y = 0
            z = self.z_final

        elif move_direction == 'left':
            if t < swing_time:
                # SWING PHASE FIRST
                swing_t = t / swing_time
                x = 0
                y = swing_t * self.step_length - self.step_length / 2
                z = self.step_height * np.sin(np.pi * swing_t) + self.z_initial
            else:
                # STANCE PHASE
                stance_t = (t - swing_time) / stance_time
                x = 0
                y = (1 - stance_t) * self.step_length - self.step_length / 2
                z = self.z_final

        elif move_direction == 'right':
            if t < swing_time:
                # SWING PHASE FIRST
                swing_t = t / swing_time
                x = 0
                y = -(swing_t * self.step_length - self.step_length / 2) 
                z = self.step_height * np.sin(np.pi * swing_t) + self.z_initial
            else:
                # STANCE PHASE
                stance_t = (t - swing_time) / stance_time
                x = 0
                y = -((1 - stance_t) * self.step_length - self.step_length / 2)
                z = self.z_final

        return np.array([x, y, z])
        

# control_points = np.array([
#     [0.0, -0.35],      # Start position (adjusted Z to -0.35)
#     [0.05, -0.35],     # Early lift
#     [0.08, -0.314],     # Mid lift
#     [0.1, -0.314],     # Peak height
#     [0.1, -0.314],     # Highest forward point
#     [0.15, -0.314],      # Descending phase start
#     [0.15, -0.314],      # Lowering further
#     [0.15, -0.318],      # Almost touching ground
#     [0.2, -0.318],    # Landing position
#     [0.2, -0.318],    # Slight after-movement
#     [0.282, -0.35],     # Return to stable position
#     [0.3, -0.35],     # Final adjustment (end position)
# ])

# num_points = 100
# Foot = Foot_trajectory_planner(control_points=control_points, num_points=num_points)
# curve = Foot.bezier_curve()
# stance = Foot.stance_trajectory()
# # Plot the 2D foot trajectory (XZ plane)
# plt.figure(figsize=(8, 5))
# plt.plot(curve[:, 0], curve[:, 1], label="Foot Trajectory", color='blue')
# plt.plot(stance[:, 0], stance[:, 1], label="Stance Trajectory", color='green')
# plt.scatter(control_points[:, 0], control_points[:, 1], c='red', label="Control Points", marker='o')
# plt.scatter(control_points[0, 0], control_points[0, 1], c='green', label="Start Point", marker='o')
# plt.scatter(control_points[-1, 0], control_points[-1, 1], c='magenta', label="End Point", marker='o')
# # Labels and View
# plt.xlabel("X (Stride)")
# plt.ylabel("Z (Height)")
# plt.title("2D Foot Swing Trajectory (XZ Plane, 12-degree Bézier Curve)")
# plt.legend()
# plt.grid(True)

# # Show plot
# plt.show()