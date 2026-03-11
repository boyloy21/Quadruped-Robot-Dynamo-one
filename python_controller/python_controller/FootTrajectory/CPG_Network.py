#!/usr/bin/env python3
import numpy as np
import matplotlib.pyplot as plt
from math import *
from .Foot_trajectory import FootTrajectoryGenerator

class CPG_Network():
    def __init__(self, cycles, trajectory='bezier'):
        
        self.R_walk = np.array([
            [1.000, -0.000, -1.000, 0.000, 0.000, 1.000, -0.000, -1.000],
            [0.000, 1.000, -0.000, -1.000, -1.000, 0.000, 1.000, -0.000],
            [-1.000, -0.000, 1.000, -0.000, 0.000, -1.000, 0.000, 1.000],
            [0.000, -1.000, 0.000, 1.000, 1.000, 0.000, -1.000, 0.000],
            [0.000, -1.000, 0.000, 1.000, 1.000, -0.000, -1.000, 0.000],
            [1.000, 0.000, -1.000, 0.000, 0.000, 1.000, -0.000, -1.000],
            [-0.000, 1.000, 0.000, -1.000, -1.000, -0.000, 1.000, -0.000],
            [-1.000, -0.000, 1.000, 0.000, 0.000, -1.000, 0.000, 1.000]
            ])
        
        self.R_walk_back = np.array([
            [1.000, -0.000, -1.000, -0.000, 0.000, -1.000, -0.000, 1.000],
            [0.000, 1.000, 0.000, -1.000, 1.000, 0.000, -1.000, -0.000],
            [-1.000, 0.000, 1.000, -0.000, 0.000, 1.000, 0.000, -1.000],
            [-0.000, -1.000, 0.000, 1.000, -1.000, 0.000, 1.000, 0.000],
            [0.000, 1.000, 0.000, -1.000, 1.000, -0.000, -1.000, -0.000],
            [-1.000, 0.000, 1.000, 0.000, 0.000, 1.000, 0.000, -1.000],
            [-0.000, -1.000, 0.000, 1.000, -1.000, 0.000, 1.000, -0.000],
            [1.000, -0.000, -1.000, 0.000, -0.000, -1.000, 0.000, 1.000]
            ])
        
        self.R_walk_left = np.array([
            [1.000, -0.000, -0.000, -1.000, -1.000, 0.000, 0.000, 1.000],
            [0.000, 1.000, 1.000, -0.000, -0.000, -1.000, -1.000, 0.000],
            [-0.000, 1.000, 1.000, -0.000, 0.000, -1.000, -1.000, -0.000],
            [-1.000, -0.000, 0.000, 1.000, 1.000, 0.000, 0.000, -1.000],
            [-1.000, -0.000, 0.000, 1.000, 1.000, -0.000, 0.000, -1.000],
            [0.000, -1.000, -1.000, 0.000, 0.000, 1.000, 1.000, 0.000],
            [0.000, -1.000, -1.000, 0.000, 0.000, 1.000, 1.000, -0.000],
            [1.000, 0.000, -0.000, -1.000, -1.000, 0.000, 0.000, 1.000],
            ])
        self.R_walk_right = np.array([
            [1.000, -0.000, -0.000, 1.000, -1.000, -0.000, 0.000, -1.000],
            [0.000, 1.000, -1.000, -0.000, 0.000, -1.000, 1.000, 0.000],
            [-0.000, -1.000, 1.000, -0.000, 0.000, 1.000, -1.000, 0.000],
            [1.000, -0.000, 0.000, 1.000, -1.000, 0.000, -0.000, -1.000],
            [-1.000, 0.000, 0.000, -1.000, 1.000, -0.000, 0.000, 1.000],
            [-0.000, -1.000, 1.000, 0.000, 0.000, 1.000, -1.000, 0.000],
            [0.000, 1.000, -1.000, -0.000, 0.000, -1.000, 1.000, -0.000],
            [-1.000, 0.000, 0.000, -1.000, 1.000, 0.000, 0.000, 1.000],
            ])
        self.R_trot = np.array([
            [1.000, -0.000, -1.000, 0.000, 1.000, -0.000, -1.000, 0.000],
            [0.000, 1.000, -0.000, -1.000, 0.000, 1.000, -0.000, -1.000],
            [-1.000, -0.000, 1.000, -0.000, -1.000, -0.000, 1.000, -0.000],
            [0.000, -1.000, 0.000, 1.000, 0.000, -1.000, 0.000, 1.000],
            [1.000, -0.000, -1.000, 0.000, 1.000, -0.000, -1.000, 0.000],
            [0.000, 1.000, -0.000, -1.000, 0.000, 1.000, -0.000, -1.000],
            [-1.000, -0.000, 1.000, -0.000, -1.000, -0.000, 1.000, -0.000],
            [0.000, -1.000, 0.000, 1.000, 0.000, -1.000, 0.000, 1.000]
        ])

        self.R_gallop = np.array([
            [1.000, -0.000, -1.000, 0.000, -1.000, 0.000, 1.000, -0.000],
            [0.000, 1.000, -0.000, -1.000, -0.000, -1.000, 0.000, 1.000],
            [-1.000, -0.000, 1.000, -0.000, 1.000, -0.000, -1.000, -0.000],
            [0.000, -1.000, 0.000, 1.000, 0.000, 1.000, 0.000, -1.000],
            [-1.000, -0.000, 1.000, -0.000, 1.000, -0.000, -1.000, -0.000],
            [0.000, -1.000, 0.000, 1.000, 0.000, 1.000, 0.000, -1.000],
            [1.000, -0.000, -1.000, 0.000, -1.000, 0.000, 1.000, -0.000],
            [0.000, 1.000, -0.000, -1.000, -0.000, -1.000, 0.000, 1.000],
        ])

        self.R_pace = np.array([
            [1.000, -0.000, 1.000, -0.000, -1.000, 0.000, -1.000, 0.000],
            [0.000, 1.000, 0.000, 1.000, -0.000, -1.000, -0.000, -1.000],
            [1.000, -0.000, 1.000, -0.000, -1.000, 0.000, -1.000, 0.000],
            [0.000, 1.000, 0.000, 1.000, -0.000, -1.000, -0.000, -1.000],
            [-1.000, -0.000, -1.000, -0.000, 1.000, -0.000, 1.000, -0.000],
            [0.000, -1.000, 0.000, -1.000, 0.000, 1.000, 0.000, 1.000],
            [-1.000, -0.000, -1.000, -0.000, 1.000, -0.000, 1.000, -0.000],
            [0.000, -1.000, 0.000, -1.000, 0.000, 1.000, 0.000, 1.000]
        ])
        # Type of Trajectory Cycloid or bezier
        self.trajectory = trajectory
        # Parameters constant
        self.alpha = 50
        self.gamma = 50
        self.b = 50 
        self.mu = 1
        self.beta_walk = 0.75
        self.beta_trot = 0.5
        self.delta_walk = 1.0
        self.delta_trot = 0.5
        self.delta_tran = 0.5

        # Parametr avaible
        self.T = 1.0
        self.dt = 0.01
        self.cycles = cycles  # simulate 10 gait cycles
        self.steps = int(self.cycles * self.T / self.dt)
        
        # Initialize CPG parameters (omitted for brevity)
        self.G = np.array([0.7, 0.7])  # [FT_gain, IMU_gain]
        self.S = np.zeros((2, 8))      # Reflex matrix: 2 sensors × 8 states (4 legs × [x,y])
        
        # Force-Torque sensor thresholds
        self.Fx_max = 20.0   # Max lateral force (N)
        self.Fy_max = 15.0    # Max longitudinal force (N)
        self.Fz_nominal = 50.0  # Expected vertical force (N)
        self.Fz_range = 30.0    # Tolerance range (N)
        
        # IMU thresholds
        self.max_roll = 0.5   # Radians
        self.max_pitch = 0.5  # Radians
        # self.steps = int(self.cycles / self.dt)
        
        #coupling intensity coefficient
        self.Q_walk = np.array([
            [1.0], [0.0], [-1.0], [0.0],
            [0.0], [1.0], [0.0], [-1.0]
        ])
        self.Q_walk_back = np.array([
            [0.0], [-1.0], [0.0], [1.0],
            [-1.0], [0.0], [1.0], [0.0]
        ])
        
        self.Q_walk_left = np.array([
            [1.0], [0.0], [0.0], [-1.0],
            [-1.0], [0.0], [0.0], [1.0]
        ])
        
        self.Q_walk_right = np.array([
            [0.0], [-1.0], [1.0], [0.0],
            [0.0], [1.0], [-1.0], [0.0]
        ])
        
        # Coupling for trot gait
        self.Q_trot = np.array([
            [1.0], [0.0], [-1.0], [0.0],
            [1.0], [0.0], [-1.0], [0.0]
        ])

        self.Q_gallop = np.array([
            [1.0], [0.0], [-1.0], [0.0],
            [-1.0], [0.0], [1.0], [0.0]
        ])

        self.Q_pace = np.array([
            [1.0], [0.0], [1.0], [0.0],
            [-1.0], [0.0], [-1.0], [0.0]
        ])
        
        self.Q = np.zeros((8, 1))

        self.Q_history = np.zeros((8, self.steps))
        self.next_Q = np.zeros_like(self.Q)

        self.step_length = 0.0
        self.step_height = 0.0

        LEG_LABELS = ["FL", "FR", "RR", "RL"]  # Labels for each leg
        self.foot_positions = {label: {"x": [], "y": [], "z": []} for label in LEG_LABELS}
        self.leg_map = {0: "FL", 2: "FR", 4: "RR", 6: "RL"} 
        
    def r(self, x, y):
        return np.sqrt(x**2 + y**2)

    def a(self, y, b, beta, T):
        epsilon = 1e-8  # Small value to prevent division by zero
        beta = np.clip(beta, epsilon, 1 - epsilon)  # Ensure beta is not 0 or 1
        exp_term1 = np.exp(-b * y)
        exp_term2 = np.exp(b * y)

        # Avoid overflow in exponential
        exp_term1 = np.clip(exp_term1, 1e-8, 1e8)
        exp_term2 = np.clip(exp_term2, 1e-8, 1e8)

        return (np.pi / (beta * T * (exp_term1 + 1))) + (np.pi / ((1 - beta) * T * (exp_term2 + 1)))

    def update_sensors(self, leg_forces, roll, pitch):
        """Simulate FT and IMU inputs for all legs."""
        # FT Sensor (k=1)
        for i, leg in enumerate(["FL", "FR", "RR", "RL"]):
            Fx, Fy, Fz = leg_forces[leg]
            # --- FT Sensor Feedback (k=1) ---
            # Swing: Detect lateral slip (combine Fx and Fy for diagonal resistance)
            resultant_force = np.sqrt(Fx**2 + Fy**2) * np.sign(Fx)
            self.S[0, 2*i] = np.clip(resultant_force / self.Fx_max, -1, 1)  # s_i1^x
            
            # Stance: Adjust stiffness based on vertical load
            self.S[0, 2*i+1] = np.clip(
                (Fz - self.Fz_nominal) / self.Fz_range, -1, 1
            )  # s_i1^y

        # Apply to all legs uniformly
        self.S[1, 0::2] = np.clip(roll / self.max_roll, -1, 1)    # s_i2^x (roll correction)
        self.S[1, 1::2] = np.clip(pitch / self.max_pitch, -1, 1)  # s_i2^y (pitch correction)
    
    def generate(self, sensor_data,  gait_type='walk', Ts=1.0, step_length=0.05, step_height=0.05, z=-0.0, move_direction='forward'):

        self.T = Ts
        self.steps = int(self.cycles * self.T / self.dt)
        foot_traj = FootTrajectoryGenerator(
            Ts=Ts,
            step_length=step_length,
            step_height=step_height,
            duty_factor=self.beta_walk if gait_type == "walk" else self.beta_trot,
            z=z,
            type=self.trajectory
        )
        
        # 1. Configure gait parameters
        if (gait_type == 'walk' ):
            delta = self.delta_walk
            beta = self.beta_walk
            if move_direction == 'forward' or move_direction == 'stop' or move_direction == 'forward_left' or move_direction == 'forward_right' or 'backward':
                R = self.R_walk
                self.Q = self.Q_walk
            # elif move_direction == 'backward':
            #     R = self.R_walk_back
            #     self.Q = self.Q_walk_back
            elif move_direction == 'left' :
                R = self.R_walk_left
                self.Q = self.Q_walk_left
            elif move_direction == 'right':
                R = self.R_walk_right
                self.Q = self.Q_walk_right
        elif (gait_type == 'trot'):
            delta = self.delta_trot
            beta = self.beta_trot
            R = self.R_trot
            self.Q = self.Q_trot
        elif (gait_type == 'bound'):
            delta = 0.4
            beta = 0.5
            R = self.R_gallop
            self.Q = self.Q_gallop
        elif (gait_type == 'pace'):
            delta = 0.4
            beta = 0.5
            R = self.R_pace
            self.Q = self.Q_pace
        else:
            delta = self.delta_walk
            beta = self.beta_walk
            R = self.R_walk
            self.Q = self.Q_walk
        
        
        for t in range(self.steps):
            self.update_sensors(sensor_data['forces'], sensor_data['imu']['roll'], sensor_data['imu']['pitch'])
            for i in range(0, 8, 2):
                x, y = self.Q[i, 0], self.Q[i + 1, 0]
                r_val = self.r(x, y)
                a_val = self.a(y, self.b, beta, self.T)
                
                # Compute F matrix
                F = np.array([[self.alpha * (self.mu - r_val**2), -a_val], 
                            [a_val, self.gamma * (self.mu - r_val**2)]])
                
                # Compute coupling term
                coupling = delta * R[i, :] @ self.Q
                feedback = self.G @ self.S[:, i:i+2].T

                # Update equations
                Q_dot = F @ np.array([x, y]) + coupling + feedback

                # Euler integration
                self.next_Q[i, 0] = x + Q_dot[0] * self.dt
                self.next_Q[i + 1, 0] = y + Q_dot[1] * self.dt
                
                phase = np.arctan2(y, x)  # phase of oscillator
                phase = (phase + np.pi) % (2 * np.pi)  # normalize to [0, 2pi]
                phase_time = phase / (2 * np.pi) * self.T  # map phase to time in gait cycle

                # Generate foot trajectory
                foot_pos = foot_traj.generate(phase_time, move_direction=move_direction)
                leg = self.leg_map[i]
                self.foot_positions[leg]["x"].append(foot_pos[0])
                self.foot_positions[leg]["y"].append(foot_pos[1])
                self.foot_positions[leg]["z"].append(foot_pos[2])
                
            # Store state history
            # Q = self.next_Q
            self.Q = self.next_Q.copy()
            self.Q_history[:, t] = self.Q.flatten()

        return self.foot_positions, self.Q_history
    
    


