#!/usr/bin/env python3
import numpy as np
class CPG_Network():
    def __init__(self, Ts, cycles, dt):
       
        self.R_walk = np.array([
            [1.000, -0.000, 0.000, 1.000, -0.000, -1.000, -1.000, 0.000],
            [0.000, 1.000, -1.000, 0.000, 1.000, -0.000, -0.000, -1.000],
            [0.000, -1.000, 1.000, -0.000, -1.000, 0.000, 0.000, 1.000],
            [1.000, 0.000, 0.000, 1.000, -0.000, -1.000, -1.000, 0.000],
            [-0.000, 1.000, -1.000, -0.000, 1.000, -0.000, 0.000, -1.000],
            [-1.000, -0.000, 0.000, -1.000, 0.000, 1.000, 1.000, 0.000],
            [-1.000, -0.000, 0.000, -1.000, 0.000, 1.000, 1.000, -0.000],
            [0.000, -1.000, 1.000, 0.000, -1.000, 0.000, 0.000, 1.000]])

        self.R_trot = np.array([
            [1, 0, -1, 0, 1, 0, -1, 0],
            [0, 1, 0, -1, 0, 1, 0, -1],
            [-1, 0, 1, 0, -1, 0, 1, 0],
            [0, -1, 0, 1, 0, -1, 0, 1],
            [1, 0, -1, 0, 1, 0, -1, 0],
            [0, 1, 0, -1, 0, 1, 0, -1],
            [-1, 0, 1, 0, -1, 0, 1, 0],
            [0, -1, 0, 1, 0, -1, 0, 1]
        ])

        self.R_walk_back = np.array([
            [1.000, -0.000, -0.000, 1.000, 0.000, -1.000, -1.000, -0.000],
            [0.000, 1.000, -1.000, -0.000, 1.000, 0.000, 0.000, -1.000],
            [-0.000, -1.000, 1.000, -0.000, -1.000, 0.000, 0.000, 1.000],
            [1.000, -0.000, 0.000, 1.000, -0.000, -1.000, -1.000, 0.000],
            [0.000, 1.000, -1.000, -0.000, 1.000, -0.000, 0.000, -1.000],
            [-1.000, 0.000, 0.000, -1.000, 0.000, 1.000, 1.000, 0.000],
            [-1.000, 0.000, 0.000, -1.000, 0.000, 1.000, 1.000, -0.000],
            [-0.000, -1.000, 1.000, 0.000, -1.000, 0.000, 0.000, 1.000]
        ])

        

        # Parameters constant
        self.alpha = 50
        self.gamma = 50
        self.b = 50 
        self.mu = 1
        self.beta_walk = 0.75
        self.beta_trot = 0.5
        self.delta_walk = 1
        self.delta_trot = 0.3
        self.delta_tran = 0.5

        # Parametr avaible
        self.T = Ts
        self.dt = dt
        self.cycles = cycles  # simulate 10 gait cycles
        self.steps = int(self.cycles * self.T  / self.dt)
        
#         self.steps = int(self.cycles / self.dt)

        #coupling intensity coefficient
        
        self.Q_trot = np.array([
            [1.0], [0.0], [-1.0], [0.0],
            [1.0], [0.0], [-1.0], [0.0]
        ])
        self.Q_walk = np.array([
            [1.0], [0.0], [0.0], [1.0],
            [0.0], [-1.0], [-1.0], [0.0]
        ])
        self.Q_walk_back = np.array([
            [0.0], [-1.0], [1.0], [0.0],
            [-1.0], [0.0], [0.0], [1.0]
        ])

        self.Q = np.zeros((8, 1))

        self.Q_history = np.zeros((8, self.steps))
        self.next_Q = np.zeros_like(self.Q)
        self.phases = [[] for _ in range(4)]

        self.step_length = 0.0
        self.step_height = 0.0
        self.current_time = 0.0  # Initialize current time
        self.phase_offsets = {
            "FL": 0.0,
            "RL": 0.0,
            "RR": 0.0,
            "FR": 0.0
        }
        LEG_LABELS = ["FL", "RL", "RR", "FR"]  # Labels for each leg
        self.LEG_LABELS = ["FL", "RL", "RR", "FR"]
        self.foot_positions = {label: {"x": [], "y": [], "z": []} for label in LEG_LABELS}
        self.leg_map = {0: "FL", 2: "RL", 4: "RR", 6: "FR"} 
        self.contact_state = {label: {"contact": []} for label in LEG_LABELS}
        
    def r(self, x, y):
        return np.sqrt(x**2 + y**2)

    def omega(self, y, b, beta, T):
        epsilon = 1e-8  # Small value to prevent division by zero
        beta = np.clip(beta, epsilon, 1 - epsilon)  # Ensure beta is not 0 or 1
        exp_term1 = np.exp(-b * y)
        exp_term2 = np.exp(b * y)

        # Avoid overflow in exponential
        exp_term1 = np.clip(exp_term1, 1e-8, 1e8)
        exp_term2 = np.clip(exp_term2, 1e-8, 1e8)

        return (np.pi / (beta * T * (exp_term1 + 1))) + (np.pi / ((1 - beta) * T * (exp_term2 + 1)))

    def generate(self, gait_type='walk', move_direction='forward'):

        
        # 1. Configure gait parameters
        if (gait_type == 'walk' ):
            delta = self.delta_walk
            beta = self.beta_walk
            if move_direction == 'forward' or move_direction == 'stop':
                R = self.R_walk
                self.Q = self.Q_walk
            elif move_direction == 'backward':
                R = self.R_walk_back
                self.Q = self.Q_walk_back
            else :
                R = self.R_walk
                self.Q = self.Q_walk
                
        elif (gait_type == 'trot'):
            delta = self.delta_trot
            beta = self.beta_trot
            R = self.R_trot
            self.Q = self.Q_trot
        else:
            raise ValueError("Invalid gait_type. Use 'walk' or 'trot'.")
        
        
        for t in range(self.steps):
            for i in range(0, 8, 2):
                j = int(i/2)
                x, y = self.Q[i, -1], self.Q[i + 1, -1]
                r_val = self.r(x, y)
                omega_val = self.omega(y, self.b, beta, self.T)
                
                # Compute F matrix
                F = np.array([
                        [self.alpha * (self.mu - r_val**2), -omega_val], 
                        [omega_val, self.gamma * (self.mu - r_val**2)]
                ])
                
                # Compute coupling term
                coupling = delta * R[i:(j+1)*2, :] @ self.Q
                
                # Update equation
                Q_dot = F @ np.array([[x], [y]]) + coupling

                # Euler integration
                self.next_Q[i, 0] = x + Q_dot[0,0] * self.dt
                self.next_Q[i + 1, 0] = y + Q_dot[1,0] * self.dt
                
                phases = np.arctan2(self.next_Q[i + 1, 0], self.next_Q[i, 0])  # phase of oscillator
                phase_range = (phases + np.pi) % (2 * np.pi) # normalize to [0, 2pi]
                self.phases[j].append((phases  + np.pi) / (2 * np.pi))
                leg = self.leg_map[i]
                if 0 <= phase_range < np.pi:  # swing
                    self.contact_state[leg]["contact"].append(0)
        
                else:  # stance
                    self.contact_state[leg]["contact"].append(1)
            # Store state history
            self.Q = self.next_Q.copy()
            self.Q_history[:, t] = self.Q.flatten()

        return self.Q_history, self.contact_state, np.array(self.phases)
    
    
    def get_contact_schedule(self, horizon, dt):
        """Generate contact schedule for MPC horizon"""
        contact_schedule = np.zeros((4, horizon))
        for i in range(horizon):
            t = self.current_time + i*dt
            for j, leg in enumerate(self.LEG_LABELS):
                phase = (self.phase_offsets[leg] + 2*np.pi*t/self.T) % (2*np.pi)
                contact_schedule[j,i] = 0 if phase < np.pi else 1
        return contact_schedule
    
    def update_time(self, dt):
        self.current_time += dt
