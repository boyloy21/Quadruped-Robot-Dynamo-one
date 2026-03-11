#!/usr/bin/env python3
import numpy as np
import casadi as ca
from .utilize import a_hat, RotationMatrix, RotZ

class Dynamo_one_SRBM():
    def __init__(self, L, W, H, m, dt, num_legs=4):
        # Parameters of the robot
        self.L = L      #  Length
        self.W = W      #  Width
        self.H = H      #  Height
        self.m = m      #  Mass
        self.Ixx = (1/12) * m * (W**2 + H**2)
        self.Iyy = (1/12) * m * (L**2 + H**2)
        self.Izz = (1/12) * m * (L**2 + W**2)
        self.n = num_legs

       
        self.dt = dt  # time step
        self.state_dim = 12
        ### *** Transition Matrix from Continuous to Discreate **** ###
        #_ Ad = II + dt*Ac , Bd = dt*Bc
        
    def approximateInertia(self, psi, type=None):
        if type == 'casadi':
            Rotz = RotZ(psi, type='casadi')
            Ib = ca.DM([[self.Ixx, 0, 0],
                            [0, self.Iyy, 0],
                            [0, 0, self.Izz]])
            I_hat = ca.mtimes(Rotz, ca.mtimes(Ib, Rotz.T))
        elif type == 'numpy':
            Rotz = RotZ(psi, type='numpy')
            Ib = np.array([[self.Ixx, 0, 0],
                               [0, self.Iyy, 0],
                               [0, 0, self.Izz]])
            I_hat = np.dot(Rotz, np.dot(Ib, Rotz.T))
        return I_hat
    
    def compute_Ad(self, psi, type=None):
        if type == 'casadi':
            Rotz = RotZ(psi, type='casadi')
            II = ca.DM.eye(3)
            O = ca.DM.zeros((3, 3))
            Ad = ca.vertcat(ca.horzcat(II, self.dt * II, O, O),
                            ca.horzcat(O, II, O, O),
                            ca.horzcat(O, O, II, self.dt*Rotz.T),
                            ca.horzcat(O, O, O, II))
            
        elif type == 'numpy':
            Rotz = RotZ(psi, type='numpy')
            II = np.eye(3)
            O = np.zeros((3, 3))
            Ad = np.block([[II, self.dt * II, O],
                       [O, II, O, O],
                       [O, O, II, self.dt*Rotz.T],
                       [O, O, II]])
        return Ad
    
    def compute_Bd(self, yaw, foot_pos, type=None):
        if type == 'casadi':
            I_hat = self.approximateInertia(yaw, type='casadi')
            II = ca.DM.eye(3)
            Bd = ca.DM.zeros(self.state_dim, self.state_dim)
            for i in range(4):
                Bd[3:6 , 3*i:3*(i+1)] = (self.dt/self.m)*II
                Bd[9:12, 3*i:3*(i+1)] = ca.mtimes(ca.inv(I_hat), a_hat(foot_pos[i, :], type='casadi'))
            
        elif type == 'numpy':
            I_hat = self.approximateInertia(yaw, type='numpy')
            II = np.eye(3)
            Bd = np.zeros(self.state_dim,self.state_dim)
            for i in range(4):
                Bd[3:6 , 3*i:3*(i+1)] = (self.dt/self.m)*II
                Bd[9:12, 3*i:3*(i+1)] = np.dot(np.linalg.inv(I_hat), a_hat(foot_pos[i, :], type='numpy'))

        return Bd
    
    def UpdateState(self,psi, foot_pos, state, control, type=None):
        Ad = self.compute_Ad(psi, type=type)
        Bd = self.compute_Bd(psi, foot_pos, type=type)
        # Update the state using the system dynamics
        if type == 'casadi':
            state = ca.mtimes(Ad, state) + ca.mtimes(Bd, control)
        elif type == 'numpy':
            state = np.dot(Ad, state) + np.dot(Bd, control)
        return state
        