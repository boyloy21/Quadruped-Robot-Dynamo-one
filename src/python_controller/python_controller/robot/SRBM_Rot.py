#!/usr/bin/env python3
import numpy as np
import casadi as ca
from .utilize2 import eta_co_xv, eta_co_R, eta_co_w

class Single_Rigid_Body_Model_Rot():
    def __init__(self, L, W, H, m ,dt):
        self.m = m
        self.dt = dt
        self.g = 9.81
        Ixx = (1/12) * m * (W**2 + H**2)
        Iyy = (1/12) * m * (L**2 + H**2)
        Izz = (1/12) * m * (L**2 + W**2)
        self.I = np.diag([Ixx, Iyy, Izz])
    
    def construct_ABD(self, Xt, Ut):
        """
        Builds A, B, D matrices for linearized dynamics.

        Args:
            Xt: Current state (30,)
            Ut: Current control (12,)
        
        Returns:
            A: State transition matrix (12x12)
            B: Control input matrix (12x12)
            D: Constant term (12,)
        """
        # Unpack state
        xop = Xt[0:3].reshape(3,1)       # Position 
        vop = Xt[3:6].reshape(3,1)       # Velocity
        Rop = Xt[6:15].reshape(3,3) # Rotation matrix
        wop = Xt[15:18].reshape(3,1)        # Angular velocity
        pf34 = Xt[18:30].reshape(3,4)       # Foot position 
        ##----- Compute submatrices ---
        # 1. Linear dynamics (position/velocity)
        Cx_x, Cx_v, Cv_v, Cv_u, Cv_c = eta_co_xv(Ut, self.dt, self.m, self.g)

        # 2. Orientation error dynamics
        CE_eta, CE_w, CE_c = eta_co_R(Rop, wop, self.dt)

        # 3. Angular velocity dynamics
        Cw_x, Cw_eta, Cw_w, Cw_u, Cw_c = eta_co_w(xop, Rop, wop, Ut, self.dt , self.I, pf34)

        ##------ Assemble A, B, D --------
        # A matrix (12x12)
        A = np.block([
            [Cx_x, Cx_v, np.zeros((3, 6))],
            [np.zeros((3, 3)), Cv_v, np.zeros((3, 6))], 
            [np.zeros((3,6)), CE_eta, CE_w],
            [np.zeros((3,3)), np.zeros((3, 3)), Cw_eta, Cw_w]
        ])

        # B matrix (12x12)
        B = np.block([
            np.zeros((3, 12)),
            [Cv_u],
            [np.zeros((3, 12))], 
            [Cw_u]
        ])

        # D vector (12,)
        D = np.concatenate([
            np.zeros(3),
            Cv_c,
            CE_c,
            Cw_c
        ])

        return A, B, D
    def Rot_Linearization(self, Xd):
        ###*** R_{k+1} = Rk*exp(\hat{omega}*DeltaT)
        pass