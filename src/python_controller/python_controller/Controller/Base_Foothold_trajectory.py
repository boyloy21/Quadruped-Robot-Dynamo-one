#!/usr/bin/env python3
import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import expm
from .utils import rot_to_rpy

class Base_Foot_Trajectory():
    def __init__(self,L, W, dt=0.02, k=0.03, hf=0.05):
        self.dt = dt
        self.k = k
        self.hf = hf    # Swing foot height
        self.psi = 0.0  # Yaw angle
        self.Pb = np.zeros(3)  # Body position [x, y, z]
        self.Rb_cur = np.eye(3)
        LEG_LABELS = ["FL", "RL", "RR", "FR"]
        self.Phip = np.array([[L/2, W/2, 0.0],
                             [L/2, -W/2, 0.0],
                             [-L/2, -W/2, 0.0],
                             [-L/2, W/2, 0.0]])
        self.Pfoot = np.zeros((4,3))
        self.Pswing = np.zeros(3)
        self.Vswing = np.zeros(3)
        self.Aswing = np.zeros(3)
        
        # Swing trajectory state
        self.swing_start_positions = [None]*4  # World-frame lift-off positions
        self.in_swing_phase = [False]*4  # Track swing/stance state
    def RotationBody(self, rpy):
        """Create rotation matrix from roll-pitch-yaw angles"""
        phi, theta, psi = rpy
        c, s = np.cos, np.sin
        
        return np.array([
            [c(psi)*c(theta), c(psi)*s(theta)*s(phi)-s(psi)*c(phi), c(psi)*s(theta)*c(phi)+s(psi)*s(phi)],
            [s(psi)*c(theta), s(psi)*s(theta)*s(phi)+c(psi)*c(phi), s(psi)*s(theta)*c(phi)-c(psi)*s(phi)],
            [-s(theta), c(theta)*s(phi), c(theta)*c(phi)]
        ])
    
    def Body_motion_ref(self, Vcmd, Omegacmd, robot_height=0.3):
        # Current Rotation Matrix
        self.psi += Omegacmd * self.dt
        rpy = [0.0, 0.0, self.psi]
        self.Rb_cur = self.RotationBody(rpy)

        # Update Position 
        Vb = self.Rb_cur @ Vcmd     # Rotate command to world frame
        if abs(Vb[0]) < 0.01:
            Vb[0] = 0.0
        if abs(Vb[1]) < 0.01:
            Vb[1] = 0.0
        self.Pb += Vb * self.dt
        self.Pb[2] = robot_height   # Maintain constant height
        Omegab = self.Rb_cur @ np.array([0, 0, Omegacmd])  # Only yaw rate considered
        Rb = self.Rb_cur * expm(Omegacmd*self.dt)

        theta = rot_to_rpy(Rb)
        

        return self.Pb, Vb, theta, Omegab
    
    def FootTarget(self,Vcur, Tst, legID = 0):
        """Calculate next foot placement target"""
        # Current foot position in world frame
        current_foot_pos = self.Pb + self.Rb_cur @ self.Phip[legID]
        # Predictive foot placement (capture point)
        target_offset = (Tst/2)*Vcur 
        
        # Ensure target is reachable
        max_reach = 0.15
        target_offset[:2] = np.clip(target_offset[:2], -max_reach, max_reach)
        target_offset[2] = 0 # Maintain groud contact
        current_foot_pos[2] = 0.0 # Maintain groud contact
#         Pfoot = self.Pb + self.Rb_cur @ self.Phip[legID] + (Tst/2)*Vcur + self.k*(Vcur - Vcmd)
       
        return current_foot_pos + target_offset
    
    def Bezier(self, phi):
        return phi**3 + 3*(phi**2)*(1-phi)
    
    def Bezierderi(self, phi):
        return 6*phi*(1-phi)
    
    def Bezierderi2(self, phi):
        return 6*(1 - 2*phi)  # Second derivative
    
    def SwingTrajectory(self, Ps, Pe, phi, Tsw, legId = 0):
        """
        Generate swing trajectory between Ps and Pe
        Args:
            Ps: Swing start position (world frame)
            Pe: Target foot position (world frame)
            phi: Normalized phase [0-1]
            Tsw: Swing duration (seconds)
            legID: Leg index (0-3)
        Returns:
            Psw: Current swing position (world frame)
            Vsw: Current swing velocity (world frame)
            Asw: Current swing acceleration (world frame)
        """
        
        # XY plane movement (Bezier interpolation)
        self.Pswing[:2] = Ps[:2] + self.Bezier(phi) * (Pe[:2] - Ps[:2])
        self.Vswing[:2] = self.Bezierderi(phi) * (Pe[:2] - Ps[:2]) / Tsw
        self.Aswing[:2] = self.Bezierderi2(phi) * (Pe[:2] - Ps[:2]) / (Tsw**2)
        # Z axis movement (with height)
        if phi <= 0.5:
            self.Pswing[2] = Ps[2] + self.Bezier(2*phi) * self.hf
            self.Vswing[2] = self.Bezierderi(2*phi) * self.hf * (2/Tsw)
            self.Aswing[2] = self.Bezierderi2(2*phi) * self.hf * (4/(Tsw**2))
        else:
            self.Pswing[2] = Ps[2] + self.Bezier(2*phi-1) * (Pe[2] - Ps[2]) + self.hf * (1 - self.Bezier(2*phi-1))
            self.Vswing[2] = self.Bezierderi(2*phi-1) * (Pe[2] - Ps[2] - self.hf) * (2/Tsw)
            self.Aswing[2] = self.Bezierderi2(2*phi-1) * (Pe[2] - Ps[2] - self.hf) * (4/(Tsw**2))

        
        return self.Pswing, self.Vswing, self.Aswing


