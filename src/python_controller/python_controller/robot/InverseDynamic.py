import numpy as np
from math import *


class InverseDynamic:
    def __init__(self, lb, wb, l1, l2, l3, lt, lc, m1, m2, I1zz, I2zz, dt):
        """
        COG : Center Of Gravity
        GRF : Ground Reaction Force
        :param lb: body length
        :param wb: body width
        :param l1: length of the thigh
        :param l2: length of the shank
        :param l3: length of the foot
        :param lt: distance from hip joint to the COG of the thigh
        :param lc: distance from knee joint to the COG of the shank
        :param m1.m2: mass of the thigh and shank link, respectively
        :param I1zz, I2zz: moment of inertia perpendicular to the z-axial direction of the thigh and shank link, respectively.
        :param dt: time step
        :theta3 : Rolling hip joint
        :theta3 : Pitching hip joint angle
        :theta3 : Pitching knee joint angle
        :Fx, Fy : are components of the GRF in the X-axial and Y-axial direction, respectively
        """
        self.lb = lb
        self.wb = wb
        self.l1 = l1
        self.l2 = l2
        self.l3 = l3
        self.lt = lt
        self.lc = lc
        self.m1 = m1
        self.m2 = m2
        self.I1zz = I1zz
        self.I2zz = I2zz
        self.g = 9.81
        self.dt = dt
        
    def InitialMatrix(self, theta3):
        M1 = np.array([[self.m1*self.lt**2 + self.m2(self.l1**2 + self.lc**2 + 2*self.l1*self.lc*np.cos(theta3[-1])) + self.I1zz + self.I2zz],
                        [self.m2*(self.lc**2 + self.l1*self.lc*np.cos(theta3[-1])) + self.I2zz]])
        M2 = np.array([[self.m2*(self.lc**2 + self.l1*self.lc*np.cos(theta3[-1])) + self.I2zz],
                        [self.m2*self.lc**2 + self.I2zz]])   
        M = np.array([M1, M2])
        return M
    
    def CoriolisMatrix(self,theta2, theta3):
        C00 = -((theta3[-1] - theta3[-2])/self.dt)*self.m2*self.l1*self.lc*np.sin(theta3[-1])
        C01 = -(((theta2[-1] - theta2[-2])/self.dt) + (theta3[-1] - theta3[-2])/self.dt)*self.m2*self.lc*self.l1*np.sin(theta3[-1])
        C10 = (theta2[-1] - theta2[-2])/self.dt*self.m2*self.lc*self.l1*np.sin(theta3[-1])
        C11 = 0
        C = np.array([[C00, C01], [C10, C11]])
        return C

    def GravityMatrix(self, theta2, theta3):
        G = np.array([[self.g*self.m1*self.lt*np.sin(theta2[-1]) + self.g*self.m2*self.l1*np.sin(theta2[-1]) + self.g*self.m2*self.lc*np.sin(theta2[-1]+theta3[-1])], [self.g*self.m2*self.lc*np.sin(theta2[-1]+theta3[-1])]])
        return G
    
    def thetadot(self, theta):
        thetadot = (theta[-1] - theta[-2])/self.dt
        return thetadot
    
    def thetadotdot(self, theta):
        thetadotdot = (theta[-1] - 2*theta[-2] + theta[-3])/self.dt**2
        return thetadotdot
    def SwingDynamic(self, theta2, theta3):
        vel_theta = np.array([[self.thetadot(theta2)],[self.thetadot(theta3)]])
        acc_theta = np.array([[self.thetadotdot(theta2)],[self.thetadotdot(theta3)]])
        tau_swing = self.InitialMatrix(theta3)@acc_theta + self.CoriolisMatrix(theta2, theta3)@vel_theta + self.GravityMatrix(theta2, theta3)

        return tau_swing

    def Jacobian(self,theta2, theta3):
        E = -self.l1*np.cos(theta2[-1]) - self.l2*np.cos(theta2[-1]+theta3[-1]) + self.l1*np.sin(theta2[-1]) + self.l2*np.sin(theta2[-1]+theta3[-1])
        F = -self.l2*np.cos(theta2[-1] + theta3[-1]) + self.l2*np.sin(theta2[-1]+theta3[-1])
        J = np.array([E, F])
        return J
    def StanceDynamic(self, Fx, Fy, theta2, theta3):
        F_Ground = np.array([[Fx], [Fy]])
        tau_stance = self.Jacobian(theta2, theta3)@F_Ground
        return tau_stance
    


