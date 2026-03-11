#!/usr/bin/env python3
import numpy as np

def hat_map(v):
        """Converta a 3D vector to skew-symetric matrix"""
        return np.array([
            [0, -v[2], v[1]],
            [v[2], 0, -v[0]],
            [-v[1], v[0], 0]
        ])
def vee_map(S):
    """Converts skew-symmetric matrix to 3D vector"""
    return np.array([[S[2,1], S[0,2], S[1,0]]])

def get_N():
    """Basis matrix for so(3) Lie algebra"""
    return np.array([
        [0, 0, 0],
        [0, 0, 1],
        [0, -1, 0], 
        [0, 0, -1],
        [0, 0, 0],
        [1, 0, 0],
        [0, 1, 0],
        [-1, 0, 0],
        [0, 0, 0]
    ]).reshape(9,3)

def get_D(w):
    """Helper matrix for orientation dynamics"""
    d, e, f = w[0], w[1], w[2]
    return np.array([
        [0, 0, 0],
        [e, -d, 0],
        [f, 0, -d],
        [-e, d, 0],
        [0, 0, 0],
        [0, f, -e],
        [-f, 0, d],
        [0, -f, e],
        [0, 0, 0]
    ])

def get_F(k):
    """Helper function for eta_co_w"""
    return np.vstack([
        np.hstack([k.T, np.zeros(3), np.zeros(3)]),
        np.hstack([np.zeros(3), k.T, np.zeros(3)]),
        np.hstack([np.zeros(3), np.zeros(3), k.T])
    ])

def eta_co_xv(fop, dt, mass, g):
    """
    Computes linear dynamics matrices for position and velocity.

    Args:
        fop: Current contact forces (12,) [f1x, f1y, f1z, f2x, ...., f4z] (Ut) input control
        dt: Time step (scalar)
        mass: Robot mass (scalar)
        g: Gravity (scalar, positive = upward)

    Returns : 
        Cx_x: ∂(position_next)/∂(current_position) (3x3)
        Cx_v: ∂(position_next)/∂(current_velocity) (3x3)
        Cv_v: ∂(velocity_next)/∂(current_velocity) (3x3)
        Cv_u: ∂(velocity_next)/∂(control_forces) (3x12)
        Cv_c: Constant term (gravity + current forces) (3,)
    """
    Cx_x = np.eye(3)        # Position -> Position
    Cx_v = np.eye(3)*dt     # Velocity -> Position
    Cv_v = np.eye(3)        # Velocity -> Velocity

    # Control Jacobian: Sum of all foot forces affects acceleration 
    Cv_u = (dt / mass) * np.hstack([np.eye(3)] * 4) # [I3, I3, I3, I3]

    # Constant term: Current forces + gravity
    Cv_c = Cv_u @ fop + np.array([0, 0, -g]) * dt

    return Cx_x, Cx_v, Cv_v, Cv_u, Cv_c
def eta_co_R(Rop, wop, dt):
    """
    Computes orientation error dynamics matrices
    Args:
        Rop: Current rotation matrix (3x3)
        wop: Current angular velocity (3,)
        dt: Time step
    Returns:
        CE_eta: ∂η_{k+1}/∂η_k (3x3)
        CE_w: ∂η_{k+1}/∂w_k (3x3)
        CE_c: Constant term (3,)
    """
    N = get_N()
    invN = np.linalg.pinv(N)

    # Term C_eta in the paper
    C_eta = np.kron(np.eye(3), Rop @ hat_map(wop)) @ N + np.kron(np.eye(3), Rop) @ get_D(wop)

    # Term C_w in the paper
    C_w = np.kron(np.eye(3), Rop) @ N

    # Constant term C_c
    C_c = (Rop @ hat_map(wop)).flatten() - np.kron(np.eye(3) , Rop) @ N @ wop

    # Final matrices
    CE_eta = np.eye(3) + invN @ dt * np.kron(np.eye(3), Rop.T) @ C_eta
    CE_w = invN @ dt * np.kron(np.eye(3), Rop.T) @ C_w
    CE_c = invN @ dt * np.kron(np.eye(3), Rop.T) @ C_c

    return CE_eta, CE_w, CE_c

def eta_co_w(xop, Rop, wop, fop, dt, J, pf):
    """
    Computes angular velocity dynamics matrices
    Args:
        xop: Current position (3,)
        Rop: Current rotation matrix (3x3)
        wop: Current angular velocity (3,)
        fop: Current contact forces (12,)
        dt: Time step
        J: Body inertia matrix (3x3)
        pf: Foot positions (3x4)
    Returns:
        Cw_x: ∂w_{k+1}/∂x_k (3x3)
        Cw_eta: ∂w_{k+1}/∂η_k (3x3)
        Cw_w: ∂w_{k+1}/∂w_k (3x3)
        Cw_u: ∂w_{k+1}/∂u_k (3x12)
        Cw_c: Constant term (3,)
    """
    N = get_N()
    r = [pf[:, i] - xop for i in range(4)] # Moment arms

    # Net moment from contact forces
    Mop = sum([hat_map(r[i]) @ fop[3*i:3*(i+1)] for i in range(4)])

    # Gyroscopic terms
    temp_J_w = hat_map(J @ wop) - hat_map(wop) @ J

    # Sum of all contact forces
    sum_fop = np.sum(fop.reshape(-1,3), axis=0) # mean (fx1+fx2+..), (fy1+fy2+..), (fz1+..+fz4)

    # Partial derivatives
    Cx = Rop.T @ hat_map(sum_fop)
    Ceta = get_F(Rop.T @ Mop) @ N - temp_J_w @ hat_map(wop)
    Cw = temp_J_w
    Cu = np.hstack([Rop.T @ hat_map(r[i] for i in range(4))])
    Cc = -hat_map(wop) @ J @ wop + Rop.T @ Mop - temp_J_w @ wop - Cx @ xop

    # Scale by dt and inertia inverse
    invJ = np.linalg.inv(J)
    Cw_x = dt * (invJ @ Cx)
    Cw_eta = dt * (invJ @ Ceta)
    Cw_w = dt * (invJ @ Cw) + np.eye(3)
    Cw_u = dt * (invJ @ Cu)
    Cw_c = dt * (invJ @ Cc)

    return Cw_x, Cw_eta, Cw_w, Cw_u, Cw_c