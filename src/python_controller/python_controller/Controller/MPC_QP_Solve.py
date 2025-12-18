#!/usr/bin/env python3
import numpy as np
import casadi as ca
from scipy.linalg import expm
from .utils import skew_symmetric, convert_rpy_to_rot

class MPC_Single_Rigid_Body_Model():
    def __init__(self, mass, I, num_legs=4, horizon=10, dt=0.02, qp_weights=None, alpha=1e-6, mu=0.6):
        # Robot parameters
        self.mass = mass
        self.inertia = I
        self.inv_mass = 1.0 / mass
        self.inv_inertia = np.linalg.inv(self.inertia)
        self.num_legs = num_legs
        self.horizon = horizon
        self.dt = dt
        self.mu = mu
        self.gravity = 9.8
        self.kMaxScale = 2
        self.kMinScale = 0.05
        self.fz_min = self.mass * self.gravity * self.kMinScale
        self.fz_max = self.mass * self.gravity * self.kMaxScale
        self.constraint_dim = 5
        
        # State dimensions
        self.state_dim = 13             # [x, y, z, vx, vy, vz, ϕ, θ, ψ, ϕ̇, θ̇, ψ̇, -g]
        self.action_dim = num_legs * 3  # [Fx1, Fy1, Fz1, Fx2, Fy2, Fz2, ..., Fx4, Fy4, Fz4]
        self.control_dim = 3 * num_legs * horizon
        
        # QP weights (13 * horizon diagonal matrix)
        self.qp_weights = np.diag(np.tile(qp_weights, self.horizon))
        self.qp_weights_single = np.diag(qp_weights)

        # Alpha regularization (num_leg*3 *horizon x num_leg*3*horizon)
        self.alpha = alpha * np.identity(self.control_dim)
        self.alpha_single = alpha * np.identity(num_legs * 3)
        

        # Dynamics matrices
        self.A_mat = np.zeros((self.state_dim, self.state_dim))
        self.B_mat = np.zeros((self.state_dim, 12))

        # Discretized matrices
        self.A_exp = np.zeros((self.state_dim, self.state_dim))
        self.B_exp = np.zeros((self.state_dim, 12))

        # QP matrices
        self.A_qp = np.zeros((self.state_dim * horizon, self.state_dim))
        self.B_qp = np.zeros((self.state_dim * horizon, 12 * horizon))
        self.anb_aux = np.zeros((self.state_dim * horizon, 12))
        self.H = np.zeros((self.control_dim, self.control_dim))     # Hessian
        self.g_vec = np.zeros(self.control_dim)     #  Gradient vector
        

        # Initialize state
        self.state = np.zeros(self.state_dim)
        self.desired_state = np.zeros(self.state_dim*horizon)
        self.contact_state = np.zeros((horizon, num_legs))
        self.foot_position_base = np.zeros((num_legs, 3))
        self.foot_position_world = np.zeros((num_legs, 3))
        self.foot_friction_coeff = np.zeros(num_legs)

        # Constraint matrices
        self.constraint = np.zeros((self.constraint_dim*num_legs * horizon, num_legs*3*horizon))
        self.lb = np.zeros((self.constraint_dim*num_legs * horizon))
        self.ub = np.zeros((self.constraint_dim*num_legs * horizon))

        #Verify inputs
        assert len(qp_weights) == self.state_dim
        assert len(I) == 3

    def calculate_a_mat(self, rpy):
        """Calculate A matrix for linearized dynamics."""
        cosyaw = np.cos(rpy[2])
        sinyaw = np.sin(rpy[2])
        cospitch = np.cos(rpy[1])
        tanpitch = np.tan(rpy[1])

        # Transformation from angular velocity to rpy rate
        angular_velocity_to_rpy_rate = np.array([
            [cosyaw/cospitch, sinyaw/cospitch, 0],
            [-sinyaw, cosyaw, 0],
            [cosyaw * tanpitch, sinyaw * tanpitch, 1]
        ])

        self.A_mat[0:3, 3:6] = np.eye(3)
        self.A_mat[5, 12] = 1
        self.A_mat[6:9, 9:12] = angular_velocity_to_rpy_rate

    
    def calculate_b_mat(self, inv_inertia_world, foot_position):
        """Calculate B matrix for linearized dynamics."""
        for i in range(self.num_legs):
            r = foot_position[i]
            skew_sym = skew_symmetric(r)
            self.B_mat[3:6, 3*i:3*(i+1)] = self.inv_mass * np.eye(3)
            self.B_mat[6:9, 3*i:3*(i+1)] = inv_inertia_world @ skew_sym
        

    def calculate_exponentail(self):
        """Calculate exponential matrix for discretized dynamics."""
        ab_mat = np.zeros((self.state_dim + self.action_dim, self.state_dim + self.action_dim))
        ab_mat[:self.state_dim, :self.state_dim] = self.A_mat * self.dt
        ab_mat[:self.state_dim, self.state_dim:] = self.B_mat * self.dt
        ab_exp = expm(ab_mat)
        self.A_exp = ab_exp[:self.state_dim, :self.state_dim]
        self.B_exp = ab_exp[:self.state_dim, self.state_dim:]
    
    def calculate_qp_matrix(self):
        """Calculate QP matrices for the MPC problem"""
        
        # Calculte A_qp matrix [A_exp, A_exp^2, ..., A_exp^horizon]
        self.A_qp[:self.state_dim, :] = self.A_exp
        for i in range(1, self.horizon):
            self.A_qp[i*self.state_dim:(i+1)*self.state_dim, :] = self.A_exp @ self.A_qp[(i-1)*self.state_dim:i*self.state_dim, :]

        # Calculate auxiliary matrix [B_exp, A_exp*B_exp, ..., A_exp^(h-1)*B_exp]
        self.anb_aux[:self.state_dim, :] = self.B_exp
        for i in range(1, self.horizon):
            self.anb_aux[i*self.state_dim:(i+1)*self.state_dim, :] = self.A_exp @ self.anb_aux[(i-1)*self.state_dim:i*self.state_dim, :]
        
        # Calculate B_qp matrix
        for i in range(self.horizon):
            # Diagonal block
            self.B_qp[i*self.state_dim:(i+1)*self.state_dim, i*12:(i+1)*12] = self.B_exp
            # Off-diagonal blocks
            for j in range(i):
                power = i - j
                self.B_qp[i*self.state_dim:(i+1)*self.state_dim, j*12:(j+1)*12] = self.anb_aux[power*self.state_dim:(power+1)*self.state_dim, :]

        # Calculate H matrix (Hessian) H = 2(B_qp.T*QP_weights*Bqp + K)
        # K = alpha*np.eye(self.control_dim)

        # First compute submatrices at last column H
        for i in range(self.horizon-1, -1, -1):
            anb_block = self.anb_aux[(self.horizon-i-1)*self.state_dim:(self.horizon-i)*self.state_dim, :]
#             block = (self.anb_aux[(self.horizon-i-1)*self.state_dim:(self.horizon-i)*self.state_dim, :].T  
#                      @ self.qp_weights_single @ self.B_exp)
            block = anb_block.T @ self.qp_weights_single @ self.B_exp
    
            self.H[i*self.action_dim:(i+1)*self.action_dim, (self.horizon-1)*self.action_dim:self.horizon*self.action_dim] = block
            if i != self.horizon - 1:
                self.H[(self.horizon-1)*self.action_dim:self.horizon*self.action_dim, 
                           i*self.action_dim:(i+1)*self.action_dim] = block.T  # Changed state_dim to action_dim
        # Fill in the rest of the matrix
        for i in range(self.horizon-2, -1, -1):

            # diagonal block
            self.H[i*self.action_dim:(i+1)*self.action_dim, i*self.action_dim:(i+1)*self.action_dim] = (
                self.H[i*self.action_dim:(i+1)*self.action_dim, (i+1)*self.action_dim:(i+2)*self.action_dim] +
                self.anb_aux[(self.horizon-i-1)*self.state_dim:(self.horizon-i)*self.state_dim, :].T @ self.qp_weights_single @ 
                self.anb_aux[(self.horizon-i-1)*self.state_dim:(self.horizon-i)*self.state_dim, :]
            )
            # off-diagonal blocks
            for j in range(i+1, self.horizon-1):
                #Diagonall Bolocks
                block = (
                    self.H[(i+1)*self.action_dim:(i+2)*self.action_dim, (j+1)*self.action_dim:(j+2)*self.action_dim] +
                    self.anb_aux[(self.horizon-i-1)*self.state_dim:(self.horizon-i)*self.state_dim, :].T @ self.qp_weights_single @ 
                    self.anb_aux[(self.horizon-j-1)*self.state_dim:(self.horizon-j)*self.state_dim, :]
                           
                )

                self.H[i*self.action_dim:(i+1)*self.action_dim,
                    j*self.action_dim:(j+1)*self.action_dim] = block
                
                self.H[j*self.action_dim:(j+1)*self.action_dim,
                    i*self.action_dim:(i+1)*self.action_dim] = block.T
                
        # Multiply by 2 and add alpha
        self.H *= 2
        for i in range(self.horizon):
            self.H[i*self.action_dim:(i+1)*self.action_dim, i*self.action_dim:(i+1)*self.action_dim] += self.alpha_single

    def update_constraints_matrix(self, friction_coeff, horizon, num_legs):
        """Update the constraints matrix for the QP problem."""
        for i in range(horizon * num_legs):
            self.constraint[i*self.constraint_dim:(i+1)*self.constraint_dim,
                           i*3:(i+1)*3] = np.array([
                [-1, 0, friction_coeff[0]],
                [1, 0, friction_coeff[1]],
                [0, -1, friction_coeff[2]],
                [0, 1, friction_coeff[3]],
                [0, 0, 1]
            ])
    
    def calculate_constrain_bounds(self, contact_state):
        """Calculate lower and upper bounds for the constraints."""
        for i in range(self.horizon):
            for j in range(self.num_legs):
                row = (i * self.num_legs + j) * self.constraint_dim

                # Lower bounds
                self.lb[row:row+4] = 0
                self.lb[row+4] = self.fz_min * contact_state[i, j]

                # Upper bounds
                friction_ub = self.mu * self.fz_max * contact_state[i, j]
                self.ub[row:row+4] = friction_ub
                self.ub[row+4] = self.fz_max * contact_state[i, j]

    def compute_contact_force(self, com_state, desired_state, foot_positions, contact_state):
        
        # State Matrix
        self.state[0:3] = com_state['position']
        self.state[3:6] = com_state['velocity'] 
        self.state[6:9] = com_state['orientation']
        self.state[9:12] = com_state['angular_velocity']
        self.state[12] = -self.gravity
        
        
        # Desired State
        for i in range(self.horizon):
            idx = i * self.state_dim
            t = (i+1)*self.dt
            
            # Linear extrapolation for desired position 
            self.desired_state[idx:idx+3] = desired_state['position'] 
            
            # Constant desired velocity
            self.desired_state[idx+3:idx+6] = desired_state['velocity']
            self.desired_state[idx+5:idx+6] = 0
            # Orientation tracking (yaw only)
            self.desired_state[idx+6:idx+9] = desired_state['orientation'] 
            self.desired_state[idx+6:idx+8] = 0  # Zero roll/pitch
            
            # Angular velocity tracking
            self.desired_state[idx+9:idx+12] = desired_state['angular_velocity']

            self.desired_state[idx + 12] = -self.gravity
            
        
        self.contact_state= contact_state
        # Calculate A matrix
        self.calculate_a_mat(com_state['orientation'])

        # Calculate B matrix
        rot = convert_rpy_to_rot(com_state['orientation'])
        inv_inertia_world = rot @ self.inv_inertia @ rot.T
        self.calculate_b_mat(inv_inertia_world, foot_positions)

        # Calculate AB exponentail
        self.calculate_exponentail()

        # Calculate QP matrices
        self.calculate_qp_matrix()
        
        #  Calculate state difference and g vector
        state_diff = self.A_qp @ self.state - self.desired_state
        self.g_vec = 2 * self.B_qp.T @ (self.qp_weights @ state_diff)

        # Calculate constraint bounds
        friction_coeff = np.array([self.mu, self.mu, self.mu, self.mu])
        self.update_constraints_matrix(friction_coeff, self.horizon, self.num_legs)
        # self.contact_state = np.array(contact_state).reshape(self.horizon, self.num_legs)
        self.calculate_constrain_bounds(self.contact_state)
   
    def solve_qp(self):
        """Solve the quadratic program using CasADi's qpOASES interface.

        Returns:
            np.ndarray: Optimal control forces (12*horizon vector)
        """
        # Convert numpy arrays to CasADi types
        H_ca = ca.DM(self.H)  # Hessian matrix
        g_ca = ca.DM(self.g_vec)  # Gradient vector
        A_ca = ca.DM(self.constraint)  # Constraint matrix
        lb_ca = ca.DM(self.lb)  # Lower bounds
        ub_ca = ca.DM(self.ub)  # Upper bounds

        # Create QP variables and problem formulation
        x = ca.MX.sym('x', self.control_dim)  # Decision variables (control forces)

        # QP problem formulation:
        # minimize 0.5*x'*H*x + g'*x
        # subject to lb <= A*x <= ub
        qp = {
            'x': x,  # Decision variables
            'f': 0.5 * ca.mtimes(ca.mtimes(x.T, H_ca), x) + ca.mtimes(g_ca.T, x),  # Cost function
            'g': ca.mtimes(A_ca, x)  # Constraints
        }

        # Solver options
        opts = {
            'printLevel': 'none',  # No output printing
            'error_on_fail': False,  # Don't raise exception on failure
            'print_time': False,  # Don't print timing info
        }

        try:
            # Create solver instance
            solver = ca.qpsol('solver', 'qpoases', qp, opts)

            # Solve QP problem
            res = solver(lbg=lb_ca, ubg=ub_ca)

            # Extract solution if successful
            if res['x'] is not None:
                solution = np.array(res['x']).flatten()

                # Verify solution satisfies constraints
                if self._check_solution(solution):
                    return solution

        except Exception as e:
            print(f"QP solve failed: {str(e)}")

        # Return zero forces if solver fails
        return np.zeros(self.control_dim)

    def _check_solution(self, solution):
        """Internal method to verify solution validity"""
        forces = solution.reshape(-1, 3)  # Reshape to (horizon*num_legs, 3)

        for i in range(len(forces)):
            leg_idx = i % self.num_legs
            time_step = i // self.num_legs

            fx, fy, fz = forces[i]

            # Check if foot is in contact
            if self.contact_state[time_step, leg_idx] > 0.5:  # Foot in contact
                # Check friction pyramid constraints
                if not (abs(fx) <= self.mu * fz + 1e-4 and 
                        abs(fy) <= self.mu * fz + 1e-4 and
                        fz >= self.fz_min - 1e-4):
                    print(f"Warning: Invalid solution at leg {leg_idx}, step {time_step}")
                    return False
            else:  # Foot not in contact
                if np.linalg.norm(forces[i]) > 1e-4:
                    print(f"Warning: Non-zero force for non-contact leg {leg_idx}")
                    return False

        return True
       
#     def solve_qp(self):
#         """Solve the QP problem."""
        
#         H = ca.DM(self.H)
#         g = ca.DM(self.g_vec)

#         A = ca.DM(self.constraint)

#         lb = ca.DM(self.lb)
#         ub = ca.DM(self.ub)
#         # Create QP Problem
#         # n_vars = self.H.shape[0]
#         # n_constraints = self.constraint.shape[0]

#         qp = {
#             'h' : H.sparsity(),
#             'a' : A.sparsity(),
#         }

#         opts = {
#             'printLevel': 'none',
#             'error_on_fail': False  # Prevent crashes on infeasibility
            
#         }

#         try:
#             solver = ca.conic('solver', 'qpoases', qp, opts)
#             res = solver(h=H, g=g, a=A, lba=lb, uba=ub)
            
#             if res['x'] is not None:
#                 return np.array(res['x'])  # First step
#         except Exception as e:
#             print(f"QP failed: {str(e)}")
            
#         return np.zeros(self.control_dim)
       