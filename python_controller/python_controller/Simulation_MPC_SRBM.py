#!/usr/bin/env python3
import rclpy
from rclpy.node import Node 
from sensor_msgs.msg import Joy, JointState
from geometry_msgs.msg import Twist
from std_msgs.msg import String, Float32MultiArray
import numpy as np
import casadi as ca
from scipy.linalg import expm
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

from python_controller.Controller.Base_Foothold_trajectory import Base_Foot_Trajectory
from python_controller.Controller.CPG_Network import CPG_Network
from python_controller.Controller.Kinematic_Model import KinematicQuadruped
from python_controller.Controller.MPC_QP_Solve import MPC_Single_Rigid_Body_Model
from python_controller.Controller.InverseDynamic import InverseDynamic
from python_controller.Controller.utils import convert_rpy_to_rot, skew_symmetric, rot_to_rpy, exp_rotation

def quaternion_to_rotation_matrix(q):
    """Convert quaternion to rotation matrix"""
    w, x, y, z = q
    return np.array([
        [1-2*y*y-2*z*z, 2*x*y-2*z*w, 2*x*z+2*y*w],
        [2*x*y+2*z*w, 1-2*x*x-2*z*z, 2*y*z-2*x*w],
        [2*x*z-2*y*w, 2*y*z+2*x*w, 1-2*x*x-2*y*y]
    ])

def quaternion_multiply(q1, q2):
    """Multiply two quaternions"""
    w1, x1, y1, z1 = q1
    w2, x2, y2, z2 = q2
    return np.array([
        w1*w2 - x1*x2 - y1*y2 - z1*z2,
        w1*x2 + x1*w2 + y1*z2 - z1*y2,
        w1*y2 - x1*z2 + y1*w2 + z1*x2,
        w1*z2 + x1*y2 - y1*x2 + z1*w2
    ])

def quaternion_to_euler(q):
    """Convert quaternion to Euler angles (roll, pitch, yaw)"""
    w, x, y, z = q
    
    # Roll (x-axis rotation)
    sinr_cosp = 2 * (w*x + y*z)
    cosr_cosp = 1 - 2*(x*x + y*y)
    roll = np.arctan2(sinr_cosp, cosr_cosp)
    
    # Pitch (y-axis rotation)
    sinp = 2 * (w*y - z*x)
    if abs(sinp) >= 1:
        pitch = np.copysign(np.pi/2, sinp)
    else:
        pitch = np.arcsin(sinp)
    
    # Yaw (z-axis rotation)
    siny_cosp = 2 * (w*z + x*y)
    cosy_cosp = 1 - 2*(y*y + z*z)
    yaw = np.arctan2(siny_cosp, cosy_cosp)
    
    return np.array([roll, pitch, yaw])

def normalize(v):
    """Normalize a vector"""
    norm = np.linalg.norm(v)
    if norm == 0:
        return v
    return v / norm

class Simulation_MPC_SRBM(Node):
    def __init__(self):
        super().__init__('Simulation_MPC_SRBM')

        # INITIALIZE VARIABLES
        self.dt = 0.01
        self.horizon = 10
        self.robot_height = 0.3
        hf = 0.05

        # CPG Parameters
        cycles = 5
        Tcpg = 0.75
        self.beta = 0.75
        self.step = int(cycles*Tcpg/self.dt)
        self.command_gait = "walk"  # Default gait
        self.move_direction = "stop"
        self.LEG_LABELS = ["FL", "RL", "RR", "FR"]
        self.current_phase = np.zeros(4)  # Initialize current phase for each leg
        self.contact_schedule = np.zeros((4, self.step))  # Contact schedule for each leg
        self.contact_state = {label: {"contact": []} for label in self.LEG_LABELS}
        self.leg_phase = np.zeros((4, self.step))  # Phase for each leg
        # INITIALIZE PARAMETERS
        self.L = 0.433
        self.W = 0.12
        L1, L2, L3 = 0.087, 0.25, 0.25
        m = 40
        g = 9.81
        self.I_body = np.diag([0.41, 2.1, 2.1])

        # INITIALIZE GAIN
        Kp = np.diag([1.0, 1.0, 1.0])
        Kd = np.diag([0.01, 0.01, 0.01])
        qp_weights = np.array([
            0.2,  # x
            0.2,  # y 
            500, # z (higher priority)
            1.0,   # vx
            1.0,   # vy
            0.02,  # vz
            1,   # roll
            1,   # pitch 
            0.0,   # yaw
            0.1,    # roll rate
            0.1,    # pitch rate
            5,   # yaw rate
            0     # gravity term
        ])

        # INITIALIZE OBJECTS
        self.model = KinematicQuadruped(self.L, self.W, L1, L2, L3)
        self.inv_dyn = InverseDynamic(self.L, self.W, L1, L2, L3, 0.1, 0.1, 0.01, 0.01, self.dt)
        self.BaseFoot = Base_Foot_Trajectory(self.L, self.W, dt=self.dt, k=0.03, hf=hf)
        self.CPG = CPG_Network(Ts=Tcpg, cycles=cycles, dt=0.01)
        self.mpc = MPC_Single_Rigid_Body_Model(mass=m, I=self.I_body, num_legs=4 , horizon=self.horizon, dt=0.02, qp_weights=qp_weights, alpha=1e-6, mu=0.6)
        self.Tst = self.CPG.T * self.beta  # Stance time
        self.Tsw = self.CPG.T * (1 - self.beta)  # Swing time
        # INITIALIZE VARIABLES
        self.com_pos = np.array([0.0, 0.0, self.robot_height])
        self.com_vel = np.array([0.0, 0.0, 0.0])
        self.com_ori = np.array([0.0, 0.0, 0.0])
        self.com_omega = np.array([0.0, 0.0, 0.0])

        self.des_pos = np.array([0.0, 0.0, self.robot_height])
        self.des_vel = np.array([0.0, 0.0, 0.0])
        self.des_ori = np.array([0.0, 0.0, 0.0])
        self.des_omega = np.array([0.0, 0.0, 0.0])

        self.force_reshaped = np.zeros((self.horizon, 4, 3))  # Forces for each leg over the horizon

        # INITIALIZE HISTORY STORAGE
        self.history_state = {
            'actual': {'position': [], 'velocity': [], 'orientation': [], 'angular_velocity': []},
            'desired': {'position': [], 'velocity': [], 'orientation': [], 'angular_velocity': []},
            'time': []
        }
        self.force_history = {
            'x': [[] for _ in range(4)],  # Fx for each leg
            'y': [[] for _ in range(4)],  # Fy for each leg
            'z': [[] for _ in range(4)]   # Fz for each leg
        }
        self.Phip = np.array([[self.L/2, self.W, 0.0],
                [-self.L/2, self.W/2, 0.0],
                [-self.L/2, -self.W/2, 0.0],
                [self.L/2, -self.W/2, 0.0]])
        self.footall_trajectories = [[] for _ in range(4)]  # One list for each leg
        self.footall_velocities = [[] for _ in range(4)]
        self.body_pos = np.zeros((3, self.step))
        self.contact_schedule = np.zeros((4, self.step))
        self.contact_schedule_horizon = np.zeros((4, self.horizon))  # 4 legs, horizon steps
        self.world_foot_pos = self.Phip.copy()  # Foot positions in world frame
        self.base_foot_pos = np.zeros((4, 3))
        self.base_foot_vel = np.zeros((4, 3))
        self.base_foot_acc = np.zeros((4, 3))
        self.current_forces = np.zeros((4, 3))  # Current forces applied to each leg
        

        # VELOCITY COMMAND
        self.Vcmd = np.array([0.0, 0.0, 0.0])  # Velocity command in world frame
        self.Omegacmd = np.array([0.0, 0.0, 0.0])  # Angular velocity command in world frame
        
        # SUBSCRIBERS
        self.sub_mode = self.create_subscription(String, '/dynamo_one/mode', self.mode_callback, 10)
        self.sub_desired = self.create_subscription(Float32MultiArray, 'dynamo_one/base_des', self.desired_callback, 10)
        
        # PUBLISHERS
        self.com_pub = self.create_publisher(Float32MultiArray, 'dynamo_one/com_pos', 1)
        # self.pub_foot = self.create_publisher(PoseArray, '/foot_pos', 1)
        # self.pub_contact_schedule = self.create_publisher(Float32MultiArray, '/contact_schedule', 1)

        # TIMER
        self.foot_timer = self.create_timer(0.01, self.foot_callback)
        self.mpc_timer = self.create_timer(0.02, self.mpc_callback)
        self.step_count = 0
        self.step_mpc = 0
        Q_history, self.contact_state, self.leg_phase = self.CPG.generate(gait_type=self.command_gait, move_direction=self.move_direction)

        self.current_orientation_quaternion = np.array([1.0, 0.0, 0.0, 0.0])  # Initialize quaternion
    
    def update_dynamics(self):
        # 1. Compute total force and torque in world frame
        total_force = np.sum(self.current_forces, axis=0)
        total_torque = np.zeros(3)
        
        # Current orientation as rotation matrix
        R = quaternion_to_rotation_matrix(self.current_orientation_quaternion)
        
        for i in range(4):
            r_i = self.world_foot_pos[i] - self.com_pos
            total_torque += np.cross(r_i, self.current_forces[i])
        
        # 2. Update linear dynamics
        gravity_force = np.array([0.0, 0.0, -self.mpc.mass * 9.81])
        net_force = total_force + gravity_force
        self.com_vel += (net_force / self.mpc.mass) * self.dt
        
        # 3. Update angular dynamics (in body frame)
        total_torque_body = R.T @ total_torque
        gyroscopic = np.cross(self.com_omega, self.I_body @ self.com_omega)
        angular_accel = np.linalg.inv(self.I_body) @ (total_torque_body - gyroscopic)
        self.com_omega += angular_accel * self.dt
        
        # 4. Update position
        self.com_pos += self.com_vel * self.dt
        
        # 5. Update orientation using quaternion integration
        omega_mag = np.linalg.norm(self.com_omega)
        if omega_mag > 1e-6:
            delta_angle = omega_mag * self.dt
            axis = self.com_omega / omega_mag
            delta_q = np.array([
                np.cos(delta_angle/2),
                *(axis * np.sin(delta_angle/2))
            ])
            self.current_orientation_quaternion = quaternion_multiply(
                delta_q, self.current_orientation_quaternion)
            self.current_orientation_quaternion /= np.linalg.norm(self.current_orientation_quaternion)
        
        # Convert to Euler angles for visualization/control
        self.com_ori = quaternion_to_euler(self.current_orientation_quaternion)
        
        # 6. Ground contact handling
        if self.com_pos[2] < self.robot_height:
            # Simple ground reaction model
            penetration = self.robot_height - self.com_pos[2]
            k_ground = 10000  # N/m
            d_ground = 100    # Ns/m
            
            # Spring-damper force
            ground_force_z = k_ground * penetration - d_ground * self.com_vel[2]
            self.com_vel[2] += (ground_force_z / self.mpc.mass) * self.dt
            
            # Friction
            friction_coeff = 0.5
            normal_force = max(0, ground_force_z)
            max_friction = friction_coeff * normal_force
            
            if np.linalg.norm(self.com_vel[:2]) > 0.01:
                friction_force = -normalize(self.com_vel[:2]) * min(
                    max_friction, 
                    0.1 * self.mpc.mass * np.linalg.norm(self.com_vel[:2])/self.dt
                )
                self.com_vel[:2] += (friction_force / self.mpc.mass) * self.dt
            
            # Position correction
            self.com_pos[2] = max(self.com_pos[2], self.robot_height)

    def mode_callback(self, msg):
        self.command_gait = msg.data
        if self.command_gait == "walk":
            self.beta = self.CPG.beta_walk
        elif self.command_gait == "trot":
            self.beta = self.CPG.beta_trot
        
        Q_history, self.contact_state, self.leg_phase = self.CPG.generate(gait_type=self.command_gait, move_direction=self.move_direction)
    def desired_callback(self, msg):
        self.des_pos = np.array([msg.data[0], msg.data[1], msg.data[2]])
        self.des_vel = np.array([msg.data[3], msg.data[4], msg.data[5]])
        self.des_ori = np.array([msg.data[6], msg.data[7], msg.data[8]])
        self.des_omega = np.array([msg.data[9], msg.data[10], msg.data[11]])
        if self.des_vel[0] >= 0.2 and self.des_vel[1] == 0.0:
            self.move_direction = "forward"
        elif self.des_vel[0] <= -0.2 and self.des_vel[1] == 0.0:
            self.move_direction = "backward"
        elif self.des_vel[1] >= 0.2 and self.des_vel[0] == 0.0:
            self.move_direction = "right"
        elif self.des_vel[1] <= -0.2 and self.des_vel[0] == 0.0:
            self.move_direction = "left"
        else:
            self.move_direction = "stop"
        self.get_logger().info(f"Desired Position: {self.des_pos}, Velocity: {self.des_vel}, Orientation: {self.des_ori}, Angular Velocity: {self.des_omega}, Move Direction: {self.move_direction}")
        
    def foot_callback(self):
        # Gait Scheduling
        Q_history, self.contact_state, self.leg_phase = self.CPG.generate(gait_type=self.command_gait, move_direction=self.move_direction)
        
        Rb = convert_rpy_to_rot(self.des_ori)
        # Determine foot states
        for i, leg in enumerate(self.LEG_LABELS):
            self.current_phase[i] = self.leg_phase[i, self.step_count]
            # self.contact_schedule[i, self.step_count] = self.contact_state[leg]["contact"][self.step_count]

            if self.current_phase[i] <= self.beta:
                self.contact_schedule[i, self.step_count] = 1
                Ps = self.footall_trajectories[i][-1] if self.step_count > 0 else self.Phip[i, :]
                # If in stance phase, use the last foot position
                Pe = self.BaseFoot.FootTarget(Vcur= self.des_vel, Tst=self.Tst, legID=i)
                self.world_foot_pos[i, :] = Ps
                Psw = Ps
                Vsw = np.zeros(3)
                self.base_foot_pos[i, :] = Rb.T @ (Psw - self.des_pos)
                self.base_foot_vel[i, :] = Rb.T @ Vsw
            else:
                self.contact_schedule[i, self.step_count] = 0
                phi_swing = (self.current_phase[i] - self.beta) / (1 - self.beta)
                Ps = self.footall_trajectories[i][-1] if self.step_count > 0 else self.Phip[i, :]
                # If in swing phase, use the last foot position
                Pe = self.BaseFoot.FootTarget(Vcur=self.des_vel, Tst=self.Tst, legID=i)
                Psw, Vsw, Asw = self.BaseFoot.SwingTrajectory(Ps=Ps, Pe=Pe, phi=phi_swing, Tsw=self.Tsw, legId=i)
                self.world_foot_pos[i, :] = Psw
                self.base_foot_pos[i, :] = Rb.T @ (Psw - self.des_pos)
                self.base_foot_vel[i, :] = Rb.T @ Vsw
            if Psw[2] >= 0.04:
                print(f"Foot {leg} is too high: {Psw[2]} m. Adjusting to ground level.")
            self.footall_trajectories[i].append(Psw.copy())
            self.footall_velocities[i].append(Vsw.copy())
            print(f"Leg {leg}: Phase {self.current_phase[i]:.2f}, Contact: {self.contact_schedule[i, self.step_count]}, Position: {Psw}, Velocity: {Vsw}")

        self.get_logger().info(f"World Foot Positions: {self.world_foot_pos}")
        # # 1. Calculate total force and torque in world frame
        # total_torque = np.zeros(3)
        # total_force = np.sum(self.force_reshaped[0], axis=0)
        # rpy_cur = self.com_ori
        # R = convert_rpy_to_rot(rpy_cur)
        # for i in range(4):
        #     r_i_world = (self.world_foot_pos[i, :] - self.com_pos)  # Position vector from COM to foot in world frame
        #     # Calculate torque for each foot
        #     total_torque += np.cross(r_i_world, self.force_reshaped[0, i])

        # # 2. Update the center of mass position and velocity
        # gravity_force = np.array([0.0, 0.0, -self.mpc.mass * 9.81])
        # net_force = total_force + gravity_force
        # self.com_vel += (net_force / self.mpc.mass) * self.dt
        
        # # self.com_vel[2] = 0.0  # Maintain constant height
        # if abs(self.com_vel[0]) < 0.01:
        #     self.com_vel[0] = 0.0
        # if abs(self.com_vel[1]) < 0.01:
        #     self.com_vel[1] = 0.0
        # if abs(self.com_vel[2]) < 0.01:
        #     self.com_vel[2] = 0.0
        # self.com_pos += self.com_vel * self.dt 
        # self.com_pos[2] = np.clip(self.com_pos[2], self.robot_height - 0.05, self.robot_height + 0.05)  # Maintain height within limits
        # # 3. Update the orientation and angular velocity
        # total_torque_body = R.T @ total_torque
        # gyroscopic_term = np.cross(self.com_omega, self.I_body @ self.com_omega)
        # self.com_omega += np.linalg.pinv(self.I_body) @ (total_torque_body - gyroscopic_term) * self.dt

        # R = R @ exp_rotation(self.com_omega, self.dt)
        # self.com_ori = rot_to_rpy(R)
        # # self.com_ori[0:3] = [0.0, 0.0, 0.0]  # Reset roll and pitch to zero for simplicity
        self.update_dynamics()
        self.get_logger().info(f"Step {self.step_count}: COM Position: {self.com_pos}, Velocity: {self.com_vel}, Orientation: {self.com_ori}, Angular Velocity: {self.com_omega}")
        # Publish COM position
        com_msg = Float32MultiArray()
        com_msg.data = [self.com_pos[0], self.com_pos[1], self.com_pos[2],
                        self.com_vel[0], self.com_vel[1], self.com_vel[2],
                        self.com_ori[0], self.com_ori[1], self.com_ori[2],
                        self.com_omega[0], self.com_omega[1], self.com_omega[2]]
        # Publish the COM position and velocity
        self.com_pub.publish(com_msg)
        self.step_count = (self.step_count + 1) % self.step
        # Update the step count
        # Store history for plotting
        
    def mpc_callback(self):
        # Update the MPC with the current state and desired state
        desired_state = {
            'position': self.des_pos,
            'velocity': self.des_vel,
            'orientation': self.des_ori,
            'angular_velocity': self.des_omega
        }
        actual_state = {
            'position': self.com_pos,
            'velocity': self.com_vel,
            'orientation': self.com_ori,
            'angular_velocity': self.com_omega
        }
        # Get contact schedule for the horizon
        horizon_contacts = np.zeros((self.horizon, 4))
        for i in range(self.horizon):
            idx = (self.step_mpc + i) % self.step
            horizon_contacts[i,:] = self.contact_schedule[:,idx]
        
        self.mpc.compute_contact_force(actual_state, desired_state, self.base_foot_pos, horizon_contacts)
        # Get the forces from the MPC
        force = self.mpc.solve_qp()
        self.force_reshaped = force.reshape((self.horizon, 4, 3))
        self.current_forces = self.force_reshaped[0]
        # Store force history
        for i in range(4):
            self.force_history['x'][i].append(self.force_reshaped[0,i,0])
            self.force_history['y'][i].append(self.force_reshaped[0,i,1])
            self.force_history['z'][i].append(self.force_reshaped[0,i,2])
        # Dynamic model update
        self.step_mpc = (self.step_mpc + 1) % self.step

def main(args=None):
    rclpy.init(args=args)
    node = Simulation_MPC_SRBM()
    
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()
if __name__ == '__main__':
    main()


        