#!/usr/bin/env python3
import rclpy
from rclpy.node import Node 
from sensor_msgs.msg import JointState
from geometry_msgs.msg import Twist
from std_msgs.msg import String
from std_msgs.msg import Float64MultiArray, Int8MultiArray
import numpy as np
from dynamo_one_control.Controller.PD_Control_torque import PD_Control_torque
from dynamo_one_control.model.Kinematic_Model import KinematicQuadruped
from dynamo_one_control.model.InverseDynamic import InverseDynamic

class Leg_PD_control(Node):
    def __init__(self):
        super().__init__('leg_pd_control')
        
        # Joint state subscriber
        self.joint_sub = self.create_subscription(
            JointState,
            '/joint_states',
            self.joint_callback,
            10)
        self.joint__pos_sub = self.create_subscription(
            Float64MultiArray, 
            'dynamo_one/joint_positions', 
            self.position_callback, 
            10)
        self.joint_vel_sub = self.create_subscription(
            Float64MultiArray,
            'dynamo_one/joint_velocities',
            self.velocity_callback,
            10)
        self.Mode_sub = self.create_subscription(
            String, 
            'dynamo_one/mode', 
            self.mode_callback, 
            10)
        self.footpos_des = self.create_subscription(
            Float64MultiArray,
            'dynamo_one/footpos_des',
            self.foot_callback,
            10
        )
        self.footvel_des = self.create_subscription(
            Float64MultiArray,
            'dynamo_one/footvel_des',
            self.footvel_callback,
            10
        )
        self.footacc_des = self.create_subscription(
            Float64MultiArray,
            'dynamo_one/footacc_des',
            self.footacc_callback,
            10
        )
        self.forcempc_sub = self.create_subscription(
            Float64MultiArray,
            'dynamo_one/forcempc',
            self.forcempc_callback,
            10
        )

        self.contact_sub = self.create_subscription(
            Int8MultiArray,
            'dynamo_one/contact_schedule',
            self.contact_callback,
            10
        )
        # Effort command publisher
        self.effort_pub = self.create_publisher(
            Float64MultiArray,
            '/joint_group_effort_controller/commands',
            10)
            
        # Control timer (100Hz)
        self.dt = 0.01
        self.control_timer = self.create_timer(self.dt, self.control_loop)
        self.t = 0.0
        # Robot parameters
        # Store both sets of gains
        # Define gain sets for different modes
        self.gain_sets = {
            'sit': {
                'hip': {'kp': 50.0, 'kd': 1.0},
                'thigh': {'kp': 60.0, 'kd': 1.2},
                'calf': {'kp': 50.0, 'kd': 1.0}
            },
            'stand': {
                'hip': {'kp': 50.0, 'kd': 1.0},
                'thigh': {'kp': 60.0, 'kd': 1.2},
                'calf': {'kp': 50.0, 'kd': 1.0}
            },
            'walk': {
                'hip': {'kp': 12.0, 'kd': 0.2},
                'thigh': {'kp': 12.0, 'kd': 0.2},
                'calf': {'kp': 6.0, 'kd': 0.15}
            },
            'trot': {
                'hip': {'kp': 8.0, 'kd': 0.2},
                'thigh': {'kp': 10.0, 'kd': 0.25},
                'calf': {'kp': 9.0, 'kd': 0.22}
            }
        }
        
        # Transition control
        self.transition_start_time = 0.0
        self.transition_duration = 0.5  # seconds for transition
        self.is_transitioning = False
        self.current_gains = self.gain_sets['stand'].copy()
        self.target_gains = self.current_gains.copy()
        self.t = 0.0
        # Initialize PD controller with current gains
        self.limit_toque = 30
        self.pd_torque = PD_Control_torque(
            self.current_gains['hip'],
            self.current_gains['thigh'],
            self.current_gains['calf'],
            limit_toque=self.limit_toque
        )
        # Initialize variables
        self.joint_pos_feedback = np.zeros(12)
        self.joint_vel_feedback = np.zeros(12)
        self.joint_names = [
            "FL_hip_joint", "FL_thigh_joint", "FL_calf_joint",
            "FR_hip_joint", "FR_thigh_joint", "FR_calf_joint",
            "RR_hip_joint", "RR_thigh_joint", "RR_calf_joint",
            "RL_hip_joint", "RL_thigh_joint", "RL_calf_joint"
        ]
        self.torque_command = np.zeros(12)
        self.joint_toque = np.zeros(12)
        # Standing configuration (modify based on your robot)
        self.joint_pos_desired = np.array([
                                0.0, -0.0, 0.0,   # FL
                                0.0, -0.0, 0.0,   # FR 
                                0.0, -0.0, 0.0,   # RL
                                0.0, -0.0, 0.0    # RR
                            ])
        self.L = 0.433
        self.W = 0.12
        self.L1 = 0.087
        self.L2 = 0.25
        self.L3 = 0.25
        self.m1 = 0.08
        self.m2 = 0.143
        self.m3 = 0.1
        self.I1zz = 0.001  # Inertia of the thigh
        self.I2zz = 0.001  # Inertia of the calf
        self.I3zz = 0.001
        self.robot_height = 0.35
        self.KpCartesian = np.diag([1.0, 1.0, 1.0])  # Proportional gain for Cartesian control
        self.KdCartesian = np.diag([0.1, 0.1, 0.1])
        self.omega = np.array([1.0, 1.0, 1.0]) * 0.5

        self.LegInvdynamic = InverseDynamic(self.L, self.W, self.L1, self.L2, self.L3, self.m1, self.m2, self.m3, self.I1zz, self.I2zz, self.I3zz, self.KpCartesian, self.KdCartesian, self.dt, self.omega)
        self.hip_origin = np.array([[self.L/2, self.W/2, 0.0],
                                    [self.L/2, -self.W/2, 0.0],
                                    [-self.L/2, -self.W/2, 0.0],
                                    [-self.L/2, self.W/2, 0.0]])
        self.KinematicModel = KinematicQuadruped(self.L, self.W, self.L1, self.L2, self.L3)
        self.joint_vel_desired = np.zeros(12)  # Desired velocities are zero for standing
        self.pDes = np.array([
            [0.0, self.L1, -self.robot_height],
            [0.0, self.L1, -self.robot_height],
            [0.0, -self.L1, -self.robot_height],
            [0.0, -self.L1, -self.robot_height]
        ])
        self.vDes = np.zeros((4, 3))
        self.aDes = np.zeros((4, 3))
        self.forcempc = np.zeros((4, 3))  # Force MPC for each leg
        self.qDes = np.zeros(12)  # Joint positions
        self.qdDes = np.zeros(12) # Joint velocities
        self.contact_state = np.array([1, 1, 1, 1])  # Contact state for each leg (1: contact, 0: no contact)
        self.joint_pos = [0.0] *12
        self.rot = [0.0, 0.0, 0.0]
        self.command_gait = 'stand'
        self.hip_limits = (-1.57, 1.20)
        self.thigh_limits = (-3.14, 3.14)
        self.calf_limits = (-2.53, -0.0872665)
        self.check = True
        self.torque_cur = np.zeros(12)
    def start_gait_transition(self, target_mode):
        """Initiate a smooth transition to new gait gains"""
        if target_mode in self.gain_sets:
            self.target_gains = self.gain_sets[target_mode].copy()
            self.transition_start_time = self.t
            self.is_transitioning = True
            # self.get_logger().info(f"Starting transition to {target_mode} mode")
        else:
            self.get_logger().warn(f"Unknown gait mode: {target_mode}")

    def update_gait_transition(self):
        """Update gain interpolation during transition"""
        if self.is_transitioning:
            elapsed = self.t - self.transition_start_time
            progress = min(elapsed / self.transition_duration, 1.0)
            
            # Interpolate all gains
            for joint_type in ['hip', 'thigh', 'calf']:
                for gain_type in ['kp', 'kd']:
                    start_val = self.current_gains[joint_type][gain_type]
                    target_val = self.target_gains[joint_type][gain_type]
                    self.current_gains[joint_type][gain_type] = \
                        start_val + (target_val - start_val) * progress
            
            # Update PD controller with interpolated gains
            self.pd_torque = PD_Control_torque(
                self.current_gains['hip'],
                self.current_gains['thigh'],
                self.current_gains['calf'],
                limit_toque=self.limit_toque
            )
            
            # Check if transition complete
            if progress >= 0.5:
                self.is_transitioning = False
                self.get_logger().info("Gait transition completed")
    def joint_callback(self, msg):
        # Update current joint states
        for i, name in enumerate(msg.name):
            if name in self.joint_names:
                idx = self.joint_names.index(name)
                self.joint_pos_feedback[idx] = msg.position[i]
                self.joint_vel_feedback[idx] = msg.velocity[i]
    def position_callback(self, msg):
        self.joint_pos_desired = np.array([
            msg.data[0], msg.data[1], msg.data[2],    # FL leg
            msg.data[3], msg.data[4], msg.data[5],     # FR leg
            msg.data[6], msg.data[7], msg.data[8],     # RR leg
            msg.data[9], msg.data[10], msg.data[11]     # RL leg
        ])  
    def velocity_callback(self, msg):
        # Assuming the first 12 elements are joint velocities
        self.joint_vel_desired = np.array([
            msg.data[0], msg.data[1], msg.data[2],    # FL leg
            msg.data[3], msg.data[4], msg.data[5],     # FR leg
            msg.data[6], msg.data[7], msg.data[8],     # RR leg
            msg.data[9], msg.data[10], msg.data[11]     # RL leg
        ])       
        if np.any(np.isnan(self.joint_vel_desired)):
            # self.get_logger().error("Received NaN values in joint velocities. Resetting to zero.")
            self.joint_vel_desired = np.zeros(12)
    def foot_callback(self, msg):
        # Update desired foot positions and velocities from the message
        FLpdes = np.array(msg.data[:3])
        FRpdes = np.array(msg.data[3:6])
        RRpdes = np.array(msg.data[6:9])
        RLpdes = np.array(msg.data[9:12])
        self.pDes = np.array([FLpdes, FRpdes, RRpdes, RLpdes]) - self.hip_origin
        # self.get_logger().info(f"Received foot positions: {self.pDes}")
    def footvel_callback(self, msg):
        # Update desired foot velocities from the message
        FLvdes = np.array(msg.data[:3])
        FRvdes = np.array(msg.data[3:6])
        RRvdes = np.array(msg.data[6:9])
        RLvdes = np.array(msg.data[9:12])
        self.vDes = np.array([FLvdes, FRvdes, RRvdes, RLvdes])
        # self.get_logger().info(f"Received foot velocities: {self.vDes}")
    def footacc_callback(self, msg):
        # Update desired foot accelerations from the message
        FLades = np.array(msg.data[:3])
        FRades = np.array(msg.data[3:6])
        RRades = np.array(msg.data[6:9])
        RLades = np.array(msg.data[9:12])
        self.aDes = np.array([FLades, FRades, RRades, RLades])
    def forcempc_callback(self, msg):
        # Update desired foot accelerations from the message
        FLades = np.array(msg.data[:3])
        FRades = np.array(msg.data[3:6])
        RRades = np.array(msg.data[6:9])
        RLades = np.array(msg.data[9:12])
        self.forcempc = np.array([FLades, FRades, RRades, RLades])
    
    def mode_callback(self, msg):
        """Handle mode change commands"""
        new_mode = msg.data.lower()
        if new_mode != self.command_gait:
            self.command_gait = new_mode
            self.start_gait_transition(new_mode)
            
            # Additional mode-specific initialization
            if new_mode in ['sit', 'stand']:
                self.torque_command = self.torque_cur
            elif new_mode in ['walk', 'trot']:
                # Initialize to slightly bent legs for locomotion
                self.torque_command = self.torque_cur

    
    def contact_callback(self, msg):
        # Update the contact state based on the message received
        self.contact_state = np.array(msg.data)
        # self.get_logger().info(f"Received contact state: {self.contact_state}")
    def body_control(self):
        # Standard standing control
        self.torque_command = self.pd_torque.compute_torque(
            self.joint_pos_desired, 
            self.joint_vel_desired, 
            self.joint_pos_feedback, 
            self.joint_vel_feedback
        )

    def gait_control(self):
        # Calculate desired joint positions
        qFL, qFR, qRR, qRL = self.KinematicModel.leg_IK(self.pDes, rot=self.rot)
        self.qDes[0:3] = qFL
        self.qDes[3:6] = qFR
        self.qDes[6:9] = qRR
        self.qDes[9:12] = qRL
        
        # Apply joint limits
        for i in [0, 3, 6, 9]:  # Hip joints
            self.qDes[i] = np.clip(self.qDes[i], *self.hip_limits)
        for i in [1, 4, 7, 10]:  # Thigh joints
            self.qDes[i] = np.clip(self.qDes[i], *self.thigh_limits)
        for i in [2, 5, 8, 11]:  # Calf joints
            self.qDes[i] = np.clip(self.qDes[i], *self.calf_limits)
        
        # Calculate desired joint velocities
        for i in range(4):
            self.qdDes[i*3:(i+1)*3] = self.KinematicModel.get_joint_velocity(
                self.vDes[i, :],
                self.qDes[i*3:(i+1)*3],
                leg_id=i
            )
        
        # Apply torque
        self.torque_command = self.pd_torque.compute_torque(
            self.qDes,
            self.qdDes, 
            self.joint_pos_feedback, 
            self.joint_vel_feedback
        )
        
        # Optionally add inverse dynamics torques
        # for i in range(4):
        #     if self.contact_state[i] == 0:  # Swing phase
        #         tauswing = self.LegInvdynamic.TorqueSwing(
        #             self.pDes[i,:], self.vDes[i,:], self.aDes[i,:],
        #             self.joint_pos_feedback[i*3:(i+1)*3], self.joint_vel_feedback[i*3:(i+1)*3], legID=i
        #         )
        #         self.torque_command[3*i:3*(i+1)] += np.array(tauswing).reshape(3)   # Add tauswing
        #     else:  # Stance phase
        #         taustance = self.LegInvdynamic.TorqueStance(
        #             self.forcempc[i,:], self.joint_pos_feedback[i*3:(i+1)*3], legID=i
        #         )
        #         # self.torque_command[3*i:3*(i+1)] = self.torque_command[3*i:3*(i+1)] # Add taustance
        #         self.torque_command[3*i:3*(i+1)] += np.array(taustance).reshape(3) 
        # self.get_logger().info(f"Torque command: {self.torque_command}")
    def control_loop(self):
        # Handle gait transitions
        self.update_gait_transition()
        
        if self.command_gait in ["stand", "sit"]:
            self.body_control()
            # self.get_logger().info(f"Desired joint velocities: {self.joint_vel_desired}")
        elif self.command_gait in ['walk', 'trot']:
            self.gait_control()
        self.torque_cur = self.torque_command
        # self.get_logger().info(f"Current gains: {self.current_gains}")
        # Publish efforts
        effort_msg = Float64MultiArray()
        effort_msg.data = self.torque_command.tolist()
        self.effort_pub.publish(effort_msg)
        # Update time
        self.t += self.dt
    
def main(args=None):
    rclpy.init(args=args)
    controller = Leg_PD_control()
    rclpy.spin(controller)
    controller.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()