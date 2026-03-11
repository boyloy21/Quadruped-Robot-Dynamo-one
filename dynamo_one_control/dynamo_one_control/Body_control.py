#!/usr/bin/env python3
import rclpy
from rclpy.node import Node 
from sensor_msgs.msg import Joy, JointState, Imu
from geometry_msgs.msg import Twist
from std_msgs.msg import String, Float64MultiArray, Float32MultiArray
import numpy as np
from dynamo_one_control.model.Kinematic_Model import KinematicQuadruped as RobotModel
from dynamo_one_control.Desired.CPG_Network import CPG_Network
from dynamo_one_control.utills.kinematic_utill import RotMatrix3D
from dynamo_one_control.Filter.moving_window_filter import MovingWindowFilter
from math import *
import time


class Gait_control(Node):
    def __init__(self):
        super().__init__('Gait_control')
        # Publishers
        self.joint_pos_pub = self.create_publisher(Float64MultiArray, 'dynamo_one/joint_positions', 10)
        self.joint_vel_pub = self.create_publisher(Float64MultiArray, 'dynamo_one/joint_velocities', 10)

        # Subscribers
        self.position_sub = self.create_subscription(Twist, 'dynamo_one/cmd_vel', self.position_callback, 10)
        self.Mode_sub = self.create_subscription(String, 'dynamo_one/mode', self.mode_callback, 10)
        self.body_des_sub = self.create_subscription(Float32MultiArray, 'dynamo_one/body_des', self.body_callback, 10)
        self.imu_sub = self.create_subscription(Imu, 'dynamo_one/imu', self.imu_callback, 10)

        # Timer for control loop
        self.timer = self.create_timer(0.01, self.control_loop)
        # Timer to print stats every 10 seconds (adjust as needed)
        self.create_timer(10.0, self.print_statistics)

        self.joint_positions = [0.0] * 12
        self.joint_velocities = [0.0] * 12
       
        self.dt = 0.01
        self.joint_names = [
            "FL_hip_joint", "FL_thigh_joint", "FL_calf_joint",
            "FR_hip_joint", "FR_thigh_joint", "FR_calf_joint",
            "RR_hip_joint", "RR_thigh_joint", "RR_calf_joint",
            "RL_hip_joint", "RL_thigh_joint", "RL_calf_joint"
        ]

        # Parameter of Robots
        self.L = 0.433
        self.W = 0.12
        self.L1, self.L2, self.L3 = 0.087, 0.25, 0.25
        self.dynamo_one = RobotModel(self.L, self.W, self.L1, self.L2, self.L3)

        self.orientation_filter = MovingWindowFilter(window_size=5, dim=4)  # Filter for angular velocity

        # Define initial position
        self.rot = [0.0, 0.0, 0.0] # [roll, pitch, yaw]
        self.rot_matrix = np.eye(4)
        
        # Clipping joint angles to limits
        self.hip_limits = (-1.57, 1.20)
        self.thigh_limits = (-3.14, 3.14)
        self.calf_limits = (-2.53, -0.0872665)

        # Initialize Goal positions
        self.x = 0.0
        self.y = 0.0
        self.z = 0.0
        self.roll = 0.0
        self.pitch = 0.0
        self.yaw = 0.0
        
        self.xyz = np.array([
                [self.x , self.y + self.L1, self.z ],  # FL
                [self.x , self.y - self.L1, self.z ],  # FR
                [self.x , self.y - self.L1, self.z ],  # RR
                [self.x , self.y + self.L1, self.z ]   # RL
            ])
        self.vel_foot = np.array([
                [0.0, 0.0, 0.0],  # FL
                [0.0, 0.0, 0.0],  # FR
                [0.0, 0.0, 0.0],  # RR
                [0.0, 0.0, 0.0]   # RL
            ])
        self.xyz_prev = np.array([
                [self.x , self.y + self.L1, self.z ],  # FL
                [self.x , self.y - self.L1, self.z ],  # FR
                [self.x , self.y - self.L1, self.z ],  # RR
                [self.x , self.y + self.L1, self.z ]   # RL
            ])
        self.y_offset = np.array([[0, self.L1, 0],
                                  [0, -self.L1, 0],
                                  [0, -self.L1, 0],
                                  [0, self.L1, 0]])
        self.command_gait = "stand"
        self.t_index = 0
        # For storing roll and pitch errors
        self.roll_errors = []
        self.pitch_errors = []


    def foot_position_in_body_frame(self, theta1, theta2, theta3, hip_offset):
        L1, L2, L3 = 0.087, 0.25, 0.25  # Lengths of the leg segments
        # Foot position in hip frame (forward kinematics)
        H_P_F = np.array([
            L2 * np.cos(theta2) + L3 * np.cos(theta2 + theta3),
            L1 * np.cos(theta1) + L2 * np.sin(theta1) * np.sin(theta2) + L3 * np.sin(theta1) * np.sin(theta2 + theta3),
            L1 * np.sin(theta1) - L2 * np.cos(theta1) * np.sin(theta2) - L3 * np.cos(theta1) * np.sin(theta2 + theta3)
        ])
        
        # Rotate by abduction (theta1) and add hip offset
        R_z = np.array([
            [np.cos(theta1), -np.sin(theta1), 0],
            [np.sin(theta1),  np.cos(theta1), 0],
            [0,              0,              1]
        ])
        B_P_F = hip_offset + R_z @ H_P_F
        return B_P_F
    def body_callback(self, msg):
        # Update body position and orientation from the message
        self.x = msg.data[0]
        self.x = -1*self.x
        self.y = msg.data[1]
        self.y = -1*self.y
        self.z = msg.data[2]
        self.z = -1*self.z
        self.roll = msg.data[3]
        self.pitch = msg.data[4]
        self.yaw = msg.data[5]
        
        # Log the received body position and orientation
        # self.get_logger().info(f"Body Position - x: {self.x}, y: {self.y}, z: {self.z}, roll: {self.roll}, pitch: {self.pitch}, yaw: {self.yaw}")
    def quaternion_to_euler(self, w, x, y, z):
        # Roll (x-axis rotation)
        roll = atan2(2 * (w * x + y * z), 1 - 2 * (x**2 + y**2))

        # Pitch (y-axis rotation)
        pitch = asin(2 * (w * y - z * x))

        # Yaw (z-axis rotation)
        yaw = atan2(2 * (w * z + x * y), 1 - 2 * (y**2 + z**2))

        return roll, pitch, yaw
    
    def imu_callback(self, msg):
        # Store raw IMU data
        self.quaternion = np.array([
            msg.orientation.w,
            msg.orientation.x,
            msg.orientation.y,
            msg.orientation.z
        ])

        # Apply filtering
        filterd_orientation = self.orientation_filter.calculate_average(self.quaternion)

        # Convert to Euler angles
        roll, pitch, yaw = self.quaternion_to_euler(
            filterd_orientation[0],
            filterd_orientation[1],
            filterd_orientation[2],
            filterd_orientation[3]
        )
        self.rot = [roll, pitch, yaw]
        # Log the Euler angles
        # Store error (only if desired values are available)
        roll_error = self.roll - roll
        pitch_error = self.pitch - pitch
        self.roll_errors.append(roll_error)
        self.pitch_errors.append(pitch_error)
        # self.get_logger().info(f"IMU feedback: Roll_back: {roll}, Pitch_back: {pitch}, Yaw_back: {yaw}")

    def position_callback(self, msg):
        # self.axis_list = msg.linear.x, msg.linear.y, msg.linear.z, msg.angular.x, msg.angular.y, msg.angular.z, msg.angular.x, msg.angular.y
        self.x = msg.linear.x
        self.x = -1*self.x
        self.y = msg.linear.y
        self.z = msg.linear.z
        self.z = -1*self.z
        self.roll = msg.angular.x
        self.pitch = msg.angular.y
        self.yaw = msg.angular.z
    
    def mode_callback(self, msg):
        self.command_gait = msg.data
        # if self.command_gait == "stand":
        #     self.get_logger().info("Stand mode activated")
        # elif self.command_gait == "sit":
        #     self.get_logger().info("Sit mode activated")
        # elif self.command_gait == "walk":
        #     self.get_logger().info("Walk mode activated")
        # elif self.command_gait == "trot":
        #     self.get_logger().info("Trot mode activated")
        # elif self.command_gait == "gallop":
        #     self.get_logger().info("Gallop mode activated")
        # elif self.command_gait == "pace":
        #     self.get_logger().info("Pace mode activated")
        # else:
        #     self.get_logger().error(f"Unknown mode: {self.command_gait}")
    
    def control_loop(self):
        if self.command_gait in ["stand", "sit"]:
            try:
                # 1. Calculate desired foot positions in body frame
                self.xyz = np.array([
                    [self.x, self.y + self.L1, self.z],  # FL
                    [self.x, self.y - self.L1, self.z],  # FR 
                    [self.x, self.y - self.L1, self.z],  # RR
                    [self.x, self.y + self.L1, self.z]   # RL
                ])
                # self.get_logger().info(f"foot positions: {self.xyz}")
                # 2. Compute smoothed foot velocities
                # if not hasattr(self, 'xyz_prev'):
                #     self.xyz_prev = self.xyz.copy()
                #     self.vel_foot = np.zeros((4,3))
                # else:
                #     # Velocity limiting
                #     max_foot_vel = 2.0  # m/s
                #     for i in range(4):
                #         vel_norm = np.linalg.norm(self.vel_foot[i])
                #         if vel_norm > max_foot_vel:
                #             self.vel_foot[i] = self.vel_foot[i] * (max_foot_vel/vel_norm)
                self.vel_foot = 0.9 * self.vel_foot + 0.1 * (self.xyz - self.xyz_prev)
                self.xyz_prev = self.xyz.copy()
                
                # 3. Compute inverse kinematics
                FL, FR, RR, RL = self.dynamo_one.leg_IK(
                    xyz=self.xyz,
                    rot=[self.roll, self.pitch, self.yaw],
                    is_radians=True
                )
                self.joint_positions = np.concatenate([FL, FR, RR, RL])
                
                # 4. Compute joint velocities with improved method
                for i in range(4):
                    self.joint_velocities[i*3:(i+1)*3] = self.dynamo_one.get_joint_velocity(
                        foot_velocity=self.vel_foot[i],
                        angles=self.joint_positions[i*3:(i+1)*3],
                        leg_id=i
                    )
                # self.get_logger().info(f"joint velocities: {self.joint_velocities}")
                # 5. Apply joint limits and publish
                self.apply_joint_limits()
                self.publish_commands()
                
            except Exception as e:
                self.get_logger().error(f"Control loop error: {str(e)}")
                self.emergency_stop()

    def apply_joint_limits(self):
        """Apply position and velocity limits"""
        # Position limits
        for i in [0, 3, 6, 9]:  # Hip joints
            self.joint_positions[i] = np.clip(self.joint_positions[i], *self.hip_limits)
        for i in [1, 4, 7, 10]:  # Thigh joints
            self.joint_positions[i] = np.clip(self.joint_positions[i], *self.thigh_limits)
        for i in [2, 5, 8, 11]:  # Calf joints
            self.joint_positions[i] = np.clip(self.joint_positions[i], *self.calf_limits)
        
        # # Velocity limits
        MAX_ACCEL = 10.0  # rad/s²
        if hasattr(self, 'prev_joint_vel'):
            vel_diff = np.array(self.joint_velocities) - self.prev_joint_vel
            vel_diff = np.clip(vel_diff, -MAX_ACCEL*self.dt, MAX_ACCEL*self.dt)
            self.joint_velocities = self.prev_joint_vel + vel_diff
        self.prev_joint_vel = np.array(self.joint_velocities.copy())
    
    def publish_commands(self):
        """Safe command publishing with NaN checks"""
        if np.any(np.isnan(self.joint_positions)) or np.any(np.isnan(self.joint_velocities)):
            # self.get_logger().warn("Position NaN detected in commands! Sending zeros.")
            self.joint_positions = np.zeros(12)
            self.joint_velocities = np.zeros(12)
            
        # Create and publish messages
        pos_msg = Float64MultiArray(data=self.joint_positions.tolist())
        vel_msg = Float64MultiArray(data=self.joint_velocities.tolist())
        
        self.joint_pos_pub.publish(pos_msg)
        self.joint_vel_pub.publish(vel_msg)
    def emergency_stop(self):
        """Immediately stop all robot motion safely by:
        1. Setting all joint velocities to zero
        2. Maintaining current positions (or moving to safe positions)
        3. Disabling any active gait
        4. Publishing the stop commands
        """
        # 1. Zero all joint velocities
        self.joint_velocities = [0.0] * 12
        
        # 2. Option 1: Hold current positions
        # (Keep current self.joint_positions as is)
        
        # Option 2: Move to safe positions (e.g., standing configuration)
        # self.joint_positions = [
        #     0.0, -0.7, -1.4,  # FL
        #     0.0, -0.7, -1.4,  # FR 
        #     0.0, -0.7, -1.4,  # RR
        #     0.0, -0.7, -1.4   # RL
        # ]
        
        # 3. Disable any active gait pattern
        self.command_gait = "stand"
        
        # 4. Immediately publish stop commands
        self.publish_commands()
        
        # 5. Log the emergency stop
        # self.get_logger().error("!!! EMERGENCY STOP ACTIVATED !!!")
        
        # 6. Optional: Flash lights/sound alarms if available
        # self.activate_alarm()
    def compute_stats(self, errors):
        errors = np.array(errors)
        mean = np.mean(errors)
        peak = np.max(np.abs(errors))
        rms = np.sqrt(np.mean(errors**2))
        return mean, peak, rms
    def print_statistics(self):
        
        roll_mean, roll_peak, roll_rms = self.compute_stats(self.roll_errors)
        pitch_mean, pitch_peak, pitch_rms = self.compute_stats(self.pitch_errors)

        self.get_logger().info(f"Roll Error - MEAN: {roll_mean:.4f}, PEAK: {roll_peak:.4f}, RMS: {roll_rms:.4f}")
        self.get_logger().info(f"Pitch Error - MEAN: {pitch_mean:.4f}, PEAK: {pitch_peak:.4f}, RMS: {pitch_rms:.4f}")
        # Clear errors for next interval
        self.roll_errors.clear()
        self.pitch_errors.clear()
def main(args=None):
    rclpy.init(args=args)
    joint_foot_publisher = Gait_control()
    rclpy.spin(joint_foot_publisher)
    joint_foot_publisher.destroy_node()
    rclpy.shutdown()
    

if __name__ == '__main__':
    main()



