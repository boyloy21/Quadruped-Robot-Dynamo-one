#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState
from std_msgs.msg import Float64MultiArray, String
from geometry_msgs.msg import Twist
from sensor_msgs.msg import Imu
import numpy as np
import math
import time
from python_controller.robot.QuadrupedModel import RobotModel
from python_controller.Controller.PID_controller import PID_controller
from python_controller.FootTrajectory.CPG_Network import CPG_Network


class DynamoControl(Node):
    def __init__(self):
        super().__init__('dynamo_control')
        self.joint_pub = self.create_publisher(JointState, '/joint_states', 10)
        self.joint_pos_pub = self.create_publisher(Float64MultiArray, '/forward_position_controller/commands', 10)
        # self.Rot_pub = self.create_publisher(Float64MultiArray, 'dynamo_one/imu', 20)
        self.mode_sub = self.create_subscription(String, '/dynamo_one/mode', self.mode_callback, 10)
        self.position_sub = self.create_subscription(Twist, '/dynamo_one/cmd_vel', self.position_callback, 10)
        self.imu_sub = self.create_subscription(Imu, '/imu', self.imu_callback, 10)
        self.timer = self.create_timer(0.001, self.timer_callback)  # Publish joint states at 1KHz
        # self.timer_traj = self.create_timer(0.03, self.timer_callback_traj) # 30Hz
        self.joint_sub = self.create_subscription(JointState, '/joint_states', self.joint_callback, 10)
        # Initialize joint positions
        self.joint_positions = [0.0] * 12
        self.joint_feedback = [0.0] * 12
        self.joint_names = [
            "FR_hip_joint", "FR_thigh_joint", "FR_calf_joint",
            "FL_hip_joint", "FL_thigh_joint", "FL_calf_joint",
            "RR_hip_joint", "RR_thigh_joint", "RR_calf_joint",
            "RL_hip_joint", "RL_thigh_joint", "RL_calf_joint"
        ]
        self.rot = [0.0, 0.0, 0.0]

        # Robot parameters 
        self.L = 0.433
        self.W = 0.295
        self.L1, self.L2, self.L3 = 0.087, 0.25, 0.25
        self.robot = RobotModel(self.L, self.W, self.L1, self.L2, self.L3)

        # Initialize Goal positions
        self.goal_x = 0.0
        self.goal_y = 0.0
        self.goal_z = -0.25
        self.goal_roll = 0.0
        self.goal_pitch = 0.0
        self.goal_yaw = 0.0
        # self.goal_x_front = 0.1
        # self.goal_x_back = -0.05
        self.goal_x_front = 0.05
        self.goal_x_back = -0.05

        # Initialize mode and position
        self.mode = "sit"
        self.x = 0.0
        self.y = 0.0
        self.z = -0.0
        self.roll = 0.0
        self.pitch = 0.0
        self.yaw = 0.0
        self.x_front_st = 0.0
        self.x_back_st = 0.0
        self.xyz = np.array([[self.x , self.y + self.L1, self.z ], 
                        [self.x , self.y - self.L1, self.z], 
                        [self.x , self.y - self.L1, self.z], 
                        [self.x , self.y + self.L1, self.z]])

       
        # PID controller
        self.pid = PID_controller(0.75, 0.0, 0.0)
        self.pid.reset()
        self.roll_feedback = 0.0
        self.pitch_feedback = 0.0
        self.linear_vel = 0.0005
        self.angular_Vel = 0.02

        # Clipping joint angles
        self.hip_limits = (-1.57, 1.20)
        self.thigh_limits = (-3.14, 3.14)
        self.calf_limits = (-2.53, 0)

        # Gait parameters
        self.t_index = 0
        self.cycles = 5
        self.Ts = 0.0
        self.step_length = 0.05
        self.step_height = 0.05
        self.LEG_LABELS = ["FL", "FR", "RR", "RL"] 
        self.command_speed = 0.5
        self.move_direction = 'forward'
        # Generate foot positions
        self.foot_positions = {label: {"x": [], "y": [], "z": []} for label in self.LEG_LABELS}
        self.leg_map = {0: "FL", 2: "FR", 4: "RR", 6: "RL"} 
        # self.cpg = CPG_Network(Ts=self.Ts, cycles=self.cycles)
        # self.foot_positions, self.Q_history = self.cpg.generate(gait_type=self.command_gait, step_length=self.step_length, step_height=self.step_height, z=self.goal_z)

    def move_toward(self, current, target, step):
        if abs(target - current) < step:
            return target
        return current + np.sign(target - current) * step
    
    def calculate_gait_parameters(self):
        min_Ts, max_Ts = 0.25, 1.5
        min_step, max_step = 0.05, 0.15

        if self.command_speed <= 0.1:
            Ts = 0.5
            step_length = 0.0
            step_height = 0.05
        else:
            Ts = max_Ts - (max_Ts - min_Ts) * self.command_speed
            step_length = min_step + (max_step - min_step) * self.command_speed
            step_height = 0.05
        return Ts, step_length, step_height
    
    def Q_update(self):
        self.Ts, self.step_length, self.step_height = self.calculate_gait_parameters()
        self.cpg = CPG_Network(Ts=self.Ts, cycles=self.cycles)
        self.foot_positions, self.Q_history = self.cpg.generate(gait_type=self.mode, step_length=self.step_length, step_height=self.step_height, z=-0.35, move_direction=self.move_direction)
   
    def quaternion_to_euler(self, w, x, y, z):
        # Roll (x-axis rotation)
        roll = math.atan2(2 * (w * x + y * z), 1 - 2 * (x**2 + y**2))

        # Pitch (y-axis rotation)
        pitch = math.asin(2 * (w * y - z * x))

        # Yaw (z-axis rotation)
        yaw = math.atan2(2 * (w * z + x * y), 1 - 2 * (y**2 + z**2))

        return roll, pitch, yaw
    def imu_callback(self, msg):
        # Extract quaternion from IMU message
        w = msg.orientation.w
        x = msg.orientation.x
        y = msg.orientation.y
        z = msg.orientation.z

        # Convert quaternion to Euler angles
        roll, pitch, yaw = self.quaternion_to_euler(w, x, y, z)
        self.roll_feedback = roll
        self.pitch_feedback = pitch
        # self.Rot_pub.publish(Float64MultiArray(data=[roll, pitch, yaw]))
        # Log the Euler angles
        # self.get_logger().info(f"Roll_back: {roll}, Pitch_back: {pitch}, Yaw_back: {yaw}")
    def mode_callback(self, msg):
        self.mode = msg.data
        self.get_logger().info(f"Mode changed to: {self.mode}")

    def joint_callback(self, msg):
        for i in range(len(msg.name)):
            if msg.name[i] in self.joint_names:
                self.joint_feedback[self.joint_names.index(msg.name[i])] = msg.position[i]
        # self.get_logger().info(f"Joint feedback: {self.joint_feedback}")
    def Balance(self):
        compensation = self.pid.run(self.roll_feedback, self.pitch_feedback)
        roll_compensation = -compensation[0]
        pitch_compensation = -compensation[1]
        self.goal_roll = roll_compensation
        self.goal_pitch = pitch_compensation
        

    def position_callback(self, msg):
        self.position = msg
        self.goal_x = msg.linear.x
        self.goal_y = msg.linear.y
        self.goal_z = msg.linear.z
        self.goal_roll = msg.angular.x
        self.goal_pitch = msg.angular.y
        self.goal_yaw = msg.angular.z
        # if ((self.mode == 'walk' and (self.goal_x >= 0.5))):
        #     self.move_direction = 'forward'
        # elif ((self.mode == 'walk' and self.goal_x <= -0.5)):
        #     self.move_direction = 'backward'
        # elif ((self.mode == 'walk' and self.goal_y >= 0.5)):
        #     self.move_direction = 'left'
        # elif ((self.mode == 'walk' and self.goal_y <= -0.5)):
        #     self.move_direction = 'right'
        if ((self.mode == 'walk')):
            if self.goal_x > 0.5:
                self.move_direction = 'forward'
                self.command_speed = abs(self.goal_x)
            elif self.goal_x < -0.5:
                self.move_direction = 'backward'
                self.command_speed = abs(self.goal_x)
            elif self.goal_y > 0.5:
                self.move_direction = 'left'
                self.command_speed = abs(self.goal_y)
            elif self.goal_y < -0.5:
                self.move_direction = 'right'
                self.command_speed = abs(self.goal_y)
            else:
                self.move_direction = 'stop'
                self.command_speed = 0.0

        self.get_logger().info(f"Move direction changed to: {self.move_direction}, Position changed to: {self.goal_x}, {self.goal_y}, {self.goal_z}, {self.goal_roll}, {self.goal_pitch}, {self.goal_yaw}")
        # if ((self.mode == 'walk' and (self.move_direction == 'forward' or self.move_direction == 'backward')) or self.mode == 'trot' ):
        #     self.command_speed = abs(self.goal_x)
        #     self.get_logger().info(f"Command speed changed to: {self.command_speed}")
            
        # elif ((self.mode == 'walk' and (self.move_direction == 'right' or self.move_direction == 'left')) or self.mode == 'trot' ):
        #     self.command_speed = abs(self.goal_y)
        #     self.get_logger().info(f"Command speed changed to: {self.command_speed}")

        
    def timer_callback(self):
            
        # self.x_front_st = self.move_toward(self.x_front_st, self.goal_x_front, self.linear_vel)
        # self.x_back_st = self.move_toward(self.x_back_st, self.goal_x_back, self.linear_vel)
        self.x = self.move_toward(self.x, self.goal_x, self.linear_vel)
        self.y = self.move_toward(self.y, self.goal_y, self.linear_vel)
        self.z = self.move_toward(self.z, self.goal_z, self.linear_vel)
        self.roll = self.move_toward(self.roll, self.goal_roll, self.angular_Vel)
        self.pitch = self.move_toward(self.pitch, self.goal_pitch, self.angular_Vel)
        self.yaw = self.move_toward(self.yaw, self.goal_yaw, self.angular_Vel)

        if (self.mode == "stand"):
            self.xyz = np.array([[self.x , self.y + self.L1, self.z ], 
                        [self.x , self.y - self.L1, self.z], 
                        [self.x , self.y - self.L1, self.z], 
                        [self.x , self.y + self.L1, self.z]])
            self.angular_Vel = 0.01
            self.rot = [self.roll, self.pitch, self.yaw]
            
        elif (self.mode == "sit"):
            self.xyz = np.array([[self.x , self.y + self.L1, self.z ], 
                        [self.x, self.y - self.L1, self.z ], 
                        [self.x, self.y - self.L1, self.z ], 
                        [self.x, self.y + self.L1, self.z]])
            self.angular_Vel = 0.01
            self.rot = [self.roll, self.pitch, self.yaw]

        
        ###=== In Gait Control ====###
        else:
            self.Q_update()
            max_index = len(self.Q_history[0]) - 1
            if self.t_index > max_index:
                self.t_index = 0  # Loop animation
            if len(self.foot_positions['FL']["x"]) == 0 or len(self.foot_positions['FL']["y"]) == 0 or len(self.foot_positions['FL']["z"]) == 0:
                self.get_logger().error("Foot positions data is empty. Check CPG_Network generation.")
                return
            
            # xyz = np.array([
            #     [self.foot_positions['FL']["x"][self.t_index] , self.foot_positions['FL']["y"][self.t_index] + self.L1, self.foot_positions['FL']["z"][self.t_index]],  # FR
            #     [self.foot_positions['FR']["x"][self.t_index] , self.foot_positions['FR']["y"][self.t_index] - self.L1, self.foot_positions['FR']["z"][self.t_index]],  # FL
            #     [self.foot_positions['RR']["x"][self.t_index] ,  self.foot_positions['RR']["y"][self.t_index] - self.L1, self.foot_positions['RR']["z"][self.t_index]],  # RR
            #     [self.foot_positions['RL']["x"][self.t_index] ,  self.foot_positions['RL']["y"][self.t_index] + self.L1, self.foot_positions['RL']["z"][self.t_index]]   # RL
            # ])
            curr_time = time.time()
            # Initialize a timer for the condition
            if not hasattr(self, 'condition_start_time'):
                self.condition_start_time = None

            if self.command_speed <= 0.6 and self.command_speed > 0.1:
                # Start the timer if not already started
                if self.condition_start_time is None:
                    self.condition_start_time = curr_time

                # Check if 5 seconds have elapsed
                if curr_time - self.condition_start_time <= 20.0:
            
                    self.goal_x = 0.0
                    self.goal_y = self.L1
                    self.goal_z = -0.35
                    self.linear_vel = 0.0005
                    self.xyz[0, 0] = self.move_toward(self.xyz[0, 0], self.goal_x, self.linear_vel)
                    self.xyz[0, 1] = self.move_toward(self.xyz[0, 1] , self.goal_y, self.linear_vel)
                    self.xyz[0, 2] = self.move_toward(self.xyz[0, 2], self.goal_z, self.linear_vel)
                    
                    self.xyz[:, 0] = self.xyz[0, 0]
                    self.xyz[:, 1] = np.array([self.xyz[0, 1], -self.xyz[0, 1], -self.xyz[0, 1] , self.xyz[0, 1] ])
                    self.xyz[:, 2] = self.xyz[0, 2]
            
            else:
                self.condition_start_time = None
                self.goal_x_front = -0.05
                self.goal_x_back = 0.1
                self.x_front_st = self.move_toward(self.x_front_st, self.goal_x_front, self.linear_vel)
                self.x_back_st = self.move_toward(self.x_back_st, self.goal_x_back, self.linear_vel)
                self.xyz = np.array([
                    [self.foot_positions['FL']["x"][self.t_index] -self.x_front_st, self.foot_positions['FL']["y"][self.t_index] + self.L1, self.foot_positions['FL']["z"][self.t_index]],  # FL
                    [self.foot_positions['FR']["x"][self.t_index] -self.x_front_st, self.foot_positions['FR']["y"][self.t_index] - self.L1, self.foot_positions['FR']["z"][self.t_index]],  # FR
                    [self.foot_positions['RR']["x"][self.t_index] -self.x_back_st ,  self.foot_positions['RR']["y"][self.t_index] - self.L1, self.foot_positions['RR']["z"][self.t_index]],  # RR
                    [self.foot_positions['RL']["x"][self.t_index] -self.x_back_st,  self.foot_positions['RL']["y"][self.t_index] + self.L1, self.foot_positions['RL']["z"][self.t_index]]   # RL
                ])
            self.Balance()  # Output is goal_roll, goal_pitch
            # self.angular_Vel = 0.05
            # self.roll = self.move_toward(self.roll, self.goal_roll, self.angular_Vel)
            # self.pitch = self.move_toward(self.pitch, self.goal_pitch, self.angular_Vel)
            # self.yaw = self.move_toward(self.yaw, self.goal_yaw, self.angular_Vel)
            
            self.rot = [self.roll, self.pitch, self.yaw]
            self.t_index += 1
            # if self.command_speed <= 0.1:
            #     self.x = self.foot_positions['FL']["x"][self.t_index] 
            #     self.y = self.foot_positions['FL']["y"][self.t_index]
            #     self.z = self.foot_positions['FL']["z"][self.t_index]
                # Balance for Gait Control
                

        FL, FR, RR, RL = self.robot.leg_IK(xyz=self.xyz, rot=self.rot, is_radians=True)

        self.joint_positions[0:3] = FL
        self.joint_positions[3:6] = FR
        self.joint_positions[6:9] = RR
        self.joint_positions[9:12] = RL

        for i in [0, 3, 6, 9]:  # Hip joints
            self.joint_positions[i] = np.clip(self.joint_positions[i], *self.hip_limits)
        for i in [1, 4, 7, 10]:  # Thigh joints
            self.joint_positions[i] = np.clip(self.joint_positions[i], *self.thigh_limits)
        for i in [2, 5, 8, 11]:  # Calf joints
            self.joint_positions[i] = np.clip(self.joint_positions[i], *self.calf_limits)
        
        # Publish joint states Testing in Rviz
        ##############################################################################
        # Create a JointState message
        # joint_state_msg = JointState()
        # joint_state_msg.header.stamp = self.get_clock().now().to_msg()
        # joint_state_msg.name = self.joint_names
        # joint_state_msg.position = self.joint_positions
        # self.joint_pub.publish(joint_state_msg)

        # Publish joint positions to the controller
        ##############################################################################
        joint_pos_msg = Float64MultiArray()
        joint_pos_msg.data = self.joint_positions
        self.joint_pos_pub.publish(joint_pos_msg)
        # self.get_logger().info(f"t_index: {self.t_index}, xyz: {xyz}, joint_positions: {self.joint_positions}")
        # self.get_logger().info(f"Joint positions: {self.joint_positions}")
        
def main(args=None):
    rclpy.init(args=args)
    dynamo_control = DynamoControl()
    rclpy.spin(dynamo_control)
    dynamo_control.destroy_node()
    rclpy.shutdown()
if __name__ == '__main__':
    main()

    