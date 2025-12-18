#!/usr/bin/env python3
import rclpy
from rclpy.node import Node 
from sensor_msgs.msg import Joy, JointState, Imu
from geometry_msgs.msg import Twist
from std_msgs.msg import String, Float64MultiArray
import numpy as np
from python_controller.robot.QuadrupedModel import RobotModel
from python_controller.FootTrajectory.CPG_Network import CPG_Network
import time


class Gait_control(Node):
    def __init__(self):
        super().__init__('Gait_control')

        self.joint_pos_pub = self.create_publisher(Float64MultiArray, '/forward_position_controller/commands', 10)
        self.position_sub = self.create_subscription(Twist, 'dynamo_one/cmd_vel', self.position_callback, 10)
        self.Mode_sub = self.create_subscription(String, 'dynamo_one/mode', self.mode_callback, 10)
        # self.imu_sub = self.create_subscription(Imu, 'dynamo_one/imu', self.imu_callback, 10)
        self.timer = self.create_timer(0.01, self.control_loop)
        
        self.joint_positions = [0.0] * 12
        self.joint_feedback = [0.0] * 12
        self.joint_prev = [0.0] * 12
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

        
        # Define initial position
        self.rot = [0.0, 0.0, 0.0] # [roll, pitch, yaw]
        self.rot_matrix = np.eye(4)
        
        # Clipping joint angles to limits
        self.hip_limits = (-1.57, 1.20)
        self.thigh_limits = (-3.14, 3.14)
        self.calf_limits = (-2.60, 0.0)

        # Initialize Goal positions
        self.x = 0.0
        self.y = 0.0
        self.z = 0.0
        self.roll = 0.0
        self.pitch = 0.0
        self.yaw = 0.0
        self.robot_height = 0.35
        
        self.xyz = np.array([
                [self.x , self.y + self.L1, self.z ],  # FL
                [self.x , self.y - self.L1, self.z ],  # FR
                [self.x , self.y - self.L1, self.z ],  # RR
                [self.x , self.y + self.L1, self.z ]   # RL
            ])
        # PS4 Button Filter
        self.K_a = 0.3
        self.command_speed = 0.0
        self.command_gait = None

        # CPG Configure
        self.Ts = 1.0
        self.cycles = 1
        self.step = int(self.cycles * self.Ts / self.dt)
        self.min_Ts, self.max_Ts = 0.3, 1.5
        self.min_step, self.max_step = 0.05, 0.15
        self.step_length = 0
        self.step_height = 0
        self.move_direction = "forward"
        self.LEG_LABELS = ["FL", "FR", "RR", "RL"] 
        # Generate foot positions
        self.foot_positions = {label: {"x": [], "y": [], "z": []} for label in self.LEG_LABELS}
        self.leg_map = {0: "FL", 2: "FR", 4: "RR", 6: "RL"} 
        self.prev_time_command = 0
        self.t_index = 0
        self.Q_update()
    def calculate_gait_parameters(self):
        
        if self.command_gait == "walk":
            self.min_Ts = 0.3
            self.max_Ts = 0.6
            self.min_step = 0.03
            self.max_step = 0.06
            self.step_height = 0.04
            
        elif self.command_gait == "trot" :
            self.min_Ts = 0.2
            self.max_Ts = 0.5
            self.min_step = 0.04
            self.max_step = 0.07
            self.step_height = 0.04
            # Cap and normalize command speed
        elif self.command_gait == "bound" or self.command_gait == "pace":
            self.min_Ts = 0.4
            self.max_Ts = 0.7
            self.min_step = 0.06
            self.max_step = 0.1
            self.step_height = 0.08
        speed_max = 1.2
        speed = max(min(self.command_speed, speed_max), 0.01)  # avoid zero division
        # speed_ratio = speed / speed_max

        # Interpolate stride period and step length
        
        if self.command_speed <= 0.1:
            Ts = self.min_Ts -0.15
            step_length = 0.0
            step_height = 0.05
        else:
            Ts = self.max_Ts - (self.max_Ts - self.min_Ts) * speed
            step_length = self.min_step + (self.max_step - self.min_step) * speed
            step_height = self.step_height
        self.step = int(self.cycles * Ts / self.dt)
        return Ts, step_length, step_height

    def Q_update(self):
        self.cpg = CPG_Network(cycles=self.cycles, trajectory='bezier')
        self.Ts, self.step_length, self.step_height = self.calculate_gait_parameters()
        self.foot_positions, self.Q_history = self.cpg.generate(gait_type=self.command_gait, Ts=self.Ts, step_length=self.step_length, step_height=self.step_height, z=0.0, move_direction=self.move_direction)
        self.get_logger().info(f"Ts: {self.Ts}, step_length: {self.step_length}, step_height: {self.step_height}")

    def position_callback(self, msg):
        # self.axis_list = msg.linear.x, msg.linear.y, msg.linear.z, msg.angular.x, msg.angular.y, msg.angular.z, msg.angular.x, msg.angular.y
        self.x = msg.linear.x
        self.y = msg.linear.y
        self.z = msg.linear.z
        self.roll = msg.angular.x
        self.pitch = msg.angular.y
        self.yaw = msg.angular.z

        self.robot_height = self.z
        
    def mode_callback(self, msg):
        self.command_gait = msg.data
        if self.command_gait == "stand":
            self.get_logger().info("Stand mode activated")
        elif self.command_gait == "sit":
            self.get_logger().info("Sit mode activated")
        elif self.command_gait == "walk":
            self.get_logger().info("Walk mode activated")
        elif self.command_gait == "trot":
            self.get_logger().info("Trot mode activated")
        elif self.command_gait == "bound":
            self.get_logger().info("Bound mode activated")
        elif self.command_gait == "pace":
            self.get_logger().info("Pace mode activated")
        else:
            self.get_logger().error(f"Unknown mode: {self.command_gait}")
        
    def joint_callback(self, msg):
        for i in range(len(msg.name)):
            if msg.name[i] in self.joint_names:
                self.joint_feedback[self.joint_names.index(msg.name[i])] = msg.position[i]
        self.get_logger().info(f"Joint feedback: {self.joint_feedback}")

    def control_loop(self):
        
        if self.command_gait == "stand" or self.command_gait == "sit":
            
            self.xyz = np.array([
                [-1*self.x , -1*self.y + self.L1, -1*self.z ],  # FL
                [-1*self.x , -1*self.y - self.L1, -1*self.z ],  # FR
                [-1*self.x , -1*self.y - self.L1, -1*self.z ],  # RR
                [-1*self.x , -1*self.y + self.L1, -1*self.z ]   # RL
            ])
        
            self.get_logger().info(f"x: {self.x}, y: {self.y}, z: {self.z}, roll: {self.roll}, pitch: {self.pitch}, yaw: {self.yaw}")
            self.rot = [self.roll, self.pitch, self.yaw]

        elif self.command_gait == "walk" or self.command_gait == "trot" or self.command_gait == "gallop" or self.command_gait == "pace":
            if self.x >= 0.2 and abs(self.y) < 0.2:
                self.move_direction = "forward"
            elif self.x <= -0.1 and abs(self.y) < 0.3:
                self.move_direction = "backward"
            elif self.y >= 0.2 and abs(self.x) < 0.2:
                self.move_direction = "left"
            elif self.y <= -0.2 and abs(self.x) < 0.2:
                self.move_direction = "right"
            elif self.x >= 0.2 and self.y >= 0.2:
                self.move_direction = "forward_left"
            elif self.x >= 0.2 and self.y <= -0.2:
                self.move_direction = "forward_right"
            # elif self.x <= -0.1 and self.y >= 0.1:
            #     self.move_direction = "backward_left"
            # elif self.x <= -0.1 and self.y <= -0.1:
            #     self.move_direction = "backward_right"
            else :      
                self.move_direction = "stop"

            
            self.command_speed = np.sqrt((abs(self.x)+0.7)**2 + (abs(self.y)+0.7)**2 + (self.K_a+self.yaw)**2)
            # self.get_logger().info(f"Move Direction: {self.move_direction}, Command Speed: {self.command_speed}")
            
            self.Q_update()
            max_index = len(self.foot_positions['FL']["x"]) - 1
            if self.t_index > max_index:
                self.t_index = 0  # Loop animation
            if len(self.foot_positions['FL']["x"]) == 0 or len(self.foot_positions['FL']["y"]) == 0 or len(self.foot_positions['FL']["z"]) == 0:
                self.get_logger().error("Foot positions data is empty. Check CPG_Network generation.")
                return
            self.xyz = np.array([
                    [self.foot_positions['FL']["x"][self.t_index] , self.foot_positions['FL']["y"][self.t_index] + self.L1, self.foot_positions['FL']["z"][self.t_index] - self.robot_height],  # FL
                    [self.foot_positions['FR']["x"][self.t_index] , self.foot_positions['FR']["y"][self.t_index] - self.L1, self.foot_positions['FR']["z"][self.t_index] - self.robot_height],  # FR
                    [self.foot_positions['RR']["x"][self.t_index] ,  self.foot_positions['RR']["y"][self.t_index] - self.L1, self.foot_positions['RR']["z"][self.t_index] - self.robot_height],  # RR
                    [self.foot_positions['RL']["x"][self.t_index] ,  self.foot_positions['RL']["y"][self.t_index] + self.L1, self.foot_positions['RL']["z"][self.t_index] - self.robot_height]   # RL
                ])
            self.get_logger().info(f"Foot Positions: {self.xyz}")
            self.rot = [0.0, 0.0, self.yaw]
            

        # Inverse Kinematics
        FL, FR, RR, RL = self.dynamo_one.leg_IK(xyz=self.xyz, rot=[self.roll, self.pitch, self.yaw], is_radians=True)
        
        # Set joint positions
        self.joint_positions[0:3] = FL
        self.joint_positions[3:6] = FR
        self.joint_positions[6:9] = RR
        self.joint_positions[9:12] = RL

        # Set positions with safety checks
        for i in range(12):  # More concise than separate loops
            if i in [0, 3, 6, 9]:
                limits = self.hip_limits
            elif i in [1, 4, 7, 10]:
                limits = self.thigh_limits
            else:
                limits = self.calf_limits
            
            # Ensure scalar value and proper clipping
            self.joint_positions[i] = float(np.clip(self.joint_positions[i], *limits))
            
            if np.isnan(self.joint_positions[i]):  # Simple check for scalar
                print(f"WARNING: NaN at joint {i}. Resetting to zero.")
                self.joint_positions[i] = 0.0  # Single float, not [0.0]

        joint_pos_msg = Float64MultiArray()
        joint_pos_msg.data = list(map(float, self.joint_positions))
        self.joint_pos_pub.publish(joint_pos_msg)
        self.t_index = (self.t_index + 1) % self.step
        self.get_logger().info("Gait: %s, Speed: %.2f" % (self.command_gait, self.command_speed))

def main(args=None):
    rclpy.init(args=args)
    joint_foot_publisher = Gait_control()
    rclpy.spin(joint_foot_publisher)
    joint_foot_publisher.destroy_node()
    rclpy.shutdown()
    

if __name__ == '__main__':
    main()



