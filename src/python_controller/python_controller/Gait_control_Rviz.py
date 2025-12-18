#!/usr/bin/env python3
import rclpy
from rclpy.node import Node 
from sensor_msgs.msg import Joy, JointState
import numpy as np
from python_controller.robot.QuadrupedModel import RobotModel
from python_controller.FootTrajectory.CPG_Network import CPG_Network
import time


class Gait_control(Node):
    def __init__(self):
        super().__init__('Gait_control')

        self.joy_sub = self.create_subscription(Joy, '/joy', self.joy_callback, 5)

        self.joint_pub = self.create_publisher(JointState, '/joint_states', 10)
        self.timer = self.create_timer(0.01, self.control_loop)
        self.timer_command = self.create_timer(0.1, self.command_loop)
        
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
        self.thigh_limits = (-2.0944, 4.71239)
        self.calf_limits = (-2.53, -0.0872665)

        # Initialize Goal positions
        self.goal_x = 0.0
        self.goal_y = 0.0
        self.goal_z = 0.0
        self.goal_roll = 0.0
        self.goal_pitch = 0.0
        self.goal_yaw = 0.0
        self.linear_vel = 0.001
        self.angular_Vel = 0.02
        
        # Current position
        self.cur_x = 0.0
        self.cur_y = 0.0
        self.cur_z = 0.0

        self.x = 0.0
        self.y = 0.0
        self.z = 0.0
        self.roll = 0.0
        self.pitch = 0.0
        self.yaw = 0.0
        self.x_offset = 0.0
        self.z_offset = 0.0
        self.xyz = np.array([
                [self.x , self.y + self.L1, self.z + self.z_offset],  # FL
                [self.x , self.y - self.L1, self.z + self.z_offset],  # FR
                [self.x , self.y - self.L1, self.z + self.z_offset],  # RR
                [self.x , self.y + self.L1, self.z + self.z_offset]   # RL
            ])
        # PS4 Button Filter
        self.speed_x = 0.7
        self.speed_y = 0.7
        self.command_speed = 0.0
        self.command_gait = "stand"
        self.debounce_button_rate = 0.3
        self.prev_time = 0
        self.axis_list = [0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0]
        # CPG Configure
        self.Ts = 1.0
        self.cycles = 5
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

    def joy_callback(self, msg):
        curr_time = time.time()
        self.axis_list = msg.axes
        # Basic joystick command mapping
        if msg.buttons[0]:  # e.g., X button
            self.command_gait = "stand"
        elif msg.buttons[1]:    # O button
            self.command_gait = "sit" 
        elif msg.buttons[2]:  # e.g., Triangle button 
            self.command_gait = "walk"
        elif msg.buttons[3]:    # Square button
            self.command_gait = "trot"
        
        # Speed with trigger or joystick axis (example)
        
        elif msg.buttons[4] == 1 and (curr_time - self.prev_time) > self.debounce_button_rate:
            self.speed_x += 0.1
            self.speed_x = np.clip(self.speed_x , 0.0, 1.2)
            self.prev_time = curr_time
        
        elif msg.buttons[5] == 1 and (curr_time - self.prev_time) > self.debounce_button_rate:
            self.speed_x -= 0.1
            self.speed_x = np.clip(self.speed_x , 0.0, 1.2)
            self.prev_time = curr_time
        
        elif msg.buttons[6] == 1 and (curr_time - self.prev_time) > self.debounce_button_rate:
            self.speed_y += 0.1
            self.speed_y = np.clip(self.speed_y , 0.0, 1.2)
            self.prev_time = curr_time
        
        elif msg.buttons[7] == 1 and (curr_time - self.prev_time) > self.debounce_button_rate:
            self.speed_y -= 0.1
            self.speed_y = np.clip(self.speed_y , 0.0, 1.2)
            self.prev_time = curr_time
        
        
        # self.Ts, self.step_length, self.step_height = self.calculate_gait_parameters()
        self.get_logger().info(f"Command gait: {self.command_gait}, Command speed: {self.command_speed}, move_direction: {self.move_direction}")
    def Q_update(self):
        self.Ts, self.step_length, self.step_height = self.calculate_gait_parameters()
        self.cpg = CPG_Network(Ts=self.Ts, cycles=self.cycles, trajectory='bezier')
        self.foot_positions, self.Q_history = self.cpg.generate(gait_type=self.command_gait, step_length=self.step_length, step_height=self.step_height, z=-0.35, move_direction=self.move_direction)
        self.get_logger().info(f"Ts: {self.Ts}, step_length: {self.step_length}, step_height: {self.step_height}")

    def joint_callback(self, msg):
        for i in range(len(msg.name)):
            if msg.name[i] in self.joint_names:
                self.joint_feedback[self.joint_names.index(msg.name[i])] = msg.position[i]
        self.get_logger().info(f"Joint feedback: {self.joint_feedback}")

    def command_loop(self):
        if self.command_gait == "walk":
            if self.axis_list[1] >= 0.5 :
                self.move_direction = "forward"
                self.command_speed = abs(self.axis_list[1])*self.speed_x
            elif self.axis_list[1] <= -0.5:
                self.move_direction = "backward"
                self.command_speed = abs(self.axis_list[1])*self.speed_x
            elif self.axis_list[0] >= 0.5:
                self.move_direction = "left"
                self.command_speed = abs(self.axis_list[0])*self.speed_y
            elif self.axis_list[0] <= -0.5:
                self.move_direction = "right"
                self.command_speed = abs(self.axis_list[0])*self.speed_y
            else :      
                self.move_direction = "stop"
                self.command_speed = 0.0
        # Speed with trigger or joystick axis (example)
        elif self.command_gait == "stand" or self.command_gait == "sit":
            self.goal_x = self.axis_list[1]*self.speed_x*0.1
            self.goal_y = self.axis_list[0]*self.speed_y*0.1
            self.goal_z = self.axis_list[4]*0.08
            self.goal_roll = self.axis_list[7]*0.1
            self.goal_pitch = self.axis_list[6]*0.1
            self.goal_yaw = self.axis_list[3]*0.1
        self.get_logger().info(f"Command speed: {self.command_speed}")
    def control_loop(self):
        joint_state_msg = JointState()
        joint_state_msg.header.stamp = self.get_clock().now().to_msg()
        joint_state_msg.name = self.joint_names
        

        if self.command_gait == "stand":
            goal_z_stand = -0.35
            self.z_offset = self.move_toward(self.z_offset, goal_z_stand, self.linear_vel)
            self.x = self.move_toward(self.x, self.goal_x, self.linear_vel)
            self.y = self.move_toward(self.y, self.goal_y, self.linear_vel)
            self.z = self.move_toward(self.z, self.goal_z, self.linear_vel)
            self.roll = self.move_toward(self.roll, self.goal_roll, self.angular_Vel)
            self.pitch = self.move_toward(self.pitch, self.goal_pitch, self.angular_Vel)
            self.yaw = self.move_toward(self.yaw, self.goal_yaw, self.angular_Vel)

            self.xyz = np.array([
                [self.x , self.y + self.L1, self.z + self.z_offset],  # FL
                [self.x , self.y - self.L1, self.z + self.z_offset],  # FR
                [self.x , self.y - self.L1, self.z + self.z_offset],  # RR
                [self.x , self.y + self.L1, self.z + self.z_offset]   # RL
            ])
        
            self.get_logger().info(f"x: {self.x}, y: {self.y}, z: {self.z_offset+self.z}, roll: {self.roll}, pitch: {self.pitch}, yaw: {self.yaw}")
            self.rot = [self.roll, self.pitch, self.yaw]

        elif self.command_gait == "sit":
            goal_x_sit, goal_z_sit = -0.1, -0.25
            self.x_offset = self.move_toward(self.x_offset, goal_x_sit, self.linear_vel)
            self.z_offset = self.move_toward(self.z_offset, goal_z_sit, self.linear_vel)
            self.x = self.move_toward(self.x, self.goal_x, self.linear_vel)
            self.y = self.move_toward(self.y, self.goal_y, self.linear_vel)
            self.z = self.move_toward(self.z, self.goal_z, self.linear_vel)
            self.roll = self.move_toward(self.roll, self.goal_roll, self.angular_Vel)
            self.pitch = self.move_toward(self.pitch, self.goal_pitch, self.angular_Vel)
            self.yaw = self.move_toward(self.yaw, self.goal_yaw, self.angular_Vel)

            self.xyz = np.array([
                [self.x + self.x_offset, self.y + self.L1, self.z + self.z_offset],  # FL
                [self.x + self.x_offset, self.y - self.L1, self.z + self.z_offset],  # FR
                [self.x - self.x_offset*-1,  self.y - self.L1, self.z + self.z_offset],  # RR
                [self.x - self.x_offset*-1,  self.y + self.L1, self.z + self.z_offset]   # RL
            ])

        else:
            self.Q_update()
            max_index = len(self.Q_history[0]) - 1
            if self.t_index > max_index:
                self.t_index = 0  # Loop animation
            if len(self.foot_positions['FL']["x"]) == 0 or len(self.foot_positions['FL']["y"]) == 0 or len(self.foot_positions['FL']["z"]) == 0:
                self.get_logger().error("Foot positions data is empty. Check CPG_Network generation.")
                return
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
            # if self.command_speed <= 0.5 and self.command_speed > 0.1 and (curr_time - self.prev_time_command) > 0.5:
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
                self.xyz = np.array([
                    [self.foot_positions['FL']["x"][self.t_index] + self.x_offset, self.foot_positions['FL']["y"][self.t_index] + self.L1, self.foot_positions['FL']["z"][self.t_index]],  # FL
                    [self.foot_positions['FR']["x"][self.t_index] + self.x_offset, self.foot_positions['FR']["y"][self.t_index] - self.L1, self.foot_positions['FR']["z"][self.t_index]],  # FR
                    [self.foot_positions['RR']["x"][self.t_index] - self.x_offset*-1,  self.foot_positions['RR']["y"][self.t_index] - self.L1, self.foot_positions['RR']["z"][self.t_index]],  # RR
                    [self.foot_positions['RL']["x"][self.t_index] - self.x_offset*-1,  self.foot_positions['RL']["y"][self.t_index] + self.L1, self.foot_positions['RL']["z"][self.t_index]]   # RL
                ])

            self.roll = self.move_toward(self.roll, self.goal_roll, self.angular_Vel)
            self.pitch = self.move_toward(self.pitch, self.goal_pitch, self.angular_Vel)
            self.yaw = self.move_toward(self.yaw, self.goal_yaw, self.angular_Vel)

            self.rot = [self.roll, self.pitch, self.yaw]
        
        # Inverse Kinematics
        FL, FR, RR, RL = self.dynamo_one.leg_IK(xyz=self.xyz, rot=[self.goal_roll, self.goal_pitch, self.goal_yaw], is_radians=True)
        
        # Set joint positions
        self.joint_positions[0:3] = FL
        self.joint_positions[3:6] = FR
        self.joint_positions[6:9] = RR
        self.joint_positions[9:12] = RL

        # joint_feedback = np.array([[self.joint_feedback[0:3]], [self.joint_feedback[3:6]], [self.joint_feedback[6:9]], [self.joint_feedback[9:12]]])
        # error = self.dynamo_one.Calculate_error(xyz, angles=joint_feedback, rot=self.rot, center_offset=[0,0,0])
        # self.get_logger().info(f"error: {error}")
        joint_state_msg.position = self.joint_positions
        self.joint_pub.publish(joint_state_msg)
        self.t_index += 1

        self.get_logger().info("Gait: %s, Speed: %.2f" % (self.command_gait, self.command_speed))

def main(args=None):
    rclpy.init(args=args)
    joint_foot_publisher = Gait_control()
    rclpy.spin(joint_foot_publisher)
    joint_foot_publisher.destroy_node()
    rclpy.shutdown()
    

if __name__ == '__main__':
    main()



