#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
import math
import threading
from sensor_msgs.msg import JointState 
from sensor_msgs.msg import Joy
from std_msgs.msg import Float64MultiArray
from geometry_msgs.msg import Twist
from sensor_msgs.msg import Imu
import numpy as np
# from QuadrupedModel import QuadrupedModel
# from python_controller.Invesekinematic import InverseKinematic
from python_controller.robot.Invesekinematic import InverseKinematic
# from python_controller.utils import RotMatrix3D
import time

def quaternion_to_euler(w, x, y, z):
    # Roll (x-axis rotation)
    roll = math.atan2(2 * (w * x + y * z), 1 - 2 * (x**2 + y**2))

    # Pitch (y-axis rotation)
    pitch = math.asin(2 * (w * y - z * x))

    # Yaw (z-axis rotation)
    yaw = math.atan2(2 * (w * z + x * y), 1 - 2 * (y**2 + z**2))

    # Convert radians to degrees
    roll = math.degrees(roll)
    pitch = math.degrees(pitch)
    yaw = math.degrees(yaw)

    return roll, pitch, yaw
class PS4Controller(Node):
    def __init__(self):
        super().__init__('ps4_controller')
        self.joy_sub = self.create_subscription(Joy, '/joy', self.joy_callback, 10)
        # self.imu_sub = self.create_subscription(Imu, '/imu', self.imu_callback, 10)
        # # Publisher for joint states
        self.joint_pub = self.create_publisher(JointState, '/joint_states', 10)
        self.joint_pos_pub = self.create_publisher(Float64MultiArray, '/forward_position_controller/commands', 10)
        self.timer = self.create_timer(0.01, self.timer_callback) # Publish joint states at 10 Hz
        # Initialize joint positions
        self.joint_positions = [0.0] * 12
        self.joint_names = [
            "FR_hip_joint", "FR_thigh_joint", "FR_calf_joint",
            "FL_hip_joint", "FL_thigh_joint", "FL_calf_joint",
            "RR_hip_joint", "RR_thigh_joint", "RR_calf_joint",
            "RL_hip_joint", "RL_thigh_joint", "RL_calf_joint"
        ]

        # note: leg IDs
        left_front = 0
        left_back  = 1
        right_front= 2
        right_back = 3
        
        self.right_legs = [right_front, right_back]

        # Parameter of Robots
        self.L = 0.433
        self.W = 0.295
        self.hip_offset = 0.065
        self.L1, self.L2, self.L3 = 0.087, 0.25, 0.25
        self.body = [self.L, self.W] # [length, width]
        self.legs = [0. , 0.087, 0.25, 0.25]
        self.x = 0.0
        self.y = 0.0
        self.z = -0.35 # When strart robot to stand
        self.delta_x = self.body[0] * 0.5
        self.delta_y = self.body[1] * 0.5 + self.legs[1]
        self.x_shift_front = 0.05
        self.x_shift_back = -0.0
        # leg origins (right_f, left_f, right_b , left_b), i.e., the coordinate of j1
        self.leg_origins = np.matrix([[self.L/2, -self.W/2, 0],
                          [self.L/2, self.W/2, 0],
                          [-self.L/2, -self.W/2, 0],
                          [-self.L/2, self.W/2, 0]])
        self.foot_position = np.array([[self.delta_x + self.x_shift_front, self.delta_x + self.x_shift_front,-self.delta_x + self.x_shift_back,-self.delta_x + self.x_shift_back],
                         [-self.delta_y                    ,self.delta_y                     ,-self.delta_y                    , self.delta_y                    ],
                         [self.z                               ,self.z                                  ,self.z                                  ,self.z                                 ]])
        # Define rotation angles

        self.rot = [0.0, 0.0, 0.0] # [roll, pitch, yaw]
        self.rot_matrix = np.eye(4)
        self.inv = InverseKinematic(self.body, self.legs)
        self.rot_feedback = [0.0, 0.0, 0.0]

        # Clipping joint angles
        self.hip_limits = (-1.57, 1.20)
        self.thigh_limits = (-2.0944, 4.71239)
        self.calf_limits = (-2.53, -0.0872665)
        # self.model = QuadrupedModel(self.L, self.W, self.L1, self.L2, self.L3)

        # Step Sizes
        self.MOVE_STEP = 0.01  # Step size for translation
        self.ROTATE_STEP = 5*np.pi/180  # Step size for rotation (5 degrees)
        
        # Limits
        self.TRANS_LIMIT = 1.0  # Limit for x, y, z movement (-0.1 to 0.1)
        self.ROT_LIMIT = 20*np.pi/180  # Limit for roll, pitch, yaw (-20 to 20 degrees)
        self.Z_Max = -0.08  # Limit for z movement
        self.Z_Min = -0.50  # Limit for z movement

        self.debounce_button_rate = 0.3
        self.prev_time = 0
    def move_toward(self, current, target, step):
        if abs(target - current) < step:
            return target
        return current + np.sign(target - current) * step
    
    
    def timer_callback(self):

        # self.get_logger().info(f"xyz = {transformed_xyz[0][0]}")
        self.joint_positions = self.inv.inverse_kinematics(self.foot_position, self.delta_x, self.delta_y+0.05, self.z, self.rot[0], self.rot[1], self.rot[2])
        for i in [0, 3, 6, 9]:  # Hip joints
            self.joint_positions[i] = np.clip(self.joint_positions[i], *self.hip_limits)
        for i in [1, 4, 7, 10]:  # Thigh joints
            self.joint_positions[i] = np.clip(self.joint_positions[i], *self.thigh_limits)
        for i in [2, 5, 8, 11]:  # Calf joints
            self.joint_positions[i] = np.clip(self.joint_positions[i], *self.calf_limits)

        # self.joint_pos_pub.publish(Float64MultiArray(data=self.joint_positions))

        
        self.publish_joint_states()
    
    def joy_callback(self, joy):
        self.axes_list = joy.axes
        self.button_list = joy.buttons
        curr_time = time.time()

        # Translation
        if (self.button_list[0] == 1 and (curr_time - self.prev_time) > self.debounce_button_rate):
            self.x += self.MOVE_STEP
            self.prev_time = curr_time
        if (self.button_list[1] == 1 and (curr_time - self.prev_time) > self.debounce_button_rate):
            self.x -= self.MOVE_STEP
            self.prev_time = curr_time
        if (self.button_list[2] == 1 and (curr_time - self.prev_time) > self.debounce_button_rate):
            self.y += self.MOVE_STEP
            self.prev_time = curr_time
        if (self.button_list[3] == 1 and (curr_time - self.prev_time) > self.debounce_button_rate):
            self.y -= self.MOVE_STEP
            self.prev_time = curr_time
        if (self.button_list[4] == 1 and (curr_time - self.prev_time) > self.debounce_button_rate):
            self.z += self.MOVE_STEP
        if (self.button_list[5] == 1 and (curr_time - self.prev_time) > self.debounce_button_rate):
            self.z -= self.MOVE_STEP
            self.prev_time = curr_time
        
        self.x = np.clip(self.x, -self.TRANS_LIMIT, self.TRANS_LIMIT)
        self.y = np.clip(self.y, -self.TRANS_LIMIT, self.TRANS_LIMIT)
        self.z = np.clip(self.z, self.Z_Min, self.Z_Max)
        # Rotation
        if (self.button_list[6] == 1 and (curr_time - self.prev_time) > self.debounce_button_rate):
            self.rot[0] += self.ROTATE_STEP
            self.prev_time = curr_time
        if (self.button_list[7] == 1 and (curr_time - self.prev_time) > self.debounce_button_rate):
            self.rot[0] -= self.ROTATE_STEP
            self.prev_time = curr_time

        if (self.axes_list[0] >= 0.9 and (curr_time - self.prev_time) > self.debounce_button_rate):
            self.rot[1] += self.ROTATE_STEP
            self.prev_time = curr_time
        if (self.axes_list[0] <= -0.9 and (curr_time - self.prev_time) > self.debounce_button_rate):
            self.rot[1] -= self.ROTATE_STEP 
            self.prev_time = curr_time
        if (self.button_list[11] == 1 and (curr_time - self.prev_time) > self.debounce_button_rate):
            self.rot[2] += self.ROTATE_STEP 
            self.prev_time = curr_time
        if (self.button_list[12] == 1 and (curr_time - self.prev_time) > self.debounce_button_rate):
            self.rot[2] -= self.ROTATE_STEP
            self.prev_time = curr_time

        self.rot[0] = np.clip(self.rot[0], -self.ROT_LIMIT, self.ROT_LIMIT)
        self.rot[1] = np.clip(self.rot[1], -self.ROT_LIMIT, self.ROT_LIMIT)
        self.rot[2] = np.clip(self.rot[2], -self.ROT_LIMIT, self.ROT_LIMIT)
        # self.delta_x = self.x * 0.5
        # self.delta_y = self.y * 0.5 + self.legs[1]
        # self.foot_position = np.array([[self.delta_x + self.x_shift_front,self.delta_x + self.x_shift_front,-self.delta_x + self.x_shift_back,-self.delta_x + self.x_shift_back],
        #                  [-self.delta_y                    ,self.delta_y                     ,-self.delta_y                    , self.delta_y                    ],
        #                  [-0.30                                ,-0.30                                   ,-0.30                                   ,-0.30                                   ]])
        # self.get_logger().info(f"x: {self.x}, y: {self.y}, z: {self.z}, roll: {self.rot[0]}, pitch: {self.rot[1]}, yaw: {self.rot[2]}")

    # def publish_joint_states(self):
    #     """ Publish the joint positions """
    #     joint_state_msg = JointState()
    #     joint_state_msg.header.stamp = self.get_clock().now().to_msg()
    #     joint_state_msg.name = self.joint_names
    #     joint_state_msg.position = self.joint_positions

    #     self.joint_pub.publish(joint_state_msg)

def main():
    rclpy.init()
    ps4 = PS4Controller()
    executor = rclpy.executors.MultiThreadedExecutor()
    executor.add_node(ps4)
    

    executor_thread = threading.Thread(target=executor.spin, daemon=True)
    executor_thread.start()
    rate = ps4.create_rate(2)
    try:
        while rclpy.ok():
            rate.sleep()
    except KeyboardInterrupt:
        pass
    
    rclpy.shutdown()
    executor_thread.join()
    

if __name__ == '__main__':
    main()