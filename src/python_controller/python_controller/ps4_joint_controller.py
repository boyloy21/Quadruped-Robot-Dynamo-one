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
from python_controller.robot.QuadrupedModel import RobotModel
from python_controller.robot.Invesekinematic import InverseKinematic
from python_controller.robot.utils import RotMatrix3D
from python_controller.Controller.PID_controller import PID_controller
import time

def quaternion_to_euler(w, x, y, z):
    # Roll (x-axis rotation)
    roll = math.atan2(2 * (w * x + y * z), 1 - 2 * (x**2 + y**2))

    # Pitch (y-axis rotation)
    pitch = math.asin(2 * (w * y - z * x))

    # Yaw (z-axis rotation)
    yaw = math.atan2(2 * (w * z + x * y), 1 - 2 * (y**2 + z**2))

    # Convert radians to degrees
    # roll = math.degrees(roll)
    # pitch = math.degrees(pitch)
    # yaw = math.degrees(yaw)

    return roll, pitch, yaw

    
class PS4Controller(Node):
    def __init__(self):
        super().__init__('ps4_controller')
        # Subscribers
        self.joy_sub = self.create_subscription(Joy, '/joy', self.joy_callback, 10)
        # self.joint_state_sub = self.create_subscription(JointState, '/joint_states', self.joint_state_callback, 10)
        self.imu_sub = self.create_subscription(Imu, 'dynamo_one/imu', self.imu_callback, 10)
        # self.rot_pub = self.create_publisher()

        # # Publisher for joint states
        # self.joint_pub = self.create_publisher(JointState, '/joint_states', 10)
        
        self.joint_pos_pub = self.create_publisher(Float64MultiArray, '/forward_position_controller/commands', 10)
        self.timer = self.create_timer(0.01, self.timer_callback) # Publish joint states at 10 Hz
        # self.roll_pitch_pub = self.create_publisher(Float64MultiArray, '/roll_pitch', 10)
        self.rot_error_pub = self.create_publisher(Float64MultiArray, '/error_roll_pitch', 10)
        # self.pid_controller = PID(0.75, 2.29, 0.0)
        # PID parameters
        # self.roll_pid = PID(kp=2.0, ki=0.0, kd=0.05)
        # self.pitch_pid = PID(kp=2.0, ki=0.0, kd=0.05)

        # Initialize joint positions and names
        self.joint_positions = [0.0] * 12
        self.joint_names = [
            "FL_hip_joint", "FL_thigh_joint", "FL_calf_joint",
            "FR_hip_joint", "FR_thigh_joint", "FR_calf_joint",
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
        # Robot parameters 
        self.L = 0.433
        self.W = 0.295
        self.L1, self.L2, self.L3 = 0.087, 0.25, 0.25
        self.robot = RobotModel(self.L, self.W, self.L1, self.L2, self.L3)

        
        # Initial position
        self.goal_front_st = 0.15
        self.goal_back_st = -0.15
        self.goal_x = 0.0 # Target position
        self.goal_y = 0.0
        self.goal_z = -0.15
        self.goal_roll = 0.0
        self.goal_pitch = 0.0
        self.goal_yaw = 0.0

        # Current position
        self.x_back_st = 0.0  # 0.05
        self.x_front_st = -0.0 # 0.05
        self.x = 0.0  # Current position
        self.y = 0.0
        self.z = -0.01
        self.desired_roll = 0.0
        self.desired_pitch = 0.0

        # PID parameters
        self.step_rate = 0.01  # Adjust for speed (lower = slower)
        self.position_tolerance = 0.001  # Threshold to stop moving

        # leg origins (right_f, left_f, right_b , left_b), i.e., the coordinate of j1
        self.leg_origins = np.matrix([[self.L/2, -self.W/2, 0],
                          [self.L/2, self.W/2, 0],
                          [-self.L/2, -self.W/2, 0],
                          [-self.L/2, self.W/2, 0]])
        
        # Define rotation angles
        self.rot = [0.0, 0.0, 0.0] # [roll, pitch, yaw]
        self.rot_matrix = np.eye(4)
        # self.inv = InverseKinematic(self.L1, self.L2, self.L3)
        self.current_roll, self.current_pitch = 0.0, 0.0
        self.rot_feedback = [0.0, 0.0, 0.0]

        # Initialize PID controller
        self.pid = PID_controller(0.7, 1.5, 0.0)
        self.pid.reset()
        self.roll_feedback = 0.0
        self.pitch_feedback = 0.0
        self.linear_vel = 0.001
        self.angular_Vel = 0.02

        # Clipping joint angles
        self.hip_limits = (-1.57, 1.20)
        self.thigh_limits = (-3.14, 3.14)
        self.calf_limits = (-2.53, 0)
        # self.model = QuadrupedModel(self.L, self.W, self.L1, self.L2, self.L3)

        # Step Sizes
        self.MOVE_STEP = 0.03  # Step size for translation
        self.ROTATE_STEP = 5*np.pi/180  # Step size for rotation (5 degrees)
        
        # Limits
        self.TRANS_LIMIT = 0.1  # Limit for x, y, z movement (-0.1 to 0.1)
        self.ROT_LIMIT = 20*np.pi/180  # Limit for roll, pitch, yaw (-20 to 20 degrees)
        self.Z_Max = -0.00  # Limit for z movement
        self.Z_Min = -0.50  # Limit for z movement

        self.debounce_button_rate = 0.3
        self.prev_time = 0
        self.dt = 0.01
        self.use_imu = False

        # Speed control joint
        self.vel = 0.001
        self.angular = 0.01

        # Current foot positions
        self.current_foot_positions = np.zeros((4, 3))  # [FR, FL, RR, RL]

    def move_toward(self, current, target, step):
        if abs(target - current) < step:
            return target
        return current + np.sign(target - current) * step
    # def joint_state_callback(self, msg):
    #     """Callback to handle joint state feedback."""
    #     # Update joint positions based on joint names
    #     for i, name in enumerate(msg.name):
    #         if name in self.joint_names:
    #             index = self.joint_names.index(name)
    #             self.joint_positions[index] = msg.position[i]

    #     # Calculate current foot positions using forward kinematics
    #     self.calculate_foot_positions()
    # def calculate_foot_positions(self):
    #     """Calculate the current foot positions using forward kinematics."""
    #     for i in range(4):  # Iterate over each leg
    #         hip_angle = self.joint_positions[i * 3]
    #         thigh_angle = self.joint_positions[i * 3 + 1]
    #         calf_angle = self.joint_positions[i * 3 + 2]

    #         # Forward kinematics for each leg
    #         x, y, z = self.inv.ForwardKinematic4(hip_angle, thigh_angle, calf_angle, right=(i % 2 == 0))
    #         self.current_foot_positions[i] = [x, y, z]

    #     # Log the current foot positions
    #     self.get_logger().info(f"Current Foot Positions: {self.current_foot_positions}")

    def imu_callback(self, imu):
        
        quaternion = [imu.orientation.x, imu.orientation.y, imu.orientation.z, imu.orientation.w]
        # Convert quaternion to Euler angles
        roll, pitch, yaw = quaternion_to_euler(*quaternion)
        self.current_roll = roll
        self.current_pitch = pitch
        
        # Convert to radians
        # self.rot_feedback = [math.radians(angle) for angle in self.rot_feedback]

        self.get_logger().info(f"roll:  {roll}, pitch:  {pitch}, yaw: {yaw}")

    def timer_callback(self):
        
        # Calculate Roll and Pitch errors
        if self.use_imu:
            compensation = self.pid.run(self.current_roll, self.current_pitch)
            self.goal_roll = -compensation[0]
            self.goal_pitch = -compensation[1]
            # self.goal_yaw = 0.0
            # roll_error = self.desired_roll - self.current_roll
            # pitch_error = self.desired_pitch - self.current_pitch

            # roll_output = self.roll_pid.compute(roll_error, self.dt)
            # pitch_output = self.pitch_pid.compute(pitch_error, self.dt)

            # # Apply PID corrections to foot z-coordinates
            # z_correction_front = pitch_output / 2.0
            # z_correction_back = -pitch_output / 2.0
            # z_correction_left = roll_output / 2.0
            # z_correction_right = -roll_output / 2.0

            # z_correction_front = np.clip(z_correction_front, -0.03, 0.03)
            # z_correction_back  = np.clip(z_correction_back, -0.03, 0.03)
            # z_correction_right = np.clip(z_correction_right, -0.03, 0.03)
            # z_correction_left  = np.clip(z_correction_left, -0.03, 0.03)
            self.get_logger().info("USE Balance")
        else:
            self.get_logger().info("NO Balance")
            # self.goal_roll = 0.0
            # self.goal_pitch = 0.0
            # self.goal_yaw = 0.0
        # Smoothly move toward goal positions
        self.x = self.move_toward(self.x, self.goal_x, self.vel)
        self.y = self.move_toward(self.y, self.goal_y, self.vel)
        self.z = self.move_toward(self.z, self.goal_z, self.vel)
        self.x_front_st = self.move_toward(self.x_front_st, self.goal_front_st, self.vel)
        self.x_back_st = self.move_toward(self.x_back_st, self.goal_back_st, self.vel)
        self.rot[0] = self.move_toward(self.rot[0], self.goal_roll, self.angular)
        self.rot[1] = self.move_toward(self.rot[1], self.goal_pitch, self.angular)
        self.rot[2] = self.move_toward(self.rot[2], self.goal_yaw, self.angular)
        # self.rot= [self.goal_roll, self.goal_pitch, self.goal_yaw]

        # xyz = np.array([[self.x, self.y, self.z]] * 4)
        # FL, FR, RR, RL
        xyz = np.array([[self.x + self.x_front_st, self.y + self.L1, self.z ], 
                        [self.x + self.x_front_st, self.y - self.L1, self.z ], 
                        [self.x - self.x_back_st, self.y - self.L1, self.z ], 
                        [self.x - self.x_back_st, self.y + self.L1, self.z ]])
        
        
       
        FL, FR, RR, RL = self.robot.leg_IK(xyz, rot=self.rot, is_radians=True)

        self.joint_positions[0:3] = FL
        self.joint_positions[3:6] = FR
        self.joint_positions[6:9] = RR
        self.joint_positions[9:12] = RL
        # Apply limits to joint angles
        for i in [0, 3, 6, 9]:  # Hip joints
            self.joint_positions[i] = np.clip(self.joint_positions[i], *self.hip_limits)
        for i in [1, 4, 7, 10]:  # Thigh joints
            self.joint_positions[i] = np.clip(self.joint_positions[i], *self.thigh_limits)
        for i in [2, 5, 8, 11]:  # Calf joints
            self.joint_positions[i] = np.clip(self.joint_positions[i], *self.calf_limits)
        
        error = Float64MultiArray()
        error.data = [0.0 - self.goal_roll, 0.0 - self.goal_pitch]
        self.rot_error_pub.publish(error)
        # Smoothly interpolate current positions toward target positions
        joint_pos_msg = Float64MultiArray()
        joint_pos_msg.data = self.joint_positions
        self.joint_pos_pub.publish(joint_pos_msg)
        
    
    def joy_callback(self, joy):
        self.axes_list = joy.axes
        self.button_list = joy.buttons
        # curr_time = self.get_clock().now().to_msg().sec
        curr_time = time.time()

        # Translation
        # Set goal positions (not direct increments)
        if (self.button_list[0] == 1 and (curr_time - self.prev_time) > self.debounce_button_rate):
            self.goal_x += self.MOVE_STEP
            self.prev_time = curr_time
            # self.use_imu = True
            # self.prev_time = curr_time
        elif (self.button_list[1] == 1 and (curr_time - self.prev_time) > self.debounce_button_rate):
            self.goal_x -= self.MOVE_STEP
            self.prev_time = curr_time
            # self.use_imu = False
            # self.prev_time = curr_time

        elif (self.button_list[2] == 1 and (curr_time - self.prev_time) > self.debounce_button_rate):
            self.goal_y += self.MOVE_STEP
            self.prev_time = curr_time
        elif (self.button_list[3] == 1 and (curr_time - self.prev_time) > self.debounce_button_rate):
            self.goal_y -= self.MOVE_STEP
            self.prev_time = curr_time

        elif (self.button_list[4] == 1 and (curr_time - self.prev_time) > self.debounce_button_rate):
            self.goal_z += self.MOVE_STEP
            self.prev_time = curr_time
        elif (self.button_list[5] == 1 and (curr_time - self.prev_time) > self.debounce_button_rate):
            self.goal_z -= self.MOVE_STEP
            self.prev_time = curr_time

        # Clamp goals to limits
        
        # Rotation
        if (self.button_list[6] == 1 and (curr_time - self.prev_time) > self.debounce_button_rate):
            self.goal_roll += self.ROTATE_STEP
            self.prev_time = curr_time
        if (self.button_list[7] == 1 and (curr_time - self.prev_time) > self.debounce_button_rate):
            self.goal_roll -= self.ROTATE_STEP
            self.prev_time = curr_time
        
        if (self.axes_list[0] >= 0.9 and (curr_time - self.prev_time) > self.debounce_button_rate):
            self.goal_yaw += self.ROTATE_STEP
            self.prev_time = curr_time
        if (self.axes_list[0] <= -0.9 and (curr_time - self.prev_time) > self.debounce_button_rate):
            self.goal_yaw -= self.ROTATE_STEP 
            self.prev_time = curr_time

        # When use IMU or not
        if (self.axes_list[1] >= 0.9 and (curr_time - self.prev_time) > self.debounce_button_rate):
            self.use_imu = True
            self.prev_time = curr_time
        if (self.axes_list[1] <= -0.9 and (curr_time - self.prev_time) > self.debounce_button_rate):
            self.use_imu = False
            self.prev_time = curr_time

        if (self.button_list[11] == 1 and (curr_time - self.prev_time) > self.debounce_button_rate):
            self.goal_pitch += self.ROTATE_STEP 
            self.prev_time = curr_time
        if (self.button_list[12] == 1 and (curr_time - self.prev_time) > self.debounce_button_rate):
            self.goal_pitch -= self.ROTATE_STEP
            self.prev_time = curr_time

        self.goal_roll = np.clip(self.goal_roll, -self.ROT_LIMIT, self.ROT_LIMIT)
        self.goal_pitch = np.clip(self.goal_pitch, -self.ROT_LIMIT, self.ROT_LIMIT)
        self.goal_yaw = np.clip(self.goal_yaw, -self.ROT_LIMIT, self.ROT_LIMIT)

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