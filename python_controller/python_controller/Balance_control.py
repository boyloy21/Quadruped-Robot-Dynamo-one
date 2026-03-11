#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
import math
import threading
from sensor_msgs.msg import JointState 
from sensor_msgs.msg import Joy
from std_msgs.msg import Float64MultiArray
from sensor_msgs.msg import Imu
from geometry_msgs.msg import Twist
import numpy as np
# from QuadrupedModel import QuadrupedModel
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
class Balance_control(Node):
    def __init__(self):
        super().__init__('balance_control')
        self.joy_sub = self.create_subscription(Joy, '/joy', self.joy_callback, 10)
        self.imu_sub = self.create_subscription(Imu, '/imu', self.imu_callback, 10)

        # # Publisher for joint states
       
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
        self.x = 0.0
        self.y = 0.0
        self.z = -0.35 # When strart robot to stand

        # leg origins (right_f, left_f, right_b , left_b), i.e., the coordinate of j1
        self.leg_origins = np.matrix([[self.L/2, -self.W/2, 0],
                          [self.L/2, self.W/2, 0],
                          [-self.L/2, -self.W/2, 0],
                          [-self.L/2, self.W/2, 0]])
        
        # Define rotation angles
        self.rot = [0.0, 0.0, 0.0] # [roll, pitch, yaw]
        self.rot_matrix = np.eye(4)
        self.rot_mesurement = [0.0, 0.0, 0.0]
        self.inv = InverseKinematic(self.L1, self.L2, self.L3)
        self.dt = 0.01
        # PID Controllers for Roll and Pitch
        roll_output_max = 0.68  # ±0.1 m / (W/2)
        pitch_output_max = 0.46  # ±0.1 m / (L/2)
        self.pitch_controller = PID_controller(
            Kp=5.0, Ki=0.0, Kd=0.01, dt=self.dt,
            output_min=-pitch_output_max, output_max=pitch_output_max
        )
        self.roll_controller = PID_controller(
            Kp=5.0, Ki=0.0, Kd=0.01, dt=self.dt,
            output_min=-roll_output_max, output_max=roll_output_max
        )
        
        # Initialize error histories (at least 2 values for [-1] and [-2])
        self.pitch_errors = [0.0, 0.0]  # [previous, current]
        self.roll_errors = [0.0, 0.0]   # [previous, current]

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

    def imu_callback(self, imu):
        quaternion = [imu.orientation.x, imu.orientation.y, imu.orientation.z, imu.orientation.w]
        # Convert quaternion to Euler angles (roll, pitch, yaw)
        self.rot_mesurement = quaternion_to_euler(*quaternion)
        # Convert to radians for control
        self.get_logger().info(f"roll: {self.rot_mesurement[0]}, pitch: {self.rot_mesurement[1]}, yaw: {self.rot_mesurement[2]}")
    
    def joy_callback(self, joy):
        self.axes_list = joy.axes
        self.button_list = joy.buttons
        # curr_time = self.get_clock().now().to_msg().sec
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
        
        self.get_logger().info(f"x: {self.x}, y: {self.y}, z: {self.z}, roll: {self.rot[0]}, pitch: {self.rot[1]}, yaw: {self.rot[2]}")

    def timer_callback(self):

        # Compute errors (desired - measured)
        # Compute current errors (desired - measured, in radians)
        roll_error = self.rot[0] - self.rot_mesurement[0]  # Desired roll = 0
        pitch_error = self.rot[1] - self.rot_mesurement[1]  # Desired pitch = 0

        # Update error histories
        self.roll_errors.append(roll_error)  # Shift: [prev, current] -> [current, new]
        self.pitch_errors.append(pitch_error)
        # Calculate PID corrections
        roll_correction = self.roll_controller.Calculate(self.roll_errors)
        pitch_correction = self.pitch_controller.Calculate(self.pitch_errors)
        # Adjust foot positions based on roll and pitch corrections
        xyz = np.array([[self.x, self.y, self.z]] * 4)  # Base foot positions
        roll_adjust = np.array([[0, 0, roll_correction * self.W/2]] * 4)  # Adjust height based on roll
        pitch_adjust = np.array([[0, 0, pitch_correction * self.L/2]] * 4)  # Adjust height based on pitch

        # Apply adjustments based on leg positions
        roll_adjust[0, 2] *= -1  # FR: opposite sign for right legs
        roll_adjust[2, 2] *= -1  # RR: opposite sign for right legs
        pitch_adjust[0, 2] *= -1  # FR: opposite sign for front legs
        pitch_adjust[1, 2] *= -1  # FL: opposite sign for front legs

        # Combine base position with corrections
        xyz_adjusted = xyz + roll_adjust + pitch_adjust
        # xyz = np.array([[self.x, self.y, self.z]] * 4)
        
        self.rot_matrix[:3, :3] = np.linalg.inv(RotMatrix3D(self.rot, is_radians=True))  # Ensure RotMatrix3D returns a proper matrix
        transformed_xyz = (self.rot_matrix @ (xyz_adjusted + self.leg_origins))
        
        xyz_tf = transformed_xyz - self.leg_origins
        # self.get_logger().info(f"xyz = {transformed_xyz[0][0]}")
        self.joint_positions[0], self.joint_positions[1], self.joint_positions[2] = self.inv.InverseKinematic4(xyz_tf[0,0], xyz_tf[0,1]-self.hip_offset, xyz_tf[0,2], right = True)
        self.joint_positions[3], self.joint_positions[4], self.joint_positions[5] = self.inv.InverseKinematic4(xyz_tf[1,0], xyz_tf[1,1]+self.hip_offset, xyz_tf[1,2], right = False)
        self.joint_positions[6], self.joint_positions[7], self.joint_positions[8] = self.inv.InverseKinematic4(xyz_tf[2,0], xyz_tf[2,1]-self.hip_offset, xyz_tf[2,2], right = True)
        self.joint_positions[9], self.joint_positions[10], self.joint_positions[11] = self.inv.InverseKinematic4(xyz_tf[3,0], xyz_tf[3,1]+self.hip_offset, xyz_tf[3,2], right = False)
        
        
        # Apply limits to the joint angles
        for i in [0, 3, 6, 9]:  # Hip joints
            self.joint_positions[i] = np.clip(self.joint_positions[i], *self.hip_limits)
        for i in [1, 4, 7, 10]:  # Thigh joints
            self.joint_positions[i] = np.clip(self.joint_positions[i], *self.thigh_limits)
        for i in [2, 5, 8, 11]:  # Calf joints
            self.joint_positions[i] = np.clip(self.joint_positions[i], *self.calf_limits)

        self.joint_pos_pub.publish(Float64MultiArray(data=self.joint_positions))

def main():
    rclpy.init()
    ballance = Balance_control()
    executor = rclpy.executors.MultiThreadedExecutor()
    executor.add_node(ballance)
    

    executor_thread = threading.Thread(target=executor.spin, daemon=True)
    executor_thread.start()
    rate = ballance.create_rate(2)
    try:
        while rclpy.ok():
            rate.sleep()
    except KeyboardInterrupt:
        pass
    
    rclpy.shutdown()
    executor_thread.join()
    

if __name__ == '__main__':
    main()