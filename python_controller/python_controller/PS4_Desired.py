import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Joy
from geometry_msgs.msg import Twist
from std_msgs.msg import String, Float32MultiArray
import time
import numpy as np

from python_controller.Controller.Base_Foothold_trajectory import Base_Foot_Trajectory


class PS4_desired(Node):
    def __init__(self):
        super().__init__('PS4_desired')
        # SUBSCRIBER
        self.joy_sub = self.create_subscription(
            Joy,
            '/joy',
            self.joy_callback,
            10)
        self.joy_sub  # prevent unused variable warning
        
        # PUBLISHER
        self.mode_pub = self.create_publisher(String, 'dynamo_one/mode', 10)
        self.body_des_pub = self.create_publisher(Float32MultiArray, 'dynamo_one/body_des', 10)
        self.base_des_pub = self.create_publisher(Float32MultiArray, 'dynamo_one/base_des', 10)
        self.dt = 0.1
        self.timer = self.create_timer(self.dt, self.timer_callback)
        # Target joystick axes
        self.Mode = 'walk'
        self.target_axes = [0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0]
        self.last_axes = [0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0]
        # Target joystick buttons
        self.target_buttons = [0]*13
        self.prev_time = 0
        self.ocillator = 0.001

        # INITIALIZE VELOCITY AND ANGULAR VELOCITY COMMAND
        self.Vx = 0.0
        self.Vy = 0.0
        self.Omega = 0.0
        self.speed_limit = 0.1
        self.omega_limit = 0.1
        self.robot_height = 0.3

        # INITIALIZE BASE POSITION 
        self.des_pos = np.array([0.0, 0.0, self.robot_height])
        self.des_vel = np.array([0.0, 0.0, 0.0])
        self.des_ori = np.array([0.0, 0.0, 0.0])
        self.des_omega = np.array([0.0, 0.0, 0.0])
        L, W = 0.433, 0.12
        self.hf = 0.05
        self.BaseFoot = Base_Foot_Trajectory(L=L, W=W, dt=self.dt, k=0.03, hf=self.hf)
        self.speed = 0.0
        # INITIALIZE BODY CONTROL
        self.body_control = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        self.x_cur = 0.0
        self.y_cur = 0.0
        self.z_cur = 0.0
        self.x_offset = -0.15
        self.roll_cur = 0.0
        self.pitch_cur = 0.0
        self.yaw_cur = 0.0
        self.kp_pos = 0.8
        self.kp_theta = 1.0
        self.pos_limit = 0.4
        self.theta_limit = 0.4


    def P_controller(self, k, target, current):
        return k*(target - current)
    
    def map(self, x, in_min, in_max, out_min, out_max):
        return (x - in_min) * (out_max - out_min) / (in_max - in_min) + out_min
    
    def update_target(self):
        # Map joystick axes to desired position
        x_target = self.map(self.target_axes[1], -1.0, 1.0, -self.pos_limit, self.pos_limit) + self.x_offset
        y_target = self.map(self.target_axes[0], -1.0, 1.0, -self.pos_limit, self.pos_limit)
        z_target = self.map(self.target_axes[4], -1.0, 1.0, -self.pos_limit + self.robot_height, self.pos_limit - self.robot_height) + self.robot_height
        roll_target = self.map(self.target_axes[7], -1.0, 1.0, -self.theta_limit, self.theta_limit)
        pitch_target = self.map(self.target_axes[6], -1.0, 1.0, -self.theta_limit, self.theta_limit)
        yaw_target = self.map(self.target_axes[3], -1.0, 1.0, -self.theta_limit, self.theta_limit)
        
        # Position Update
        x = self.x_cur + self.P_controller(self.kp_pos, x_target, self.x_cur) * self.dt 
        y = self.y_cur + self.P_controller(self.kp_pos, y_target, self.y_cur) * self.dt
        z = self.z_cur + self.P_controller(self.kp_pos, z_target, self.z_cur) * self.dt
        roll = self.roll_cur + self.P_controller(self.kp_theta, roll_target, self.roll_cur) * self.dt
        pitch = self.pitch_cur + self.P_controller(self.kp_theta, pitch_target, self.pitch_cur) * self.dt
        yaw = self.yaw_cur + self.P_controller(self.kp_theta, yaw_target, self.yaw_cur) * self.dt

        # Current Position Update
        self.x_cur = x
        self.y_cur = y
        self.z_cur = z
        self.roll_cur = roll
        self.pitch_cur = pitch
        self.yaw_cur = yaw

        return x, y, z, roll, pitch, yaw
    def joy_callback(self, msg):
        # Update target axes and buttons based on joystick input
        self.target_axes = msg.axes
        self.target_buttons = msg.buttons

    def timer_callback(self):
        # Check for mode change
        cur_time = time.time()
        if self.target_buttons[0] == 1:
            self.Mode = 'stand'
            self.robot_height = 0.3
            self.x_offset = 0.0
        elif self.target_buttons[1] == 1:
            self.Mode = 'sit'
            self.robot_height = 0.1
            self.x_offset = 0.25
        elif self.target_buttons[2] == 1:
            self.Mode = 'walk'
            self.robot_height = 0.3
            self.x_offset = 0.0
        elif self.target_buttons[3] == 1:
            self.Mode = 'trot'
            self.robot_height = 0.3
            self.x_offset = 0.0
        elif self.target_buttons[4] == 1 and (cur_time - self.prev_time) > 0.3 and (self.Mode == 'walk' or self.Mode == 'trot'):
            self.speed += 0.1
            self.prev_time = cur_time
        elif self.target_buttons[6] == 1 and (cur_time - self.prev_time) > 0.3 and (self.Mode == 'walk' or self.Mode == 'trot'):
            self.speed -= 0.1
            self.prev_time = cur_time
        self.speed = np.clip(self.speed, 0.0, 1.0)

        mode = String()
        mode.data = self.Mode
        self.mode_pub.publish(mode)
        # Update body control
        if (self.Mode == 'stand' or self.Mode == 'sit'):
            x, y, z, roll, pitch, yaw = self.update_target()
            self.body_control = [x, y, z, roll, pitch, yaw]

            body = Float32MultiArray()
            body.data = self.body_control
            self.body_des_pub.publish(body)
            self.get_logger().info(f"Position: {x}, {y}, {z}, Roll: {roll}, Pitch: {pitch}, Yaw: {yaw}")

        elif (self.Mode == 'walk' or self.Mode == 'trot'):
            self.Vx = self.map(self.target_axes[1] , -1.0, 1.0, -self.speed_limit-self.speed, self.speed_limit+self.speed)
            self.Vy = self.map(self.target_axes[0] , -1.0, 1.0, -self.speed_limit-self.speed, self.speed_limit+self.speed)
            self.Omega = self.map(self.target_axes[3] , -1.0, 1.0, -self.omega_limit-self.speed, self.omega_limit+self.speed)
           
            Pb, Vb, thetab, Omegab = self.BaseFoot.Body_motion_ref(np.array([self.Vx, self.Vy, 0.0]), self.Omega, self.robot_height)
            
            base = Float32MultiArray()
            base.data = [Pb[0], Pb[1], Pb[2], Vb[0], Vb[1], Vb[2], thetab[0], thetab[1], thetab[2], Omegab[0], Omegab[1], Omegab[2]]
            self.base_des_pub.publish(base)

            self.get_logger().info(f"Position: {Pb[0]}, {Pb[1]}, {Pb[2]}, Roll: {thetab[0]}, Pitch: {thetab[1]}, Yaw: {thetab[2]}")
            self.get_logger().info(f"Velocity: {Vb[0]}, {Vb[1]}, {Vb[2]}, Omega: {Omegab[0], Omegab[1], Omegab[2]}")

        self.get_logger().info("Mode: %s" % self.Mode)
def main(args=None):
    rclpy.init(args=args)
    ps4_desired = PS4_desired()
    rclpy.spin(ps4_desired)
    ps4_desired.destroy_node()
    rclpy.shutdown()
if __name__ == "__main__":
    main()