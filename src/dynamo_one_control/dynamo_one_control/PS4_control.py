import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Joy
from geometry_msgs.msg import Twist
from std_msgs.msg import String
import time
import numpy as np

class PS4Controller(Node):
    def __init__(self):
        super().__init__('ps4_controller')
        self.joy_sub = self.create_subscription(Joy, '/joy', self.joy_callback, 10)

        self.mode_pub = self.create_publisher(String, 'dynamo_one/mode', 10)
        self.position_pub = self.create_publisher(Twist, 'dynamo_one/cmd_vel', 20)
        self.timer = self.create_timer(0.01, self.timer_callback)
        # Target joystick axes
        self.Mode = 'sit'
        self.target_axes = [0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0]
        # Target joystick buttons
        self.target_buttons = [0]*13
        self.prev_time = 0

        self.last_axes = [0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0]
        self.robot_height = -0.15
        self.x_offset = 0.0
        # Initialize Velocity
        self.P_target = np.array([0.0, 0.0, 0.0])
        self.P_current = np.array([0.0, 0.0, -0.05])
        self.P_desired = np.array([0.3, 0.3, 0.15])
        self.theta_target = np.array([0.0, 0.0, 0.0])
        self.theta_current = np.array([0.0, 0.0, 0.0])
        self.theta_desired = np.array([0.4, 0.4, 0.4])
        
        # self.Linear_velocity = [0.1, 0.1, 0.1]
        # self.Angular_velocity = [0.3, 0.3, 0.2]
        self.Kp = 0.15 # Proportional gain for position control
        self.Ktheta = 1.0  # Proportional gain for orientation control
        self.dt = 0.01  # Time step for control updates

    # Position controller
    def Position_controller(self, Kp, p_target, p_current):
        """
        Proportional controller to compute the control input based on the error between target and current position.
        :param Kp: Proportional gain
        :return: Control input
        """
        error = p_target - p_current
        control_input = Kp * error
        return control_input

    def Orentation_controller(self, Ktheta, theta_target, theta_current):
        """
        Proportional controller to compute the control input based on the error between target and current orientation.
        :param Kp: Proportional gain
        :return: Control input
        """
        error = theta_target - theta_current
        control_input = Ktheta * error
        return control_input
    
    def Update_target(self, dt):
        vel_output = self.Position_controller(self.Kp, self.P_target, self.P_current)
        omega_output = self.Orentation_controller(self.Ktheta, self.theta_target, self.theta_current)
        P_update = self.P_current + vel_output*dt
        theta_update = self.theta_current + omega_output*dt
        self.P_current = P_update
        self.theta_current = theta_update
        return P_update, theta_update
    def joy_callback(self, msg):
        # Update target axes and buttons based on joystick input
        self.target_axes = msg.axes
        self.target_buttons = msg.buttons
    
    def timer_callback(self):
        # Create a Twist message
        mode = String()
        twist_msg = Twist()

        # Mode selection based on button presses
        if self.target_buttons[0] == 1:
            self.Mode = "stand"
            self.robot_height = -0.35
            self.x_offset = -0.1
        elif self.target_buttons[1] == 1:
            self.Mode= "sit"
            self.robot_height = -0.25
            self.x_offset = 0.1
        elif self.target_buttons[2] == 1:
            self.Mode = "walk"
            self.robot_height = -0.35
            self.x_offset = -0.1
        elif self.target_buttons[3] == 1:
            self.Mode = "trot"
            self.robot_height = -0.35
            self.x_offset = -0.1
        elif self.target_buttons[4] == 1:
            self.Mode = "gallop"
            self.robot_height = -0.35
            self.x_offset = 0.0
        elif self.target_buttons[5] == 1:
            self.Mode = "pace"
            self.robot_height = -0.35
            self.x_offset = 0.0
        
        self.P_desired = np.array([0.3- abs(self.x_offset), 0.3, 0.15])
        self.P_target[0] = self.P_desired[0]*self.target_axes[1] + self.x_offset
        self.P_target[1] = self.P_desired[1]*self.target_axes[0] 
        self.P_target[2] = self.P_desired[2]*self.target_axes[4] + self.robot_height
        self.theta_target[0] = self.theta_desired[0]*self.target_axes[7]
        self.theta_target[1] = self.theta_desired[1]*self.target_axes[6]
        self.theta_target[2] = self.theta_desired[2]*self.target_axes[3]
        P_update, theta_update = self.Update_target(self.dt)
        # self.get_logger().info('Target Position: %f, %f, %f' % (self.P_target[0], self.P_target[1], self.P_target[2]))
        # self.get_logger().info('Target Orientation: %f, %f, %f' % (self.theta_target[0], self.theta_target[1], self.theta_target[2]))
        self.get_logger().info('Position: %f, %f, %f' % (P_update[0], P_update[1], P_update[2]))
        self.get_logger().info('Orientation: %f, %f, %f' % (theta_update[0], theta_update[1], theta_update[2]))
        # Publish the mode
        mode.data = self.Mode
        self.mode_pub.publish(mode)
        # Map joystick axes to linear and angular velocities
        twist_msg.linear.x = P_update[0] 
        twist_msg.linear.y = P_update[1]
        twist_msg.linear.z = P_update[2] 

        twist_msg.angular.x = theta_update[0]
        twist_msg.angular.y = theta_update[1]
        twist_msg.angular.z = theta_update[2]

        # Publish the Twist message
        self.position_pub.publish(twist_msg)

def main(args=None):
    rclpy.init(args=args)
    ps4_controller = PS4Controller()
    rclpy.spin(ps4_controller)
    ps4_controller.destroy_node()
    rclpy.shutdown()
if __name__ == '__main__':
    main()