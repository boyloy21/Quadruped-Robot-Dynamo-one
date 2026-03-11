#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Joy
from geometry_msgs.msg import Twist
from std_msgs.msg import Float64MultiArray

class PS4Controller(Node):
    def __init__(self):
        super().__init__('remote_controller')
        self.sub_joy = self.create_subscription(Joy, 'joy', self.joy_callback, 10)
        self.pub_position = self.create_publisher(Twist, '/cmd_vel', 10)
        self.pub_velocity = self.create_publisher(Float64MultiArray, '/velocity', 10)
        self.timer = self.create_timer(0.01, self.timer_callback)
        self.button = [0] * 13
        self.axis = [0] * 8

    def joy_callback(self, msg):
        self.button = msg.buttons
        self.axis = msg.axes

    def timer_callback(self):
        # Create messages
        position = Twist()
        velocity = Float64MultiArray()
        velocity.data = [0.0, 0.0, 0.0]  # Initialize velocity data correctly

        # Set velocity based on axis movement
        if abs(self.axis[0]) >= 0.5:
            velocity.data[0] = self.axis[0]
        if abs(self.axis[1]) >= 0.5:
            velocity.data[1] = self.axis[1]
        if abs(self.axis[3]) >= 0.5:
            velocity.data[2] = self.axis[3]

        # Set position based on button presses
        if self.button[0] == 1:
            position.linear.x += 1.0
        if self.button[1] == 1:
            position.linear.x -= 1.0
        if self.button[2] == 1:
            position.linear.y += 1.0
        if self.button[3] == 1:
            position.linear.y -= 1.0
        if self.button[4] == 1:
            position.angular.z += 1.0
        if self.button[5] == 1:
            position.angular.z -= 1.0

        # Publish messages
        self.pub_position.publish(position)
        self.pub_velocity.publish(velocity)

        # Log the button and axis states
        # self.get_logger().info(f'Buttons: {self.button}')
        # self.get_logger().info(f'Axes: {self.axis}')
        self.get_logger().info(f'Position: {position}')
        self.get_logger().info(f'Velocity: {velocity}')

def main(args=None):
    rclpy.init(args=args)
    remote_controller = PS4Controller()
    rclpy.spin(remote_controller)
    remote_controller.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
