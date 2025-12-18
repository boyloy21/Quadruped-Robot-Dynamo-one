#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from rclpy.timer import Timer
from numpy import array_equal
from math import fabs

from sensor_msgs.msg import Joy

class PS4_controller(Node):
    def __init__(self, rate):
        super().__init__('joystick_ramped')
        self.sub_joy = self.create_subscription(Joy, 'joy', self.joy_callback, 10)
        self.publish = self.create_publisher(Joy, 'dynamo_one/ramped_joy', 10)

        self.timer = self.create_timer(1.0 / rate, self.publish_joy)

        # Target Remote
        self.target_joy = Joy()
        self.target_joy.axes = [0., 0., 1., 0., 0., 1., 0., 0.]
        self.target_joy.buttons = [0] * 13

        # Last Remote
        self.last_joy = Joy()
        self.last_joy.axes = [0., 0., 1., 0., 0., 1., 0., 0.]
        self.last_joy.buttons = [0] * 13
        self.last_send_timer = self.get_clock().now()

        self.use_button = True
        self.speed_index = 2
        self.available_speeds = [0.5, 1.0, 2.0, 4.0]

    def joy_callback(self, msg):
        
        if self.use_button:
            if msg.buttons[4]:  # Decrease speed
                self.speed_index -= 1
                if self.speed_index < 0:
                    self.speed_index = len(self.available_speeds) - 1
                self.get_logger().info(f"Joystick speed: {self.available_speeds[self.speed_index]}")
                self.use_button = False

            elif msg.buttons[5]:  # Increase speed
                self.speed_index += 1
                if self.speed_index >= len(self.available_speeds):
                    self.speed_index = 0
                self.get_logger().info(f"Joystick speed: {self.available_speeds[self.speed_index]}")
                self.use_button = False

        if not self.use_button:
            if not (msg.buttons[4] or msg.buttons[5]):
                self.use_button = True

        self.target_joy.axes = msg.axes
        self.target_joy.buttons = msg.buttons

    def ramped_vel(self, v_prev, v_target, t_prev, t_now):
        """ Smooth velocity ramping """
        step = (t_now - t_prev).nanoseconds / 1e9 # Convert nanoseconds to seconds
        sign = self.available_speeds[self.speed_index] if (v_target > v_prev) else -self.available_speeds[self.speed_index]
        error = fabs(v_target - v_prev)

        if error < self.available_speeds[self.speed_index] * step:
            return v_target
        else:
            return v_prev + sign * step  # Take a step toward the target

    def publish_joy(self):
        t_now = self.get_clock().now()

        # Check if the joystick state has changed
        button_changed = array_equal(self.target_joy.buttons, self.last_joy.buttons)
        axes_changed = array_equal(self.target_joy.axes, self.last_joy.axes)

        if not button_changed or not axes_changed:
            joy = Joy()

            if not axes_changed:
                joy.axes = [
                    self.ramped_vel(self.last_joy.axes[i], self.target_joy.axes[i], self.last_send_time, t_now)
                    if self.target_joy.axes[i] != self.last_joy.axes[i]
                    else self.last_joy.axes[i]
                    for i in range(len(self.target_joy.axes))
                ]
            else:
                joy.axes = self.last_joy.axes

            joy.buttons = self.target_joy.buttons
            self.last_joy = joy
            self.publish.publish(self.last_joy)

        self.last_send_time = t_now

def main():
    rclpy.init()
    rate = 30  # Hz
    ps4_controller = PS4_controller(rate)
    rclpy.spin(ps4_controller)
    ps4_controller.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()