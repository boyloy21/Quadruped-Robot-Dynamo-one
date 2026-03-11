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
        self.timer = self.create_timer(0.02, self.timer_callback)
        # Target joystick axes
        self.Mode = 'sit'
        self.target_axes = [0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0]
        # Target joystick buttons
        self.target_buttons = [0]*13
        self.speed_x = 0.7
        self.speed_y = 0.7
        self.prev_time = 0

        self.last_axes = [0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0]
        self.robot_height = -0.15
        self.x_offset = -0.15
        # Initialize Velocity
        self.Linear_velocity = [0.08, 0.1, 0.08]
        self.Angular_velocity = [0.3, 0.3, 0.2]

    def joy_callback(self, msg):
        # Update target axes and buttons based on joystick input
        self.target_axes = msg.axes
        self.target_buttons = msg.buttons
    
    def timer_callback(self):
        # Create a Twist message
        mode = String()
        twist_msg = Twist()
        cur_time = time.time()
        if self.target_buttons[0] == 1:
            self.Mode = "stand"
            self.Linear_velocity = [0.03 + (self.speed_x*0.1), 0.005 + (self.speed_y*0.1), 0.08]
            self.Angular_velocity = [0.3, 0.3, 0.3]
            self.robot_height = -0.35
            self.x_offset = 0.0
        elif self.target_buttons[1] == 1:
            self.Mode= "sit"
            self.Linear_velocity = [0.03 + (self.speed_x*0.1), (self.speed_y*0.1), 0.07]
            self.Angular_velocity = [0.3, 0.3, 0.2]
            self.robot_height = -0.25
            self.x_offset = 0.1
        elif self.target_buttons[2] == 1:
            self.Mode = "walk"
            self.Linear_velocity = [1.0*self.speed_x , 1.0*self.speed_y, 0.0]  # Move forward
            self.Angular_velocity = [0.0, 0.0, 0.05]  # Just control only yaw
        elif self.target_buttons[3] == 1:
            self.Mode = "trot"
            self.Linear_velocity = [1.0*self.speed_x, 0.0, 0.0]   # Move forward
            self.Angular_velocity = [0.0, 0.0, 0.8]
        
            
        if self.target_buttons[4] == 1 and (cur_time - self.prev_time) > 0.3:
            self.speed_x += 0.1
            self.speed_x = np.clip(self.speed_x , 0.0, 1.0)
            self.prev_time = cur_time
        elif self.target_buttons[5] == 1 and (cur_time - self.prev_time) > 0.3:
            self.speed_x -= 0.1
            self.speed_x = np.clip(self.speed_x , 0.0, 1.0)
            self.prev_time = cur_time
        if self.target_buttons[6] == 1 and (cur_time - self.prev_time) > 0.3:
            self.speed_y += 0.1
            self.speed_y = np.clip(self.speed_y , 0.0, 1.0)
            self.prev_time = cur_time
        elif self.target_buttons[7] == 1 and (cur_time - self.prev_time) > 0.3:
            self.speed_y -= 0.1
            self.speed_y = np.clip(self.speed_y , 0.0, 1.0)
            self.prev_time = cur_time
        
        self.get_logger().info('Speedx: %f, Speedy: %f' % (self.speed_x, self.speed_y))
        # Publish the mode
        mode.data = self.Mode
        self.mode_pub.publish(mode)
        # Map joystick axes to linear and angular velocities
        twist_msg.linear.x = self.target_axes[1] * self.Linear_velocity[0] + self.x_offset
        twist_msg.linear.y = self.target_axes[0] * self.Linear_velocity[1]
        twist_msg.linear.z = self.target_axes[4] * self.Linear_velocity[2] + self.robot_height

        twist_msg.angular.x = self.target_axes[7] * self.Angular_velocity[0]
        twist_msg.angular.y = self.target_axes[6] * self.Angular_velocity[1]
        twist_msg.angular.z = self.target_axes[3] * self.Angular_velocity[2]

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