#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState

class InitialJointStatePublisher(Node):
    def __init__(self):
        super().__init__('initial_joint_state_publisher')
        self.publisher = self.create_publisher(JointState, '/joint_states', 10)
        self.timer = self.create_timer(0.1, self.publish_joint_states)
        self.publish_count = 0  # Counter to track how many times the message is published

    def publish_joint_states(self):
        if self.publish_count >= 5:  # Publish 5 times, then stop
            self.get_logger().info("Initial joint states published. Shutting down.")
            rclpy.shutdown()  # Stop the node
            return

        joint_state_msg = JointState()
        joint_state_msg.header.stamp = self.get_clock().now().to_msg()
        joint_state_msg.name = [
            "FL_hip_joint", "FL_thigh_joint", "FL_calf_joint",
            "FR_hip_joint", "FR_thigh_joint", "FR_calf_joint",
            "RR_hip_joint", "RR_thigh_joint", "RR_calf_joint",
            "RL_hip_joint", "RL_thigh_joint", "RL_calf_joint"
        ]
        joint_state_msg.position = [0.0, 1.0, -1.57] * 4  # Example initial positions
        
        self.publisher.publish(joint_state_msg)
        self.publish_count += 1  # Increment the counter
    
def main():
    rclpy.init()
    node = InitialJointStatePublisher()
    rclpy.spin(node)

if __name__ == '__main__':
    main()