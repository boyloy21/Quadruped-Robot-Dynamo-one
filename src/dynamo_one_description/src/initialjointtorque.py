#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from std_msgs.msg import Float64MultiArray
from sensor_msgs.msg import JointState
import numpy as np

class PD_Control_torque():
    def __init__(self, kp, kd, limit_toque=30):
        self.kp = kp
        self.kd = kd
        self.limit_toque = limit_toque
        
    def compute_torque(self, joint_pos_desired, joint_vel_desired, joint_pos_feedback, joint_vel_feedback):
        """
        Compute the torque command using PD control.

        :param joint_pos_desired: Desired joint positions (numpy array)
        :param joint_vel_desired: Desired joint velocities (numpy array)
        :param joint_pos_feedback: Current joint positions (numpy array)
        :param joint_vel_feedback: Current joint velocities (numpy array)
        :return: Torque command (numpy array)
        """
        position_error = joint_pos_desired - joint_pos_feedback
        velocity_error = joint_vel_desired - joint_vel_feedback
        

        torque_command = self.kp * position_error + self.kd * velocity_error
        # Limit the torque command to a specified range
        for i in range(12):
            if torque_command[i] >= self.limit_toque:
                torque_command[i] = self.limit_toque
            elif torque_command[i] <= -self.limit_toque:
                torque_command[i] = -self.limit_toque
            else:
                torque_command[i] = torque_command[i]
    
        return torque_command

class InitialJointStatePublisher(Node):
    def __init__(self):
        super().__init__('initial_joint_state_publisher')
        self.joint_sub = self.create_subscription(
            JointState,
            '/joint_states',
            self.joint_callback,
            10)
        self.publisher = self.create_publisher(Float64MultiArray, '/joint_group_effort_controller/commands', 10)
        self.timer = self.create_timer(0.01, self.publish_joint_states)
        self.publish_count = 0  # Counter to track how many times the message is published
        self.pd_torque = PD_Control_torque(kp=50.0, kd=1.0, limit_toque=8.0)
        # Initialize variables
        self.joint_pos_feedback = np.zeros(12)
        self.joint_vel_feedback = np.zeros(12)
        self.joint_names = [
            "FL_hip_joint", "FL_thigh_joint", "FL_calf_joint",
            "FR_hip_joint", "FR_thigh_joint", "FR_calf_joint",
            "RR_hip_joint", "RR_thigh_joint", "RR_calf_joint",
            "RL_hip_joint", "RL_thigh_joint", "RL_calf_joint"
        ]
        self.torque_command = np.zeros(12)
        
        # Standing configuration (modify based on your robot)
        self.joint_pos_desired = np.array([
                                0.0, -0.2, 2.0,   # FL
                                0.0, -0.2, 2.0,   # FR 
                                0.0, -0.2, 2.0,   # RL
                                0.0, -0.2, 2.0    # RR
                            ])

        self.joint_vel_desired = np.zeros(12)  # Desired velocities are zero for standing
    
    def joint_callback(self, msg):
        # Update current joint states
        for i, name in enumerate(msg.name):
            if name in self.joint_names:
                idx = self.joint_names.index(name)
                self.joint_pos_feedback[idx] = msg.position[i]
                self.joint_vel_feedback[idx] = msg.velocity[i]

    def publish_joint_states(self):
        if self.publish_count >= 5:  # Publish 5 times, then stop
            self.get_logger().info("Initial joint states published. Shutting down.")
            rclpy.shutdown()  # Stop the node
            return
        
        self.joint_pos_desired = np.array([
                0.0, -0.2, -2.0,   # FL
                0.0, -0.2, -2.0,   # FR 
                0.0, -0.2, -2.0,   # RL
                0.0, -0.2, -2.0    # RR
            ])
        
        # # Calculate position error
        self.torque_command = self.pd_torque.compute_torque(
            self.joint_pos_desired, 
            self.joint_vel_desired, 
            self.joint_pos_feedback, 
            self.joint_vel_feedback
        )
        joint_state_msg = Float64MultiArray()
        joint_state_msg.data = self.torque_command.tolist()
        self.publisher.publish(joint_state_msg)
        self.publish_count += 0.1  # Increment the counter
    
def main():
    rclpy.init()
    node = InitialJointStatePublisher()
    rclpy.spin(node)

if __name__ == '__main__':
    main()