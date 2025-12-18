import rclpy
from rclpy.node import Node
import numpy as np
from math import *
from std_msgs.msg import Float64, Int8MultiArray
from geometry_msgs.msg import Wrench
from nav_msgs.msg import Odometry
from sensor_msgs.msg import Imu, JointState
from dynamo_one_control.model.Kinematic_Model import KinematicQuadruped
from dynamo_one_control.Filter.moving_window_filter import MovingWindowFilter

class Sensor_Feedback(Node):
    def __init__(self):
        super().__init__('Sensor_Feedback')
        self.Imu_sub = self.create_subscription(
            Imu,
            'dynamo_one/imu',
            self.imu_callback,
            10)
        self.joint_sub = self.create_subscription(
            JointState,
            '/joint_states',
            self.joint_callback,
            10)
        self.force_sub = [
            self.create_subscription(Wrench, f'/dynamo_one/{leg}_ft_sensor', getattr(self, f"{leg}_force_callback"), 10)
            for leg in ['FL', 'FR', 'RL', 'RR']
        ]

        # Publishers for for estimate state
        self.odom_pub = self.create_publisher(Odometry, 'dynamo_one/odometry', 10)
        self.contact_state_pub = self.create_publisher(Int8MultiArray, 'dynamo_one/contact_state', 10)
        # Timer for publishing odometry
        self.timer = self.create_timer(0.01, self.timer_callback)  # 10 Hz
        # Robot paameters
        self.declare_parameter('robot_height', 0.35)  # Height of the robot
        self.declare_parameter('L', 0.433)  # Length of the robot
        self.declare_parameter('W', 0.12)  # Width of the robot
        self.declare_parameter('L1', 0.087)  # Length of the first link
        self.declare_parameter('L2',0.25)
        self.declare_parameter('L3', 0.25)
        self.robot_height = self.get_parameter('robot_height').get_parameter_value().double_value
        self.L = self.get_parameter('L').get_parameter_value().double_value
        self.W = self.get_parameter('W').get_parameter_value().double_value
        self.L1 = self.get_parameter('L1').get_parameter_value().double_value
        self.L2 = self.get_parameter('L2').get_parameter_value().double_value
        self.L3 = self.get_parameter('L3').get_parameter_value().double_value
        self.leg_geometry = np.array([
            [self.L / 2, self.W / 2, 0.0],   # FL
            [self.L / 2, -self.W / 2, 0.0],  # FR
            [-self.L / 2, -self.W / 2, 0.0], # RR
            [-self.L / 2, self.W / 2, 0.0]   # RL
        ])
        self.kinematic_model = KinematicQuadruped(self.L, self.W, self.L1, self.L2, self.L3)

        # State variables
        self.position = np.zeros(3)  # [x, y, z]
        self.velocity = np.zeros(3)  # [vx, vy, vz]
        self.orientation = np.zeros(3)  # [roll, pitch, yaw]
        self.angular_velocity = np.zeros(3)  # [wx, wy, wz]
        self.quaternion = np.zeros(4)  # [w, x, y, z]
        self.foot_positions = np.zeros((4, 3))  # Positions of the four feet in world frame
        self.contact_states = np.zeros(4, dtype=bool)  # Contact states for each foot (0: no contact, 1: contact)
        self.ground_normal = np.array([0.0, 0.0, 1.0])  # Normal vector of the ground plane

        # Filter parameters
        self.velocity_filter = MovingWindowFilter(window_size=10, dim=3)  # Filter for velocity
        self.orientation_filter = MovingWindowFilter(window_size=5, dim=4)  # Filter for angular velocity

        
        self.rot = [0.0, 0.0, 0.0]

        # Joint feedback
        self.joint_positions = [0.0] * 12
        self.joint_velocities = [0.0] * 12
        self.joint_names = [
            "FL_hip_joint", "FL_thigh_joint", "FL_calf_joint",
            "FR_hip_joint", "FR_thigh_joint", "FR_calf_joint",
            "RR_hip_joint", "RR_thigh_joint", "RR_calf_joint",
            "RL_hip_joint", "RL_thigh_joint", "RL_calf_joint"
        ]

        # Force feedback
        self.FL_force = np.array([0.0, 0.0, 0.0])
        self.FR_force = np.array([0.0, 0.0, 0.0])
        self.RL_force = np.array([0.0, 0.0, 0.0])
        self.RR_force = np.array([0.0, 0.0, 0.0])


    def quaternion_to_euler(self, w, x, y, z):
        # Roll (x-axis rotation)
        roll = atan2(2 * (w * x + y * z), 1 - 2 * (x**2 + y**2))

        # Pitch (y-axis rotation)
        pitch = asin(2 * (w * y - z * x))

        # Yaw (z-axis rotation)
        yaw = atan2(2 * (w * z + x * y), 1 - 2 * (y**2 + z**2))

        return roll, pitch, yaw
    def imu_callback(self, msg):
        # Store raw IMU data
        self.quaternion = np.array([
            msg.orientation.w,
            msg.orientation.x,
            msg.orientation.y,
            msg.orientation.z
        ])

        self.angular_velocity = np.array([
            msg.angular_velocity.x,
            msg.angular_velocity.y,
            msg.angular_velocity.z
        ])

        # Apply filtering
        filterd_orientation = self.orientation_filter.calculate_average(self.quaternion)

        # Convert to Euler angles
        roll, pitch, yaw = self.quaternion_to_euler(
            filterd_orientation[0],
            filterd_orientation[1],
            filterd_orientation[2],
            filterd_orientation[3]
        )
        self.rot = [roll, pitch, yaw]
        # Log the Euler angles
        self.get_logger().info(f"IMU feedback: Roll_back: {roll}, Pitch_back: {pitch}, Yaw_back: {yaw}")

    def joint_callback(self, msg):
        # Update joint positions and velocities
        joint_positions = np.zeros(12)
        joint_velocities = np.zeros(12)
        for i, name in enumerate(msg.name):
            if name in self.joint_names:
                joint_positions[self.joint_names.index(name)] = msg.position[i]
                joint_velocities[self.joint_names.index(name)] = msg.velocity[i]

        # Compute forward kinematics to get foot positions
        for leg_idx in range(4):
            self.foot_positions[leg_idx] = self.kinematic_model.ForwardKinematic(
                joint_positions[leg_idx * 3: leg_idx * 3 + 3],
                rot=self.rot,
                legID=leg_idx
            )
        self.get_logger().info(f"Joint feedback: {self.joint_positions}")

    def update_contact_states(self):
        """Update which feet are in contact with the ground based on force feedback."""
        force_threshold = 10.0 # Newton threshold for contact detection
        self.contact_states = np.array([
            np.linalg.norm(self.FL_force) > force_threshold,
            np.linalg.norm(self.FR_force) > force_threshold,
            np.linalg.norm(self.RL_force) > force_threshold,
            np.linalg.norm(self.RR_force) > force_threshold
        ])
        self.get_logger().info(f"Contact states: {self.contact_states}")

    def estimate_body_position(self):
        """Estimate the body position based on foot positions and contact states."""
        if np.any(self.contact_states):
            # Get height of feet in contact (z coordinate in body frame)
            contact_heights = -self.foot_positions[self.contact_states, 2]  # z-coordinates of feet in contact

            # Estimate body height as the average of contact heights
            if len(contact_heights) > 0:
                self.position[2] = np.mean(contact_heights)
            else:
                self.position[2] = self.robot_height
        else:
            # If no feet are in contact, maintain the last known position
            self.get_logger().warn("No feet in contact, maintaining last known position.")
    
    def estimate_ground_normal(self):
        """Estimate the ground normal vector based on foot positions."""
        if np.sum(self.contact_states) >= 3:   # Need at least 3 contacts fro plane fitting
            contact_points = self.foot_positions[self.contact_states]
            
            # Center point around mean 
            centroid = np.mean(contact_points, axis=0)
            centroid = contact_points - centroid

            # Perform SVD to find normal 
            _, _, vh = np.linalg.svd(centroid)
            normal = vh[2, :]

            # Ensure  normal points upwards 
            if normal[2] < 0:
                normal = -normal

            self.ground_normal = normal / np.linalg.norm(normal)

    def FL_force_callback(self, msg):
        self.FL_force = np.array([msg.force.x, msg.force.y, msg.force.z])
        self.get_logger().info(f"FL_force: {self.FL_force}")

    def FR_force_callback(self, msg):
        self.FR_force = np.array([msg.force.x, msg.force.y, msg.force.z])
        self.get_logger().info(f"FR_force: {self.FR_force}")

    def RL_force_callback(self, msg):
        self.RL_force = np.array([msg.force.x, msg.force.y, msg.force.z])
        self.get_logger().info(f"RL_force: {self.RL_force}")

    def RR_force_callback(self, msg):
        self.RR_force = np.array([msg.force.x, msg.force.y, msg.force.z])
        self.get_logger().info(f"RR_force: {self.RR_force}")

    def publish_odometry(self):
        """Publish the estimated odometry."""
        odom_msg = Odometry()
        odom_msg.header.stamp = self.get_clock().now().to_msg()
        odom_msg.header.frame_id = 'odom'
        odom_msg.child_frame_id = 'base_link'

        # Set position and orientation
        odom_msg.pose.pose.position.x = self.position[0]
        odom_msg.pose.pose.position.y = self.position[1]
        odom_msg.pose.pose.position.z = self.position[2]

        
        odom_msg.pose.pose.orientation.x = self.quaternion[1]
        odom_msg.pose.pose.orientation.y = self.quaternion[2]
        odom_msg.pose.pose.orientation.z = self.quaternion[3]
        odom_msg.pose.pose.orientation.w = self.quaternion[0]

        # Set linear and angular velocity
        odom_msg.twist.twist.linear.x = self.velocity[0]
        odom_msg.twist.twist.linear.y = self.velocity[1]
        odom_msg.twist.twist.linear.z = self.velocity[2]
        
        # Angular velocity in body frame
        odom_msg.twist.twist.angular.x = self.angular_velocity[0]
        odom_msg.twist.twist.angular.y = self.angular_velocity[1]
        odom_msg.twist.twist.angular.z = self.angular_velocity[2]

        # Publish the message
        self.odom_pub.publish(odom_msg)
    def publish_contact_state(self):
        contact_state_msg = Int8MultiArray()
        for i in range(4):
            if self.contact_states[i]:
                contact_state_msg.data.append(1)
            else:
                contact_state_msg.data.append(0)
        self.contact_state_pub.publish(contact_state_msg)
    def timer_callback(self):
        """Timer callback to update state and publish odometry."""
        # Update contact states based on force feedback
        self.update_contact_states()

        # Estimate body position based on foot positions and contact states
        self.estimate_body_position()

        # Estimate ground normal vector
        self.estimate_ground_normal()

        # Publish the estimated odometry
        self.publish_odometry()

        self.publish_contact_state()

        # Log the current state
        self.get_logger().info(f"Position: {self.position}, Velocity: {self.velocity}, Orientation: {self.rot}")
        self.get_logger().info(f"Ground Normal: {self.ground_normal}")
def main(args=None):
    rclpy.init(args=args)
    sensor_feedback = Sensor_Feedback()
    rclpy.spin(sensor_feedback)
    sensor_feedback.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()

