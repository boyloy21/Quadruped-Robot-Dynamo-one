import rclpy 
from rclpy.node import Node
from sensor_msgs.msg import Joy
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
from builtin_interfaces.msg import Duration
import numpy as np
from python_controller.robot.Invesekinematic import InverseKinematic
from python_controller.robot.utils import RotMatrix3D
from python_controller.FootTrajectory.Foot_trajectory_planner import Foot_trajectory_planner
import time

class FootTrajectory_control(Node):
    def __init__(self):
        super().__init__('FootTrajectory_control')

        self.joy_sub = self.create_subscription(Joy, '/joy', self.joy_callback, 10)

        self.publisher = self.create_publisher(JointTrajectory, '/joint_trajectory_controller/joint_trajectory', 10)
        self.timer = self.create_timer(1, self.publish_trajectory) # Publish joint states at 10 Hz

        self.joint_positions = [0.0] * 12
        self.joint_names = [
            "FR_hip_joint", "FR_thigh_joint", "FR_calf_joint",
            "FL_hip_joint", "FL_thigh_joint", "FL_calf_joint",
            "RR_hip_joint", "RR_thigh_joint", "RR_calf_joint",
            "RL_hip_joint", "RL_thigh_joint", "RL_calf_joint"
        ]

        # note: leg IDs
        self.left_front = 0
        self.left_back  = 1
        self.right_front= 2
        self.right_back = 3
        self.right_legs = [self.right_front, self.right_back]
        self.left_legs = [self.left_front, self.left_back]
        self.FR_cnt = 0
        self.FL_cnt = 0
        self.RR_cnt = 0
        self.RL_cnt = 0
        # Parameter of Robots
        self.L = 0.433
        self.W = 0.295
        self.hip_offset = 0.065
        self.L1, self.L2, self.L3 = 0.087, 0.25, 0.25
        self.inv = InverseKinematic(self.L1, self.L2, self.L3)

        # leg origins (right_f, left_f, right_b , left_b), i.e., the coordinate of j1
        self.leg_origins = np.matrix([[self.L/2, -self.W/2, 0],
                          [self.L/2, self.W/2, 0],
                          [-self.L/2, -self.W/2, 0],
                          [-self.L/2, self.W/2, 0]])
        
        # Define initial position
        self.rot = [0.0, 0.0, 0.0] # [roll, pitch, yaw]
        self.rot_matrix = np.eye(4)
        self.x = 0.0
        self.y = 0.0
        self.z = -0.35
        
        # Clipping joint angles to limits
        self.hip_limits = (-1.57, 1.20)
        self.thigh_limits = (-2.0944, 4.71239)
        self.calf_limits = (-2.53, -0.0872665)

        # self.center_foot_trajectory = [
        #     [0.0, 0.0, -0.35],  # Initial position
        #     [0.05, 0.0, -0.30],
        #     [0.10, 0.0, -0.25],
        #     [0.15, 0.0, -0.30],
        #     [0.2, 0.0, -0.35],
        #     [0.0, 0.0, -0.35]  # Return to stance
        # ]

        # Create Foot Trajectory
        control_points = np.array([
            [0.0, -0.35],      # Start position (adjusted Z to -0.35)
            [0.02, -0.35],     # Early lift
            [0.05, -0.3],     # Mid lift
            [0.08, -0.3],     # Peak height
            [0.1, -0.3],     # Highest forward point
            [0.12, -0.3],      # Descending phase start
            [0.12, -0.3],      # Lowering further
            [0.12, -0.318],      # Almost touching ground
            [0.15, -0.318],    # Landing position
            [0.15, -0.318],    # Slight after-movement
            [0.18, -0.32],     # Return to stable position
            [0.18, -0.32],     # Final adjustment (end position)
        ])

        self.foot_planer = Foot_trajectory_planner(control_points=control_points, num_points=100)

        # PS4 Button Filter
        self.debounce_button_rate = 0.3
        self.prev_time = 0
        self.is_in_swing = False  # Track if the leg is in swing mode
        self.is_in_stance = False  # Track if the leg is in stance mode
        self.is_moving = False  # Track if any movement is in progress
        self.is_robot_start = True
    
    # Improved debounce function
    def debounce(self, button_id, current_time):
        if self.button_list[button_id] == 1 and (current_time - self.prev_time) > self.debounce_button_rate:
            self.prev_time = current_time
            return True
        return False

    def foot_trajectory(self):
        Swing_trajec = self.foot_planer.bezier_curve()
        Stance_trajec = self.foot_planer.stance_trajectory()

        return Swing_trajec, Stance_trajec
    def transform_center_to_foot(self, x, y, z):
        """Transform the central trajectory to each foot using the transformation matrix."""
        xyz = np.array([[x, y, z]] * 4)
        
        self.rot_matrix[:3, :3] = np.linalg.inv(RotMatrix3D(self.rot, is_radians=True))  # Ensure RotMatrix3D returns a proper matrix
        transformed_xyz = (self.rot_matrix @ (xyz + self.leg_origins))
        
        xyz_tf = transformed_xyz - self.leg_origins
        return xyz_tf

    def publish_trajectory(self):
        # Get foot trajectory from Bezier curve
        swing_trajectory, stance_trajectory = self.foot_trajectory()

        msg = JointTrajectory()
        msg.joint_names = self.joint_names

        # Loop through swing trajectory and publish joint angles
        if (self.is_in_swing == True):
            for i, point in enumerate(swing_trajectory):
                self.x, self.z = point[0], point[1]
                xyz_tf = self.transform_center_to_foot(self.x, self.y, self.z)
                self.get_logger().info(f"x: {xyz_tf[0,0]}, y: {xyz_tf[0,1]}, z: {xyz_tf[0,2]}")
                if (i >= 99):
                    self.is_moving = False
                    self.is_in_swing = False
                if (self.right_front == 1 and self.is_moving == True):
                    self.joint_positions[0], self.joint_positions[1], self.joint_positions[2] = self.inv.InverseKinematic4(xyz_tf[0,0], xyz_tf[0,1]-self.hip_offset, xyz_tf[0,2], right = True)
                elif (self.left_front == 1 and self.is_moving == True):
                    self.joint_positions[3], self.joint_positions[4], self.joint_positions[5] = self.inv.InverseKinematic4(xyz_tf[1,0], xyz_tf[1,1]+self.hip_offset, xyz_tf[1,2], right = False)
                elif (self.right_back == 1 and self.is_moving == True):
                    self.joint_positions[6], self.joint_positions[7], self.joint_positions[8] = self.inv.InverseKinematic4(xyz_tf[2,0], xyz_tf[2,1]-self.hip_offset, xyz_tf[2,2], right = True)
                elif (self.left_back == 1 and self.is_moving == True):
                    self.joint_positions[9], self.joint_positions[10], self.joint_positions[11] = self.inv.InverseKinematic4(xyz_tf[3,0], xyz_tf[3,1]+self.hip_offset, xyz_tf[3,2], right = False)

                
                #  Create a JointTrajectoryPoint message for the current point in the trajectory
                joint_point = JointTrajectoryPoint()
                joint_point.positions = self.joint_positions
                joint_point.time_from_start = Duration(sec=int(i), nanosec=int((i % 1) * 1e9))  # Ensure integer value

                # Add the JointTrajectoryPoint to the message
                msg.points.append(joint_point)
                

        elif (self.is_in_stance == True):
            for i, point in enumerate(stance_trajectory):
                self.x, self.z = point[0], point[1]
                xyz_tf = self.transform_center_to_foot(self.x, self.y, self.z)
                self.get_logger().info(f"x: {xyz_tf[0,0]}, y: {xyz_tf[0,1]}, z: {xyz_tf[0,2]}")
                if (i >= 99):
                    self.is_moving = False
                    self.is_in_stance = False
                if (self.right_front == 1 and self.is_moving == True):
                    self.joint_positions[0], self.joint_positions[1], self.joint_positions[2] = self.inv.InverseKinematic4(xyz_tf[0,0], xyz_tf[0,1]-self.hip_offset, xyz_tf[0,2], right = True)
                elif (self.left_front == 1 and self.is_moving == True):
                    self.joint_positions[3], self.joint_positions[4], self.joint_positions[5] = self.inv.InverseKinematic4(xyz_tf[1,0], xyz_tf[1,1]+self.hip_offset, xyz_tf[1,2], right = False)
                elif (self.right_back == 1 and self.is_moving == True):
                    self.joint_positions[6], self.joint_positions[7], self.joint_positions[8] = self.inv.InverseKinematic4(xyz_tf[2,0], xyz_tf[2,1]-self.hip_offset, xyz_tf[2,2], right = True)
                elif (self.left_back == 1 and self.is_moving == True):
                    self.joint_positions[9], self.joint_positions[10], self.joint_positions[11] = self.inv.InverseKinematic4(xyz_tf[3,0], xyz_tf[3,1]+self.hip_offset, xyz_tf[3,2], right = False)

                #  Create a JointTrajectoryPoint message for the current point in the trajectory
                joint_point = JointTrajectoryPoint()
                joint_point.positions = self.joint_positions
                joint_point.time_from_start = Duration(sec=int(i), nanosec=int((i % 1) * 1e9))  # Ensure integer value

                # Add the JointTrajectoryPoint to the message
                msg.points.append(joint_point)
        
        
        elif(self.is_robot_start == True) :
            xyz_tf = self.transform_center_to_foot(self.x, self.y, self.z)
            self.joint_positions[0], self.joint_positions[1], self.joint_positions[2] = self.inv.InverseKinematic4(xyz_tf[0,0], xyz_tf[0,1]-self.hip_offset, xyz_tf[0,2], right = True)
            self.joint_positions[3], self.joint_positions[4], self.joint_positions[5] = self.inv.InverseKinematic4(xyz_tf[1,0], xyz_tf[1,1]+self.hip_offset, xyz_tf[1,2], right = False)
            self.joint_positions[6], self.joint_positions[7], self.joint_positions[8] = self.inv.InverseKinematic4(xyz_tf[2,0], xyz_tf[2,1]-self.hip_offset, xyz_tf[2,2], right = True)
            self.joint_positions[9], self.joint_positions[10], self.joint_positions[11] = self.inv.InverseKinematic4(xyz_tf[3,0], xyz_tf[3,1]+self.hip_offset, xyz_tf[3,2], right = False)
            
        
            self.is_robot_start = False
        # Publish the trajectory
        self.publisher.publish(msg)

    def joy_callback(self, joy):
        self.axes_list = joy.axes
        self.button_list = joy.buttons
        # curr_time = self.get_clock().now().to_msg().sec
        curr_time = time.time()

        # Walk Control manual
        # Left Front Leg Move
        if (self.debounce(0, current_time=curr_time)):
            self.right_front = 0
            self.left_front = 1
            self.right_back = 0
            self.left_back = 0
            self.FL_cnt += 1
            self.is_in_swing = True
            self.is_moving = True
            if (self.FL_cnt == 2):
                self.FL_cnt = 0
                self.is_in_stance = True
                self.is_in_swing = False
            self.prev_time = curr_time
        
        # Right Rear Leg Move
        if (self.button_list[1] == 1 and (curr_time - self.prev_time) > self.debounce_button_rate):
            self.right_front = 0
            self.left_front = 0
            self.right_back = 1
            self.left_back = 0
            self.RR_cnt += 1
            self.is_in_swing = True
            self.is_moving = True
            if (self.RR_cnt == 2):
                self.RR_cnt = 0
                self.is_in_stance = True
                self.is_in_swing = False
            self.prev_time = curr_time

        # Right Front Leg Move
        if (self.button_list[2] == 1 and (curr_time - self.prev_time) > self.debounce_button_rate):
            self.right_front = 1
            self.left_front = 0
            self.right_back = 0
            self.left_back = 0
            self.FR_cnt += 1
            self.is_in_swing = True
            self.is_moving = True
            if (self.FR_cnt == 2):
                self.FR_cnt = 0
                self.is_in_stance = True
                self.is_in_swing = False
            self.prev_time = curr_time

        # Left Rear Leg Move
        if (self.button_list[3] == 1 and (curr_time - self.prev_time) > self.debounce_button_rate):
            self.right_front = 0
            self.left_front = 0
            self.right_back = 0
            self.left_back = 1
            self.RL_cnt += 1
            self.is_in_swing = True
            self.is_moving = True
            if (self.RL_cnt == 2):
                self.RL_cnt = 0
                self.is_in_stance = True
                self.is_in_swing = False
            self.prev_time = curr_time

def main(args=None):
    rclpy.init(args=args)
    joint_trajectory_publisher = FootTrajectory_control()
    rclpy.spin(joint_trajectory_publisher)
    joint_trajectory_publisher.destroy_node()
    rclpy.shutdown()
    

if __name__ == '__main__':
    main()