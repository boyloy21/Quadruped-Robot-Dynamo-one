#!/usr/bin/env python3
import rclpy
from rclpy.node import Node 
from sensor_msgs.msg import JointState
from std_msgs.msg import Float32MultiArray, Float64MultiArray
import numpy as np
from python_controller.Controller.InverseDynamic import InverseDynamic
from python_controller.Controller.Kinematic_Model import KinematicQuadruped


class Leg_PIDController(Node):
    def __init__(self):
        super().__init__('leg_controller')

        # Subscribers for joint states and foot positions
        self.joint_sub = self.create_subscription(
            JointState,
            '/joint_states',
            self.joint_callback,
            10)
        self.footpos_des = self.create_subscription(
            Float32MultiArray,
            '/dynamo_one/footpos_des',
            self.foot_callback,
            10
        )
        self.footvel_des = self.create_subscription(
            Float32MultiArray,
            '/dynamo_one/footvel_des',
            self.foot_callback,
            10
        )
        self.footacc_des = self.create_subscription(
            Float32MultiArray,
            '/dynamo_one/footacc_des',
            self.foot_callback,
            10
        )
        self.forcempc_sub = self.create_subscription(
            Float32MultiArray,
            '/dynamo_one/forcempc',
            self.foot_callback,
            10
        )

        # Publisher for joint commands
        self.joint_pub = self.create_publisher(Float64MultiArray, 'forward_position_controller/commands', 10)
        # self.foot_sub = self.create_subscription(Float32MultiArray, '/dynamo_one/foot_des', self.foot_callback, 10)
        # self.joint_pub = self.create_publisher(JointState, '/dynamo_one/joint_commands', 10)
        self.timer = self.create_timer(0.01, self.calculate)  # 10 Hz
        self.declare_parameter('L', 0.433)  # Length of the leg
        self.declare_parameter('W', 0.12)  # Width of the leg
        self.declare_parameter('L1', 0.087)  # Length of the hip
        self.declare_parameter('L2', 0.25)  # Length of the thigh
        self.declare_parameter('L3', 0.25)  # Length of the calf
        self.declare_parameter('m_thigh', 0.143)  # Mass of the thigh
        self.declare_parameter('m_calf', 0.1)

        self.L = self.get_parameter('L').get_parameter_value().double_value
        self.W = self.get_parameter('W').get_parameter_value().double_value
        self.L1 = self.get_parameter('L1').get_parameter_value().double_value
        self.L2 = self.get_parameter('L2').get_parameter_value().double_value
        self.L3 = self.get_parameter('L3').get_parameter_value().double_value
        self.m1 = self.get_parameter('m_thigh').get_parameter_value().double_value
        self.m2 = self.get_parameter('m_calf').get_parameter_value().double_value
        self.I1zz = 0.001  # Inertia of the thigh
        self.I2zz = 0.001  # Inertia of the calf
        dt = 0.01  # Time step for the controller

        # Initialize joint positions
        self.joint_positions = [0.0] * 12
        self.joint_velocities = [0.0] * 12
        self.joint_torques = [0.0] * 12
        self.joint_names = [
            "FR_hip_joint", "FR_thigh_joint", "FR_calf_joint",
            "FL_hip_joint", "FL_thigh_joint", "FL_calf_joint",
            "RR_hip_joint", "RR_thigh_joint", "RR_calf_joint",
            "RL_hip_joint", "RL_thigh_joint", "RL_calf_joint"
        ]
        self.pDes = np.zeros((4, 3))
        self.vDes = np.zeros((4, 3))
        self.aDes = np.zeros((4, 3))
        self.qDes = np.zeros((4, 3))  # Joint positions
        self.qdDes = np.zeros((3))  # Joint velocities

        self.KpJoint = np.diag([0.85, 0.85, 0.85])  # Proportional gain for position control
        self.KdJoint = np.diag([0.1, 0.1, 0.1])
        self.KpCartesian = np.diag([0.85, 0.85, 0.85])  # Proportional gain for Cartesian control
        self.KdCartesian = np.diag([0.1, 0.1, 0.1])

        self.LegInvdynamic = InverseDynamic(self.L, self.W, self.L1, self.L2, self.L3, self.m1, self.m2, self.I1zz, self.I2zz, self.KpCartesian, self.KdCartesian, dt)
        self.KinematicModel = KinematicQuadruped(self.L, self.W, self.L1, self.L2, self.L3)
        self.tau = np.zeros((3, 1))
        self.contact_state = np.array([1, 1, 1, 1])  # Contact state for each leg (1: contact, 0: no contact)
        self.forcempc = np.zeros((4, 3))  # Force MPC for each leg
        self.rot = [0.0, 0.0, 0.0]

    def joint_callback(self, msg):
        for i in range(len(msg.name)):
            if msg.name[i] in self.joint_names:
                self.joint_positions[self.joint_names.index(msg.name[i])] = msg.position[i]
                self.joint_velocities[self.joint_names.index(msg.name[i])] = msg.velocity[i]
        self.get_logger().info(f"Joint feedback: {self.joint_positions}")
    def foot_callback(self, msg):
        # Update desired foot positions and velocities from the message
        FLpdes = np.array(msg.data[:3])
        FRpdes = np.array(msg.data[3:6])
        RRpdes = np.array(msg.data[6:9])
        RLpdes = np.array(msg.data[9:12])
        self.pDes = np.array([FLpdes, FRpdes, RRpdes, RLpdes])
    def footvel_callback(self, msg):
        # Update desired foot velocities from the message
        FLvdes = np.array(msg.data[:3])
        FRvdes = np.array(msg.data[3:6])
        RRvdes = np.array(msg.data[6:9])
        RLvdes = np.array(msg.data[9:12])
        self.vDes = np.array([FLvdes, FRvdes, RRvdes, RLvdes])
    def footacc_callback(self, msg):
        # Update desired foot accelerations from the message
        FLades = np.array(msg.data[:3])
        FRades = np.array(msg.data[3:6])
        RRades = np.array(msg.data[6:9])
        RLades = np.array(msg.data[9:12])
        self.aDes = np.array([FLades, FRades, RRades, RLades])
    def forcempc_callback(self, msg):
        # Update desired foot accelerations from the message
        FLades = np.array(msg.data[:3])
        FRades = np.array(msg.data[3:6])
        RRades = np.array(msg.data[6:9])
        RLades = np.array(msg.data[9:12])
        self.forcempc = np.array([FLades, FRades, RRades, RLades])
    def calculate(self):
        # Calculate the desired joint positions and velocities
        self.JointTau = np.zeros((12, 1))
        qFL, qFR, qRR, qRL = self.KinematicModel.leg_IK(self.pDes, rot=self.rot)
        self.qDes = np.array([qFL, qFR, qRR, qRL])
        for i in range(4):
            self.qdDes = self.KinematicModel.get_joint_velocity(self.vDes[i,:], self.qDes[i,:], leg_id=i)  # Reshape to match the joint velocity dimensions
            
            if self.contact_state[i] == 0:  # If the leg is in the air
                tauswing = self.LegInvdynamic.TorqueSwing(self.pDes[i,:], self.vDes[i,:], self.aDes[i,:], self.qDes[i,:], self.qdDes, legID=i)
                Tau = tauswing.reshape(3, 1)  # Reshape to match the joint torque dimensions
            else:  # If the leg is in contact
                taustance = self.LegInvdynamic.TorqueStance(self.forcempc[i,:], self.qDes[i,:], legID=i)
                Tau = taustance.reshape(3, 1)  # Reshape to match the joint torque dimensions

            self.JointTau[i*3:(i+1)*3] = Tau + self.KpJoint @ (self.qDes[i,:].reshape(3, 1) - np.array(self.joint_positions[i*3:(i+1)*3]).reshape(3, 1)) + self.KdJoint @ (self.qdDes.reshape(3, 1) - np.array(self.joint_velocities[i*3:(i+1)*3]).reshape(3, 1))
        # # Create a JointState message
        self.joint_torques = self.JointTau.flatten().tolist()
        joint_state_msg = Float64MultiArray()
        joint_state_msg.data = self.joint_torques
        # Publish the joint commands
        self.joint_pub.publish(joint_state_msg)
        self.get_logger().info(f'Joint Torques: {self.JointTau.flatten()}')
        

def main(args=None):
    rclpy.init(args=args)
    leg_controller = Leg_PIDController()
    rclpy.spin(leg_controller)
    leg_controller.destroy_node()
    rclpy.shutdown()
if __name__ == '__main__':
    main()
