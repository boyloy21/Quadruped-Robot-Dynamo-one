#!/usr/bin/env python3
import rclpy
from rclpy.node import Node 
from sensor_msgs.msg import JointState
from std_msgs.msg import Float32MultiArray, Float64MultiArray, String, Int8MultiArray
import numpy as np
from dynamo_one_control.Controller.PD_Control_torque import PD_Control_torque
from dynamo_one_control.model.InverseDynamic import InverseDynamic
from dynamo_one_control.model.Kinematic_Model import KinematicQuadruped


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
            Float64MultiArray,
            'dynamo_one/footpos_des',
            self.foot_callback,
            10
        )
        self.footvel_des = self.create_subscription(
            Float64MultiArray,
            'dynamo_one/footvel_des',
            self.footvel_callback,
            10
        )
        self.footacc_des = self.create_subscription(
            Float64MultiArray,
            'dynamo_one/footacc_des',
            self.footacc_callback,
            10
        )
        self.forcempc_sub = self.create_subscription(
            Float32MultiArray,
            '/dynamo_one/forcempc',
            self.forcempc_callback,
            10
        )

        self.contact_sub = self.create_subscription(
            Int8MultiArray,
            '/dynamo_one/contact_schedule',
            self.contact_callback,
            10
        )

        self.mode_sub = self.create_subscription(
            String,
            'dynamo_one/mode',
            self.mode_callback,
            10
        )
        
        # Publisher for joint commands
        self.joint_pub = self.create_publisher(
            Float64MultiArray, 
            '/joint_group_effort_controller/commands', 
            10)
        
        # Control timer (100Hz)
        self.timer = self.create_timer(0.01, self.calculate)  # 100 Hz
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
        self.joint_positions = np.zeros(12)
        self.joint_velocities = np.zeros(12)
        self.joint_torques = np.zeros(12)
        self.joint_names = [
            "FL_hip_joint", "FL_thigh_joint", "FL_calf_joint",
            "FR_hip_joint", "FR_thigh_joint", "FR_calf_joint",
            "RR_hip_joint", "RR_thigh_joint", "RR_calf_joint",
            "RL_hip_joint", "RL_thigh_joint", "RL_calf_joint"
        ]
        self.hip_origin = np.array([[self.L/2, self.W/2, 0.0],
                                    [self.L/2, -self.W/2, 0.0],
                                    [-self.L/2, -self.W/2, 0.0],
                                    [-self.L/2, self.W/2, 0.0]])
        self.pDes = np.zeros((4, 3))
        self.vDes = np.zeros((4, 3))
        self.aDes = np.zeros((4, 3))
        self.qDes = np.zeros(12)  # Joint positions
        self.qdDes = np.zeros(12) # Joint velocities
        self.joint_pos = [0.0] *12
        # Clipping joint angles to limits
        self.hip_limits = (-1.57, 1.20)
        self.thigh_limits = (-3.14, 3.14)
        self.calf_limits = (-2.53, -0.0872665)

        self.KpJoint = 80  # Proportional gain for position control
        self.KdJoint = 1.5
        self.KpCartesian = np.diag([0.85, 0.85, 0.85])  # Proportional gain for Cartesian control
        self.KdCartesian = np.diag([0.1, 0.1, 0.1])

        # self.limit_torque = 30.0
        self.hip_gains={'kp': 50, 'kd': 1.0}
        self.thigh_gains={'kp': 150, 'kd': 4.5}
        self.calf_gains={'kp': 80, 'kd': 2.5}
        self.limit_torque = 30
        self.pd_torque = PD_Control_torque(self.hip_gains, self.thigh_gains, self.calf_gains, limit_toque=self.limit_torque)
        self.LegInvdynamic = InverseDynamic(self.L, self.W, self.L1, self.L2, self.L3, self.m1, self.m2, self.I1zz, self.I2zz, self.KpCartesian, self.KdCartesian, dt)
        self.KinematicModel = KinematicQuadruped(self.L, self.W, self.L1, self.L2, self.L3)
        self.tau = np.zeros((3, 1))
        self.contact_state = np.array([1, 1, 1, 0])  # Contact state for each leg (1: contact, 0: no contact)
        self.forcempc = np.zeros((4, 3))  # Force MPC for each leg
        self.rot = [0.0, 0.0, 0.0]
        self.joint_torques = np.zeros(12)

        self.mode = 'walk'  # Default mode

    def mode_callback(self, msg):
        # Update the mode based on the message received
        self.mode = msg.data
    
    def contact_callback(self, msg):
        # Update the contact state based on the message received
        self.contact_state = np.array(msg.data)

    def joint_callback(self, msg):
        # Update current joint states
        for i, name in enumerate(msg.name):
            if name in self.joint_names:
                idx = self.joint_names.index(name)
                self.joint_positions[idx] = msg.position[i]
                self.joint_velocities[idx] = msg.velocity[i]
    
    def foot_callback(self, msg):
        # Update desired foot positions and velocities from the message
        FLpdes = np.array(msg.data[:3])
        FRpdes = np.array(msg.data[3:6])
        RRpdes = np.array(msg.data[6:9])
        RLpdes = np.array(msg.data[9:12])
        self.pDes = np.array([FLpdes, FRpdes, RRpdes, RLpdes]) - self.hip_origin
        self.get_logger().info(f"Received foot positions: {self.pDes}")
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
        if self.mode == 'walk' or self.mode == 'trot' or self.mode == 'bound' or self.mode == 'pace':
            # Calculate the desired joint positions and velocities
            self.JointTau = np.zeros((12, 1))
            qFL, qFR, qRR, qRL = self.KinematicModel.leg_IK(self.pDes, rot=self.rot)
            self.qDes[0:3] = qFL
            self.qDes[3:6] = qFR
            self.qDes[6:9] = qRR
            self.qDes[9:12] = qRL
            
            # Set joint positions
            self.joint_pos[0:3] = qFL
            self.joint_pos[3:6] = qFR
            self.joint_pos[6:9] = qRR
            self.joint_pos[9:12] = qRL
            # Set positions with safety checks

            for i in [0, 3, 6, 9]:  # Hip joints
                self.joint_pos[i] = np.clip(self.joint_pos[i], *self.hip_limits)
            for i in [1, 4, 7, 10]:  # Thigh joints
                self.joint_pos[i] = np.clip(self.joint_pos[i], *self.thigh_limits)
            for i in [2, 5, 8, 11]:  # Calf joints
                self.joint_pos[i] = np.clip(self.joint_pos[i], *self.calf_limits)

            for i in range(4):
                # desired_joints = self.qDes[i*3:(i+1)*3].tolist()
                self.qdDes[i*3:(i+1)*3] = self.KinematicModel.get_joint_velocity(
                        self.vDes[i, :],
                        self.joint_pos[i*3:(i+1)*3],  # Using qDes instead of current positions
                        leg_id=i
                    )

            self.torque_joint = self.pd_torque.compute_torque(
                self.qDes,
                self.qdDes, 
                self.joint_positions, 
                self.joint_velocities
            )
            
            for i in range(4):
                if self.contact_state[i] == 0:  # If the leg is in the air
                    tauswing = self.LegInvdynamic.TorqueSwing(self.pDes[i,:], self.vDes[i,:], self.aDes[i,:], self.qDes[i*3:(i+1)*3], self.qdDes[i*3:(i+1)*3], legID=i)
                    Tau_task = tauswing.reshape(3, 1)  # Reshape to match the joint torque dimensions
                else:  # If the leg is in contact
                    taustance = self.LegInvdynamic.TorqueStance(self.forcempc[i,:], self.qDes[i*3:(i+1)*3], legID=i)
                    Tau_task = taustance.reshape(3, 1)  # Reshape to match the joint torque dimensions

                self.JointTau[i*3:(i+1)*3] = Tau_task + self.torque_joint[i*3:(i+1)*3].reshape(3, 1)
            # Create a JointState message
            self.joint_torques = self.torque_joint.flatten().tolist()
            joint_state_msg = Float64MultiArray()
            joint_state_msg.data = self.torque_joint.tolist()
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
