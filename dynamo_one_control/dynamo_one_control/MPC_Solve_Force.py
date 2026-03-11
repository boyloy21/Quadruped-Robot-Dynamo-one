#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from std_msgs.msg import String, Float64MultiArray, Float32MultiArray, Int8MultiArray
from nav_msgs.msg import Odometry
import numpy as np
from dynamo_one_control.Desired.CPG_Network import CPG_Network
from dynamo_one_control.Controller.MPC_QP_Solve import MPC_Single_Rigid_Body_Model
from dynamo_one_control.utills.quaternion_utill import quaternion_to_euler


class MPC_Solve_Force(Node):
    def __init__(self):
        super().__init__('mpc_solve_force')

        # SUBSCRIBERS
        self.mode_sub = self.create_subscription(
            String,
            'dynamo_one/mode',
            self.mode_callback,
            10
        )
        self.sub_desired = self.create_subscription(
            Float32MultiArray, 
            'dynamo_one/base_des', 
            self.desired_callback, 
            10)
        
        self.odometry_sub = self.create_subscription(
            Odometry, 
            'dynamo_one/odometry', 
            self.odometry_callback, 
            10)
        
        self.foot_sub = self.create_subscription(
            Float64MultiArray, 
            'dynamo_one/footpos_des', 
            self.foot_callback, 
            10
        )

        self.contact_sub = self.create_subscription(
            Int8MultiArray,
            'dynamo_one/contact_schedule',
            self.contact_callback,
            1
        )
        # PUBLISHERS
        self.force_pub = self.create_publisher(
            Float64MultiArray, 
            'dynamo_one/forcempc', 
            10)

        # TIMER
        self.dt = 1/100  # 100 Hz
        self.timer = self.create_timer(self.dt, self.force_calculate)

        self.initialize_parameters()

        """Initialize all control components"""
        # MPC weights - tuned for stability
        qp_weights = np.array([
            0.2, 0.2, 100,  # Position (x,y,z)
            10.0, 10.0, 0.0,   # Velocity (vx,vy,vz)
            0.0, 0.0, 0.0,    # Orientation (roll,pitch,yaw)
            0.1, 0.1, 1.0,    # Angular velocity
            0.0               # Gravity term
        ])
        
        # Initialize MPC with optimized parameters
        self.mpc = MPC_Single_Rigid_Body_Model(
            mass=self.m,
            I=self.I_body,
            num_legs=4,
            horizon=self.horizon,
            dt=self.dt,
            qp_weights=qp_weights,
            alpha=1e-6,
            mu=0.6
        )
        

    def initialize_parameters(self):

        # INITIALIZE PARAMETERS
        self.horizon = 10  # Prediction horizon
        self.num_legs = 4
        self.declare_parameter('L', 0.433)  # Length of the leg
        self.declare_parameter('W', 0.12)  # Width of the leg
        self.declare_parameter('L1', 0.087)  # Length of the hip
        self.declare_parameter('L2', 0.25)  # Length of the thigh
        self.declare_parameter('L3', 0.25)
        self.declare_parameter('m', 20.0)  # Mass of the robot
        self.declare_parameter('Ixx', 0.25)  # Moment of inertia around x-axis
        self.declare_parameter('Iyy', 1.05)  # Moment of inertia around y-axis
        self.declare_parameter('Izz', 1.05)  # Moment of inertia around z-axis
        
        self.L = self.get_parameter('L').get_parameter_value().double_value
        self.W = self.get_parameter('W').get_parameter_value().double_value
        self.L1 = self.get_parameter('L1').get_parameter_value().double_value
        self.L2 = self.get_parameter('L2').get_parameter_value().double_value
        self.L3 = self.get_parameter('L3').get_parameter_value().double_value
        self.m = self.get_parameter('m').get_parameter_value().double_value
        self.Ixx = self.get_parameter('Ixx').get_parameter_value().double_value
        self.Iyy = self.get_parameter('Iyy').get_parameter_value().double_value
        self.Izz = self.get_parameter('Izz').get_parameter_value().double_value
        self.robot_height = 0.35
        self.I_body = np.diag([self.Ixx, self.Iyy, self.Izz])

        # CPG parameters
        self.cycles = 3
        self.Tcpg = 0.7
        self.dt_cpg = 0.01
        self.CPG = CPG_Network(Ts=self.Tcpg, cycles=self.cycles, dt=self.dt_cpg)

        self.step = int(self.cycles*self.CPG.T/self.dt_cpg)

        # Initialize step counters
        self.step_count = 0
        self.mpc_step = 0
        
        # Gait control
        self.command_gait = "trot"  # Default to trot for better stability
        self.move_direction = "stop"
        self.LEG_LABELS = ["FL", "FR", "RR", "RL"]
        self.current_phase = np.zeros(4)  # Initialize current phase for each leg
        self.contact_horizon = np.zeros((self.horizon, 4))  # Contact schedule for each leg
        self.contact_schedule = np.zeros((self.step, 4))
        self.contact_state = np.zeros(4)
        self.current_forces = np.zeros((4, 3))  # Forces applied at each foot
        
        # State variables
        self.com_pos = np.array([0.0, 0.0, self.robot_height])
        self.com_vel = np.zeros(3)
        self.com_ori = np.zeros(3)
        self.com_omega = np.zeros(3)
        self.current_orientation_quaternion = np.array([1.0, 0.0, 0.0, 0.0])
        
        # Desired states
        self.des_pos = np.array([0.0, 0.0, self.robot_height])
        self.des_vel = np.zeros(3)
        self.des_ori = np.zeros(3)
        self.des_omega = np.zeros(3)
        self.base_foot_pos = np.zeros((4, 3))
        # Foot parameters
        self.Phip = np.array([
            [self.L/2, self.L1 + self.W/2, -0.05],
            [self.L/2,-self.L1 - self.W/2, -0.05],
            [-self.L/2, -self.L1 - self.W/2, -0.05],
            [-self.L/2, self.L1 + self.W/2, -0.05]
        ])
       
    
    def get_horizon_contacts(self):
        """Get contact schedule for MPC horizon"""
        horizon_contacts = np.zeros((self.horizon, 4))
        for i in range(self.horizon):
            # idx = (self.mpc_step + i) % self.step
            horizon_contacts[i] = self.contact_state[:]
            print(horizon_contacts[i])
        return horizon_contacts
    
    def prepare_mpc_state(self):
        """Prepare state information for MPC"""
        return {
            'actual': {
                'position': self.com_pos,
                'velocity': self.com_vel,
                'orientation': self.com_ori,
                'angular_velocity': self.com_omega
            },
            'desired': {
                'position': self.des_pos,
                'velocity': self.des_vel,
                'orientation': self.des_ori,
                'angular_velocity': self.des_omega
            },
            'foot_positions': self.base_foot_pos,
            'contact_schedule': self.get_horizon_contacts()
        }
    
    def solve_mpc(self, state):
        """Solve MPC problem and update forces"""
        self.mpc.compute_contact_force(
            state['actual'],
            state['desired'],
            state['foot_positions'],
            state['contact_schedule']
        )
        
        # Solve QP and update forces
        force = self.mpc.solve_qp()
        self.force_reshaped = force.reshape((self.horizon, 4, 3))
        self.current_forces = self.force_reshaped[0]
        
        
    def contact_callback(self, msg):
        self.contact_state = np.array(msg.data)
        # Store contact schedule
        self.get_horizon_contacts()

    def foot_callback(self, msg):
        # Update desired foot positions and velocities from the message
        self.base_foot_pos[0] = np.array(msg.data[:3])
        self.base_foot_pos[1] = np.array(msg.data[3:6])
        self.base_foot_pos[2] = np.array(msg.data[6:9])
        self.base_foot_pos[3] = np.array(msg.data[9:12])
        self.get_logger().info(f"Base Foot Positin: {self.base_foot_pos}")

    def mode_callback(self, msg):
        self.command_gait = msg.data
        self.get_logger().info(f"Mode changed to: {self.command_gait}")
    
    def desired_callback(self, msg):
        # Update desired position, velocity, orientation, and omega from the message
        self.des_pos = np.array(msg.data[:3])
        self.des_vel = np.array(msg.data[3:6])
        self.des_ori = np.array(msg.data[6:9])
        self.des_omega = np.array(msg.data[9:12])
        
        # Log the received desired state
        self.get_logger().info(f"Desired Position: {self.des_pos}, Desired Velocity: {self.des_vel}, Desired Orientation: {self.des_ori}, Desired Omega: {self.des_omega}")
    
    def odometry_callback(self, msg):
        # Update current position, velocity, orientation, and omega from the odometry message
        self.com_pos = np.array([msg.pose.pose.position.x, msg.pose.pose.position.y, msg.pose.pose.position.z])
        self.com_vel = np.array([msg.twist.twist.linear.x, msg.twist.twist.linear.y, msg.twist.twist.linear.z])
        self.com_pos[2] = self.robot_height
        # Convert quaternion to euler angles
        self.current_orientation_quaternion = np.array([
            msg.pose.pose.orientation.x,
            msg.pose.pose.orientation.y,
            msg.pose.pose.orientation.z,
            msg.pose.pose.orientation.w
        ])
        self.com_ori = quaternion_to_euler(self.current_orientation_quaternion)
        self.com_omega = np.array([msg.twist.twist.angular.x, msg.twist.twist.angular.y, msg.twist.twist.angular.z])
        
        # Log the received odometry data
        self.get_logger().info(f"Odometry - Position: {self.com_pos}, Velocity: {self.com_vel}, Orientation: {self.com_ori}, Omega: {self.com_omega}")

    def force_calculate(self):
        if self.command_gait == "walk" or self.command_gait == "trot" or self.command_gait == "bound" or self.command_gait == "pace":
            state = self.prepare_mpc_state()
            # Solve mpc
            self.solve_mpc(state)
            self.get_logger().info(f"Force: {self.current_forces}")
            # Publish Force
            force_msg = Float64MultiArray()
            force_msg.data = self.current_forces.flatten().tolist()
            self.force_pub.publish(force_msg)
            self.get_logger().info(f"Force: {self.current_forces}")


def main(args=None):
    rclpy.init(args=args)
    mpc_solver = MPC_Solve_Force()
    rclpy.spin(mpc_solver)
    mpc_solver.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()


        
            

    

