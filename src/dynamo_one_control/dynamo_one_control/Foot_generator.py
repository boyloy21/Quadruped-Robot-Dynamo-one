#!/usr/bin/env python3
import rclpy
from rclpy.node import Node 
from std_msgs.msg import Float64MultiArray, Int8MultiArray, Float32MultiArray, String
from dynamo_one_control.Desired.CPG_Network import CPG_Network
from dynamo_one_control.Desired.Base_Foothold_trajectory import Base_Foot_Trajectory
from dynamo_one_control.utills.mpc_utill import convert_rpy_to_rot
import numpy as np

class FootGenerator(Node):
    def __init__(self):
        super().__init__('foot_generator')
        # Initialize parameters
        self.declare_parameter('L', 0.433)  # Length of the leg
        self.declare_parameter('W', 0.12)
        self.L = self.get_parameter('L').get_parameter_value().double_value
        self.W = self.get_parameter('W').get_parameter_value().double_value

        # CPG PARAMETERS
        self.Tcpg = 0.6  # CPG period
        self.Cycles = 3 # Number of CPG cycles
        self.dt = 0.01
        
        self.cpg = CPG_Network(Ts=self.Tcpg, cycles=self.Cycles, dt=self.dt)
        self.BaseFoot = Base_Foot_Trajectory(self.L, self.W, dt=self.dt, k=0.03, hf=0.05)
        
        self.step = int(self.Cycles*self.cpg.T/self.dt)
        # Initialize step counters
        self.step_count = 0
        self.mpc_step = 0

        # Timing calculations
        self.Tst = self.cpg.T * self.cpg.beta  # Stance time
        self.Tsw = self.cpg.T * (1 - self.cpg.beta)  # Swing time

        # Gait control
        self.command_gait = None  # Default to trot for better stability
        self.move_direction = "stop"
        self.LEG_LABELS = ["FL", "FR", "RR", "RL"]
        self.current_phase = np.zeros(4)  # Initialize current phase for each leg
        self.contact_schedule = np.zeros((4, self.step))  # Contact schedule for each leg
        self.contact_state = {label: {"contact": []} for label in self.LEG_LABELS}
        self.leg_phase = np.zeros((4, self.step))  # Phase for each leg
        self.current_forces = np.zeros((4, 3))  # Forces applied at each foot
        
        # Desired states
        self.robot_height = 0.35
        self.des_pos = np.array([0.0, 0.0, self.robot_height])
        self.des_vel = np.zeros(3)
        self.des_ori = np.zeros(3)
        self.des_omega = np.zeros(3)
        
        # Foot parameters
        self.L1 = 0.087
        self.Phip = np.array([
            [self.L/2, self.L1 + self.W/2, -0.05],
            [self.L/2,-self.L1 - self.W/2, -0.05],
            [-self.L/2, -self.L1 - self.W/2, -0.05],
            [-self.L/2, self.L1 + self.W/2, -0.05]
        ])
        
        # self.world_foot_pos = self.Phip.copy()
        self.world_foot_pos = [self.des_pos + convert_rpy_to_rot(self.des_ori) @ self.Phip[i] for i in range(4)]
        self.offset_foot = np.array([0.0, 0.0, self.robot_height])
        self.foot_trajectories = [[] for _ in range(4)]
        self.base_foot_pos = np.zeros((4, 3))
        self.base_foot_vel = np.zeros((4, 3))
        self.base_foot_acc = np.zeros((4, 3))
        self.foot_positions = np.zeros((4, 3))  # Foot positions in world frame
        self.contact_schedule = np.zeros((4, self.step), dtype=int)
        self.Ps = self.Phip.copy()
        self.Pe = np.zeros(4)
        self.des_pos_foot = np.array([0.0, 0.0, 0.0])
        
        # SUBSCRIBER
        self.mode_sub = self.create_subscription(
            String,
            'dynamo_one/mode',
            self.mode_callback,
            10)
        self.foot_pos_sub = self.create_subscription(
            Float32MultiArray,
            'dynamo_one/base_des',
            self.base_des_callback,
            10)

        # PUBLISHER
        self.contact_schedule_pub = self.create_publisher(Int8MultiArray, 'dynamo_one/contact_schedule', 10)
        self.foot_pos_pub = self.create_publisher(Float64MultiArray, 'dynamo_one/footpos_des', 10)
        self.foot_vel_pub = self.create_publisher(Float64MultiArray, 'dynamo_one/footvel_des', 10)
        self.foot_acc_pub = self.create_publisher(Float64MultiArray, 'dynamo_one/footacc_des', 10)
        self.timer = self.create_timer(self.dt, self.generate_foot_trajectory)

        self.footpos_pub = [0.0]*12
        self.footvel_pub = [0.0]*12
        self.footacc_pub = [0.0]*12
        self.footpos_prev = np.array([
            [self.L/2, self.L1 + self.W/2, -0.35],
            [self.L/2,-self.L1 - self.W/2, -0.35],
            [-self.L/2, -self.L1 - self.W/2, -0.35],
            [-self.L/2, self.L1 + self.W/2, -0.35]
        ])

        # Velocity smoothing and bounds parameters
        self.velocity_smoothing_factor = 0.2  # Alpha for exponential smoothing (0-1)
        self.max_foot_velocity = 2.0  # m/s
        self.max_foot_acceleration = 10.0  # m/s²
        self.offset_x = np.array([0.06, 0.0, 0.0])
        # Initialize velocity history for smoothing
        self.smoothed_velocities = np.zeros((4, 3))
    def update_gait(self):
        """Update CPG network based on current gait command"""
        if self.command_gait in ["walk", "trot", "bound", "pace"]:
            self.cpg = CPG_Network(Ts=self.Tcpg, cycles=self.Cycles, dt=self.dt)
            self.leg_phase = self.cpg.generate(
                gait_type=self.command_gait, 
                move_direction=self.move_direction
            )
    def reset_foot_positions(self):
        """Reset foot positions to nominal stance positions"""
        Rb = convert_rpy_to_rot(self.des_ori)
        self.world_foot_pos = np.array([
            self.des_pos + Rb @ self.Phip[i] 
            for i in range(4)
        ])
        self.foot_trajectories = [[pos.copy()] for pos in self.world_foot_pos]
        self.step_count = 0

    def mode_callback(self, msg):
        # Update the mode based on the message received
        self.command_gait = msg.data
        
    def base_des_callback(self, msg):
        """Update desired base states"""
        data = msg.data
        self.des_pos = np.array(data[:3])
        self.des_vel = np.array(data[3:6])
        self.des_ori = np.array(data[6:9])
        self.des_omega = np.array(data[9:12])
        self.des_pos_foot[0:2] = self.des_pos[0:2]

    def generate_foot_trajectory(self):

        if self.command_gait in ["walk", "trot", "bound", "pace"]:
            # Update CPG network
            self.update_gait()
            
            # Get current body rotation
            Rb = convert_rpy_to_rot(self.des_ori)
            
            # Generate foot trajectories
            for i, leg in enumerate(self.LEG_LABELS):
                phase = self.leg_phase[i, self.step_count % self.step]
                self.current_phase[i] = phase
                
                # Get PROPER initial foot position
                if len(self.foot_trajectories[i]) == 0:
                    # First time - start from nominal position at ground level
                    current_foot_world = self.des_pos + Rb @ self.Phip[i]
                    current_foot_world[2] = 0.0  # Force z=0 in world frame
                else:
                    current_foot_world = self.foot_trajectories[i][-1]

                 # Calculate target foot placement
                Pe_world = self.BaseFoot.FootTarget(
                    Pb=self.des_pos,
                    Vcur=self.des_vel,
                    Vcmd=self.des_vel,
                    Tst=self.Tst,
                    legID=i
                )
                Pe_world[2] = 0.0  # Ensure target is on ground
                
                # STANCE PHASE - foot stays at ground level
                if phase <= self.cpg.beta:  # Stance phase
                    self.contact_schedule[i, self.step_count] = 1
                    # In stance, foot moves with the ground (fixed in world frame)
                    new_foot_world = current_foot_world.copy()
                    new_foot_world[2] = 0.0  # Keep foot on ground

                    raw_velocity = -Rb.T @ self.des_vel
                    self.base_foot_acc[i] = -Rb.T @ np.zeros(3)  # Assuming zero body acceleration

                # SWING PHASE - use your original SwingTrajectory  
                else:  # Swing phase
                    self.contact_schedule[i, self.step_count] = 0
                    phi_swing = (phase - self.cpg.beta) / (1 - self.cpg.beta)
                    
                    # Find the last stance position (where foot was on ground)
                    swing_start_pos = self.find_last_stance_position(i)

                    # Generate swing trajectory
                    new_foot_world, world_vel, Asw = self.BaseFoot.SwingTrajectory(
                        Ps=swing_start_pos, 
                        Pe=Pe_world, 
                        phi=phi_swing, 
                        Tsw=self.Tsw
                    )
                
                    # Convert to body and foot frames
                    raw_velocity = Rb.T @ (world_vel - self.des_vel)
                    self.base_foot_acc[i] = Rb.T @ Asw
                
                
                self.world_foot_pos[i] = new_foot_world
                self.base_foot_pos[i] = Rb.T @ (self.world_foot_pos[i] - self.des_pos)
                # Apply smoothing and bounds
                self.base_foot_vel[i] = self.apply_velocity_limits(i, raw_velocity) + self.offset_x
                # Store new foot position (limit history length)
                self.foot_trajectories[i].append(self.world_foot_pos[i].copy())
                # self.get_logger().info(f"Foot position start: {Ps_world}, end: {Pe_world}")
            self.get_logger().info(f"Foot trajectories in world frame: {self.world_foot_pos[:]}")
            # Publish messages (unchanged)
            self.publish_contact_schedule()
            self.publish_foot_states()
            
            # Update step counter
            self.get_logger().info(f"Step {self.step_count}: Contacts {self.contact_schedule[:, self.step_count]}")
            self.get_logger().info(f"Foot position in body frame: {self.base_foot_pos}")
            self.get_logger().info(f"Foot velocity in body frame: {self.base_foot_vel}")
            self.step_count = (self.step_count + 1) % self.step
    def find_last_stance_position(self, leg_idx):
        """Find the last position where this foot was on the ground"""
        # Look backwards through history to find last stance position
        for i in range(len(self.foot_trajectories[leg_idx])-1, -1, -1):
            if self.contact_schedule[leg_idx, i % self.step] == 1:  # Was in stance
                return self.foot_trajectories[leg_idx][i].copy()
        
        # Fallback to nominal position if no history
        Rb = convert_rpy_to_rot(self.des_ori)
        nominal_pos = self.des_pos + Rb @ self.Phip[leg_idx]
        nominal_pos[2] = 0.0
        return nominal_pos
    def apply_velocity_limits(self, leg_index, raw_velocity):
        """Apply smoothing and bounds to foot velocity"""
        # 1. Exponential smoothing
        smoothed_vel = (self.velocity_smoothing_factor * raw_velocity + 
                    (1 - self.velocity_smoothing_factor) * self.smoothed_velocities[leg_index])
        
        # 2. Velocity magnitude clipping
        vel_magnitude = np.linalg.norm(smoothed_vel)
        if vel_magnitude > self.max_foot_velocity:
            smoothed_vel = smoothed_vel * (self.max_foot_velocity / vel_magnitude)
        
        # 3. Update stored velocity
        self.smoothed_velocities[leg_index] = smoothed_vel
        
        return smoothed_vel
    def publish_contact_schedule(self):
        contact_state = np.array([self.contact_schedule[:, self.step_count]])
        msg = Int8MultiArray(data=contact_state.flatten().tolist())
        self.contact_schedule_pub.publish(msg)

    def publish_foot_states(self):
        self.foot_pos_pub.publish(Float64MultiArray(data=self.base_foot_pos.flatten().tolist()))
        self.foot_vel_pub.publish(Float64MultiArray(data=self.base_foot_vel.flatten().tolist())) 
        self.foot_acc_pub.publish(Float64MultiArray(data=self.base_foot_acc.flatten().tolist()))
def main(args=None):
    rclpy.init(args=args)
    foot_generator = FootGenerator()
    rclpy.spin(foot_generator)
    foot_generator.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()




