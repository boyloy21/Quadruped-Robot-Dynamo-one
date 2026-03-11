#!/usr/bin/env python3
import rclpy
from rclpy.node import Node 
from sensor_msgs.msg import Joy, JointState
from geometry_msgs.msg import Twist
from std_msgs.msg import String, Float32MultiArray
import numpy as np
import casadi as ca
from dynamo_one_control.utills.mpc_utill import convert_rpy_to_rot, skew_symmetric, rot_to_rpy, exp_rotation
from dynamo_one_control.Desired.Base_Foothold_trajectory import Base_Foot_Trajectory
from dynamo_one_control.Desired.CPG_Network import CPG_Network
from dynamo_one_control.Controller.MPC_QP_Solve import MPC_Single_Rigid_Body_Model
def quaternion_to_rotation_matrix(q):
    """Convert quaternion to rotation matrix"""
    w, x, y, z = q
    return np.array([
        [1-2*y*y-2*z*z, 2*x*y-2*z*w, 2*x*z+2*y*w],
        [2*x*y+2*z*w, 1-2*x*x-2*z*z, 2*y*z-2*x*w],
        [2*x*z-2*y*w, 2*y*z+2*x*w, 1-2*x*x-2*y*y]
    ])

def quaternion_multiply(q1, q2):
    """Multiply two quaternions"""
    w1, x1, y1, z1 = q1
    w2, x2, y2, z2 = q2
    return np.array([
        w1*w2 - x1*x2 - y1*y2 - z1*z2,
        w1*x2 + x1*w2 + y1*z2 - z1*y2,
        w1*y2 - x1*z2 + y1*w2 + z1*x2,
        w1*z2 + x1*y2 - y1*x2 + z1*w2
    ])

def quaternion_to_euler(q):
    """Convert quaternion to Euler angles (roll, pitch, yaw)"""
    w, x, y, z = q
    
    # Roll (x-axis rotation)
    sinr_cosp = 2 * (w*x + y*z)
    cosr_cosp = 1 - 2*(x*x + y*y)
    roll = np.arctan2(sinr_cosp, cosr_cosp)
    
    # Pitch (y-axis rotation)
    sinp = 2 * (w*y - z*x)
    if abs(sinp) >= 1:
        pitch = np.copysign(np.pi/2, sinp)
    else:
        pitch = np.arcsin(sinp)
    
    # Yaw (z-axis rotation)
    siny_cosp = 2 * (w*z + x*y)
    cosy_cosp = 1 - 2*(y*y + z*z)
    yaw = np.arctan2(siny_cosp, cosy_cosp)
    
    return np.array([roll, pitch, yaw])

def normalize(v):
    """Normalize a vector"""
    norm = np.linalg.norm(v)
    if norm == 0:
        return v
    return v / norm

class OptimizedSimulation(Node):
    def __init__(self):
        super().__init__('optimized_quadruped_simulation')
        
        # ========== INITIALIZATION ==========
        self.initialize_parameters()
        self.initialize_controllers()
        self.setup_ros_communication()
        
        # Warm start MPC
        self.mpc_warm_start()

    def initialize_parameters(self):
        """Initialize all simulation parameters"""
        # Timing parameters
        self.dt = 0.01  # Main control loop period (100Hz)
        self.mpc_dt = 0.02  # MPC update period (50Hz)
        self.horizon = 2
        
        # Robot physical parameters
        self.robot_height = 0.35
        self.L = 0.433  # Body length
        self.W = 0.12   # Body width
        self.L1, self.L2, self.L3 = 0.087, 0.25, 0.25
        self.mass = 20.0
        self.I_body = np.diag([0.25, 1.05, 1.05])
        
        # CPG parameters
        self.cycles = 5
        self.Tcpg = 0.75
        self.beta = 0.75
        self.step = int(self.cycles*self.Tcpg/self.dt)
        
        # Gait control
        self.command_gait = "trot"  # Default to trot for better stability
        self.move_direction = "stop"
        self.LEG_LABELS = ["FL", "FR", "RR", "RL"]
        self.current_phase = np.zeros(4)  # Initialize current phase for each leg
        self.contact_schedule = np.zeros((4, self.step))  # Contact schedule for each leg
        self.contact_state = {label: {"contact": []} for label in self.LEG_LABELS}
        self.leg_phase = np.zeros((4, self.step))  # Phase for each leg
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
        
        # Foot parameters
        self.Phip = np.array([
            [self.L/2, self.L1 + self.W/2, -0.05],
            [self.L/2,-self.L1 - self.W/2, -0.05],
            [-self.L/2, -self.L1 - self.W/2, -0.05],
            [-self.L/2, self.L1 + self.W/2, -0.05]
        ])
        self.world_foot_pos = [self.des_pos + convert_rpy_to_rot(self.des_ori) @ self.Phip[i] for i in range(4)]
        self.base_foot_pos = np.zeros((4, 3))
        self.base_foot_vel = np.zeros((4, 3))
        self.base_foot_acc = np.zeros((4, 3))
        
        # Velocity smoothing and bounds parameters
        self.velocity_smoothing_factor = 0.4  # Alpha for exponential smoothing (0-1)
        self.max_foot_velocity = 2.0  # m/s
        self.max_foot_acceleration = 10.0  # m/s²
        
        # Initialize velocity history for smoothing
        self.smoothed_velocities = np.zeros((4, 3))
        # Initialize history storage
        self.initialize_history_arrays()

    def initialize_controllers(self):
        """Initialize all control components"""
        # MPC weights - tuned for stability
        qp_weights = np.array([
            0.1, 0.1, 120,  # Position (x,y,z)
            10.0, 10.0, 0.0,   # Velocity (vx,vy,vz)
            0.0, 0.0, 0.0,    # Orientation (roll,pitch,yaw)
            0.1, 0.1, 1.0,    # Angular velocity
            0.0               # Gravity term
        ])
        
        # Initialize MPC with optimized parameters
        self.mpc = MPC_Single_Rigid_Body_Model(
            mass=self.mass,
            I=self.I_body,
            num_legs=4,
            horizon=self.horizon,
            dt=self.mpc_dt,
            qp_weights=qp_weights,
            alpha=1e-4,
            mu=0.6
        )
        
        # Initialize other controllers
        self.BaseFoot = Base_Foot_Trajectory(self.L, self.W, dt=self.dt, k=0.03, hf=0.05)
        self.CPG = CPG_Network(Ts=self.Tcpg, cycles=self.cycles, dt=self.dt)
        
        # Timing calculations
        self.Tst = self.CPG.T * self.beta  # Stance time
        self.Tsw = self.CPG.T * (1 - self.beta)  # Swing time

    def setup_ros_communication(self):
        """Set up ROS2 publishers and subscribers"""
        # Subscribers
        self.sub_mode = self.create_subscription(
            String, '/dynamo_one/mode', self.mode_callback, 10)
        self.sub_desired = self.create_subscription(
            Float32MultiArray, 'dynamo_one/base_des', self.desired_callback, 10)
        
        # Publishers
        self.com_pub = self.create_publisher(
            Float32MultiArray, 'dynamo_one/com_pos', 1)
        
        # Timers - using ROS2 wall timers for better timing accuracy
        self.control_timer = self.create_timer(
            self.dt, self.control_callback)
        self.mpc_timer = self.create_timer(
            self.mpc_dt, self.mpc_callback)

    def initialize_history_arrays(self):
        """Initialize arrays for data logging"""
        self.history_state = {
            'actual': {
                'position': np.zeros((self.step, 3)),
                'velocity': np.zeros((self.step, 3)),
                'orientation': np.zeros((self.step, 3)),
                'angular_velocity': np.zeros((self.step, 3))
            },
            'desired': {
                'position': np.zeros((self.step, 3)),
                'velocity': np.zeros((self.step, 3)),
                'orientation': np.zeros((self.step, 3)),
                'angular_velocity': np.zeros((self.step, 3))
            },
            'time': np.zeros(self.step)
        }
        
        self.force_history = np.zeros((self.step, 4, 3))
        self.foot_trajectories = [[] for _ in range(4)]
        self.contact_schedule = np.zeros((4, self.step))
        
        # Initialize step counters
        self.step_count = 0
        self.mpc_step = 0

    def mpc_warm_start(self):
        """Initialize MPC with reasonable starting values"""
        # Q_history, self.contact_state, self.leg_phase = self.CPG.generate(
        #     gait_type=self.command_gait,
        #     move_direction=self.move_direction
        # )
        
        # Initialize forces with weight distribution
        self.current_forces = np.zeros((4, 3))
        self.current_forces[:, 2] = self.mass * 9.81 / 4  # Even weight distribution
        
        self.force_reshaped = np.tile(
            self.current_forces[np.newaxis, :, :],
            (self.horizon, 1, 1)
        )

    # ========== CORE CONTROL FUNCTIONS ==========
    def control_callback(self):
        """Main control loop running at 100Hz"""
        # Update gait and foot placement
        self.update_gait()
        self.update_foot_placement()
        
        # Update dynamics
        self.update_dynamics()
        
        # Publish state and update history
        self.publish_state()
        self.update_history()
        
        # Increment step counter
        self.step_count = (self.step_count + 1) % self.step

    def mpc_callback(self):
        """MPC update running at 50Hz"""
        self.get_logger().info(f"COM of Actual: {self.com_pos}, COM of Desired: {self.des_pos}")
        # Prepare state information
        state = self.prepare_mpc_state()
        
        # Solve QP problem
        self.solve_mpc(state)
        
        # Update MPC step counter
        self.mpc_step = (self.mpc_step + 1) % self.step

    # ========== IMPLEMENTATION DETAILS ==========
    def update_dynamics(self):
        """Improved robot dynamics with better orientation handling"""
        # Compute total force and torque in world frame
        total_force = np.zeros(3)
        total_torque = np.zeros(3)
        
        # Current orientation as rotation matrix (world to body)
        R = quaternion_to_rotation_matrix(self.current_orientation_quaternion)
        
        # Sum forces from all legs (already in world frame)
        for i in range(4):
            total_force += self.current_forces[i]
            
            # Calculate torque contribution in world frame
            r_i = R.T @ (self.world_foot_pos[i] - self.com_pos)
            total_torque += np.cross(r_i, self.current_forces[i])
        
        # Add gravity force (in world frame)
        gravity_force = np.array([0.0, 0.0, -self.mass * 9.81])
        net_force = total_force + gravity_force
        
        # Add velocity damping to prevent drift
          
        self.com_vel += (net_force / self.mass ) * self.dt
        
        # Then update position using NEW velocity
        self.com_pos += self.com_vel * self.dt
        
        # Handle ground contact (may modify both velocity and position)
        # self.handle_ground_contact()
        
        # Improved rotational dynamics
        self.update_rotational_dynamics(total_torque)
        
        # Publish updated state
        self.publish_state()
        
        # Update history
        # self.update_history()
    def update_rotational_dynamics(self, total_torque):
        """Improved rotational dynamics update"""
        # Convert torque to body frame
        R = quaternion_to_rotation_matrix(self.current_orientation_quaternion)
        total_torque_body = R @ total_torque  # World to body rotation
        
        # Compute angular acceleration with gyroscopic term
        gyroscopic = np.cross(self.com_omega, self.I_body @ self.com_omega)
        angular_accel = np.linalg.inv(self.I_body) @ (total_torque_body - gyroscopic)
        
        # Update angular velocity using semi-implicit Euler
        new_omega = self.com_omega + angular_accel * self.dt
        
        # Improved quaternion integration
        self.update_orientation_quaternion(new_omega)
        
        # Update stored omega
        self.com_omega = new_omega
        self.com_ori = quaternion_to_euler(self.current_orientation_quaternion)

    def update_orientation_quaternion(self, omega):
        """More accurate quaternion integration using exponential map"""
        if np.linalg.norm(omega) < 1e-6:
            return  # No rotation
        
        # Convert omega to quaternion rate
        omega_norm = np.linalg.norm(omega)
        axis = omega / omega_norm
        angle = omega_norm * self.dt
        
        # Create rotation quaternion
        half_angle = angle / 2
        delta_q = np.array([
            np.cos(half_angle),
            *(axis * np.sin(half_angle))
        ])
        
        # Apply rotation and normalize
        self.current_orientation_quaternion = quaternion_multiply(
            delta_q, self.current_orientation_quaternion)
        self.current_orientation_quaternion /= np.linalg.norm(self.current_orientation_quaternion)
    def handle_ground_contact(self):
        """Properly handle ground contact and friction for x,y,z motion"""
        # Only apply if we're below nominal height
        # Add small corrective force when near nominal height
        height_error = self.robot_height - self.com_pos[2]
        if abs(height_error) < 0.01:  # Near nominal height
            k_correction = 1000  # N/m
            correction_force = np.array([0, 0, k_correction * height_error])
            self.com_vel += (correction_force / self.mass) * self.dt
        # if self.com_pos[2] < self.robot_height:
        #     penetration = self.robot_height - self.com_pos[2]
            
        #     # Spring-damper model parameters (tuned for 40kg robot)
        #     k_ground = 20000  # N/m (stiffer for less penetration)
        #     d_ground = 500    # Ns/m (higher damping for less oscillation)
            
        #     # Normal force calculation
        #     ground_force_z = k_ground * penetration - d_ground * self.com_vel[2]
        #     ground_force_z = max(ground_force_z, 0)  # Can't pull on ground
            
        #     # Apply normal force to z-velocity
        #     self.com_vel[2] += (ground_force_z / self.mass) * self.dt
            
        #     # Friction model (only if we have normal force)
        #     if ground_force_z > 0:
        #         friction_coeff = 0.5  # Reasonable for rubber feet
        #         max_friction = friction_coeff * ground_force_z
                
        #         # Current horizontal speed
        #         v_horizontal = np.linalg.norm(self.com_vel[:2])
                
        #         if v_horizontal > 0.001:  # Small threshold to avoid jitter
        #             # Friction direction opposes motion
        #             friction_dir = -self.com_vel[:2] / v_horizontal
                    
        #             # Friction magnitude (static + dynamic)
        #             friction_mag = min(
        #                 max_friction,  # Static friction limit
        #                 0.1 * self.mass * v_horizontal / self.dt  # Dynamic friction
        #             )
        #             friction_force = friction_dir * friction_mag
                    
        #             # Apply friction to x,y velocities
        #             self.com_vel[:2] += (friction_force / self.mass) * self.dt
            
        #     # Position correction (ensure we don't go below ground)
        #     self.com_pos[2] = max(self.com_pos[2], self.robot_height)
    def update_foot_placement(self):
        """Update foot placement based on gait phase"""
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
            if phase <= self.CPG.beta:  # Stance phase
                self.contact_schedule[i, self.step_count] = 1
                # In stance, foot moves with the ground (fixed in world frame)
                new_foot_world = current_foot_world.copy()
                new_foot_world[2] = 0.0  # Keep foot on ground

                raw_velocity = -Rb.T @ self.des_vel
                self.base_foot_acc[i] = -Rb.T @ np.zeros(3)  # Assuming zero body acceleration

            # SWING PHASE - use your original SwingTrajectory  
            else:  # Swing phase
                self.contact_schedule[i, self.step_count] = 0
                phi_swing = (phase - self.CPG.beta) / (1 - self.CPG.beta)
                
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
            self.base_foot_vel[i] = self.apply_velocity_limits(i, raw_velocity)
            # Store new foot position (limit history length)
            self.foot_trajectories[i].append(self.world_foot_pos[i].copy())
            # self.get_logger().info(f"Foot position start: {Ps_world}, end: {Pe_world}")
        self.get_logger().info(f"Foot trajectories in world frame: {self.world_foot_pos[:]}")
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
    def update_gait(self):
        """Update CPG network based on current gait command"""
        self.cpg = CPG_Network(Ts=self.Tcpg, cycles=self.cycles, dt=self.dt)
        self.leg_phase = self.CPG.generate(gait_type=self.command_gait, move_direction=self.move_direction)
        self.step = int(self.cycles*self.CPG.T/self.dt)
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
        # Apply threshold to small forces
        force_threshold = 0.1  # Newtons
        self.current_forces = np.where(
            np.abs(self.force_reshaped[0]) < force_threshold,
            0,
            self.force_reshaped[0]
        )
        
        # Store force history
        self.force_history[self.mpc_step] = self.current_forces.copy()

    def get_horizon_contacts(self):
        """Get contact schedule for MPC horizon"""
        horizon_contacts = np.zeros((self.horizon, 4))
        for i in range(self.horizon):
            idx = (self.mpc_step + i) % self.step
            horizon_contacts[i] = self.contact_schedule[:, idx]
        return horizon_contacts

    # ========== ROS CALLBACKS ==========
    def mode_callback(self, msg):
        """Handle gait mode changes"""
        self.command_gait = msg.data
        if self.command_gait == "walk":
            self.beta = self.CPG.beta_walk
        elif self.command_gait == "trot":
            self.beta = self.CPG.beta_trot
        
        # Regenerate gait pattern
        self.leg_phase = self.CPG.generate(
            gait_type=self.command_gait,
            move_direction=self.move_direction
        )

    def desired_callback(self, msg):
        """Handle desired state updates"""
        self.des_pos = np.array(msg.data[0:3]) 
        # self.des_pos[0:2] = self.des_pos[0:2]
        self.des_vel = np.array(msg.data[3:6])
        self.des_ori = np.array(msg.data[6:9])
        self.des_omega = np.array(msg.data[9:12])
        
        # Update movement direction
        self.update_movement_direction()

    def update_movement_direction(self):
        """Determine movement direction from desired velocity"""
        if self.des_vel[0] >= 0.2 and self.des_vel[1] == 0.0:
            self.move_direction = "forward"
        elif self.des_vel[0] <= -0.2 and self.des_vel[1] == 0.0:
            self.move_direction = "backward"
        elif self.des_vel[1] >= 0.2 and self.des_vel[0] == 0.0:
            self.move_direction = "right"
        elif self.des_vel[1] <= -0.2 and self.des_vel[0] == 0.0:
            self.move_direction = "left"
        else:
            self.move_direction = "stop"

    def publish_state(self):
        """Publish current state"""
        com_msg = Float32MultiArray()
        com_msg.data = [
            *self.com_pos, *self.com_vel,
            *self.com_ori, *self.com_omega
        ]
        self.com_pub.publish(com_msg)

    def update_history(self):
        """Update history arrays for logging"""
        idx = self.step_count
        self.history_state['actual']['position'][idx] = self.com_pos
        self.history_state['actual']['velocity'][idx] = self.com_vel
        self.history_state['actual']['orientation'][idx] = self.com_ori
        self.history_state['actual']['angular_velocity'][idx] = self.com_omega
        
        self.history_state['desired']['position'][idx] = self.des_pos
        self.history_state['desired']['velocity'][idx] = self.des_vel
        self.history_state['desired']['orientation'][idx] = self.des_ori
        self.history_state['desired']['angular_velocity'][idx] = self.des_omega
        
        self.history_state['time'][idx] = self.step_count * self.dt

def main(args=None):
    rclpy.init(args=args)
    controller = OptimizedSimulation()
    
    try:
        rclpy.spin(controller)
    except KeyboardInterrupt:
        controller.get_logger().info("Shutting down controller...")
    finally:
        controller.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()