import os
import xacro
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, OpaqueFunction, IncludeLaunchDescription, RegisterEventHandler, ExecuteProcess
from launch.event_handlers import OnProcessExit
from launch.substitutions import PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare
from launch.launch_description_sources import PythonLaunchDescriptionSource

def launch_setup(context, *args, **kwargs):
    package_description = 'dynamo_one_description'  # Hardcoded package name
    init_height = '0.3'  # Set height explicitly to 1.0
    pkg_path = os.path.join(get_package_share_directory(package_description))

    # Load robot description from XACRO file
    xacro_file = os.path.join(pkg_path, 'xacro', 'robot.xacro')
    robot_description = xacro.process_file(
        xacro_file, 
        mappings={
            'GAZEBO': 'true',   # Enable Gazebo-specific configurations
            'CONTROL': 'true'  # select gazebo_position.xacro
        }
    ).toxml()
    # Load RViz Configuration File
    # rviz_config_file = os.path.join(get_package_share_directory(package_description), "config", "visualize_urdf.rviz")

    
    gz_spawn_entity = Node(
        package='ros_gz_sim',
        executable='create',
        output='screen',
        arguments=['-string', robot_description,
                   '-x', '0.0',
                   '-y', '0.0',
                   '-z', init_height,
                   '-R', '0.0',
                   '-P', '0.0',
                   '-Y', '0.0',
                   '-name', 'dynamo_one',
                   '-allow_renaming', 'true'],
    )

    
    # Publish Robot State
    robot_state_publisher = Node(
        package='robot_state_publisher',
        executable='robot_state_publisher',
        name='robot_state_publisher',
        parameters=[
            {
                'publish_frequency': 50.0,
                'use_tf_static': True,
                'robot_description': robot_description,
                'ignore_timestamp': True
            }
        ],
        # parameters=[{'robot_description': robot_description}],
    )


    load_joint_state_broadcaster = ExecuteProcess(
        cmd=['ros2', 'control', 'load_controller', '--set-state', 'active', 'joint_state_broadcaster'],
        output='screen',
    )

    load_dynamo_one_joint_effort = ExecuteProcess(
        cmd=['ros2', 'control', 'load_controller', '--set-state', 'active',
             'forward_position_controller'],
        output='screen',
    )
    
    initial_joint_state_publisher = Node(
        package='dynamo_one_description',
        executable='Initialjointposition.py',
        name='initial_joint_state_publisher',
        output='screen',
    )
    
    
    # Bridge for Gazebo communication
    bridge = Node(
    package='ros_gz_bridge',
    executable='parameter_bridge',
    arguments=[
        '/imu@sensor_msgs/msg/Imu@ignition.msgs.IMU',
        '/clock@rosgraph_msgs/msg/Clock@ignition.msgs.Clock',
        '/world/world_demo/model/dynamo_one/joint/FR_foot_fixed/sensor/FR_ft_sensor/forcetorque@geometry_msgs/msg/Wrench[ignition.msgs.Wrench',
        '/world/world_demo/model/dynamo_one/joint/FL_foot_fixed/sensor/FL_ft_sensor/forcetorque@geometry_msgs/msg/Wrench[ignition.msgs.Wrench',
        '/world/world_demo/model/dynamo_one/joint/RR_foot_fixed/sensor/RR_ft_sensor/forcetorque@geometry_msgs/msg/Wrench[ignition.msgs.Wrench',
        '/world/world_demo/model/dynamo_one/joint/RL_foot_fixed/sensor/RL_ft_sensor/forcetorque@geometry_msgs/msg/Wrench[ignition.msgs.Wrench',
    ],
    remappings=[
        ('/imu', 'dynamo_one/imu'),
        ('/world/world_demo/model/dynamo_one/joint/FR_foot_fixed/sensor/FR_ft_sensor/forcetorque', 'dynamo_one/FR_ft_sensor'),
        ('/world/world_demo/model/dynamo_one/joint/FL_foot_fixed/sensor/FL_ft_sensor/forcetorque', 'dynamo_one/FL_ft_sensor'),
        ('/world/world_demo/model/dynamo_one/joint/RR_foot_fixed/sensor/RR_ft_sensor/forcetorque', 'dynamo_one/RR_ft_sensor'),
        ('/world/world_demo/model/dynamo_one/joint/RL_foot_fixed/sensor/RL_ft_sensor/forcetorque', 'dynamo_one/RL_ft_sensor'),
    ],

    output='screen'
    )
    
    
    return [
        IncludeLaunchDescription(
            PythonLaunchDescriptionSource(
                [PathJoinSubstitution([FindPackageShare('ros_gz_sim'),
                                    'launch',
                                    'gz_sim.launch.py'])]),
            launch_arguments=[('gz_args', '-r -v 4 ' + os.path.join(pkg_path, 'worlds', 'world.sdf'))]
        ),
        # initial_joint_state_publisher,
        robot_state_publisher,
        gz_spawn_entity,
        bridge,
        RegisterEventHandler(
            event_handler=OnProcessExit(
                target_action=initial_joint_state_publisher,
                on_exit=[load_joint_state_broadcaster],
            )
        ),
        RegisterEventHandler(
            event_handler=OnProcessExit(
                target_action=gz_spawn_entity,
                on_exit=[load_joint_state_broadcaster],
            )
        ),
        RegisterEventHandler(
            event_handler=OnProcessExit(
               target_action=load_joint_state_broadcaster,
               on_exit=[load_dynamo_one_joint_effort],
            )
        ),
        
    ]

def generate_launch_description():
    return LaunchDescription([
        OpaqueFunction(function=launch_setup),
    ])
