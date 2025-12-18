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
    init_height = '0.15'  # Set height explicitly to 1.0
    pkg_path = os.path.join(get_package_share_directory(package_description))

    # Load robot description from XACRO file
    xacro_file = os.path.join(pkg_path, 'xacro', 'robot.xacro')
    robot_description = xacro.process_file(xacro_file, mappings={'GAZEBO': 'true'}).toxml()

    # Load RViz Configuration File
    # rviz_config_file = os.path.join(get_package_share_directory(package_description), "config", "visualize_urdf.rviz")

    # Launch RViz
    # rviz = Node(
    #     package='rviz2',
    #     executable='rviz2',
    #     name='rviz_ocs2',
    #     output='screen',
    #     arguments=["-d", rviz_config_file]
    # )
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
        parameters=[{'robot_description': robot_description}],
    )


    load_joint_state_broadcaster = ExecuteProcess(
        cmd=['ros2', 'control', 'load_controller', '--set-state', 'active', 'joint_state_broadcaster'],
        output='screen',
    )

    load_dynamo_one_joint_positions = ExecuteProcess(
        cmd=['ros2', 'control', 'load_controller', '--set-state', 'active',
             'forward_position_controller'],
        output='screen',
    )
    
    # Bridge for Gazebo communication
    bridge = Node(
    package='ros_gz_bridge',
    executable='parameter_bridge',
    arguments=[
        '/imu@sensor_msgs/msg/Imu@ignition.msgs.IMU',
        '/clock@rosgraph_msgs/msg/Clock@ignition.msgs.Clock',
    ],
    output='screen'
)
    
    
    return [
        RegisterEventHandler(
            event_handler=OnProcessExit(
                target_action=gz_spawn_entity,
                on_exit=[load_joint_state_broadcaster],
            )
        ),
        RegisterEventHandler(
            event_handler=OnProcessExit(
               target_action=load_joint_state_broadcaster,
               on_exit=[load_dynamo_one_joint_positions],
            )
        ),
        # rviz,
        # joint_state_publisher_gui,
        robot_state_publisher,
        IncludeLaunchDescription(
            PythonLaunchDescriptionSource(
                [PathJoinSubstitution([FindPackageShare('ros_gz_sim'),
                                    'launch',
                                    'gz_sim.launch.py'])]),
            launch_arguments=[('gz_args', '-r -v 4 ' + os.path.join(pkg_path, 'worlds', 'world.sdf'))]
        ),
        
        gz_spawn_entity,
        bridge,
        
    ]

def generate_launch_description():
    return LaunchDescription([
        OpaqueFunction(function=launch_setup),
    ])
