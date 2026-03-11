import os
from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from ament_index_python.packages import get_package_share_directory
from launch_ros.actions import Node

def generate_launch_description():
    # controller_pkg = get_package_share_directory('controller')  # Replace with your package name
    remote_pkg = get_package_share_directory('remote_control')  # Replace with your package name
    dynamo_control_gazebo = Node(
        package='controller',
        executable='dynamo_control_gazebo.py',
        name='robot_controller',
        output='screen',
    )

    remote_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(os.path.join(remote_pkg, 'launch', 'ramped_joystick.launch.py'))
    )

    return LaunchDescription([
        dynamo_control_gazebo,
        remote_launch
    ])
