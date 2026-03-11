from launch import LaunchDescription
from launch_ros.actions import Node

def generate_launch_description():
    return LaunchDescription([
        Node(
            package='joy',
            executable='joy_node',
            name='joy_node',
            parameters=[{'device_id': 0}]
        ),
         Node(
            package='remote_control',
            executable='remote_controller.py',
            name='ps4_controller',
            output='screen'
        ),
    ])