from launch_ros.actions import Node
from launch import LaunchDescription

def generate_launch_description():
    return LaunchDescription([
        Node(
            package='python_controller',
            executable='legcontroller',
            name='leg_controller',
            parameters=['config/parameters.yaml'],
            output='screen',
        )
    ])
    