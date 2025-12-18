from launch import LaunchDescription
from launch_ros.actions import Node


def generate_launch_description():

    
    ps4_controller = Node(
        package='dynamo_one_control',
        executable='ps4_desired',
        name='ps4_desired',
        output='screen',
    )

    body_control = Node(
        package='dynamo_one_control',
        executable='body_control',
        name='body_control',
        output='screen',
    )

    foot_trajectory = Node(
        package='dynamo_one_control',
        executable='foot_generator',
        name='ps4_control',
        output='screen',
    )

    return LaunchDescription([
        ps4_controller,
        body_control,
        foot_trajectory
    ])
