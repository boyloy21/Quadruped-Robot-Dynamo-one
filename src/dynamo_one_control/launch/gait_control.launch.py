from launch import LaunchDescription
from launch_ros.actions import Node


def generate_launch_description():

    state_estimator = Node(
        package='dynamo_one_control',
        executable='state_estimate',
        name='state_estimate',
        output='screen',
    )

    mpc = Node(
        package='dynamo_one_control',
        executable='mpc_force',
        name='mpc_force',
        output='screen',
    )

    leg_control_body = Node(
        package='dynamo_one_control',
        executable='leg_control',
        name='leg_control',
        output='screen',
    )

    

    return LaunchDescription([
        leg_control_body,
        state_estimator,
        mpc
    ])
    