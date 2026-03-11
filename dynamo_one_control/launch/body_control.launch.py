from launch import LaunchDescription
from launch_ros.actions import Node


def generate_launch_description():

    # joy = Node(
    #     package='joy',
    #     executable='joy_node',
    #     name='joy_node',
    #     parameters=[{'device_id': 0}],
    #     output='screen'
    # )
    body_control = Node(
        package='dynamo_one_control',
        executable='body_control',
        name='body_control',
        output='screen',
    )

    # ps4_controller = Node(
    #     package='dynamo_one_control',
    #     executable='ps4_control',
    #     name='ps4_control',
    #     output='screen',
    # )

    ps4_controller = Node(
        package='dynamo_one_control',
        executable='ps4_desired',
        name='ps4_controller',
        output='screen',
    )

    leg_control = Node(
        package='dynamo_one_control',
        executable='leg_control',
        name='leg_control',
        output='screen',
    )


    
    return LaunchDescription([
        ps4_controller,
        body_control,
        leg_control
    ])
