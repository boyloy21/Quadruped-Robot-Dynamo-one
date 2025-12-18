from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from ament_index_python.packages import get_package_share_directory
from launch_ros.actions import Node
from launch.actions import DeclareLaunchArgument


def generate_launch_description():

    # joy = Node(
    #     package='joy',
    #     executable='joy_node',
    #     name='joy_node',
    #     parameters=[{'device_id': 0}],
    #     output='screen'
    # )
    state_estimate = Node(
        package='python_controller',
        executable='state_estimate',
        name='state_estimate',
        output='screen',
    )
    DynamoControl = Node(
        package='python_controller',
        executable='dynamo_balance',
        name='dynamo_balance',
        output='screen',
    )

    ps4_controller = Node(
        package='python_controller',
        executable='ps4_controllerV2',
        name='ps4_controllerV2',
        output='screen',
    )

    
    return LaunchDescription([
        ps4_controller,
        DynamoControl,
        state_estimate
    ])
