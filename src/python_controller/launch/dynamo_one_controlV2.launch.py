from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from ament_index_python.packages import get_package_share_directory
from launch_ros.actions import Node
from launch.actions import DeclareLaunchArgument


def generate_launch_description():

    state_estimate = Node(
        package='python_controller',
        executable='state_estimate',
        name='state_estimate',
        output='screen'
    )
    DynamoControl = Node(
        package='python_controller',
        executable='dynamo_controlv3',
        name='dynamo_control2',
        output='screen',
    )

    ps4_controller = Node(
        package='python_controller',
        executable='ps4_controllerV2',
        name='ps4_controllerV2',
        output='screen',
    )

    
    return LaunchDescription([
        state_estimate,
        DynamoControl,
        ps4_controller
    ])
