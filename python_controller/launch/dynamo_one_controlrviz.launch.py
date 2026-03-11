from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from ament_index_python.packages import get_package_share_directory
from launch_ros.actions import Node
from launch.actions import DeclareLaunchArgument


def generate_launch_description():

    DynamoControl = Node(
        package='python_controller',
        executable='rviz_control2',
        name='rviz_control',
        output='screen',
    )

    ps4_controller = Node(
        package='python_controller',
        executable='ps4_controllerV2',
        name='ps4_controllerV2',
        output='screen',
    )

    # joy = Node(
    #     package='joy',
    #     executable='joy_node',
    #     name='joy_node',
    #     parameters=[{'device_id': 0}],
    #     output='screen'
    # )
    return LaunchDescription([
        DynamoControl,
        ps4_controller,
        # joy,
    ])
