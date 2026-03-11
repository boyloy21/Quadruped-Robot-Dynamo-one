from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from ament_index_python.packages import get_package_share_directory
from launch_ros.actions import Node
from launch.actions import DeclareLaunchArgument

def generate_launch_description():
    joy = Node(
        package='joy',
        executable='joy_node',
        name='joy_node',
        parameters=[{'device_id': 0}],
        output='screen'
    )
    ps4_controller = Node(
        package='python_controller',
        executable='ps4_desired',
        name='ps4_desired',
        output='screen',
    )
    mpc = Node(
        package='python_controller',
        executable='sml_mpc',
        name='sml_mpc',
        output='screen',
    )
    return LaunchDescription([
        joy,
        ps4_controller,
        mpc
    ])