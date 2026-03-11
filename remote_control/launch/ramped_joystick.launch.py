import launch
import launch_ros.actions

def generate_launch_description():
    return launch.LaunchDescription([
        # Joystick node
        launch_ros.actions.Node(
            package="joy",
            executable="joy_node",
            name="joystick",
            parameters=[{"dev": "/dev/input/js0"}],
        ),

        # Ramped Joystick node
        launch_ros.actions.Node(
            package="remote_control",
            executable="ramped_joystick.py",
            name="ramped_joystick",
            output="screen",
        ),
    ])
