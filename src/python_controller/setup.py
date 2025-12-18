from setuptools import find_packages, setup
import os
from glob import glob
package_name = 'python_controller'

setup(
    name=package_name,
    version='0.0.0',
    # packages=find_packages(exclude=['test']),
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name, 'launch'), glob('launch/*.launch.py')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='yin',
    maintainer_email='yin@todo.todo',
    description='TODO: Package description',
    license='Apache-2.0',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'dynamo_one_test1 = python_controller.ps4_joint_controller:main',
            'foot_trajectory_control = python_controller.FootTrajectory_control:main',
            'foot_rviz = python_controller.Foot_trajectory_rviz:main',
            'balance_control = python_controller.Balance_control:main',
            'foot_trajectory_rviz = python_controller.FootTrajectory_rviz:main',
            'ps4_testinv = python_controller.ps4_testjoint:main',
            'ps4_torque = python_controller.ps4_test_torque:main',
            'dynamo_control = python_controller.Dynamo_control:main',
            'ps4_controller = python_controller.PS4_Controller:main',
            'gait_control_rviz = python_controller.Gait_control_Rviz:main',
            'state_estimate = python_controller.State_estimate:main',
            'ps4_controllerV2 = python_controller.PS4_controllerV2:main',
            'rviz_control = python_controller.Rviz_control_robot:main',
            'rviz_control2 = python_controller.Rviz_Control_RobotV2:main',
            'dynamo_control2 = python_controller.Dynamo_controlV2:main',
            'ps4_desired = python_controller.PS4_Desired:main',
            'sml_mpc = python_controller.Simulation_MPC_SRBM:main',
            'sim_mpcv2 = python_controller.Simulation_MPC_SRBMV2:main',
            'legcontroller = python_controller.Leg_PIDController:main',
            'dynamo_balance = python_controller.Dynamo_control_BalancePID:main',
            'dynamo_controlv3 = python_controller.Dynamo_controlV3:main',
        ],
    },
)
