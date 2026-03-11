from setuptools import find_packages, setup
import os
from glob import glob
package_name = 'dynamo_one_control'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name, 'launch'), glob('launch/*.launch.py')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='yinchheanyun',
    maintainer_email='yinchheanyun@todo.todo',
    description='TODO: Package description',
    license='Apache-2.0',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'ps4_desired = dynamo_one_control.PS4_Desired:main',
            'leg_pd_controller = dynamo_one_control.Leg_PD_Control:main',
            'state_estimate = dynamo_one_control.State_estimate:main',
            'mpc_force = dynamo_one_control.MPC_Solve_Force:main',
            'body_control = dynamo_one_control.Body_control:main',
            'leg_control = dynamo_one_control.PD_Control_Leg:main',
            'ps4_control = dynamo_one_control.PS4_control:main',
            'foot_generator = dynamo_one_control.Foot_generator:main',
            'sim_mpc = dynamo_one_control.Simulation_MPC_Script:main',
        ],
    },
)
