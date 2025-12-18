#!/usr/bin/evn python3

import numpy as np
from Controller.StateCommand import State, Command, BehaviorState
from Controller.Rest_controller import RestController
from Controller.Stand_controller import StandController

def euler_from_quaternion(quaternion):
    """
    Converts quaternion (w in last place) to euler roll, pitch, yaw
    quaternion = [x, y, z, w]
    Bellow should be replaced when porting for ROS 2 Python tf_conversions is done.
    """
    x = quaternion.x
    y = quaternion.y
    z = quaternion.z
    w = quaternion.w

    sinr_cosp = 2 * (w * x + y * z)
    cosr_cosp = 1 - 2 * (x * x + y * y)
    roll = np.arctan2(sinr_cosp, cosr_cosp)

    sinp = 2 * (w * y - z * x)
    pitch = np.arcsin(sinp)

    siny_cosp = 2 * (w * z + x * y)
    cosy_cosp = 1 - 2 * (y * y + z * z)
    yaw = np.arctan2(siny_cosp, cosy_cosp)

    return roll, pitch, yaw
class RobotController(object):
    def __init__(self, body, legs, imu):
        self.body = body
        self.legs = legs
        self.imu = imu

        self.delta_x = self.body[0] * 0.5
        self.delta_y = self.body[1] * 0.5 + self.legs[1]
        self.x_shift_front = 0.02
        self.x_shift_back = -0.0
        self.default_height = 0.25

        self.rest_controller = RestController(self.default_stance)
        self.stand_controller = StandController(self.default_stance)

        self.currentController = self.rest_controller
        self.state = State(self.default_height)
        self.state.foot_localtions = self.default_stance
        self.command = Command(self.default_height)

    def change_controller(self):

        if self.command.rest_event:
            self.state.behavior_state = BehaviorState.REST
            self.currentController = self.rest_controller
            self.currentController.pid_controller.reset()
            self.command.rest_event = False
        elif self.command.stand_event:
            if self.state.behavior_state == BehaviorState.REST:
                self.state.behavior_state = BehaviorState.STAND
                self.currentController = self.stand_controller
                self.state.body_local_position[2] = 0.08 * self.default_height
            self.command.stand_event = False

    def joystick_command(self, msg):
        if msg.buttons[0]: # Rest
            self.command.rest_event = True
            self.command.stand_event = False
        elif msg.buttons[1]: # Stand
            self.command.stand_event = True
            self.command.rest_event = False

        self.currentController.updateStateCommand(msg, self.state, self.command)

    def imu_orientation(self, msg):
        q = msg.orientation
        roll, pitch, yaw = euler_from_quaternion(q)
        self.state.imu_roll = roll
        self.state.imu_pitch = pitch

    def run(self):
        return self.currentController.run(self.state, self.command)
    
    @property
    def default_stance(self):
        ### FR, FL, BR, BL
        return np.array([[self.delta_x + self.x_shift_front, self.delta_x + self.x_shift_front, -self.delta_x + self.x_shift_back, -self.delta_x + self.x_shift_back],
                         [-self.delta_y , self.delta_y, -self.delta_y, self.delta_y],
                         [-0.25, -0.25, -0.25, -0.25]])

        
        
