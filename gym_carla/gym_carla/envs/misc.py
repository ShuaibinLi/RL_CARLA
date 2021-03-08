#!/usr/bin/env python

# This file is modified by Dongjie yu (yudongjie.moon@foxmail.com)
# from <https://github.com/cjy1992/gym-carla>:
# Copyright (c) 2019:
# authors: Jianyu Chen (jianyuchen@berkeley.edu).

import math
import numpy as np
import carla


def command2Vector(command):
    """
    Convert command(scalar) to vector to be used in FC-net
    param: command(1, float)
        REACH_GOAL = 0.0
        GO_STRAIGHT = 5.0
        TURN_RIGHT = 4.0
        TURN_LEFT = 3.0
        LANE_FOLLOW = 2.0
    return: command vector(np.array, 5*1) [1 0 0 0 0]
        0-REACH_GOAL
        1-LANE_FOLLOW
        2-TURN_LEFT
        3-TURN_RIGHT
        4-GO_STRAIGHT
    """
    command_vec = np.zeros((5, 1))
    REACH_GOAL = 0.0
    GO_STRAIGHT = 5.0
    TURN_RIGHT = 4.0
    TURN_LEFT = 3.0
    LANE_FOLLOW = 2.0
    if command == REACH_GOAL:
        command_vec[0] = 1.0
    elif command > 1 and command < 6:
        command_vec[int(command) - 1] = 1.0
    else:
        raise ValueError("Command Value out of bound!")

    return command_vec


def _vec_decompose(vec_to_be_decomposed, direction):
    """
    Decompose the vector along the direction vec
    params:
        vec_to_be_decomposed: np.array, shape:(2,)
        direction: np.array, shape:(2,); |direc| = 1
    return:
        vec_longitudinal
        vec_lateral
            both with sign
    """
    assert vec_to_be_decomposed.shape[0] == 2, direction.shape[0] == 2
    lon_scalar = np.inner(vec_to_be_decomposed, direction)
    lat_vec = vec_to_be_decomposed - lon_scalar * direction
    lat_scalar = np.linalg.norm(lat_vec) * np.sign(lat_vec[0] * direction[1] -
                                                   lat_vec[1] * direction[0])
    return np.array([lon_scalar, lat_scalar], dtype=np.float32)


def delta_angle_between(theta_1, theta_2):
    """
    Compute the delta angle between theta_1 & theta_2(both in degree)
    params:
        theta: float
    return:
        delta_theta: float, in [-pi, pi]
    """
    theta_1 = theta_1 % 360
    theta_2 = theta_2 % 360
    delta_theta = theta_2 - theta_1
    if 180 <= delta_theta and delta_theta <= 360:
        delta_theta -= 360
    elif -360 <= delta_theta and delta_theta <= -180:
        delta_theta += 360
    return delta_theta


if __name__ == '__main__':
    print(command2Vector(4.0))
    print(command2Vector(5.0))
