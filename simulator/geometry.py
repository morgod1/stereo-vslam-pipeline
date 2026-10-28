"""Explicit CARLA-to-right-handed transforms for stereo data."""

import math

import numpy as np


# World/vehicle: x forward, y left, z up. Optical: x right, y down, z forward.
WORLD_FROM_CARLA = np.diag([1.0, -1.0, 1.0, 1.0])
CARLA_FROM_OPTICAL = np.array([
    [0, 0, 1, 0], [1, 0, 0, 0], [0, -1, 0, 0], [0, 0, 0, 1],
], dtype=float)


def optical_pose(carla_matrix):
    return WORLD_FROM_CARLA @ np.asarray(carla_matrix) @ CARLA_FROM_OPTICAL


def vehicle_pose(carla_matrix):
    return WORLD_FROM_CARLA @ np.asarray(carla_matrix) @ WORLD_FROM_CARLA


def calibration(config):
    focal = config["width"] / (2 * math.tan(math.radians(config["fov"]) / 2))
    mounts = {}
    for name, sign in (("left", -1), ("right", 1)):
        mount = np.eye(4)
        mount[:3, 3] = [config["mount_x"], sign * config["baseline_m"] / 2, config["mount_z"]]
        mounts[name] = optical_pose(mount).tolist()
    right_from_left = np.eye(4)
    right_from_left[0, 3] = -config["baseline_m"]
    return {
        "width": config["width"], "height": config["height"],
        "sample_interval_seconds": config["fixed_delta_seconds"],
        "K": [[focal, 0, config["width"] / 2], [0, focal, config["height"] / 2], [0, 0, 1]],
        "distortion_model": "pinhole", "distortion_coefficients": [0, 0, 0, 0, 0],
        "baseline_m": config["baseline_m"], "T_right_left": right_from_left.tolist(),
        "T_vehicle_left": mounts["left"], "T_vehicle_right": mounts["right"],
        "convention": "T_target_source maps source coordinates into target; column vectors; meters",
        "world_vehicle_axes": "right-handed: x forward, y left, z up",
        "camera_axes": "right-handed optical: x right, y down, z forward",
    }
