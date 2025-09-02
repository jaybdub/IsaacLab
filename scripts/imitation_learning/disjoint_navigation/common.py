# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Script to replay demonstrations with Isaac Lab environments."""

"""Launch Isaac Sim Simulator first."""


import enum
import numpy as np
import random
import torch
from dataclasses import dataclass

from isaacsim.replicator.mobility_gen.impl.path_planner import compress_path, generate_paths
from occupancy_map import OccupancyMap, intersect_occupancy_maps

import isaaclab.utils.math as math_utils


def transform_to_matrix(transform: torch.Tensor):
    """Convert a translation, quaternion pose representation into a transformation matrix."""
    pose_matrix = math_utils.make_pose(transform[..., :3], math_utils.matrix_from_quat(transform[..., 3:]))
    return pose_matrix


def transform_from_matrix(matrix: torch.Tensor):
    """Convert a transformation matrix to a translation, quaternion pose."""
    pos, rot = math_utils.unmake_pose(matrix)
    quat = math_utils.quat_from_matrix(rot)
    return torch.cat([pos, quat], dim=-1)


def transform_inv(transform: torch.Tensor):
    """Invert a translation, quaternion format transformation."""
    matrix = transform_to_matrix(transform)
    matrix = math_utils.pose_inv(matrix)
    return transform_from_matrix(matrix)


def transform_mul(transform_a, transform_b):
    """Multiply a two translation, quaternion pose representations to apply the transformation."""
    return transform_from_matrix(torch.matmul(transform_to_matrix(transform_a), transform_to_matrix(transform_b)))


def transform_relative_pose(world_pose: torch.Tensor, src_frame_pose: torch.Tensor, dst_frame_pose: torch.Tensor):
    """Compute the relative pose with respect to a source frame, and apply this relative pose to a destination frame."""
    pose = transform_mul(dst_frame_pose, transform_mul(transform_inv(src_frame_pose), world_pose))

    return pose


@dataclass
class DisjointNavRecordingItem:
    """Data container for in-place manipulation recording state.  Used during locomanipulation replay."""

    left_hand_pose_target: torch.Tensor  # The pose of the left hand in world coordinates.
    right_hand_pose_target: torch.Tensor  # The pose of the right hand in world coordinates.
    left_hand_joint_positions_target: torch.Tensor  # The left hand joint positions.
    right_hand_joint_positions_target: torch.Tensor  # The right hand joint positions.
    base_pose: torch.Tensor  # The robot base pose in world coordinates.
    object_pose: torch.Tensor  # The target object pose in world coordinates.
    fixture_pose: torch.Tensor  # The fixture (ie: table) pose in world coordinates.


class DisjointNavRecording:
    """An abstract class representing an in-place manipulation recording.  Used during locomanipulation replay."""

    def get_initial_state(self):
        """Get the initial state from the recording..  Used to reset the environment to an initial state."""
        raise NotImplementedError

    def get_item(self, step: int) -> DisjointNavRecordingItem:
        """Get the recording item for a particular timestep.  This recording item contains the relevant information
        needed for upper body replay."""
        raise NotImplementedError


class HasOccupancyMap:
    """An abstract base class for entities that have an associated occupancy map."""

    def get_occupancy_map(self) -> OccupancyMap:
        raise NotImplementedError


class HasPose2d:
    """An abstract base class for entities that have an associated 2D pose."""

    def get_pose_2d(self) -> torch.Tensor:
        """Get the 2D pose of the entity."""
        raise NotImplementedError

    def get_transform_2d(self):
        """Get the 2D transformation matrix of the entity."""

        pose = self.get_pose_2d()

        x = pose[..., 0]
        y = pose[..., 1]
        theta = pose[..., 2]
        ctheta = torch.cos(theta)
        stheta = torch.sin(theta)

        dims = tuple(list(pose.shape)[:-1] + [3, 3])
        transform = torch.zeros(dims)

        transform[..., 0, 0] = ctheta
        transform[..., 0, 1] = -stheta
        transform[..., 1, 0] = stheta
        transform[..., 1, 1] = ctheta
        transform[..., 0, 2] = x
        transform[..., 1, 2] = y
        transform[..., 2, 2] = 1.0

        return transform


class HasPose(HasPose2d):
    """An abstract base class for entities that have an associated 3D pose."""

    def get_pose(self):
        """Get the 3D pose of the entity."""
        raise NotImplementedError

    def get_pose_2d(self):
        """Get the 2D pose of the entity."""
        pose = self.get_pose()
        axis_angle = math_utils.axis_angle_from_quat(pose[..., 3:])

        yaw = axis_angle[..., 2:3]
        xy = pose[..., :2]

        pose_2d = torch.cat([xy, yaw], dim=-1)

        return pose_2d


class SceneBody(HasPose):
    """A helper class for working with rigid body objects in a scene."""

    def __init__(self, scene, entity_name: str, body_name: str):
        self.scene = scene
        self.entity_name = entity_name
        self.body_name = body_name

    def get_pose(self):
        """Get the 3D pose of the entity."""
        pose = self.scene[self.entity_name].data.body_link_state_w[
            :,
            self.scene[self.entity_name].data.body_names.index(self.body_name),
            :7,
        ]
        return pose


class SceneAsset(HasPose):
    """A helper class for working with assets in a scene."""

    def __init__(self, scene, entity_name: str):
        self.scene = scene
        self.entity_name = entity_name

    def get_pose(self):
        """Get the 3D pose of the entity."""
        xform_prim = self.scene[self.entity_name]
        position, orientation = xform_prim.get_world_poses()
        pose = torch.cat([position, orientation], dim=-1)
        return pose

    def set_pose(self, pose: torch.Tensor):
        """Set the 3D pose of the entity."""
        xform_prim = self.scene[self.entity_name]
        position = pose[..., :3]
        orientation = pose[..., 3:]
        xform_prim.set_world_poses(position, orientation, None)


class RelativePose(HasPose):
    """A helper class for computing the pose of an entity given it's relative pose to a parent."""

    def __init__(self, relative_pose: torch.Tensor, parent: HasPose):
        self.relative_pose = relative_pose
        self.parent = parent

    def get_pose(self):
        """Get the 3D pose of the entity."""

        parent_pose = self.parent.get_pose()

        pose = transform_mul(parent_pose, self.relative_pose)

        return pose


class SceneFixture(SceneAsset, HasOccupancyMap):
    """A helper class for working with assets in a scene that have an associated occupancy map."""

    pass


def plan_path(start: HasPose2d, end: HasPose2d, occupancy_map: OccupancyMap):
    """Plan a path between two entities that have 2D poses given an occupancy map."""

    start_pose = start.get_pose_2d()[:, :2].numpy()
    end_pose = end.get_pose_2d()[:, :2].numpy()

    start_xy_px = occupancy_map.world_to_pixel_numpy(start_pose)
    end_xy_px = occupancy_map.world_to_pixel_numpy(end_pose)

    # xy -> yx
    start_yx_px = start_xy_px[..., 0, ::-1]
    end_yx_px = end_xy_px[..., 0, ::-1]

    path_planner_output = generate_paths(start=start_yx_px, freespace=occupancy_map.freespace_mask())

    path_yx_px = path_planner_output.unroll_path(end_yx_px)
    path_yx_px, _ = compress_path(path_yx_px)

    # yx -> xy
    path_xy_px = path_yx_px[:, ::-1]

    path = occupancy_map.pixel_to_world_numpy(path_xy_px)

    path = torch.from_numpy(path)

    return path


def place_randomly(
    fixture: SceneFixture, background_occupancy_map: OccupancyMap, num_iter: int = 100, area_threshold: float = 1e-5
):
    """Place a scene fixture randomly in an unoccupied region of an occupancy."""

    # sample random xy in bounds
    bottom_left = background_occupancy_map.bottom_left_pixel_world_coords()
    top_right = background_occupancy_map.top_right_pixel_world_coords()

    initial_pose = fixture.get_pose()

    for i in range(num_iter):
        x = random.uniform(bottom_left[0], top_right[0])
        y = random.uniform(bottom_left[1], top_right[1])

        yaw = torch.tensor([random.uniform(-torch.pi, torch.pi)])
        roll = torch.zeros_like(yaw)
        pitch = torch.zeros_like(yaw)

        quat = math_utils.quat_from_euler_xyz(roll, pitch, yaw)

        new_pose = initial_pose.clone()
        new_pose[0, 0] = x
        new_pose[0, 1] = y
        new_pose[0, 3:] = quat

        fixture.set_pose(new_pose)

        intersection_map = intersect_occupancy_maps([fixture.get_occupancy_map(), background_occupancy_map])

        intersection_area = np.count_nonzero(intersection_map.occupied_mask()) * (intersection_map.resolution**2)

        if intersection_area < area_threshold:
            return True

    return False


class DisjointNavScenario:
    """An abstract base class that wraps the underlying environment, exposing methods needed for integration with
    locomanipulation replay.

    This class defines the core methods needed to integrate an environment with the disjoint navigation pipeline for
    locomanipulation replay.  By implementing these methods for a new environment, the environment can be used with
    the disjoint navigation replay function.
    """

    def set_left_hand_pose_target(self, pose: torch.Tensor):
        """Set the left hand pose target in world coordinates."""
        raise NotImplementedError

    def set_right_hand_pose_target(self, pose: torch.Tensor):
        """Set the right hand pose target in world coordinates."""
        raise NotImplementedError

    def set_left_hand_joint_positions_target(self, joint_positions: torch.Tensor):
        """Set the left hand joint position target."""
        raise NotImplementedError

    def set_right_hand_joint_positions_target(self, joint_positions: torch.Tensor):
        """Set the right hand joint position target."""
        raise NotImplementedError

    def set_base_velocity_target(self, velocity: torch.Tensor):
        """Set the base velocity in local robot frame."""
        raise NotImplementedError

    def get_base(self) -> HasPose:
        """Get the robot base body."""
        raise NotImplementedError

    def get_left_hand(self) -> HasPose:
        """Get the robot left hand body."""
        raise NotImplementedError

    def get_right_hand(self) -> HasPose:
        """Get the robot right hand body."""
        raise NotImplementedError

    def get_object(self) -> HasPose:
        """Get the target object body."""
        raise NotImplementedError

    def get_start_fixture(self) -> SceneFixture:
        """Get the start fixture body."""
        raise NotImplementedError

    def get_end_fixture(self) -> SceneFixture:
        """Get the end fixture body."""
        raise NotImplementedError

    def get_obstacle_fixtures(self) -> list[SceneFixture]:
        """Get the set of obstacle fixtures."""
        raise NotImplementedError

    def step(self):
        """Step the environment."""
        raise NotImplementedError

    def close(self):
        """Close the environment"""
        raise NotImplementedError

    def reset(self, intial_state=None):
        """Reset the environment (optionally to a provided initial state)."""
        raise NotImplementedError

    def get_env(self):
        """Get the underlying Isaac Lab environment.  This is used primarily for attaching recording managers for
        data output recording."""
        raise NotImplementedError


class DisjointNavReplayTask(enum.IntEnum):
    """The current state of the locomanipulation replay."""

    GRASP_OBJECT = 0  # Initial state.
    LIFT_OBJECT = 1  # The object is grasped and is being lifted.
    NAVIGATE = 2  # The object is lifted and the robot is navigating.
    APPROACH = 3  # The robot has finished navigating and is approaching the destination fixture.
    DROP_OFF_OBJECT = 4  # The robot has reached it's final position and is dropping off the object.
    DONE = 5  # Finished.


@dataclass
class DisjointNavReplayState:
    """A container for data that is recorded during locomanipulation replay.  This is the final output of the pipeline"""

    left_hand_pose_target: torch.Tensor | None = None  # The left hand's target pose.
    right_hand_pose_target: torch.Tensor | None = None  # The right hand's target pose.
    left_hand_joint_positions_target: torch.Tensor | None = None  # The left hand's target joint positions
    right_hand_joint_positions_target: torch.Tensor | None = None  # The right hand's target joint positions
    base_velocity_target: torch.Tensor | None = (
        None  # The target velocity of the robot base.  This value is provided to the underlying base controller or policy.
    )
    start_fixture_pose: torch.Tensor | None = None  # The pose of the start fixture (ie: pick-up table).
    end_fixture_pose: torch.Tensor | None = None  # The pose of the end / destination fixture (ie: drop-off table)
    object_pose: torch.Tensor | None = None  # The pose of the target object.
    base_pose: torch.Tensor | None = None  # The pose of the robot base.
    task: int | None = None  # The state of the the disjoint navigation replay script's state machine.
    base_goal_pose: torch.Tensor | None = (
        None  # The goal pose of the robot base (ie: the final destination before dropping off the object)
    )
    base_goal_approach_pose: torch.Tensor | None = (
        None  # The goal pose provided to the path planner (this may differ from the final pose, so the robot can "approach" the final point)
    )
    base_path: torch.Tensor | None = None  # The robot base path as determined by the path planner.
    recording_step: int | None = None  # The current recording step used for upper body replay.
    obstacle_fixture_poses: torch.Tensor | None = None  # The pose of all obstacle fixtures in the scene.
