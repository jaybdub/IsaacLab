# Copyright (c) 2024-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

import torch

from isaaclab.envs.manager_based_rl_env import ManagerBasedRLEnv
from isaaclab.managers.recorder_manager import RecorderTerm
from isaaclab.utils.datasets import EpisodeData

from isaaclab_mimic.locomanipulation_sdg.data_classes import LocomanipulationSDGInputData, LocomanipulationSDGOutputData
from isaaclab_mimic.locomanipulation_sdg.scene_utils import HasPose, SceneFixture


class LocomanipulationSDGOutputDataRecorder(RecorderTerm):
    """Recorder for Locomanipulation SDG output data."""

    def record_pre_step(self):
        output_data: LocomanipulationSDGOutputData | None = getattr(
            self._env, "_locomanipulation_sdg_output_data", None
        )
        if output_data is None:
            # Policy rollout or other use: no SDG output to record; return dummy so recorder manager doesn't break.
            device = getattr(self._env, "device", None) or torch.device("cpu")
            output_data_dict = {
                "left_hand_pose_target": torch.zeros(1, 7, device=device),
                "right_hand_pose_target": torch.zeros(1, 7, device=device),
                "left_hand_joint_positions_target": torch.zeros(1, 7, device=device),
                "right_hand_joint_positions_target": torch.zeros(1, 7, device=device),
                "base_velocity_target": torch.zeros(1, 3, device=device),
                "start_fixture_pose": torch.zeros(7, device=device),
                "end_fixture_pose": torch.zeros(7, device=device),
                "object_pose": torch.zeros(7, device=device),
                "base_pose": torch.zeros(7, device=device),
                "task": torch.zeros(1, 1, dtype=torch.long, device=device),
                "base_goal_pose": torch.zeros(7, device=device),
                "base_goal_approach_pose": torch.zeros(7, device=device),
                "base_path": torch.zeros(1, 3, device=device),
                "recording_step": torch.zeros(1, 1, dtype=torch.long, device=device),
                "obstacle_fixture_poses": torch.zeros(1, 7, device=device),
            }
            return "locomanipulation_sdg_output_data", output_data_dict

        output_data_dict = {
            "left_hand_pose_target": output_data.left_hand_pose_target[None, :],
            "right_hand_pose_target": output_data.right_hand_pose_target[None, :],
            "left_hand_joint_positions_target": output_data.left_hand_joint_positions_target[None, :],
            "right_hand_joint_positions_target": output_data.right_hand_joint_positions_target[None, :],
            "base_velocity_target": output_data.base_velocity_target[None, :],
            "start_fixture_pose": output_data.start_fixture_pose,
            "end_fixture_pose": output_data.end_fixture_pose,
            "object_pose": output_data.object_pose,
            "base_pose": output_data.base_pose,
            "task": torch.tensor([[output_data.data_generation_state]]),
            "base_goal_pose": output_data.base_goal_pose,
            "base_goal_approach_pose": output_data.base_goal_approach_pose,
            "base_path": output_data.base_path[None, :],
            "recording_step": torch.tensor([[output_data.recording_step]]),
            "obstacle_fixture_poses": output_data.obstacle_fixture_poses,
        }

        return "locomanipulation_sdg_output_data", output_data_dict


class LocomanipulationSDGEnv(ManagerBasedRLEnv):
    """An abstract base class that wraps the underlying environment, exposing methods needed for integration with
    locomanipulation replay.

    This class defines the core methods needed to integrate an environment with the locomanipulation SDG pipeline for
    locomanipulation replay.  By implementing these methods for a new environment, the environment can be used with
    the locomanipulation SDG replay function.
    """

    def load_input_data(self, episode_data: EpisodeData, step: int) -> LocomanipulationSDGInputData:
        raise NotImplementedError

    def build_action_vector(
        self,
        left_hand_pose_target: torch.Tensor,
        right_hand_pose_target: torch.Tensor,
        left_hand_joint_positions_target: torch.Tensor,
        right_hand_joint_positions_target: torch.Tensor,
        base_velocity_target: torch.Tensor,
    ):
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
