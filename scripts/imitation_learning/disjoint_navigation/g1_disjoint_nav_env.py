# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

# Copyright (c) 2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import numpy as np
import torch
from pathlib import Path

from common import (
    DisjointNavRecording,
    DisjointNavRecordingItem,
    DisjointNavReplayState,
    DisjointNavScenario,
    HasPose,
    SceneBody,
    SceneFixture,
)
# from mdp.actions import G1_UPPER_BODY_IK_ACTION_CFG, LowerBodyActionCfg
from occupancy_map import OccupancyMap

import isaaclab.envs.mdp as base_mdp
import isaaclab.sim as sim_utils
from isaaclab.actuators import DCMotorCfg, ImplicitActuatorCfg
from isaaclab.assets import ArticulationCfg, AssetBaseCfg, RigidObjectCfg
from isaaclab.devices.openxr import XrCfg
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.envs.manager_based_rl_mimic_env import ManagerBasedRLMimicEnv
from isaaclab.envs.mdp.recorders.recorders_cfg import ActionStateRecorderManagerCfg as ActionStateRecorderManagerCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.managers.recorder_manager import RecorderTerm, RecorderTermCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sim.spawners.from_files.from_files_cfg import GroundPlaneCfg, UsdFileCfg
from isaaclab.utils import configclass
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR, ISAACLAB_NUCLEUS_DIR, retrieve_file_path
from isaaclab.utils.datasets import HDF5DatasetFileHandler

from isaaclab_tasks.manager_based.locomanipulation.pick_place.locomanipulation_g1_env_cfg import (
    LocomanipulationG1SceneCfg,
    LocomanipulationG1EnvCfg
)

NUM_FORKLIFTS = 6
NUM_BOXES = 12

##
# Scene definition
##
@configclass
class DisjointNavG1SceneCfg(LocomanipulationG1SceneCfg):
    
    packing_table_2 = AssetBaseCfg(
        prim_path="/World/envs/env_.*/PackingTable2",
        init_state=AssetBaseCfg.InitialStateCfg(
            pos=[-2, -3.55, -0.3],
            # rot=[0, 0, 0, 1]),
            rot=[0.9238795, 0, 0, -0.3826834],
        ),
        spawn=UsdFileCfg(
            usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/PackingTable/packing_table.usd",
            rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
        ),
    )


# Add forklifts
for i in range(NUM_FORKLIFTS):
    setattr(
        DisjointNavG1SceneCfg,
        f"forklift_{i}",
        AssetBaseCfg(
            prim_path=f"/World/envs/env_.*/Forklift{i}",
            init_state=AssetBaseCfg.InitialStateCfg(pos=[0.0, 0.0, 0.0], rot=[1.0, 0.0, 0.0, 0.0]),
            spawn=UsdFileCfg(
                usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/Forklift/forklift.usd",
                rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
            ),
        ),
    )

# Add boxes
for i in range(NUM_BOXES):
    setattr(
        DisjointNavG1SceneCfg,
        f"box_{i}",
        AssetBaseCfg(
            prim_path=f"/World/envs/env_.*/Box{i}",
            init_state=AssetBaseCfg.InitialStateCfg(pos=[0.0, 0.0, 0.0], rot=[1.0, 0.0, 0.0, 0.0]),
            spawn=UsdFileCfg(
                usd_path=f"{ISAAC_NUCLEUS_DIR}/Environments/Simple_Warehouse/Props/SM_CardBoxB_01_681.usd",
                rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
            ),
        ),
    )



@configclass
class DisjointNavG1TerminationsCfg:
    """Termination terms for the MDP."""

    time_out = DoneTerm(func=base_mdp.time_out, time_out=True)


@configclass
class DisjointNavG1EventCfg:
    """Configuration for events."""

    reset_all = EventTerm(func=base_mdp.reset_scene_to_default, mode="reset")


class PreStepLowerBodyPolicyObservationsRecorder(RecorderTerm):
    """Recorder term that records the policy group observations in each step."""

    def record_pre_step(self):
        return "obs_lower_body", self._env.obs_buf["lower_body_policy"]


@configclass
class PreStepLowerBodyPolicyObservationsRecorderCfg(RecorderTermCfg):
    """Configuration for the step policy observation recorder term."""

    class_type: type[RecorderTerm] = PreStepLowerBodyPolicyObservationsRecorder


class DisjointNavReplayStateRecorder(RecorderTerm):

    def record_pre_step(self):
        replay_state: DisjointNavReplayState = self._env._replay_state

        replay_state_dict = {
            "left_hand_pose_target": replay_state.left_hand_pose_target[None, :],
            "right_hand_pose_target": replay_state.right_hand_pose_target[None, :],
            "left_hand_joint_positions_target": replay_state.left_hand_joint_positions_target[None, :],
            "right_hand_joint_positions_target": replay_state.right_hand_joint_positions_target[None, :],
            "base_velocity_target": replay_state.base_velocity_target[None, :],
            "start_fixture_pose": replay_state.start_fixture_pose,
            "end_fixture_pose": replay_state.end_fixture_pose,
            "object_pose": replay_state.object_pose,
            "base_pose": replay_state.base_pose,
            "task": torch.tensor([[replay_state.task]]),
            "base_goal_pose": replay_state.base_goal_pose,
            "base_goal_approach_pose": replay_state.base_goal_approach_pose,
            "base_path": replay_state.base_path[None, :],
            "recording_step": torch.tensor([[replay_state.recording_step]]),
            "obstacle_fixture_poses": replay_state.obstacle_fixture_poses,
        }

        return "replay_state", replay_state_dict


@configclass
class DisjointNavReplayStateRecorderCfg(RecorderTermCfg):
    """Configuration for the step policy observation recorder term."""

    class_type: type[RecorderTerm] = DisjointNavReplayStateRecorder


class DisjointNavRecorderManagerCfg(ActionStateRecorderManagerCfg):
    record_pre_step_lower_body_policy_observations = PreStepLowerBodyPolicyObservationsRecorderCfg()
    record_pre_step_disjoint_nav_replay_state = DisjointNavReplayStateRecorderCfg()


@configclass
class DisjointNavG1EnvCfg(LocomanipulationG1EnvCfg):
    """Configuration for the G1 29DoF environment."""

    # Scene settings
    scene: DisjointNavG1SceneCfg = DisjointNavG1SceneCfg(num_envs=1, env_spacing=2.5, replicate_physics=True)
    # MDP settings
    terminations: DisjointNavG1TerminationsCfg = DisjointNavG1TerminationsCfg()
    events: DisjointNavG1EventCfg = DisjointNavG1EventCfg()

    def __post_init__(self):
        """Post initialization."""
        # general settings
        self.decimation = 4
        self.episode_length_s = 100.0
        # simulation settings
        self.sim.dt = 1 / 200  # 200Hz
        self.sim.render_interval = 6

        # Set the URDF and mesh paths for the IK controller
        urdf_omniverse_path = f"{ISAACLAB_NUCLEUS_DIR}/Controllers/LocomanipulationAssets/unitree_g1_kinematics_asset/g1_29dof_with_hand_only_kinematics.urdf"

        # Retrieve local paths for the URDF and mesh files. Will be cached for call after the first time.
        self.actions.upper_body_ik.controller.urdf_path = retrieve_file_path(urdf_omniverse_path)

class PackingTable(SceneFixture):

    def get_occupancy_map(self):

        local_occupancy_map = OccupancyMap.from_occupancy_boundary(
            boundary=np.array([[-1.45, -0.45], [1.45, -0.45], [1.45, 0.45], [-1.45, 0.45]]), resolution=0.05
        )

        transform = self.get_transform_2d().detach().cpu().numpy()

        occupancy_map = local_occupancy_map.transformed(transform)

        return occupancy_map


class Forklift(SceneFixture):

    def get_occupancy_map(self):

        local_occupancy_map = OccupancyMap.from_occupancy_boundary(
            boundary=np.array([[-1.0, -1.9], [1.0, -1.9], [1.0, 2.1], [-1.0, 2.1]]), resolution=0.05
        )

        transform = self.get_transform_2d().detach().cpu().numpy()

        occupancy_map = local_occupancy_map.transformed(transform)

        return occupancy_map


class CardboardBox(SceneFixture):

    def get_occupancy_map(self):

        local_occupancy_map = OccupancyMap.from_occupancy_boundary(
            boundary=np.array([[-0.5, -0.5], [0.5, -0.5], [0.5, 0.5], [-0.5, 0.5]]), resolution=0.05
        )

        transform = self.get_transform_2d().detach().cpu().numpy()

        occupancy_map = local_occupancy_map.transformed(transform)

        return occupancy_map


class G1DisjointNavRecording(DisjointNavRecording):

    def __init__(self, path: str, demo: str = "demo_0", device: str = "cpu"):
        self.dataset_file_handler = HDF5DatasetFileHandler()
        self.dataset_file_handler.open(path)
        self.episode_data = self.dataset_file_handler.load_episode(demo, device)
        self._robot_base_pose = self.episode_data.get_initial_state()["articulation"]["robot"]["root_pose"]

    def get_initial_state(self):

        initial_state = self.episode_data.get_initial_state()
        return initial_state

    def get_item(self, step: int) -> DisjointNavRecordingItem | None:

        dataset_action = self.episode_data.get_action(step)
        dataset_state = self.episode_data.get_state(step)

        if dataset_action is None:
            return None

        if dataset_state is None:
            return None

        object_pose = dataset_state["rigid_object"]["object"]["root_pose"]

        target = DisjointNavRecordingItem(
            left_hand_pose_target=dataset_action[0:7],
            right_hand_pose_target=dataset_action[7:14],
            left_hand_joint_positions_target=dataset_action[14:21],
            right_hand_joint_positions_target=dataset_action[21:28],
            base_pose=self._robot_base_pose,
            object_pose=object_pose,
            fixture_pose=torch.tensor(
                [0.0, 0.55, -0.3, 1.0, 0.0, 0.0, 0.0]
            ),  # Table pose is not recorded for this env.
        )

        return target


class G1DisjointNavScenario(DisjointNavScenario):

    def __init__(self, output_dir: str, output_file_name: str):
        self._env_cfg = DisjointNavG1EnvCfg()
        self._env_cfg.sim.device = "cpu"
        # self._env_cfg.sim.render.rendering_mode = "performance"

        self._env_cfg.scene.num_envs = 1

        self._env_cfg.recorders = DisjointNavRecorderManagerCfg()
        self._env_cfg.recorders.dataset_export_dir_path = output_dir
        self._env_cfg.recorders.dataset_filename = output_file_name

        self._env = ManagerBasedRLMimicEnv(cfg=self._env_cfg)

        self._env.sim.set_camera_view([10.5, 10.5, 10.5], [0.0, 0.0, 0.5])
        self._upper_body_dim = self._env.action_manager.get_term("upper_body_ik").action_dim
        self._waist_dim = 0#self._env.action_manager.get_term("waist_joint_pos").action_dim
        self._lower_body_dim = self._env.action_manager.get_term("lower_body_joint_pos").action_dim
        self._frame_pose_dim = 7
        self._number_of_finger_joints = 7
        self._env_action = torch.zeros(self._env.action_space.shape)
        self.set_base_height_target( torch.tensor([0.8]))

    def set_left_hand_pose_target(self, pose: torch.Tensor):
        assert pose.shape == (self._frame_pose_dim,), f"Expected pose shape ({self._frame_pose_dim},), got {pose.shape}"
        self._env_action[0, : self._frame_pose_dim] = pose

    def set_right_hand_pose_target(self, pose: torch.Tensor):
        assert pose.shape == (self._frame_pose_dim,), f"Expected pose shape ({self._frame_pose_dim},), got {pose.shape}"
        self._env_action[0, self._frame_pose_dim : 2 * self._frame_pose_dim] = pose

    def set_left_hand_joint_positions_target(self, joint_positions: torch.Tensor):
        assert joint_positions.shape == (
            self._number_of_finger_joints,
        ), f"Expected joint_positions shape ({self._number_of_finger_joints},), got {joint_positions.shape}"
        self._env_action[0, 2 * self._frame_pose_dim : 2 * self._frame_pose_dim + self._number_of_finger_joints] = (
            joint_positions
        )

    def set_right_hand_joint_positions_target(self, joint_positions: torch.Tensor):
        assert joint_positions.shape == (
            self._number_of_finger_joints,
        ), f"Expected joint_positions shape ({self._number_of_finger_joints},), got {joint_positions.shape}"
        self._env_action[
            0,
            2 * self._frame_pose_dim
            + self._number_of_finger_joints : 2 * self._frame_pose_dim
            + 2 * self._number_of_finger_joints,
        ] = joint_positions

    def set_base_velocity_target(self, velocity: torch.Tensor):
        assert velocity.shape == (3,), f"Expected velocity shape (3,), got {velocity.shape}"
        lower_body_index_offset = self._upper_body_dim + self._waist_dim
        self._env_action[0, lower_body_index_offset : lower_body_index_offset + 3] = velocity

    def set_base_height_target(self, height: torch.Tensor = torch.tensor([0.72])):
        assert height.shape == (1,), f"Expected height shape (1,), got {height.shape}"
        lower_body_index_offset = self._upper_body_dim + self._waist_dim
        self._env_action[0, lower_body_index_offset + 3 : lower_body_index_offset + 4] = height

    def get_base(self) -> HasPose:
        return SceneBody(self._env.scene, "robot", "pelvis")

    def get_left_hand(self) -> HasPose:
        return SceneBody(self._env.scene, "robot", "left_wrist_yaw_link")

    def get_right_hand(self) -> HasPose:
        return SceneBody(self._env.scene, "robot", "right_wrist_yaw_link")

    def get_object(self) -> HasPose:
        return SceneBody(self._env.scene, "object", "sm_steeringwheel_a01_01")

    def get_start_fixture(self) -> SceneFixture:
        return PackingTable(self._env.scene, "packing_table")

    def get_end_fixture(self) -> SceneFixture:
        return PackingTable(self._env.scene, "packing_table_2")

    def get_obstacle_fixtures(self):
        obstacles = [Forklift(self._env.scene, f"forklift_{i}") for i in range(NUM_FORKLIFTS)]
        obstacles += [CardboardBox(self._env.scene, f"box_{i}") for i in range(NUM_BOXES)]
        return obstacles

    def reset(self, initial_state=None):
        if initial_state is not None:
            self._env.reset_to(initial_state, env_ids=torch.tensor([0]))
        else:
            self._env.reset()

    def get_env(self):
        return self._env

    def step(self):
        self._env.step(self._env_action)

    def close(self):
        self._env.close()
