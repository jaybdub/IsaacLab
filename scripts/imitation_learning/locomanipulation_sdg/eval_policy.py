# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Script to replay demonstrations with Isaac Lab environments."""

"""Launch Isaac Sim Simulator first."""


import argparse
import os

from isaaclab.app import AppLauncher

# Launch Isaac Lab
parser = argparse.ArgumentParser(description="Disjoint navigation")
parser.add_argument("--task", type=str, help="The Isaac Lab disjoint navigation task to load for data generation.")
parser.add_argument("--dataset", type=str, help="The static manipulation dataset recorded via teleoperation.")
parser.add_argument("--output_file", type=str, help="The file name for the generated output dataset.")
parser.add_argument(
    "--lift_step",
    type=int,
    help=(
        "The step index in the input recording where the robot is ready to lift the object.  Aka, where the grasp is"
        " finished."
    ),
)
parser.add_argument("--demo", type=str, default="demo_0", help="The demo in the input dataset to use.")
parser.add_argument(
    "--enable_pinocchio",
    action="store_true",
    default=False,
    help="Enable Pinocchio.",
)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

if args_cli.enable_pinocchio:
    # Import pinocchio before AppLauncher to force the use of the version installed by IsaacLab and not the one installed by Isaac Sim
    # pinocchio is required by the Pink IK controllers and the GR1T2 retargeter
    import pinocchio  # noqa: F401

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import enum
import gymnasium as gym
import torch

from policy import Policy
import omni.kit

from isaaclab.utils import configclass
from isaaclab.utils.datasets import EpisodeData, HDF5DatasetFileHandler

import isaaclab_mimic.locomanipulation_sdg.envs  # noqa: F401
from isaaclab_mimic.locomanipulation_sdg.envs.locomanipulation_sdg_env import LocomanipulationSDGEnv
from isaaclab_mimic.locomanipulation_sdg.data_classes import LocomanipulationSDGOutputData
from isaaclab_mimic.locomanipulation_sdg.path_utils import ParameterizedPath, plan_path
from isaaclab_mimic.locomanipulation_sdg.scene_utils import RelativePose, place_randomly
from isaaclab_mimic.locomanipulation_sdg.transform_utils import (
    transform_inv,
    transform_mul,
    transform_relative_pose,
)
from isaaclab_mimic.locomanipulation_sdg.occupancy_map_utils import (
    OccupancyMap,
    merge_occupancy_maps,
    occupancy_map_add_to_stage,
)

from isaaclab_tasks.utils import parse_env_cfg


def setup_navigation_scene(
    env: LocomanipulationSDGEnv, input_episode_data: EpisodeData, approach_distance: float, randomize_placement: bool = True
) -> tuple[OccupancyMap, RelativePose]:
    # Create base occupancy map
    occupancy_map = merge_occupancy_maps([
        OccupancyMap.make_empty(start=(-7, -7), end=(7, 7), resolution=0.05),
        env.get_start_fixture().get_occupancy_map(),
    ])

    # Randomize fixture placement if enabled
    if randomize_placement:
        fixtures = [env.get_end_fixture()] + env.get_obstacle_fixtures()
        for fixture in fixtures:
            place_randomly(fixture, occupancy_map.buffered_meters(1.0))
            occupancy_map = merge_occupancy_maps([occupancy_map, fixture.get_occupancy_map()])

    # Compute goal poses from initial state
    initial_state = env.load_input_data(input_episode_data, 0)
    base_goal = RelativePose(
        relative_pose=transform_mul(transform_inv(initial_state.fixture_pose), initial_state.base_pose),
        parent=env.get_end_fixture(),
    )

    return occupancy_map, base_goal


def build_model_input(env, base_goal):
    obs = env.obs_buf
    left_hand_pose = torch.cat([
        obs['policy']['left_eef_pos'],
        obs['policy']['left_eef_quat']
    ], dim=-1)
    right_hand_pose = torch.cat([
        obs['policy']['right_eef_pos'],
        obs['policy']['right_eef_quat']
    ], dim=-1)
    left_hand_joint_positions = obs['policy']['hand_joint_state'][:, 0:7]
    right_hand_joint_positions = obs['policy']['hand_joint_state'][:, 7:14]

    base_pose = env.get_base().get_pose()
    object_pose = env.get_object().get_pose()
    goal_pose = base_goal.get_pose()
    end_fixture_pose = env.get_end_fixture().get_pose()

    base_pose_inv = transform_inv(base_pose)

    # TODO: transform poses relative to base
    model_input = {
        "video.ego_view": obs['policy']['robot_pov_cam'],
        "state.left_hand_pose": transform_mul(base_pose_inv, left_hand_pose),
        "state.right_hand_pose": transform_mul(base_pose_inv, right_hand_pose),
        "state.left_hand_joint_positions": left_hand_joint_positions,
        "state.right_hand_joint_positions": right_hand_joint_positions,
        "state.object_pose": transform_mul(base_pose_inv, object_pose),
        "state.goal_pose": transform_mul(base_pose_inv, goal_pose),
        "state.end_fixture_pose": transform_mul(base_pose_inv, end_fixture_pose)
    }

    print(model_input['state.goal_pose'])
    dummy_action = torch.zeros(1, 32)
    dummy_action[:, :28] = torch.cat([
        left_hand_pose, right_hand_pose, left_hand_joint_positions, right_hand_joint_positions
    ], dim=1)
    dummy_action[:, 31] = 0.8

    return model_input, dummy_action


def eval_policy(
    env: LocomanipulationSDGEnv,
    policy: Policy,
    input_episode_data: EpisodeData,
    randomize_placement: bool = True,
) -> None:
    # Initialize environment to starting state
    obs, _ = env.reset_to(state=input_episode_data.get_initial_state(), env_ids=torch.tensor([0]), is_relative=True)

    # Set up navigation scene and path planning
    occupancy_map, base_goal = setup_navigation_scene(
        env, input_episode_data, randomize_placement
    )

    # Main simulation loop with state machine
    step = 0

    action_idx = 0
    action_buffer_avg = None
    inference_interval = 1

    while simulation_app.is_running() and not simulation_app.is_exiting():

        if step % inference_interval == 0:
                
            model_input, dummy_action = build_model_input(env, base_goal)
            action_dict = policy.policy.get_action(model_input)
            action_dict['action.base_height'] = action_dict['action.base_height'][:, None] # expand missing dim
            action_buffer = torch.cat([torch.from_numpy(v) for v in action_dict.values()], dim=-1)
            action_idx = 0
        
        # moving avg
        if step < 10:
            env.step(dummy_action)
        else:
            # if action_buffer_avg is None:
            #     action_buffer_avg = action_buffer
            # else:
            #     action_buffer_avg = 0.5 * action_buffer_avg + 0.5 * action_buffer_avg
            base_pose = env.get_base().get_pose()
            action = action_buffer.clone()
            action[:, 0:7] = transform_mul(base_pose, action[:, 0:7]) # convert poses to world coordinates
            action[:, 7:14] = transform_mul(base_pose, action[:, 7:14])
            action[:, 28:31] = action[:, 28:31] * 0.0
            env.step(action[action_idx:action_idx+1].mean(dim=0, keepdim=True))

        step += 1
        action_idx += 1


if __name__ == "__main__":


    with torch.no_grad():

        # Create environment
        if args_cli.task is not None:
            env_name = args_cli.task.split(":")[-1]
        if env_name is None:
            raise ValueError("Task/env name was not specified nor found in the dataset.")

        policy = Policy()
        # policy = None
        
        env_cfg = parse_env_cfg(env_name, device=args_cli.device, num_envs=1)
        env_cfg.sim.device = "cpu"
        env_cfg.recorders.dataset_export_dir_path = os.path.dirname(args_cli.output_file)
        env_cfg.recorders.dataset_filename = os.path.basename(args_cli.output_file)

        env = gym.make(args_cli.task, cfg=env_cfg).unwrapped

        # Load input data
        input_dataset_file_handler = HDF5DatasetFileHandler()
        input_dataset_file_handler.open(args_cli.dataset)
        input_episode_data = input_dataset_file_handler.load_episode(args_cli.demo, args_cli.device)

        eval_policy(
            env=env,
            policy=policy,
            input_episode_data=input_episode_data,
            randomize_placement=True
        )

        env.reset()  # FIXME: hack to handle missing final recording
        env.close()

        simulation_app.close()
