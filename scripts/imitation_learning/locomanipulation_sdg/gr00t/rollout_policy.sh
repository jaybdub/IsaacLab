#!/bin/bash

python scripts/imitation_learning/locomanipulation_sdg/gr00t/rollout_policy.py \
    --kit_args="--enable isaacsim.replicator.mobility_gen" \
    --model_path /home/jwelsh/isaaclab_3_migration/models/g1_locomanip_finetune_v1_simple_high_var/g1_locomanip_finetune_20260129_231610/checkpoint-20000 \
    --embodiment_tag new_embodiment \
    --dataset datasets/dataset_annotated_g1_locomanip.hdf5 \
    --demo demo_0 \
    --output_file datasets/dataset_annotated_g1_locomanip_demo_0_gr00t.hdf5 \
    --task Isaac-G1-SteeringWheel-Locomanipulation \
    --device cuda:0 \
    --randomize_placement False \
    --enable_cameras \
    --visualizer kit \
    --policy_quat_format wxyz