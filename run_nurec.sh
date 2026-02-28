#!/bin/bash

set -ex

OUTPUT_FOLDER=datasets/2026-02-23_11-16-14_babyboom_l5
ASSET_DIR="datasets/bb/hand_hold-babyboom"
USD_PATH=${ASSET_DIR}/stage.usdz
OCCUPANCY_MAP_PATH=${ASSET_DIR}/occupancy_map.yaml

./isaaclab.sh -p \
    scripts/imitation_learning/locomanipulation_sdg/generate_data.py \
    --device cpu \
    --kit_args="--enable isaacsim.replicator.mobility_gen" \
    --task="Isaac-G1-SteeringWheel-Locomanipulation" \
    --dataset ./datasets/dataset_annotated_g1_locomanip.hdf5 \
    --num_runs 1 \
    --lift_step 60 \
    --navigate_step 130 \
    --enable_pinocchio \
    --output_file $OUTPUT_FOLDER/generated_dataset_g1_locomanipulation_sdg.hdf5 \
    --enable_cameras \
    --draw_visualization \
    --randomize_placement \
    --background_usd_path=$USD_PATH \
    --background_occupancy_yaml_file=$OCCUPANCY_MAP_PATH \
    --demo demo_2 \
    --visualizer kit \
    --init_camera_view