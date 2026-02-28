#!/bin/bash

./isaaclab.sh -p \
    scripts/imitation_learning/locomanipulation_sdg/generate_data.py \
    --device cuda \
    --kit_args="--enable isaacsim.replicator.mobility_gen" \
    --task="Isaac-G1-SteeringWheel-Locomanipulation" \
    --dataset ./datasets/generated_dataset_g1_locomanip.hdf5 \
    --num_runs 1 \
    --lift_step 60 \
    --navigate_step 130 \
    --enable_pinocchio \
    --output_file ./datasets/generated_dataset_g1_locomanipulation_sdg_test_2.hdf5 \
    --enable_cameras \
    --visualizer kit \
    --randomize_placement 
