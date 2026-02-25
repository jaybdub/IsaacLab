#!/bin/bash

./isaaclab.sh -p scripts/imitation_learning/isaaclab_mimic/generate_dataset.py \
   --device cpu --headless --num_envs 20 --generation_num_trials 200 --enable_pinocchio \
   --input_file ./datasets/dataset_annotated_g1_locomanip.hdf5 --output_file ./datasets/generated_dataset_g1_locomanip.hdf5

