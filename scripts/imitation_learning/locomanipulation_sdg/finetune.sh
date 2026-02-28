#!/bin/bash

python scripts/gr00t_finetune.py \
    --dataset-path /home/jwelsh/isaaclab_3_migration/IsaacLab/datasets/datasets_train_200_lerobot \
    --output-dir /home/jwelsh/isaaclab_3_migration/Isaac-GR00T/models/g1_locomanip_finetune_v3 \
    --data-config g1_locomanipulation_sdg  --embodiment-tag new_embodiment  \
    --num-gpus 1   \
    --max-steps 10000 \
    --save-steps 1000 \
    --video-backend decord \
    --report-to tensorboard