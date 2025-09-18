./isaaclab.sh -p \
    scripts/imitation_learning/disjoint_navigation/generate_navigation.py \
    --device cpu \
    --kit_args="--enable isaacsim.replicator.mobility_gen" \
    --task="Isaac-G1-Disjoint-Navigation" \
    --dataset="datasets/dataset_generated_g1_locomanipulation_teacher_release.hdf5" \
    --demo=demo_3 \
    --num_runs=10 \
    --lift_step=75 \
    --navigate_step=120 \
    --enable_pinocchio \
    --output_file=datasets/dataset_generated_disjoint_nav_v7.hdf5 \
    --enable_cameras
