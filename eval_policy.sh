./isaaclab.sh -p \
    scripts/imitation_learning/disjoint_navigation/eval_policy.py \
    --device cpu \
    --kit_args="--enable isaacsim.replicator.mobility_gen" \
    --task="Isaac-G1-Disjoint-Navigation" \
    --dataset="datasets/dataset_generated_g1_locomanipulation_teacher_release.hdf5" \
    --demo=demo_0 \
    --enable_pinocchio \
    --output_file=datasets/dataset_generated_disjoint_nav_policy_v1.hdf5 \
    --enable_cameras
