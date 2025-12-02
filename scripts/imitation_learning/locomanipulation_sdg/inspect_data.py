import h5py


data = h5py.File("/home/john/Nvidia/isaaclab-fork/datasets/dataset_generated_g1_locomanipulation_teacher_release.hdf5", 'r')

print(data['data']['demo_0']['obs'].keys())