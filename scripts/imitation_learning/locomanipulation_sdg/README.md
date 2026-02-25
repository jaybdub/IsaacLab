Clone Isaac-GR00T

```bash
git clone -b n1.5-release https://github.com/NVIDIA/Isaac-GR00T
```

Copy the data config for the G1 locomanipulation.

```bash
cp scripts/imitation_learning/locomanipulation_sdg/gr00t_policy/data_config.py Isaac-GR00T/gr00t/experiment/data_config.py
```

Install GR00T N1.5

```bash
uv pip install -e .
uv pip install wheel
MAX_JOBS=4 uv pip install --no-build-isolation flash-attn==2.7.1.post4 
MAX_JOBS=4 uv pip install --no-build-isolation pytorch3d
uv pip install diffusers decord zmq
# uv pip install numpy --upgrade
```

./_isaac_sim/python.sh -m pip 
