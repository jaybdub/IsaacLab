
# GROOT IMPORTS
import warnings
from dataclasses import dataclass, field
from typing import List, Literal

import torch
import numpy as np
import tyro
import os

from gr00t.data.dataset import LeRobotSingleDataset
from gr00t.data.embodiment_tags import EMBODIMENT_TAG_MAPPING
from gr00t.eval.robot import RobotInferenceClient
from gr00t.experiment.data_config import load_data_config
from gr00t.model.policy import BasePolicy, Gr00tPolicy
from gr00t.utils.eval import calc_mse_for_single_trajectory

from gr00t.experiment.data_config import DATA_CONFIG_MAP


class Policy:
    def __init__(self):
        MODEL_PATH = "/home/john/Nvidia/disjoint-nav-training/models/disjoint_nav_model_v2/checkpoint-20000"
        EMBODIMENT_TAG = "new_embodiment"
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.data_config = DATA_CONFIG_MAP["disjoint_nav"]
        self.modality_config = self.data_config.modality_config()
        self.modality_transform = self.data_config.transform()
        self.policy = Gr00tPolicy(
            model_path=MODEL_PATH,
            embodiment_tag=EMBODIMENT_TAG,
            modality_config=self.modality_config,
            modality_transform=self.modality_transform,
            device=self.device,
        )

