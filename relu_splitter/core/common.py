import random
import logging
import os
from pathlib import Path
from typing import Union
from functools import reduce
from tqdm import tqdm

import onnx
import torch
import numpy as np

from ..utils.read_vnnlib import read_vnnlib
from ..utils.onnx_utils import check_models_closeness
from ..utils.errors import NOT_ENOUGH_NEURON, INVALID_PARAMETER, MODEL_NOT_EQUIV
from ..utils.misc import adjust_mask_random_k, adjust_mask_first_k, find_feasible_point
from ..model import WarppedOnnxModel
from auto_LiRPA import BoundedTensor, PerturbationLpNorm
from .default_config import default_config

TOOL_NAME = os.environ.get("TOOL_NAME", "ReluSplitter")
default_logger = logging.getLogger(__name__)
logger = default_logger