import copy
from enum import Enum, auto
from easydict import EasyDict
from typing import List
import numpy as np
import torch
from models import *
from torch import nn
from core_scripts.data_loaders import *

# I should be able to set the name and the associated parameters directly. then when I say what I want it auto selects this option. 
# thus it is like an enum where it equals a dictionary. 

dataset_params = EasyDict(

    WHIT_TOYFEATURES = dict(
        use_ffcv=False, 
        directory_path = "BleepBlop",
        dataset_class = ToyFeatures,
        input_size =6, # dim of each feature
        n_features = 6,
        classification=False, 
        no_validation = True,
    ),

)

