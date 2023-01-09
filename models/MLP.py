import torch
import torch.nn as nn
from torch.nn import functional as F
import wandb
import numpy as np
from .Base_Model import BaseModel
from .TrackedMLP_Layer import TrackedMLPLayer
from .TrackedAct_Func import TrackedActFunc
import ipdb
import copy

class MLP(BaseModel):
    def __init__(self, params):
        super().__init__(params)

        self.net = nn.Sequential( 
            *[
                TrackedMLPLayer(params.input_size, params.nneurons[0], use_bias=True, layer_ind=0, params=params ),
                TrackedActFunc(params.act_func, params.nneurons[0], params, layer_ind=0 ),
                 nn.Linear(params.nneurons[0], params.output_size, bias=True)
                ])

    @BaseModel.parse_mosaic
    def forward(self, x):
        return self.net(x)