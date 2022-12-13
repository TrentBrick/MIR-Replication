import numpy as np
import torch
from torch.nn import functional as F
import wandb
from .model_utils import *
import torchmetrics
from composer.models import ComposerModel
import ipdb
import functools

######## BASE MODEL FOR FUNCTIONS THAT ARE USED ACROSS DIFFERENT NETWORK ARCHITECTURES #######

class BaseModel(ComposerModel):
    def __init__(self, params):
        super().__init__()
        self.params = params

    def loss(self, logits, batch):
        
        x, y = self.true_x, self.true_y

        loss = F.mse_loss(logits, y, reduction='mean')
        loss += extra_loss_terms(self)

        return loss

    def forward(self, batch):
        raise NotImplementedError()

    def eval_forward(self, batch, outputs = None):
        # called during eval
        return outputs if outputs is not None else self.forward(batch)

    def parse_mosaic(forward):
        @functools.wraps(forward)
        def parse_inp(self, inp):
            if type(inp) is list and len(inp)==2: 
                return forward(self, inp[0])
            else: 
                return forward(self, inp)

        return parse_inp