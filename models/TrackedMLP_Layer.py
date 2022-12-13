import torch
import torch.nn as nn
from torch.nn import functional as F
import wandb
from core_scripts import global_timestamp

class TrackedMLPLayer(nn.Module):
    def __init__(self, inp_dim, out_dim, use_bias, layer_ind, params):
        super().__init__()

        self.layer_ind = layer_ind
        self.use_bias = use_bias
        self.device = params.device

        self.params = params 

        self.layer = nn.Linear(inp_dim, out_dim, bias=self.use_bias)

        # pointer to the model weights. 
        self.weight = self.layer.weight
        self.bias = self.layer.bias 

    def log_weight_info(self):
        
        l2_norm = torch.norm(self.layer.weight.detach(), dim=1)

        wandb.log( {
            'global_step':global_timestamp.CURRENT_TRAIN_STEP,
            'epoch':global_timestamp.CURRENT_EPOCH,

            f"layer_{self.layer_ind}/Mean_L2_norm": l2_norm.mean(),

            f"layer_{self.layer_ind}/L2_norm": l2_norm,

        })


    def log_bias_params(self, x):

        bias_vals = self.layer.bias 
        
        wandb.log( {

            'global_step':global_timestamp.CURRENT_TRAIN_STEP,
            'epoch':global_timestamp.CURRENT_EPOCH,

            f"layer_{self.layer_ind}/Mean_neuron_bias_terms": bias_vals.detach().mean(), 

            f"layer_{self.layer_ind}/neuron_bias_terms": bias_vals.detach(),

            })

    def forward(self, x):

        
        out_x = self.layer(x)

        if self.training and self.params.use_wandb and global_timestamp.CURRENT_TRAIN_STEP%self.params.wandb_log_every_n_steps==0:
            self.log_weight_info()

            if self.use_bias:
                self.log_bias_params(x)

        # need to pass it through to the higher up layers. 
        return out_x