import torch
import torch.nn as nn
from torch.nn import functional as F
import wandb
import numpy as np
from core_scripts import global_timestamp

class TrackedActFunc(nn.Module):
    def __init__(self, act_func, nneurons, params, layer_ind=0, wandb_prefix=""):
        # allows for a learnable beta parameter if it is a softmax
        super().__init__()
        self.act_func = act_func
        self.params = params
        self.layer_ind = layer_ind
        self.wandb_prefix = wandb_prefix
   
        self.activation_summer = torch.zeros( nneurons ).to(params.device)
        self.last_epoch = 0 # local epoch tracker to push the dead neurons!
        self.activity_threshold = 0.0
        

    def wandb_push_dead_neurons(self):
        # log Neuron activations at end of epoch:

        wandb.log(
            {
                'global_step':global_timestamp.CURRENT_TRAIN_STEP,
                'epoch':global_timestamp.CURRENT_EPOCH,

                f"layer_{self.layer_ind}/{self.wandb_prefix}fraction_dead_train_neurons": (
                    self.activation_summer < 0.00001
                ).type(torch.float).mean(),

            }
        )
        self.activation_summer *=0

    def count_dead_neurons(self, neurons):
        neurons = neurons.detach()
        if str(self.act_func) == "ReLU()": 
            # storing the actual neuron activity in this case
            self.activation_summer += neurons.sum(0)
        else: 
            self.activation_summer += (torch.abs(neurons)>self.activity_threshold).sum(0)

    def log_active_neurons(self, neuron_acts):
        
        batched_mean_active_neurons = (torch.abs( neuron_acts.detach()  )>self.activity_threshold).type(torch.float).mean(dim=1)

        #import ipdb 
        #ipdb.set_trace()

        wb_dict = {
            'global_step':global_timestamp.CURRENT_TRAIN_STEP,
            'epoch':global_timestamp.CURRENT_EPOCH,

            f"layer_{self.layer_ind}/{self.wandb_prefix}mean_Active_Neurons": batched_mean_active_neurons.mean(), 

            f"layer_{self.layer_ind}/{self.wandb_prefix}Active_Neurons": batched_mean_active_neurons
            }

        wandb.log(wb_dict)

    def forward(self, x):

        x = self.act_func(x)
        
        # used for L1 loss
        self.neuron_acts = torch.clone( x )
        
        if self.training and self.params.use_wandb:
            #self.count_dead_neurons( x )

            if global_timestamp.CURRENT_TRAIN_STEP%self.params.wandb_log_every_n_steps==0:
                self.log_active_neurons( x )

            if global_timestamp.CURRENT_EPOCH > self.last_epoch:
                # time to log dead neurons from this epoch
                #self.wandb_push_dead_neurons()
                self.last_epoch = global_timestamp.CURRENT_EPOCH

        return x