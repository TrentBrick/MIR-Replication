import copy
from enum import Enum, auto
from typing import List
import numpy as np
import torch
from models import *
from torch import nn
from easydict import EasyDict

global_default_settings = EasyDict(
    entity = "kreiman-sdm",
    project_name = "SparseDisentangle",
    output_directory = "../scratch_link/wandb_Logger/SparseDisentangle/",

    #dataloader
    shuffle_train = True, 
    shuffle_val=True, 

    act_func = nn.ReLU(),

    # model loading
    dont_modify_loaded_models = True, # otherwise will modify the model parameters!
    load_optimizer = True, 
    checkpoint_has_only_weights = False,
    load_existing_optimizer_state = True, 
    
    # optimizer/epoch
    starting_epoch = 0, 
    opt="Adam",
    adam_betas = (0.9, 0.999),
    lr=0.0001,
    
    gradient_clip=0.0, # means no clipping 
    clipping_type = "norm", # can also be 'adaptive' or 'value'
    adamw_l2_loss_weight=0.001,

    # Dataset
    dataset_size = None, # can make dataset smaller for faster tests. 

    # Logging
    #metrics_to_log = ["loss", "accuracy"],
    check_val_every_n_epoch = 10, # how often to run the validation loop
    save_model_checkpoints = True, 
    num_checkpoints_to_keep = 1, # for saving the model
    checkpoint_every_n_epochs = 10,
    use_wandb = True, 
    wandb_log_every_n_steps = 10,

    random_seed = None, 

    # activation penalties
    activation_l1_coefficient = None,
    

    heatmap_logging_threshold = 100, # for the number of neurons

)

# these will all overwrite the defaults. 
model_params = EasyDict(

    WHIT_TOYMODEL = dict(
        model_class = ToyModel, 
        log_disentangle_every_n_steps = 100,
        neuron_act_penalty = True, 
        batch_size=5096,
        dataset_size = 5096*2,

        opt='Adam',
        lr=0.001,
        nneurons=[16],
        classification = False,
    ),

    
)