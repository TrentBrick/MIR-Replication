import copy
from typing import List
import numpy as np
import torch
from models import *
from torch import nn
import torchvision.models as models
from core_scripts import generate_dataloaders
from core_params.model_params import *
from core_params.dataset_params import *
from core_scripts.configure_optimizers import *
import ipdb
from composer.utils import reproducibility

def update_loaded_model_epochs(params, full_cpkt):
    if "timestamp" in full_cpkt['state']:
        params.starting_epoch=full_cpkt['state']['timestamp']['Timestamp']['epoch']
        if params.load_optimizer:
            params.epochs_to_train_for += params.starting_epoch
    return params

def set_model_output_size(params):
    if params.classification:
        params["output_size"] = params["nclasses"]
    else: 
        params["output_size"] = params["input_size"]

def get_params_net_dataloader(
    model_name,
    dataset_name,
    load_from_checkpoint=None,
    dataset_path="data/",
    verbose=True,
    experiment_param_modifications=None,
    ):
    """
    Returns model parameters for given model_style and an optional list of regularizers. 
    """

    #return params, optimizer, model, data_module

    # else init params or integrate them with the loaded in model. 

    params = copy.deepcopy(global_default_settings)
    params.dataset_name = dataset_name
    params.model_name = model_name
    
    params.update( model_params[model_name] )
    params.update( dataset_params[dataset_name] )
    d_params = dataset_params[dataset_name]

    # set up here so it applies automatically and is not over written by any loaded in model (custom kwards overwrites the loaded in model parameters.)

    # THIS NEEDS TO BE THE LAST THING THAT IS MODIFIED
    if verbose: 
        print("Custom args are:", experiment_param_modifications)
    for key, value in experiment_param_modifications.items():
        params[key] = value

    params.device_type = "gpu" if torch.cuda.is_available() else "cpu"
    params.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    params.load_from_checkpoint = load_from_checkpoint


    # setting random seed. Want to do this before the models or data loaders are initialized. 
    if params.random_seed is not None: 
        reproducibility.configure_deterministic_mode()
        reproducibility.seed_all(params.random_seed)
        #pl.utilities.seed.seed_everything(seed=params.random_seed, workers=False)

    # getting whatever was set from either the loaded in model or the dataset: 
    
    # needs to be able to handle the model loading. 
    #if params["classification"] and "accuracy" not in params.metrics_to_log:
    #    params.metrics_to_log.append('accuracy')
    '''if not params["classification"] and "accuracy" in params.metrics_to_log:
        params.metrics_to_log.remove('accuracy')'''

    set_model_output_size(params)
    
    dataloaders = generate_dataloaders(
        params,
        data_path=dataset_path
    )

    model = params.model_class(params)

    # init optimizer
    optimizer = configure_optimizers_(model, verbose)

    non_zero_weights = 0
    for p in list(model.parameters()):
        if len(p.shape)>1: 
            non_zero_weights+= (torch.abs(p)>0.0000001).sum()
    if verbose: 
        print("Number of non zero weights is:", non_zero_weights)
        print("Final params being used", params)

    # useful for wandb later. 
    params.first_layer_nneurons = params.nneurons[0]

    params.eval_interval_str = f"{params.check_val_every_n_epoch}ep"

    #if dataloaders['val'] is None:
    if "no_validation" in params and params.no_validation:
        params.eval_interval_str = 0

    return model, optimizer, dataloaders, params