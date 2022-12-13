# shared by ray and slurm runners
import copy 
import wandb 
from core_scripts.combine_params import *
#import pytorch_lightning as pl
#from pytorch_lightning.loggers import WandbLogger
from composer.loggers import WandBLogger
from composer.callbacks import CheckpointSaver
import numpy as np 
import random
import torch
from composer.algorithms import GradientClipping
from models.model_utils.callbacks import *

def init_exp_settings(exp_ind,job_script):
    # takes in a job script and preps a specific experiment from the settings for all of them. 
    exp_settings = job_script.exp_list[exp_ind]
    settings_for_all = copy.deepcopy(job_script.settings_for_all)
    settings_for_all.update(exp_settings)
    exp_settings = settings_for_all

    if "test_name" in exp_settings.keys():
        exp_settings['name_prefix'] = copy.deepcopy(exp_settings['test_name'])
    if job_script.name_suffix:
        exp_settings['name_suffix'] = job_script.name_suffix
        exp_settings['test_name']+=job_script.name_suffix

    return exp_settings

def compile_experiment(exp_settings, num_workers):

    model_name = exp_settings.pop("model_name")
    
    if "dataset_name" in exp_settings.keys():
        dataset_name = exp_settings.pop("dataset_name")

    print("Init experiment", model_name, "special params:", exp_settings)

    exp_settings["num_workers"] = num_workers

    if "load_path" in exp_settings.keys():
        print("LOADING IN A MODEL!!!")
    else:
        exp_settings["load_path"] = None

    model, optimizer, data_module, params = get_params_net_dataloader(
        model_name, dataset_name, load_from_checkpoint=exp_settings["load_path"], experiment_param_modifications=exp_settings
    )

    tags = [model_name] #params.opt, # dataset_name.name, 

    if "test_description" not in exp_settings.keys():
        exp_settings["test_description"] = None

    if "test_name" not in exp_settings.keys():
        exp_settings["test_name"] = None

    if params.use_wandb:
        params.logger = WandBLogger(
            project=params.project_name,
            entity=params.entity,
            log_artifacts=False,
            tags=tags,
            name=exp_settings["test_name"],
            init_kwargs = {
                "dir":params.output_directory,
                "notes":exp_settings["test_description"],
                "config":params.__dict__,
                },#params.__dict__,
        )
    else: 
        params.logger = None

    callbacks = [SyncGlobalTimeStamp(), SaveOutHParams()]
    # PushDeadNeurons(), LogMetricsFromActivationFunctions()
        
    if params.save_model_checkpoints:
        model_checkpoint_obj = CheckpointSaver(
            folder=f'{params.output_directory}/{params.test_name}/checkpoints',
            save_interval=f'{params.checkpoint_every_n_epochs}ep',
            overwrite=True,
            num_checkpoints_to_keep=params.num_checkpoints_to_keep,
        )
        callbacks.append(model_checkpoint_obj)

    # algorithms 
    algorithms = [] #ModelSurgery()

    if params.gradient_clip>0.0: 
        algorithms.append( GradientClipping(clipping_type=params.clipping_type, clipping_threshold=params.gradient_clip) )

    return model, optimizer, data_module, params, callbacks, algorithms