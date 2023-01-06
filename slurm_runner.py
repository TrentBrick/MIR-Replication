# Running multiple experiments, have to be on different GPUs each. Used for SLURM.
import argparse
import os
import subprocess as sp
import time

import matplotlib.pyplot as plt
import numpy as np
import ray
import torch
from torch import nn
import copy
import wandb
from core_scripts import *
#from exp_commands.temp_slurm_jobs import *
#from exp_commands import *
import importlib
from composer import Trainer
#from core_scripts import trainer

args = None  # makes args into a global variable that is set during the experiment run

def train_func():
 
    if args.sweep_id:
        # need to find a way to avoid having to hard code this here. 
        wandb.init(
            project="SparseDisentangle",
            entity="kreiman-sdm",
            dir="../scratch_link/wandb_Logger/",
        )
        exp_settings = vars(wandb.config)
        exp_settings = exp_settings["_items"]
        print("Sweep Exp settings", exp_settings)
        exp_settings["save_model_checkpoints"] = False
    else:
        # This is using the list provided above. 
        job_script = importlib.import_module(args.job_script) # "exp_commands."+
        exp_settings = init_exp_settings(args.exp_ind,job_script)


    model, optimizer, dataloaders, params, callbacks, algorithms = compile_experiment(exp_settings, args.num_workers)

    trainer = Trainer(
        model=model,
        run_name=params.test_name,
        loggers = params.logger,
        train_dataloader=dataloaders['train'],
        max_duration=f"{params.epochs_to_train_for}ep",
        eval_dataloader=dataloaders['val'],
        optimizers=optimizer,
        schedulers=None,
        device=params.device_type,
        eval_interval=params.eval_interval_str,
        callbacks = callbacks, # includes the Checkpointer
        algorithms=algorithms,
        load_path = params.load_path,
        progress_bar = False,
        log_to_console = False, 
    )
    trainer.fit()
    wandb.finish()
    print("Training run finished.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--job_script", type=str, required=False, help="The job script.")
    parser.add_argument("--exp_ind", type=int, required=True, help="The job index.")
    parser.add_argument(
        "--num_workers", type=int, required=True, help="Num CPUS for the workers."
    )
    parser.add_argument(
        "--total_tasks",
        type=int,
        required=True,
        help="Can check the number of tasks equals the number of experiments.",
    )
    parser.add_argument(
        "--sweep_id",
        type=str,
        required=False,
        default=None,
        help="Can check the number of tasks equals the number of experiments.",
    )
    args = parser.parse_args()

    if args.sweep_id:  
        # run hyperparameter sweep with settings provided for by the hyper sweep agent as the exp_settings.
        print("Sweep id is:", args.sweep_id)
        wandb.agent(
            args.sweep_id,
            function=train_func,
            project=wandb_project,
            entity="kreiman-sdm",
        )

    else:  # run all of the experiments defined at the top.
        train_func()

