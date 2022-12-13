# simple test script to make sure that everything is workign or easy debugging: 

import torch
import wandb
from core_scripts.combine_params import *
from core_scripts.prep_experiment import compile_experiment
import random 
import numpy as np 
from composer import Trainer
from composer.loggers import WandBLogger


load_path = None 

extras = dict(

    num_workers=0, 

    test_name = "ToyTest",

    model_name= "LEE_TOYMODEL",
    dataset_name = "OG_LEE_TOYFEATURES",

    log_disentangle_every_n_steps = 1,

    opt='Adam',
    
    check_val_every_n_epoch = 1,
    nneurons=[32],
    
    lr=0.001,
    epochs_to_train_for = 5_000,
)

if "test_name" not in extras.keys():
    extras["test_name"] = None

if load_path:
    print("LOADING IN A MODEL!!!")

model, optimizer, dataloaders, params, callbacks, algorithms = compile_experiment(extras, extras['num_workers'])

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

if torch.cuda.is_available():
    print("using cuda", device)
    gpu = [0]
else: 
    print("using cpu", device)
    gpu=None

# SETUP TRAINER
if params.load_from_checkpoint and params.load_existing_optimizer_state:
    fit_load_state = load_path
else: 
    fit_load_state = None

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
            eval_interval=f"{params.check_val_every_n_epoch}ep",
            callbacks = callbacks, # includes the Checkpointer
            algorithms=algorithms,
            load_path = params.load_path,
            progress_bar = True, 
            log_to_console = False, 
        )
trainer.fit()
wandb.finish()
