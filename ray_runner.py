# Running multiple experiments across GPUs
import ray
import torch
import matplotlib.pyplot as plt
from core_scripts import * 

from composer import Trainer

#import pytorch_lightning as pl
import wandb
import multiprocessing
from torch import nn
import numpy as np
import time 
import os 
import subprocess as sp
import argparse
import copy 
from core_scripts import *
#from exp_commands.temp_slurm_jobs import *
#from exp_commands import *
import importlib

#from composer.datasets.ffcv_utils import ffcv_monkey_patches
#ffcv_monkey_patches()
 
ncpus_per_worker = 1  # number of cpus each job is allocated
# assuming they are all quite homogenous: 

gpu_capacity = 16160 # 
enforce_memory_per_job = False  # false enables dynamic memory allocation but I will still estimate memory per job to not overload a given GPU.
activity_threshold = 50 #out of 100
ncpus_to_allocate = 16

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--job_script", "--js", type=str, default=None, required=False, help="The job script.")

    args = parser.parse_args()

    if not args.job_script:
        job_script = importlib.import_module("experiment_generator")

    else: 
        job_script = importlib.import_module("exp_commands."+args.job_script)

    memory_per_job = job_script.memory_per_job

    # Getting the GPUs that are free
    mem_threshold_to_use_gpu = gpu_capacity-memory_per_job  # in MB. Must be below to use the GPU.
    gpus_to_use, gpu_str = get_free_gpus(mem_threshold_to_use_gpu, activity_threshold, use_all=True)
    ngpus = len(gpus_to_use)
    os.environ["CUDA_VISIBLE_DEVICES"] = gpu_str

    fraction_of_gpu = memory_per_job / gpu_capacity
    print("Fraction of each GPU to use:", round(fraction_of_gpu,2))

    print("3 Seconds to Cancel")
    time.sleep(3)

    if enforce_memory_per_job:
        for new_gpu_ind in range(ngpus):
            torch.cuda.set_per_process_memory_fraction(fraction_of_gpu, new_gpu_ind)

    print('number of gpus being used', ngpus)

    # Use local_mode = True to debug. 
    ray.init(local_mode=False, num_cpus=ncpus_to_allocate, num_gpus=ngpus)

    @ray.remote(num_gpus=fraction_of_gpu, num_cpus=ncpus_per_worker, max_calls=1)
    def start_experiment(exp_ind):

        # pulls specific details for this experiment index. 
        exp_settings = init_exp_settings(exp_ind,job_script)

        model, optimizer, dataloaders, params, callbacks, algorithms = compile_experiment(exp_settings, ncpus_per_worker)

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


    out = ray.get([start_experiment.remote(exp_ind) for exp_ind in range(len(job_script.exp_list)) ])
    print("Should now be running ray shutdown!")
    ray.shutdown()