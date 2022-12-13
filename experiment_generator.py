import os
import sys
import copy
import torch 
from torch import nn

settings_for_all = dict(
    
    model_name= "LEE_TOYMODEL",
    dataset_name = "OG_LEE_TOYFEATURES",

    epochs_to_train_for = 10_000,
    log_disentangle_every_n_steps = 500,
    
)

memory_per_job = 4000 # for GPU

main_title = "L1_FullTest"

simple_iterable = [0.0, 0.01, 0.001, 0.0001, 0.00001]
iterable_param = 'activation_l1_coefficient'#"activation_l1_coefficient" 

name_suffix = f"{main_title}"  

init_exp_list = [

    dict(
        test_name= "Model",
    ),

]

if simple_iterable is not None and iterable_param is not None:
    exp_list = [] 
    for it in simple_iterable:
        for iexp in init_exp_list: 
            temp = copy.deepcopy(iexp)
            temp["test_name"] += f"{it}_{iterable_param}_"
            temp[iterable_param] = it
            exp_list.append(temp)
else: 
    exp_list = init_exp_list
    
if __name__ == '__main__':
    print(len(exp_list)) # needed for the slurm pickup. 
    sys.exit(0)