import os
import sys
import copy
import torch 
from torch import nn

settings_for_all = dict(
    
    model_name= "WHIT_TOYMODEL",
    dataset_name = "WHIT_TOYFEATURES",

    epochs_to_train_for = 80_000,
    log_disentangle_every_n_steps = 500,

    opt='Adam',
    lr=3e-3,

    batch_size=1024,
    dataset_size = 1024*2,
    
)

memory_per_job = 4000 # for GPU

main_title = "WHIT_Full_Test_CopyCode_FlippedMatrices"

simple_iterable = None #[0.0, 0.01, 0.001, 0.0001, 0.00001]
iterable_param = 'activation_l1_coefficient'#"activation_l1_coefficient" 

name_suffix = f"{main_title}"  

init_exp_list = [

    dict(
        test_name= "ReLU",
    ),

    dict(
        test_name= "ReLU+WD-4",
        adamw_l2_loss_weight=0.0001,
    ),

    dict(
        test_name= "ReLU+L1-3",
        activation_l1_coefficient = 0.001,
    ),

    dict(
        test_name= "ReLU+WD-4+L1-2",
        activation_l1_coefficient = 0.01,
        adamw_l2_loss_weight=0.0001,
    ),

    dict(
        test_name= "ReLU+WD-4+L1-3",
        activation_l1_coefficient = 0.001,
        adamw_l2_loss_weight=0.0001,
    ),

    dict(
        test_name= "ReLU+WD-4+L1-4",
        activation_l1_coefficient = 0.0001,
        adamw_l2_loss_weight=0.0001,
    ),

    dict(
        test_name= "Sigmoid",
        act_func = nn.Sigmoid()
    ),

    dict(
        test_name= "Linear",
        act_func = nn.Identity()
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