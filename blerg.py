import os
import sys
import copy
import torch 
from torch import nn

settings_for_all = dict(
    # Model and Dataset
    model_name= "MLP",
    dataset_name = "LatentCIFAR10" ,
 
    # Learning Params
    epochs_to_train_for = 250,
    opt='AdamW',#SparseAdam',
    lr=0.01,#0.0001,#0.15,
    #batch_size=512,

    # Model Params
    diffusion_noise=0.0,
    nneurons=[1000],
    
    # Logging params
    use_wandb = True, 
    #check_val_every_n_epoch = 1000,
    
)

memory_per_job = 4000 # for GPU

main_title = "Whittington_16N_RestoredOutBias_RightHparams"

epoch_and_step = "epoch=5989-step=293510.ckpt"#"epoch=799-step=39200.ckpt"
exp_name_path = "ReconCIFAR10Long_EvenMore_Adam_lr0.0001_datas=None_10000Neurons_projM=False_nlayers1"#"Poisson_Adam_lr0.0001_datas=None_10000Neurons_projM=False_nlayers1" #"Baseline_Adam_lr0.0001_datas=None_10000Neurons_projM=False_nlayers1"

load_all_models = False

simple_iterable = [0.1, 0.01, 0.001, 0.0001]
iterable_param = None#"activation_l1_coefficient" 

# for this specific set of experiments. 
modifications = dict(

    model_name= "WHIT_TOYMODEL",
    dataset_name = "WHIT_TOYFEATURES",
    opt='SGD',
    lr=0.003,

    use_inhib_circuit = False, 

    classification=False,
    nneurons=[16],
    epochs_to_train_for = 80_000,
    log_disentangle_every_n_steps = 500,

    batch_size=1024,
    dataset_size = 1024*2,

    #check_val_every_n_epoch = 1,
    #activation_l1_coefficient = 0.01,
    #learn_subgraphs = True,

)

settings_for_all.update(modifications)

# sigma={settings_for_all['diffusion_noise']}
name_suffix = f"{main_title}_{settings_for_all['opt']}_lr{settings_for_all['lr']}"  
# _kmin={settings_for_all['k_min']}


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

'''

dict(
        test_name= "MLP_Baseline",
        model_name = "MLP",
    ),
    
'''

if simple_iterable is not None and iterable_param is not None:
    exp_list = [] 
    for it in simple_iterable:
        for iexp in init_exp_list: 
            temp = copy.deepcopy(iexp)
            temp["test_name"] += f"{it}_{iterable_param}_"
            temp[iterable_param] = it
            exp_list.append(temp)

elif load_all_models: 
    exp_list = []
    #for rand_seed in [10, 55, 78]:
    for e in init_exp_list: 
        temp = copy.deepcopy(e)
        temp['load_path'] = f"../scratch_link/Foundational-SDM/wandb_Logger/{temp['test_name']}{exp_name_path}/version_None/checkpoints/{epoch_and_step}"
        exp_list.append( temp )
else: 
    exp_list = init_exp_list
    
if __name__ == '__main__':
    print(len(exp_list)) # needed for the slurm pickup. 
    sys.exit(0)