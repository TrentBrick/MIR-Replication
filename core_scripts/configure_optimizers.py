import torch.optim as optim 

'''
When adding a new optimizer or LR, simply add a line to this dict with the relevant parameters. This is used to fill out `config_optimizers`
in the LightningModule body of `Base_model.py` and is abstracted away into this file for brevity and modularity. 
'''
def configure_optimizers_(model, verbose=False):
    params = model.params
    lr = params.lr

    # adding everything
    params_to_opt = list(model.parameters())

    ########### Set Optimizer ############
    if params.opt =="SGD": 
        optimizer = optim.SGD(params_to_opt,    lr=lr)
    elif params.opt == "SGDM":
        optimizer = optim.SGD(
            params_to_opt, lr=lr, momentum=params.sgdm_momentum)
    elif params.opt == "Adam": 
        optimizer = optim.Adam(params_to_opt, lr=lr,betas=params.adam_betas)
    elif params.opt ==  "RMSProp": 
        optimizer = optim.RMSprop(params_to_opt, lr=lr) 
    elif params.opt == "AdamW": 
        optimizer = optim.AdamW(params_to_opt, lr=lr, weight_decay=params.adamw_l2_loss_weight)
    elif params.opt == "AdaGrad": 
        optimizer = optim.Adagrad(params_to_opt, lr=lr)
    elif params.opt == "AdaFactor": 
        optimizer = add_opts.Adafactor(params_to_opt, lr=lr)
    elif params.opt == "Shampoo": 
        optimizer =add_opts.Shampoo(params_to_opt, lr=lr)

    elif params.opt == "SparseAdam": optimizer =SparseAdam(params_to_opt,
        lr=lr, betas=params.adam_betas)

    elif params.opt == "SparseSGDM": 
        optimizer =SparseSGDM(params_to_opt, lr=lr, momentum=params.sgdm_momentum)

    elif params.opt == "DemonAdam": 
        optimizer =DemonRanger(params_to_opt, 
                    lr=lr, 
                    #weight_decay=config.wd,
                    epochs = params.epochs_to_train_for,
                    #step_per_epoch = step_per_epoch, 
                    betas=(0.9,0.999,0.999), # restore default AdamW betas
                    nus=(1.0,1.0), # disables QHMomentum
                    k=0,  # disables lookahead
                    alpha=1.0, 
                    IA=False, # enables Iterate Averaging
                    rectify=False, # disables RAdam Recitification
                    AdaMod=False, #disables AdaMod
                    AdaMod_bias_correct=False, #disables AdaMod bias corretion (not used originally)
                    use_demon=True, #enables Decaying Momentum (DEMON)
                    use_gc=False, #disables gradient centralization
                    amsgrad=False # disables amsgrad
                    )
    else: 
        raise NotImplementedError("Need to implement optimizer")
    
    ########### Return to Base_model.py ############
    if verbose:
        print("length of net parameters", len(params_to_opt))

    return optimizer 