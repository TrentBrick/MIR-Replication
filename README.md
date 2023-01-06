Trying to replicate MIR results. 


## Main scripts: 
* models/ToyModel.py -- does most of the heavy lifting for the model. Includes generating the dataset features. 
* models/model_utils/compute_mir_n_migs.py -- computes MIR. 
* core_scripts/ -- sets up the experiments, other admin overhead. 
* core_params/ -- has model and dataset parameters. 
* experiment_generator.py -- has the main experiment tested. 