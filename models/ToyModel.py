import torch
import torch.nn as nn
from torch.nn import functional as F
import wandb
import numpy as np
from models import BaseModel
import ipdb
from .MLP import MLP
from core_scripts import global_timestamp
import seaborn as sns
import wandb
import matplotlib.pyplot as plt
from .model_utils.compute_mir_n_migs import *
from scipy.stats import ortho_group

class ToyModel(MLP):
    def __init__(self, params):
        super().__init__(params)
        # this is a wrapper that applies the non linearities here. 

        assert len(params.nneurons) ==1, "too many networok layers for the toy model!"

        w_true=torch.randn([params.input_size, params.input_size])

        self.register_buffer("W_true", w_true )

        use_ortho_project = True 
        if use_ortho_project:
            assert params.n_features == params.input_size, "need features and inputs to be the same for this ortho transform matrix D to work"
            dmat = torch.Tensor(ortho_group.rvs(dim=params.n_features))
        else: 
            dmat = torch.randn([params.input_size, self.params.n_features])
        self.register_buffer("D", dmat )

    def get_alive_neuron_inds(self, neuron_acts):
        nneurons = neuron_acts.shape[0]
        dead_neuron_threshold = 0.1
        dead_mask = neuron_acts.abs().sum(1)<dead_neuron_threshold
        alive_neuron_inds = np.arange(nneurons)[~dead_mask]
        return alive_neuron_inds

    def compute_and_log_mir_and_mig(self, feats, neuron_acts):
        # features come first. batch comes second, 
        # neuron acts is neurons and then batch. 
        num_bins = 20
        
        discretized_mus = histogram_discretize(neuron_acts, num_bins=num_bins)
        discretized_facts = histogram_discretize(feats, num_bins=num_bins)
        m = discrete_mutual_info(discretized_mus, discretized_facts)

        mi_mat = m.T # rows are now factors

        alive_neuron_inds = self.get_alive_neuron_inds(neuron_acts)
        if len(alive_neuron_inds)==0:
            # all dead. continue 
            return

        mi_mat = mi_mat[:,alive_neuron_inds]
        neuron_acts = neuron_acts.abs().sum(1)[alive_neuron_inds]

        mir, mig = compute_mir_n_mig(self.params.test_name, mi_mat, self.params, alive_neuron_inds, plot=False)

        act_w_mir, _ = compute_mir_n_mig(self.params.test_name, mi_mat, self.params, alive_neuron_inds, neuron_acts=neuron_acts, plot=False)

        wb_dict ={
            'global_step':global_timestamp.CURRENT_TRAIN_STEP,
            'epoch':global_timestamp.CURRENT_EPOCH,
            "mutual_info/MIR": mir, 
            "mutual_info/MIG": mig, 
            "mutual_info/MIR_act_weighted": act_w_mir, 
        }

        if self.params.nneurons[0]<self.params.heatmap_logging_threshold:
            heatmap_fig, ax = plt.subplots(figsize=(16,10)) # figsize=(16,10)
            sns.heatmap(m, annot=True, ax=ax)
            wb_dict["mutual_info/MI Heatmap"] = wandb.Image(heatmap_fig, caption=self.params.test_name )

        wandb.log(wb_dict)

        plt.close()

    @MLP.parse_mosaic
    def forward(self, e):
        # just need to create the x and y for the dataset
        with torch.no_grad():
            x = e@self.D.T
            self.true_x = torch.clone(x)
            # does it need to learn a transform here? Can it not do a reconstruction objective? 
            # NOTE: may need for D to be orthogonal. 
            self.true_y = e@self.W_true.T
       
        y_pred = super().forward(x)

        if self.training and self.params.use_wandb and global_timestamp.CURRENT_TRAIN_STEP%self.params.log_disentangle_every_n_steps==0:

                self.compute_and_log_mir_and_mig( e.T.detach().cpu(), self.net[1].neuron_acts.T.detach().cpu() )
               

        return y_pred

