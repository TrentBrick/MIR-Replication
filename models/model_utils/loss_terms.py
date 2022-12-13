import torch 
import torchmetrics
import numpy as np 
import wandb 


def activation_penalty(self, neuron_activations):
    extra_loss = 0

    if self.params.activation_l1_coefficient is not None: 
        extra_loss += self.params.activation_l1_coefficient*neuron_activations.abs().sum(dim=1).mean()

    return extra_loss 


def extra_loss_terms(self):
    # these are scalars
    extra_loss = activation_penalty(self, self.net[1].neuron_acts)

    return extra_loss


