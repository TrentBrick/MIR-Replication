
from sklearn.metrics import mutual_info_score
import torch 
import numpy as np
import matplotlib.pyplot as plt


def discrete_mutual_info(mus, ys):
    """Compute discrete mutual information."""
    num_codes = mus.shape[0]
    num_factors = ys.shape[0]
    m = np.zeros([num_codes, num_factors])
    for i in range(num_codes):
        for j in range(num_factors):
            m[i, j] = mutual_info_score(ys[j, :], mus[i, :])
    return m

def histogram_discretize(target, num_bins=20):
    """Discretization based on histograms."""
    discretized = np.zeros_like(target)
    for i in range(target.shape[0]):
        discretized[i, :] = np.digitize(target[i, :], np.histogram(
            target[i, :], num_bins)[1][:-1])
    return discretized

def compute_mir(model_template, mi_mat, params, alive_neuron_inds, neuron_acts=None, plot=True):

    # nb. nothing in here is a pytorch tensor. 

    nactive_neurons = mi_mat.shape[1]
    assert params.n_features == mi_mat.shape[0]

    rns = []
    for neuron_ind in range( nactive_neurons ):
        n_mis = mi_mat[:,neuron_ind]
        rn = max(n_mis)/ sum(n_mis)
        
        if neuron_acts is not None: 
            #import ipdb 
            #ipdb.set_trace()
            rn *= (neuron_acts[neuron_ind]/sum(neuron_acts)).item()
        else: 
            rn *= 1/nactive_neurons
        rns.append(rn)

    mir = ( sum(rns) - (1/params.n_features) )/(1 - (1/params.n_features))

    if plot: 
        xax = range(nactive_neurons)
        plt.scatter(xax, rns)
        plt.title(f"Neurons vs MI Ratio (Active) Neurons")
        plt.xlabel("Alive Neurons")

        plt.xticks(xax, alive_neuron_inds)

        plt.ylabel("MI Ratio Terms")
        plt.show() 
        print(f"MIR for the model: {model_template} \n is { round(mir, 3) }") 
    
    return round(mir, 3)


def compute_mig(model_template, mi_mat, params, neuron_acts=None, plot=True):
    # Mutual info Gap

    # where am I getting this entropy from? 
    vk_entropy = 0.5*np.log(2*np.pi)+0.5 

    #print(mi_mat.shape )

    f_migs = []
    for factor_ind in range( params.n_features) :

        f_mis = mi_mat[factor_ind,:] 
        f_mis = np.sort(f_mis)
        # max minus next closest 
        if len(f_mis)<2: 
            # only have one active neuron
            f_mig = (f_mis[-1] - 0) / vk_entropy
        else: 
            f_mig = (f_mis[-1] - f_mis[-2]) / vk_entropy

        f_migs.append(f_mig)

    mig = sum(f_migs) / params.n_features

    if plot: 
        plt.scatter(range(params.n_features), f_migs)
        plt.title(f"Neurons vs MI Gap Features")
        plt.xlabel("Factors")
        plt.ylabel("MI Gap Terms")
        plt.show() 
        print(f"MIG for the model: {model_template} \n is { round(mig, 3) }")

    return round(mig, 3)

def compute_mir_n_mig(model_template, mi_mat, params, alive_neuron_inds, neuron_acts=None, plot=False):
    mir = compute_mir(model_template, mi_mat, params, alive_neuron_inds, neuron_acts=neuron_acts, plot=plot)
    mig = compute_mig(model_template, mi_mat, params, neuron_acts=neuron_acts, plot=plot)
    return mir, mig 