from torch.utils.data import DataLoader, Dataset
import torchvision
from torchvision.datasets import MNIST, CIFAR10
from torchvision import transforms
import torch 
import pandas as pd
import numpy as np
import itertools
from typing import List
import pickle 
#from torchvision.io import read_image

class ToyFeatures(Dataset):

    def __init__(self, dataset_path, train_or_val="train", transform=None, data_size=None, params=None):
        self.data_size = data_size
        self.params = params
        
    def __len__(self):
        return self.data_size

    def __getitem__(self, idx):
        
        e = torch.rand([self.params.n_features])

        return e, e

def make_torchified_loader(params, dataset_name, train_or_val="train"):

    # TODO: load in the transforms like FFCV has. 

    dataset = params.dataset_class(
                dataset_name, train_or_val=train_or_val,transform=None,data_size=params.dataset_size, params=params
                )

    return DataLoader(
            dataset, batch_size=params.batch_size, shuffle=params[f"shuffle_{train_or_val}"], num_workers=params.num_workers
        )

def generate_dataloaders(params, data_path="data/"):

    dataloader_fn = make_torchified_loader
    data_path += params.directory_path

    dataloaders = {
        "train": dataloader_fn(params, 
            data_path, train_or_val="train"),
        }

    dataloaders['val'] = dataloader_fn(params, 
            data_path,train_or_val="val"),

    return dataloaders