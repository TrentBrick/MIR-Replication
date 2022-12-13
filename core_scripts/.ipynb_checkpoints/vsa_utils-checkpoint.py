"""
Library functions to perform circular convolution operations.

https://github.com/NeuromorphicComputationResearchProgram/Learning-with-Holographic-Reduced-Representations/blob/main/lib/mathops.py

"""

__author__ = "Ashwinkumar Ganesan, Sunil Gandhi, Hang Gao"
__email__ = "gashwin1@umbc.edu,sunilga1@umbc.edu,hanggao@umbc.edu"

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import itertools
import matplotlib.pyplot as plt
import copy 
"""
Pytorch functions.
"""

def get_node_neighbours(target, graph):
    edges = []
    for pair in graph:
        if target in pair: 
            tmp = list(copy.copy(pair))
            tmp.remove(target)
            edges.append(tmp[0])
    return edges 

class Graph():

    def __init__(self, data, d, use_complex_proj=True):
        self.data = data
        self.nedges = len(data)
        self.d = d
        self.complex_proj = use_complex_proj
        self.invert = get_inv if use_complex_proj else get_appx_inv

        merged = list(itertools.chain(*data))
        self.nodes = sorted(list(set(merged)))
        self.nnodes = len(self.nodes)
        self.einds = {k:i for i, k in enumerate(self.nodes)}
        self.inds_to_nodes = {v:k for k, v in self.einds.items()}

        self.node_to_neighbours = {node:get_node_neighbours(node, self.data) for node in self.nodes}

        self.create_graph()

    def word_to_hrr(self, word):
        return self.vecs[self.einds[ word ]].unsqueeze(0)

    def create_graph(self, given_vecs = None):
        if given_vecs is not None: 
            self.vecs = given_vecs 

        else: 
            self.vecs = torch.randn(self.nnodes, self.d) * (1/self.d) 
            if self.complex_proj:
                self.vecs = complexMagProj(self.vecs)

        self.edges = [ circular_conv(self.word_to_hrr(pair[0]), self.word_to_hrr(pair[1]) ) for pair in self.data]
        self.edges = torch.stack(self.edges).squeeze()
        self.G = self.edges.sum(0, keepdim=True)

        self.edge_sums = [ (self.word_to_hrr(pair[0]) + self.word_to_hrr(pair[1]) ) for pair in self.data]
        self.edge_sums = torch.stack(self.edge_sums).squeeze()

    def cleanup(self, query, plot = False, threshold=None):

        cosims = (normalize(query) * normalize(self.vecs) ).sum(1)
        argm = cosims.argmax()
        print("Argmax element=", self.inds_to_nodes[int(argm)])

        if threshold is not None: 
            inds = np.arange(len(cosims))[cosims>threshold]
            for i in inds:
                print(round(float(cosims[i]),2) , self.inds_to_nodes[int(i)])

        if plot: 
            plt.scatter(range(self.nnodes), cosims)
            plt.xticks(range(self.nnodes), self.nodes, rotation='vertical', fontsize=12)
            plt.show()


def unbind(query, sup):
    # uses exact inverse!
    return circular_conv(get_inv(query), sup)

def complex_multiplication(left, right):
    """
    Multiply two vectors in complex domain.
    """
    left = torch.view_as_real(left)
    right = torch.view_as_real(right)
    left_real, left_complex = left[..., 0], left[..., 1]
    right_real, right_complex = right[..., 0], right[..., 1]

    output_real = left_real * right_real - left_complex * right_complex
    output_complex = left_real * right_complex + left_complex * right_real
    return torch.view_as_complex( torch.stack([output_real, output_complex], dim=-1))

def complex_division(left, right):
    """
    Divide two vectors in complex domain.
    """
    #print(left,right)
    #left = torch.view_as_real(left)
    #right = torch.view_as_real(right)
    left_real, left_complex = left[..., 0], left[..., 1]
    right_real, right_complex = right[..., 0], right[..., 1]

    output_real = torch.div((left_real * right_real + left_complex * right_complex),(right_real**2 + right_complex**2))
    output_complex = torch.div((left_complex * right_real - left_real * right_complex ),(right_real**2 + right_complex**2))
    return torch.view_as_complex( torch.stack([output_real, output_complex], dim=-1) )

def circular_conv(a, b):
    """ Defines the circular convolution operation
    a: tensor of shape (batch, D)
    b: tensor of shape (batch, D)
    """
    assert len(a.shape)==2 and len(b.shape)==2, "need matrices!"
    left = torch.fft.fft(a)
    right = torch.fft.fft(b)
    output = complex_multiplication(left, right)
    output = torch.fft.ifft(output)
    return torch.view_as_real(output)[...,0]

def get_appx_inv(a):
    """
    Compute approximate inverse of vector a.
    """
    return torch.roll(torch.flip(a, dims=[-1]), 1,-1)

def get_inv(a, typ=torch.DoubleTensor):
    """
    Compute exact inverse of vector a.
    """
    assert len(a.shape)==2, "need matrices!"
    left = torch.fft.fft(a)
    left = torch.view_as_real(left)
    complex_1 = np.zeros(left.shape)
    complex_1[...,0] = 1
    op = complex_division(typ(complex_1),left)
    return torch.view_as_real(torch.fft.ifft(op))[...,0]

def complexMagProj(x):
    """
    Normalize a vector x in complex domain.
    """
    assert len(x.shape)==2, "need matrices!"
    c = torch.fft.fft(x)
    c_ish=c/torch.norm(c, dim=-1,keepdim=True)
    output = torch.fft.ifft(c_ish)
    return torch.view_as_real(output)[...,0]

def normalize(x, dim=1):
    return x/torch.norm(x, dim=dim, keepdim=True)

"""
Numpy Functions.
"""
# Make them work with batch dimensions
def cc(a, b):
    return np.fft.irfft(np.fft.rfft(a) * np.fft.rfft(b))

def np_inv(a):
    return np.fft.irfft((1.0/np.fft.rfft(a)),n=a.shape[-1])

def np_appx_inv(a):
    #Faster implementation
    return np.roll(np.flip(a, axis=-1), 1,-1)

def npcomplexMagProj(x):
    """
    Normalize a vector x in complex domain.
    """
    c = np.fft.rfft(x)

    # Look at real and image as if they were real
    c_ish = np.vstack([c.real, c.imag])

    # Normalize magnitude of each complex/real pair
    c_ish=c_ish/np.linalg.norm(c_ish, axis=0)
    c_proj = c_ish[0,:] + 1j * c_ish[1,:]
    return np.fft.irfft(c_proj,n=x.shape[-1])

def nrm(a):
    return a / np.linalg.norm(a)




