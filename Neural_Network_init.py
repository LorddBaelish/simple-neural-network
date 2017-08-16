# -*- coding: utf-8 -*-
"""
Created on Wed Aug 16 15:25:36 2017

@author: Achintha Iroshan
"""

import numpy as np


def initialize_network(n_inputs,n_hidden,n_outputs):
    
    network=[[],[]]
    for i in range(n_hidden):
        network[0].append({'weights':np.random.rand(n_inputs),'output':0})
    for i in range(n_outputs):
        network[1].append({'weights':np.random.rand(n_hidden),'output':0})
    return network

def forward_pass(network, inputs):
    
    z1=[]
    z2=[]
    a1=[]
    output=[]
    n_hidden=len(network[0])
    n_outputs = len(network[1])     
     
    for i in range(n_hidden):
        z1.append(np.dot(inputs,network[0][i]['weights']))
        
    for i in range(n_hidden):
        network[0][i]['output']= sigmoid(z1[i])
        a1.append(network[0][i]['output'])
    for i in range(n_outputs):
        z2.append(np.dot(a1,network[1][i]['weights']))
    
    for i in range(n_outputs):
        network[1][i]['output']= sigmoid(z2[i])
        output.append(network[1][i]['output'])
    return output

def sigmoid (x): return 1/(1 + np.exp(-x))  
network1=initialize_network(1,2,1)
output1 = forward_pass(network1,[2])


