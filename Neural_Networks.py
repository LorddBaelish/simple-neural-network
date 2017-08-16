# -*- coding: utf-8 -*-S
"""
Created on Mon Aug 14 22:13:37 2017

@author: Achintha Iroshan
"""
import numpy as np


def initialize_network(n_inputs,n_hidden,n_outputs):
    
    network=[[],[]]
    for i in range(n_hidden):
        network[0][i].append({'weights':np.random.rand(n_inputs),'output':np.zeros((n_hidden,), dtype=np.int)})
    for i in range(n_outputs):
        network[1][i].append({'weights':np.random.rand(n_hidden),'output':np.zeros((n_outputs,), dtype=np.int)})
    return network

network1=initialize_network(2,3,2)

def initialize_network_a(n_inputs,n_hidden,n_outputs):
    
    network={}
    W1 = np.random.randn(n_inputs,n_hidden)/np.sqrt(n_inputs)
    W2= np.random.randn(n_hidden,n_outputs)/np.sqrt(n_hidden)
    
    network={'W1':W1,'W2':W2}
    
    return network

#Assume that activation funtion is sigmoid
def forward_pass_a(network, inputs):
    W1,W2=network['W1'],network['W2']
    z1=np.dot(inputs, W1)
    a1 = sigmoid(z1)
    z2=np.dot(a1, W2)
    output= sigmoid(z2)
    return output
    
# Assume that  
def sigmoid (x): return 1/(1 + np.exp(-x))  

def forward_pass(network, inputs):
    
    z1=[]
    z2=[]
    n_hidden=len(network[1][0].get('weights'))
    n_outputs = len(network[1][0].get('output')) 
    
     
    for i in range(n_hidden):
        z1.append(np.dot(inputs,network[0][i]['weights']))
    a1=sigmoid(z1)
    for i in range(n_outputs):
        z2.append(np.dot(a1,network[1][i]['weights']))
    output = sigmoid(z2)
    return output

