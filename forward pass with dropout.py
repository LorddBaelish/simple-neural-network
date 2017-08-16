import numpy as np

#Forward pass for the testing phase
def forward_pass(network, inputs, dropout): # Assumed dropout applied for each layer
    layer_output = inputs
    for layer in network:
        layer_output = layer.output(layer_output) * dropout
    return layer_output

#Forward pass for the training phase
def forward_pass_training(network, inputs, dropout): # Assumed dropout applied for each layer
    layer_output = inputs
    for layer in network:
        dropout_factor = np.random.binomial(1, dropout, size=layer.shape())
        layer_output = np.matmul(np.transpose(dropout_factor)  , layer.output(layer_output))
    return layer_output