import numpy as np


def softmax_classification(network, ground_truth):
    network_output = network.output() #An array containing output activation values of the last layer
    exponential_values = np.exp(network_output)
    softmax_vector = [round(i/np.sum(exponential_values), 3) for i in exponential_values]
    return softmax_vector

print softmax_classification(None, None)