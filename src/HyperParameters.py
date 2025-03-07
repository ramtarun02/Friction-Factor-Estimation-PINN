import torch
import torch.nn as nn
import numpy as np

### Model 1
layers = np.array([2, 8, 8, 8, 8, 3])  # Number of nodes in each layer of Neural Networks
train_val = [0.85, 0.15]   # Ratio of Validation Set to be taken from the Training Set, [Train set Size, Val Set Size]
train_size = 0.75 #Ratio of Train Dataset
epochs= 7500
learning_rate= 1e-4     #Comment If Not Needed for the model.
ff_learning_rate = 1e-3   #Comment If Not Needed for the model.
batch_size = 8
input_size = 2
hidden_sizes = [4, 4, 4, 4, 4, 4, 4, 4]
output_size = 4
activations = [nn.ReLU(), nn.Tanh(), nn.ReLU(), nn.Tanh(), nn.ReLU(), nn.Tanh(), nn.ReLU(), nn.Tanh()]
lda = torch.tensor([0.001])



### Model 2
# RNN_input_size = 3
# RNN_hidden_sizes =  [2, 2, 2, 2, 2, 2, 2, 2]
# RNN_output_size = 1
# RNN_activations = [nn.ReLU(), nn.Tanh(), nn.ReLU(), nn.Tanh(), nn.ReLU(), nn.Tanh(), nn.ReLU(), nn.Sigmoid]



### Model 3
<<<<<<< HEAD
# train_val = [0.85, 0.15]   # Ratio of Validation Set to be taken from the Training Set, [Train set Size, Val Set Size]
# train_size = 0.75 #Ratio of Train Dataset
# batch_size = 32
# learning_rate= 1e-4    #Comment If Not Needed for the model.
# input_size = 2
# hidden_size = 4
# num_layers = 2 
# output_size = 4
# # activations = [nn.ReLU(), nn.Tanh(), nn.ReLU(), nn.Tanh()]
# lda = 0.1
# epochs= 500
# seq_length = 8
=======
train_val = [0.85, 0.15]   # Ratio of Validation Set to be taken from the Training Set, [Train set Size, Val Set Size]
train_size = 0.75 #Ratio of Train Dataset
batch_size = 32
learning_rate= 1e-4    #Comment If Not Needed for the model.
input_size = 2
hidden_size = 8
num_layers = 2 
output_size = 4
# activations = [nn.ReLU(), nn.Tanh(), nn.ReLU(), nn.Tanh()]
lda = 0.1
epochs= 250
seq_length = 8
>>>>>>> e4d3ce2871796c3baf74b5eb7a9b9028fc8e6fb8
