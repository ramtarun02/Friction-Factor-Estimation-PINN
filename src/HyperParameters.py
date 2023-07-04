import torch
import torch.nn as nn

### Model 1
# #layers = np.array([2, 8, 8, 8, 8, 3])  # Number of nodes in each layer of Neural Networks
# train_val = [0.85, 0.15]   # Ratio of Validation Set to be taken from the Training Set, [Train set Size, Val Set Size]
# train_size = 0.75 #Ratio of Train Dataset
# epochs= 7500
# learning_rate= 1e-2     #Comment If Not Needed for the model.
# ff_learning_rate = 1e-3   #Comment If Not Needed for the model.
# batch_size = 4
# input_size = 2
# hidden_sizes = [8, 8, 8, 8, 8, 8, 8, 8]
# output_size = 4
# activations = [nn.ReLU(), nn.Tanh(), nn.ReLU(), nn.Tanh(), nn.ReLU(), nn.Tanh(), nn.ReLU(), nn.Tanh()]
# lda = torch.tensor([0.001])



### Model 2
# RNN_input_size = 3
# RNN_hidden_sizes =  [2, 2, 2, 2, 2, 2, 2, 2]
# RNN_output_size = 1
# RNN_activations = [nn.ReLU(), nn.Tanh(), nn.ReLU(), nn.Tanh(), nn.ReLU(), nn.Tanh(), nn.ReLU(), nn.Sigmoid]


### Model 3
train_val = [0.85, 0.15]   # Ratio of Validation Set to be taken from the Training Set, [Train set Size, Val Set Size]
train_size = 0.75 #Ratio of Train Dataset
batch_size = 8
learning_rate= 1e-1     #Comment If Not Needed for the model.
input_size = 2
hidden_sizes =  [6, 6, 6, 6, 6, 6, 6, 6]
output_size = 4
activations = [nn.ReLU(), nn.Tanh(), nn.ReLU(), nn.Tanh(), nn.ReLU(), nn.Tanh(), nn.ReLU(), nn.Sigmoid()]
lda = torch.tensor([0.0001])
epochs= 7500
