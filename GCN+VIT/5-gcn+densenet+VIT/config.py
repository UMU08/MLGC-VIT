import torch


epochs = 100
batch_size = 64
lr = 0.001
weight_decay = 8e-5
drop_rate = 0.25 
num_workers = 0
device = ('cuda:0' if torch.cuda.is_available() else 'cpu' )
K = 2
