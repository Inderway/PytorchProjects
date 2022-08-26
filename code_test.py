import torch
import os
from torch.nn.utils.rnn import pad_sequence

a=torch.tensor([0,1,2,3,4,10])
b=torch.tensor([0,4,5,6,7,8,10])
c=torch.tensor([0,7,8,9,9,11,12,10])
de_batch=[a,b,c]
de_batch = pad_sequence(de_batch, padding_value=99)
print(de_batch)

