import torch
import os
from torch.nn.utils.rnn import pad_sequence

def generate_square_subsequent_mask(sz):
    """
    Used to mask target sentence
    """
    mask = (torch.triu(torch.ones((sz, sz))) == 1).transpose(0, 1)
    mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
    return mask

def create_mask(src, tgt):
    src_seq_len = src.shape[0]
    tgt_seq_len = tgt.shape[0]

    tgt_mask = generate_square_subsequent_mask(tgt_seq_len)
    src_mask = torch.zeros((src_seq_len, src_seq_len)).type(torch.bool)
    
    # batch_size x man_len
    src_padding_mask = (src == 99).transpose(0, 1)
    tgt_padding_mask = (tgt == 99).transpose(0, 1)
    return src_mask, tgt_mask, src_padding_mask, tgt_padding_mask

a=torch.tensor([0,1,2,3,4,10])
b=torch.tensor([0,4,5,6,7,8,10])
c=torch.tensor([0,7,8,9,9,11,12,10])
de_batch=[a,b,c]
de_batch = pad_sequence(de_batch, padding_value=99)
print(create_mask(de_batch,de_batch[:-1,:]))




