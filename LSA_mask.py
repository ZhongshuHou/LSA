import torch
import torch.nn as nn

class FreqLocalMask(nn.Module):
    
    def __init__(self, mask_value=-1e9):

        super(FreqLocalMask, self).__init__()

        self.mask_value = mask_value
        if not isinstance(mask_value, float): raise ValueError("Mask value must be a float.")

    def forward(self, inp):

        # inp.shape = (bs, F, T)

        freq_dim = inp.shape[1]
        local_num = 64
        causal_mask = self.lower_triangular_mask([freq_dim,freq_dim], local_num=local_num)
        return causal_mask



    def lower_triangular_mask(self, shape, local_num=2):
        
        row_index = torch.cumsum(torch.ones(shape,dtype=int), -2)
        col_index = torch.cumsum(torch.ones(shape,dtype=int), -1)
        x1 = torch.greater_equal(row_index, col_index+local_num)
        x2 = torch.greater_equal(row_index, col_index-local_num+1)
        x = x1 ^ x2
        return ~x