from operator import matmul
from this import d
import torch
from torch import nn
import numpy as np
from torch.nn.modules.conv import ConvTranspose2d
# device = torch.device("cuda:2" if torch.cuda.is_available() else "cpu")
# device = torch.device('cuda:0')
torch.set_default_tensor_type(torch.FloatTensor)
import math

import math
from collections import OrderedDict
from typing import Callable, Iterable, List, Optional, Tuple, Union

import numpy as np
import torch
from torch import Tensor, nn
from torch.nn import functional as F
from torch.nn import init
from torch.nn.parameter import Parameter
from typing_extensions import Final


ERB_fb = np.load('./erb.npy').astype(np.float32)


class AttentionMask(nn.Module):
    
    def __init__(self, causal, mask_value=-1e9):

        super(AttentionMask, self).__init__()
        self.causal = causal
        self.mask_value = mask_value
        if not isinstance(mask_value, float): raise ValueError("Mask value must be a float.")

    def forward(self, inp):

        # inp.shape = (bs, F, T)
        batch_size = inp.shape[0]
        max_seq_len = inp.shape[2]
        if self.causal:
            causal_mask = self.lower_triangular_mask([1,max_seq_len,max_seq_len])
            unmasked = torch.zeros([batch_size, max_seq_len, max_seq_len])
            masked = torch.fill_(torch.empty([batch_size, max_seq_len, max_seq_len]), self.mask_value)
            att_mask = torch.where(causal_mask, unmasked, masked)
            return att_mask
        else:
            return torch.zeros([batch_size, max_seq_len, max_seq_len], dtype=torch.float32)


    def lower_triangular_mask(self, shape):

        row_index = torch.cumsum(torch.ones(shape,dtype=int), -2)
        col_index = torch.cumsum(torch.ones(shape,dtype=int), -1)
        return torch.greater_equal(row_index, col_index)

def apply_complex(fr, fi, input, dtype = torch.complex64):
    return (fr(input.real)-fi(input.imag)).type(dtype) \
            + 1j*(fr(input.imag)+fi(input.real)).type(dtype)


class ComplexConv2d(nn.Module):
    
    def __init__(self,in_channels, out_channels, kernel_size=3, stride=1, padding = 'same',
                 dilation=1, groups=1, bias=True):
        super(ComplexConv2d, self).__init__()
        self.conv_r = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias)
        self.conv_i = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias)
        
    def forward(self,input):    
        return apply_complex(self.conv_r, self.conv_i, input)

class PE(nn.Module):
    
    def __init__(self):
        super(PE, self).__init__()
        self.complex_conv = ComplexConv2d(in_channels=1, out_channels=4, kernel_size=(3,1), stride=(1,1))
        
    def forward(self,x):
        #x.shape = (Bs, F, T), dtype=complex64
        x = x.view(x.shape[0], x.shape[1], x.shape[2], 1) #x.shape = (Bs, F, T, 1)
        x = x.permute(0,3,2,1) #(Bs, 1, T, F)
        x = self.complex_conv(x)  #(Bs, 4, T, F)
        x = torch.abs(x) #(Bs, 4, T, F), dtype=real
        x = torch.pow(x + 1e-12, 0.5) #(Bs, 4, T, F), dtype=real
        
        return x

class BM(nn.Module):
    
    def __init__(self, fft_num, f_c):
        super(BM, self).__init__()
        self.bm = nn.Linear(fft_num//2+1, f_c, bias=False)
        self.bm.weight = nn.Parameter(torch.from_numpy(ERB_fb), requires_grad=False)  

        #---------------------------high learnt-----------------------
        # self.flc_low = nn.Linear(fft_num//2+1, 160, bias=False)
        # self.flc_low.weight = nn.Parameter(torch.from_numpy(scm[:160, :]), requires_grad=False)

        # self.flc_high = nn.Linear(fft_num//2+1, 96, bias=False)
        # self.flc_high.weight = nn.Parameter(torch.from_numpy(scm[160:, :]), requires_grad=True)
        
    def forward(self,x):
        #x.shape = (Bs, 4, T, F), dtype=real
        x = self.bm(x) #(Bs, 4, T, F_c), dtype=real

        # x = x.to(torch.float32)
        # x_low = self.flc_low(x)
        # x_high = self.flc_high(x)
        # x = torch.cat([x_low, x_high], -1)
        
        return x
class BS(nn.Module):
    
    def __init__(self, fft_num, f_c):
        super(BS, self).__init__()
        self.bs = nn.Linear(f_c, fft_num//2+1, bias=False)
        self.bs.weight = nn.Parameter(torch.from_numpy(ERB_fb.T), requires_grad=False)  
        
    def forward(self,x):
        #x.shape = (Bs, 4, T, F), dtype=real
        x = self.bs(x) #(Bs, 4, T, F_c), dtype=real
        
        return x

class FD(nn.Module):
    
    def __init__(self, in_channels, out_channels):
        super(FD, self).__init__()
        self.conv_1 = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=(1,7), stride=(1,4), padding=(0,2),groups=2)
        self.bn_1 = nn.BatchNorm2d(out_channels)
        self.act_1 = nn.PReLU()


    def forward(self,x):
        #x.shape = (Bs, 4, T, F_c), dtype=real
        x = self.act_1(self.bn_1(self.conv_1(x))) #(Bs, C, T, F1), dtype=real
        
        return x


class TFCM_block(nn.Module):
    
    def __init__(self, in_channels, kernel_size, dilation):
        super(TFCM_block, self).__init__()
        self.dilation = dilation[0]
        self.k_t = kernel_size[0]
        self.pointwise_1 = nn.Conv2d(in_channels, in_channels, 1,1,0,1,1, bias=False)
        self.bn_1 = nn.BatchNorm2d(in_channels)
        self.act_1 = nn.PReLU()
        self.pointwise_2 = nn.Conv2d(in_channels, in_channels, 1,1,0,1,1, bias=False)
        self.depthwise = nn.Conv2d(in_channels, in_channels, kernel_size, dilation=dilation, padding='valid', groups=in_channels,bias=False) 
        self.bn_2 = nn.BatchNorm2d(in_channels)
        self.act_2 = nn.PReLU()

    def forward(self,x):
        #x.shape = (Bs, C, T, F'), dtype=real
        x1 = self.act_1(self.bn_1(self.pointwise_1(x))) #(Bs, C, T, F'), dtype=real
        x1 = torch.nn.functional.pad(x1,pad=(1,1,int(self.dilation*(self.k_t-1)),0,0,0), mode='constant', value=0)
        x1 = self.act_2(self.bn_2(self.depthwise(x1)))
        x1 = self.pointwise_2(x1)
        x = x + x1
        
        return x


class TFCM(nn.Module):
    
    def __init__(self, B, in_channels):
        super(TFCM, self).__init__()
        self.blocks = nn.ModuleList([TFCM_block(in_channels, (3,3), (2**b,1)) for b in range(B)])

    def forward(self,x):
        #x.shape = (Bs, C, T, F'), dtype=real
        for block in self.blocks:
            x = block(x)
        
        return x

class ASA(nn.Module):
    
    def __init__(self, in_channels):
        super(ASA, self).__init__()
        self.pointwise_1 = nn.Conv2d(in_channels, in_channels//4, 1,1,0,1,1, bias=False)
        self.bn_1 = nn.BatchNorm2d(in_channels//4)
        self.act_1 = nn.PReLU()
        self.pointwise_2 = nn.Conv2d(in_channels, in_channels//4, 1,1,0,1,1, bias=False)
        self.bn_2 = nn.BatchNorm2d(in_channels//4)
        self.act_2 = nn.PReLU()
        self.pointwise_3 = nn.Conv2d(in_channels//4, in_channels, 1,1,0,1,1, bias=False)
        self.bn_3 = nn.BatchNorm2d(in_channels)
        self.act_3 = nn.PReLU()

    def forward(self,x):
        #x.shape = (Bs, Ci, T, F), dtype=real
        C = x.shape[1]
        F = x.shape[3]
        T = x.shape[2]
        x1 = self.act_1(self.bn_1(self.pointwise_1(x)))
        qf = x1.permute(0,2,3,1)
        v = x1.permute(0,2,3,1)
        kf = x1.permute(0,2,1,3)
        mmf = torch.softmax(torch.matmul(qf,kf) / math.sqrt(C*F/2), dim=-1)
        mmf = torch.matmul(mmf,v) #shape = (Bs, T, F, C)

        x2 = self.act_2(self.bn_2(self.pointwise_2(x)))
        qt = x2.permute(0,3,2,1) #shape = (Bs, F, T, C)
        kt = x2.permute(0,3,1,2) #shape = (Bs, F, C, T)
        mmt = torch.matmul(qt,kt) / math.sqrt(C*T/2) #shape = (Bs, F, T, T)
        mask = AttentionMask(causal=True)(mmt).to(mmt.device)               #shape = (Bs, T, T)
        mask = mask.view(mask.shape[0],1, mask.shape[2],mask.shape[2])       #shape = (Bs, 1, T, T)
        mmt = torch.softmax(mmt + mask, dim=-1) #shape = (Bs, F, T, T)

        mmf = mmf.permute(0,2,1,3) #shape = (Bs, F, T, C)
        m = torch.matmul(mmt, mmf) #shape = (Bs, F, T, C)
        m = m.permute(0,3,2,1) #shape = (Bs, C, T, F)
        m = self.act_3(self.bn_3(self.pointwise_3(m)))

        out = m + x
        return out

class FU(nn.Module):
    
    def __init__(self, in_channels, out_channels):
        super(FU, self).__init__()
        self.pointwise_1 = nn.Conv2d(2*in_channels, in_channels, 1,1,0,1,1, bias=False)
        self.bn_1 = nn.BatchNorm2d(in_channels)
        self.act_1 = nn.Tanh()

        self.pointwise_2 = nn.Conv2d(in_channels, in_channels, 1,1,0,1,1, bias=False)
        self.bn_2 = nn.BatchNorm2d(in_channels)
        self.act_2 = nn.PReLU()

        self.pointwise_3 = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=(1,7), stride=(1,4),groups=2)
        self.bn_3 = nn.BatchNorm2d(out_channels)
        self.act_3 = nn.PReLU()



    def forward(self,x, fd):
        x = torch.cat([x,fd], dim=1)
        x = self.act_1(self.bn_1(self.pointwise_1(x)))
        x = x * fd
        x = self.act_2(self.bn_2(self.pointwise_2(x)))
        x = self.act_3(self.bn_3(self.pointwise_3(x)))
        
        return x

class FDTA(nn.Module):
    
    def __init__(self, in_channels, out_channels, B):
        super(FDTA, self).__init__()
        self.fd = FD(in_channels, out_channels)
        self.tf_conv = TFCM(B=B, in_channels=out_channels)
        self.asa = ASA(out_channels)
    def forward(self,x):
        fd_out = self.fd(x)
        oup = self.asa(self.tf_conv(fd_out))
                    
        return [oup,fd_out]

class FUTA(nn.Module):
    
    def __init__(self, in_channels, out_channels, B):
        super(FUTA, self).__init__()
        self.fu = FU(in_channels, out_channels)
        self.tf_conv = TFCM(B=B, in_channels=out_channels)
        self.asa = ASA(out_channels)
    def forward(self,x, fd_out):
        fu_out = self.fu(x, fd_out)
        oup = self.asa(self.tf_conv(fu_out))
                    
        return oup

class MTFAA_Net(nn.Module):
    
    def __init__(self):
        super(MTFAA_Net, self).__init__()
        self.pe = PE()
        self.bm = BM(fft_num=1536, f_c=256)
        self.fdta1 = FDTA(in_channels=4, out_channels=48, B=6)
        self.fdta2 = FDTA(in_channels=48, out_channels=96, B=6)
        self.fdta3 = FDTA(in_channels=96, out_channels=192, B=6)
        self.tf_conv1 = TFCM(B=6, in_channels=192)
        self.asa1 = ASA(192)
        self.tf_conv2 = TFCM(B=6, in_channels=192)
        self.asa2 = ASA(192)
        self.futa1 = FUTA(in_channels=192, out_channels=96, B=6)
        self.futa2 = FUTA(in_channels=96, out_channels=48, B=6)
        self.futa3 = FUTA(in_channels=48, out_channels=6, B=6)
        self.bs = BS(fft_num=1536, f_c=256)
        self.m = Mask()


    def forward(self,x):
        y = self.pe(x)
        y = self.bm(y)  
        [y,fd_out1] = self.fdta1(y)
        [y,fd_out2] = self.fdta2(y)
        [y,fd_out3] = self.fdta3(y)
        y = self.asa2(self.tf_conv2(self.asa1(self.tf_conv1(y))))
        y = self.futa1(y,fd_out3)[:,:,:,:-3]     
        y = self.futa2(y,fd_out2)[:,:,:,:-3]  
        y = self.futa3(y,fd_out1)[:,:,:,:-3]
        y = self.bs(y)
        mask = y.permute(0,1,3,2)
        oup = self.m(mask, x)
        return oup

class Mask(nn.Module):
    
    def __init__(self):
        super(Mask, self).__init__()
        self.unfold = torch.nn.Unfold(kernel_size=(3,1), padding=(1,0))


    def forward(self,mask, spec):
        # mask.shape = [bs,6,F,T], dtype=real
        # spec.shape = [bs,F,T], dtype=complex
        mask_s1 = mask[:,:3,:,:]
        mask_s2_mag = mask[:,3,:,:]
        mask_s2_pha= mask[:,4,:,:]
        spec = spec.reshape([spec.shape[0], 1, spec.shape[1],spec.shape[2]])
        mag = torch.abs(spec)
        pha = torch.angle(spec)[:,0,:,:]

        mag_unfold = self.unfold(mag).reshape([spec.shape[0], 3, -1,spec.shape[3]])
        x = torch.sum(mag_unfold * mask_s1, dim=1) # x.shape = [bs,F,T], dtype=real
        real = x * mask_s2_mag * torch.cos(pha + mask_s2_pha)
        imag = x * mask_s2_mag * torch.sin(pha + mask_s2_pha)
        real = real.reshape([real.shape[0], real.shape[1],real.shape[2],1])
        imag = imag.reshape([imag.shape[0], imag.shape[1],imag.shape[2],1])
        return  torch.cat([real, imag],dim=-1)

