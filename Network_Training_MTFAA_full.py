from typing import OrderedDict
from unicodedata import name
import torch
import torch.optim as optim
# from torch.utils.tensorboard import SummaryWriter
import numpy as np
from torch.optim.lr_scheduler import StepLR
from tqdm import tqdm
import soundfile as sf
from Dataloader import Dataset, collate_fn
from MTFAA_Net_full import MTFAA_Net
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
torch.set_default_tensor_type(torch.FloatTensor)
from signal_processing import iSTFT_module_1_8
WINDOW = torch.sqrt(torch.hann_window(1536,device=device) + 1e-8)

import librosa
import pesq
from collections import OrderedDict
import os


class NoamOpt:
    "Optim wrapper that implements rate."
    def __init__(self, model_size, factor, warmup, optimizer):
        self.optimizer = optimizer
        self._step = 0
        self.warmup = warmup
        self.factor = factor
        self.model_size = model_size
        self._rate = 0
        
    def step(self):
        "Update parameters and rate"
        self._step += 1
        rate = self.rate()
        for p in self.optimizer.param_groups:
            p['lr'] = rate
        self._rate = rate
        self.optimizer.step()
        
    def rate(self, step = None):
        "Implement `lrate` above"
        if step is None:
            step = self._step
        return self.factor * \
            (self.model_size ** (-0.5) *
            min(step ** (-0.5), step * self.warmup ** (-1.5)))

def train(end_epoch = 100):


    def Loss(y_pred, y_true, train = True, idx = -1, epoch = 0):
        snr = torch.div(torch.mean(torch.square(y_pred - y_true), dim=1, keepdim=True),(torch.mean(torch.square(y_true), dim=1, keepdim=True) + 1e-7))
        snr_loss = 10 * torch.log10(snr + 1e-7)
        
        pred_stft = torch.stft(y_pred,1536,384,win_length=1536,window=WINDOW,center=True)
        true_stft = torch.stft(y_true,1536,384,win_length=1536,window=WINDOW,center=True)
        pred_stft_real, pred_stft_imag = pred_stft[:,:,:,0], pred_stft[:,:,:,1]
        true_stft_real, true_stft_imag = true_stft[:,:,:,0], true_stft[:,:,:,1]
        pred_mag = torch.sqrt(pred_stft_real**2 + pred_stft_imag**2 + 1e-12)
        true_mag = torch.sqrt(true_stft_real**2 + true_stft_imag**2 + 1e-12)
        pred_real_c = pred_stft_real / (pred_mag**(0.7))
        pred_imag_c = pred_stft_imag / (pred_mag**(0.7))
        true_real_c = true_stft_real / (true_mag**(0.7))
        true_imag_c = true_stft_imag / (true_mag**(0.7))
        real_loss = torch.mean((pred_real_c - true_real_c)**2)
        imag_loss = torch.mean((pred_imag_c - true_imag_c)**2)
        mag_loss = torch.mean((pred_mag**(0.3)-true_mag**(0.3))**2)

        # if train & (idx%100==0):
        #     with torch.no_grad():
        #         s = model(N)
        #         s = s.cpu().detach().numpy()
        #         np.save('/data/hdd0/zhongshu.hou/Torch_DPCRN/mag_checkpoint/epoch_' + str(epoch) + '_step_' + str(idx) + '.npy', s)

        return 0.3*(real_loss + imag_loss) + 0.7*mag_loss, snr_loss

    '''model'''
    model = MTFAA_Net() 

    ''' train from checkpoints'''
    # checkpoint = torch.load('./.pth',map_location=device)
    # model.load_state_dict(checkpoint['state_dict'])

    model = model.to(device)

    '''optimizer & lr_scheduler'''
    optimizer = NoamOpt(model_size=32, factor=1., warmup=6000,
                    optimizer=torch.optim.Adam(model.parameters(), lr=0, betas=(0.9, 0.98), eps=1e-9))   

    '''load train data'''
    dataset = Dataset(length_in_seconds=3.2, num_clip_per_epoch=4000, random_start_point=True, train=True)
    data_loader = torch.utils.data.DataLoader(dataset, batch_size=3, shuffle=True, drop_last=True, num_workers=2)

    '''start train'''
    for epoch in range(end_epoch):
        train_loss = []
        asnr_loss = []
        model.train()
        dataset.sample()
        dataset.train = True
        dataset.random_start_point = True
        idx = 0

        '''train'''
        print('epoch %s--training' %(epoch))
        for i, data in enumerate(tqdm(data_loader)):
            noisy, clean = data
            noisy = noisy.to(device)
            clean = clean.to(device)
            optimizer.optimizer.zero_grad() 

            noisy_stft = torch.stft(noisy,1536,384,win_length=1536,window=WINDOW,center=True,return_complex=True)
            enh_stft = model(noisy_stft)
            enh_s = iSTFT_module_1_8(n_fft=1536, hop_length=384, win_length=1536,window=WINDOW,center = True,length = noisy.shape[-1])(enh_stft.permute([0, 3, 2, 1]).contiguous())

            stft_loss, snr_loss = Loss(enh_s, clean, train=True, idx = idx, epoch = epoch)  
            loss_overall = stft_loss
            loss_overall.backward() 
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=3)
            optimizer.step() 
            train_loss.append(loss_overall.cpu().detach().numpy())
            idx += 1
        train_loss = np.mean(train_loss) 
        asnr_loss = np.mean(asnr_loss)
        lr_scheduler.step()
        '''eval'''
        valid_loss = []
        model.eval()  
        print('epoch %s--validating' %(epoch))
        dataset.train = False
        dataset.random_start_point = False
        with torch.no_grad():
            for i, data in enumerate(tqdm(data_loader)):

                noisy, clean = data
                noisy = noisy.to(device)
                clean = clean.to(device)
                noisy_stft = torch.stft(noisy,1536,384,win_length=1536,window=WINDOW,center=True, return_complex=True)
                enh_stft = model(noisy_stft)
                enh_s = iSTFT_module_1_8(n_fft=1536, hop_length=384, win_length=1536,window=WINDOW,center = True,length = noisy.shape[-1])(enh_stft.permute([0, 3, 2, 1]).contiguous())

                stft_loss, snr_loss = Loss(enh_s, clean, train=True, idx = idx, epoch = epoch) 
                loss_overall = stft_loss
                valid_loss.append(loss_overall.cpu().detach().numpy())
                asnr_loss.append(snr_loss.cpu().detach().numpy())
            valid_loss = np.mean(valid_loss)
            asnr_loss = np.mean(asnr_loss)
        print('train loss: %s, valid loss %s, snr loss: %s' %(train_loss, valid_loss, asnr_loss))


        torch.save(
            {'epoch': epoch,
                'state_dict': model.state_dict(),
                'optimizer': optimizer.optimizer.state_dict()},
            './model_epoch_%s_trainloss_%s_validloss_%s.pth' %(str(epoch), str(train_loss), str(valid_loss)))


if __name__ == '__main__':
    train(end_epoch=300)
