from typing import OrderedDict
from unicodedata import name
import torch
import torch.optim as optim
# from torch.utils.tensorboard import SummaryWriter
import numpy as np
from torch.optim.lr_scheduler import StepLR
from tqdm import tqdm
import soundfile as sf
from Dataloader_DNS import Dataset, collate_fn
from MTFAA_Net_MSLSA import MTFAA_Net
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
torch.set_default_tensor_type(torch.FloatTensor)
from signal_processing import iSTFT_module_1_8
WINDOW = torch.sqrt(torch.hann_window(1536,device=device) + 1e-8)

import librosa
import pesq
from collections import OrderedDict
import os
# WINDOW = torch.sqrt(torch.hamming_window(1200,device=device))
#从训练结果来看hamming比hann能达到更低的loss#

# audio_test_dir = '/data/hdd0/zhongshu.hou/Torch_DPCRN/Valid_enh_FTCRN/noisy/nsy12.wav'
# au, _ = sf.read(audio_test_dir)
# N = torch.stft(torch.from_numpy(au).reshape([1,len(au)]).to(device),1200,600,win_length=1200,window=WINDOW,center=True)
noisy_dir = '/data/hdd0/zhongshu.hou/g9_copy/MTFAA_full/eval_dns/noisy'
clean_test_dir = '/data/hdd0/zhongshu.hou/g9_copy/MTFAA_full/eval_dns/clean_16k'
clean_list = librosa.util.find_files(clean_test_dir, ext='wav')
noisy_list = librosa.util.find_files(noisy_dir, ext='wav')

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
    model = MTFAA_Net() # 定义模型

    ''' train from checkpoints'''
    # checkpoint_DPCRN = torch.load('/data/hdd0/zhongshu.hou/Torch_Convtasnet/checkpoints_48k_DPCRN4_power_compress_loss_istft_sc_lrned/model_epoch_99_trainloss_0.06429266_validloss_0.060150404_snr_loss_-14.194487.pth',map_location=device)
    # model_DPCRN.load_state_dict(checkpoint_DPCRN['state_dict'])

    '''multi gpu'''
    # model_DPCRN = torch.nn.DataParallel(model_DPCRN, device_ids=device_ids)
    # model_MHA = torch.nn.DataParallel(model_MHA, device_ids=device_ids)
    # model = torch.nn.DataParallel(model, device_ids=device_ids)
    model = model.to(device)


    '''optimizer & lr_scheduler'''
    # optimizer = torch.optim.Adam(model.parameters(), lr=5e-4)
    optimizer = NoamOpt(model_size=32, factor=1., warmup=6000,
                    optimizer=torch.optim.Adam(model.parameters(), lr=0, betas=(0.9, 0.98), eps=1e-9))   

    # lr_scheduler = StepLR(optimizer, step_size=30, gamma=0.5) # 定义学习率
    # lr_scheduler = lr_decay()  # 也可以是自己定义的学习率下降方式,比如定义了一个列表

    '''load train data'''
    dataset = Dataset(length_in_seconds=3.2, num_clip_per_epoch=4000, random_start_point=True, train=True)
    data_loader = torch.utils.data.DataLoader(dataset, batch_size=3, shuffle=True, drop_last=True, num_workers=2)


    '''logs'''
    # logger = create_logger()  # 自己定义创建的log日志
    # summary_writer = SummaryWriter() # tensorboard


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
            optimizer.optimizer.zero_grad() #使用之前先清零 warm up
            # optimizer.zero_grad() # no warm up

            noisy_stft = torch.stft(noisy,1536,384,win_length=1536,window=WINDOW,center=True,return_complex=True)

            #-------------mapping based-----------------------
            enh_stft = model(noisy_stft)
            #------------------------------------


            #-------------mask based-----------------------
            # enh_mask = model_DPCRN(noisy_stft)            
            # enh_real = enh_mask[:,:,:,0] * noisy_stft[:,:,:,0] - enh_mask[:,:,:,1] * noisy_stft[:,:,:,1]
            # enh_real = enh_real.view(enh_real.shape[0], enh_real.shape[1], enh_real.shape[2], 1)
            # enh_imag = enh_mask[:,:,:,0] * noisy_stft[:,:,:,1] + enh_mask[:,:,:,1] * noisy_stft[:,:,:,0]
            # enh_imag = enh_imag.view(enh_imag.shape[0], enh_imag.shape[1], enh_imag.shape[2], 1)
            # enh_stft = torch.cat([enh_real, enh_imag], -1)
            #-------------------------------------------------

            # enh_s = torch.istft(enh_stft,1200,600,1200,window=WINDOW,center=True,onesided=True)
            enh_s = iSTFT_module_1_8(n_fft=1536, hop_length=384, win_length=1536,window=WINDOW,center = True,length = noisy.shape[-1])(enh_stft.permute([0, 3, 2, 1]).contiguous())

            stft_loss, snr_loss = Loss(enh_s, clean, train=True, idx = idx, epoch = epoch)  # 自己定义损失函数
            loss_overall = stft_loss
            loss_overall.backward() # loss反传，计算模型中各tensor的梯度
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=3)
            optimizer.step() #用在每一个mini batch中，只有用了optimizer.step()，模型才会更新
            train_loss.append(loss_overall.cpu().detach().numpy())
            # asnr_loss.append(snr_loss.cpu().detach().numpy())
            idx += 1
        train_loss = np.mean(train_loss) # 对各个mini batch的loss求平均
        # asnr_loss = np.mean(asnr_loss)
        # lr_scheduler.step() # 更新optimizer的学习率，一般以epoch为单位，即多少个epoch后换一次学习率
        '''eval'''
        # valid_loss = []
        # model.eval()  # 注意model的模式从train()变成了eval()
        # print('epoch %s--validating' %(epoch))
        # dataset.train = False
        # dataset.random_start_point = False
        # with torch.no_grad():
        #     for i, data in enumerate(tqdm(data_loader)):

        #         noisy, clean = data
        #         noisy = noisy.to(device)
        #         clean = clean.to(device)
        #         noisy_stft = torch.stft(noisy,1536,384,win_length=1536,window=WINDOW,center=True, return_complex=True)

        #         #-------------mapping based-----------------------
        #         enh_stft = model(noisy_stft)
        #         #------------------------------------


        #         #-------------mask based-----------------------
        #         # enh_mask = model_DPCRN(noisy_stft)            
        #         # enh_real = enh_mask[:,:,:,0] * noisy_stft[:,:,:,0] - enh_mask[:,:,:,1] * noisy_stft[:,:,:,1]
        #         # enh_real = enh_real.view(enh_real.shape[0], enh_real.shape[1], enh_real.shape[2], 1)
        #         # enh_imag = enh_mask[:,:,:,0] * noisy_stft[:,:,:,1] + enh_mask[:,:,:,1] * noisy_stft[:,:,:,0]
        #         # enh_imag = enh_imag.view(enh_imag.shape[0], enh_imag.shape[1], enh_imag.shape[2], 1)
        #         # enh_stft = torch.cat([enh_real, enh_imag], -1)
        #         #-------------------------------------------------

        #         # enh_s = torch.istft(enh_stft,1200,600,1200,window=WINDOW,center=True,onesided=True)
        #         enh_s = iSTFT_module_1_8(n_fft=1536, hop_length=384, win_length=1536,window=WINDOW,center = True,length = noisy.shape[-1])(enh_stft.permute([0, 3, 2, 1]).contiguous())

        #         stft_loss, snr_loss = Loss(enh_s, clean, train=True, idx = idx, epoch = epoch)  # 自己定义损失函数
        #         loss_overall = stft_loss
        #         valid_loss.append(loss_overall.cpu().detach().numpy())
        #         asnr_loss.append(snr_loss.cpu().detach().numpy())
        #     valid_loss = np.mean(valid_loss)
        #     asnr_loss = np.mean(asnr_loss)
        # print('train loss: %s, valid loss %s, snr loss: %s' %(train_loss, valid_loss, asnr_loss))
        # # print('current step:{}, current lr:{}'.format(optimizer_dpcrn._step, optimizer_dpcrn._rate))
        # # summary_writer.add_scalars('loss', {'train_loss': train_loss, 'valid_loss': valid_loss}, epoch) #写入tensorboard
        
        ##------------------Test on VCTK-DEMAND testset---------------------
        model.eval()
        pesqs = 0
        with torch.no_grad():
            for i in tqdm(range(len(noisy_list))):
                clean_s = sf.read(clean_list[i])[0]                  
                noisy_s = sf.read(noisy_list[i])[0].astype('float32')
                noisy_s = torch.from_numpy(noisy_s.reshape((1,len(noisy_s)))).to(device)
                noisy_stft = torch.stft(noisy_s,1536,384,win_length=1536,window=WINDOW,center=True ,return_complex=True)

                #----------------mapping based-----------------
                enh_stft = model(noisy_stft)
                #----------------------------------------------

                #---------------mask based---------------------
                # enh_mask = model_DPCRN(mha_out)            
                # enh_real = enh_mask[:,:,:,0] * mha_out[:,:,:,0] - enh_mask[:,:,:,1] * mha_out[:,:,:,1]
                # enh_real = enh_real.view(enh_real.shape[0], enh_real.shape[1], enh_real.shape[2], 1)
                # enh_imag = enh_mask[:,:,:,0] * mha_out[:,:,:,1] + enh_mask[:,:,:,1] * mha_out[:,:,:,0]
                # enh_imag = enh_imag.view(enh_imag.shape[0], enh_imag.shape[1], enh_imag.shape[2], 1)
                # enh_stft = torch.cat([enh_real, enh_imag], -1)
                #----------------------------------------------
                
                # enh_stft = noisy_stft
                enh_s = iSTFT_module_1_8(n_fft=1536, hop_length=384, win_length=1536,window=WINDOW,center = True,length = noisy_s.shape[-1])(enh_stft.permute([0, 3, 2, 1]).contiguous())

                enh_s = enh_s[0,:].cpu().detach().numpy()

                enh_s = librosa.resample(enh_s, 48000, 16000)

                l = min(len(clean_s), len(enh_s))
                clean_s = clean_s[:l]
                enh_s = enh_s[:l]
                pesqs = pesqs + pesq.pesq(16000,clean_s,enh_s,'wb')

        torch.save(
            {'epoch': epoch,
                'state_dict': model.state_dict(),
                # 'optimizer': optimizer_dpcrn.state_dict()}, # optimizer_dpcrn.optimizer.state_dict() 如果是warmup用这句
                'optimizer': optimizer.optimizer.state_dict()},
            '/data/hdd0/zhongshu.hou/g9_copy/MTFAA_full/chkpt_full_F_MSLSA_online_dns/model_epoch_%s_trainloss_%s_pesq_%s.pth' %(str(epoch), str(train_loss), str(pesqs/1000)))


if __name__ == '__main__':
    train(end_epoch=300)
