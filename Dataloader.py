from random import random
import soundfile as sf
import librosa
import torch
import os
import numpy as np
from scipy import signal
import matplotlib.pyplot as plt
import pandas as pd
import random

TRAIN_CLEAN_CSV = './train_clean_data.csv'
TRAIN_NOISE_CSV = './train_noise_data.csv'
VALID_CLEAN_CSV = './valid_clean_data.csv'
VALID_NOISE_CSV = './valid_noise_data.csv'
RIR_DIR = 'direction to RIR .wav audios'


T = int(500 * 48000 / 1000) 
t = np.arange(48000)
h = np.exp(-6 * np.log(10) * t / T)

FIR_LOW = []
for cut_freq in [4,8,16,24,32]:
    fir = signal.firwin(128, cut_freq /48.0)
    FIR_LOW.append(fir)




def add_pyreverb(clean_speech, rir):
    max_index = np.argmax(np.abs(rir))
    rir = rir[max_index:]
    reverb_speech = signal.fftconvolve(clean_speech, rir, mode="full")
    
    # make reverb_speech same length as clean_speech
    reverb_speech = reverb_speech[: clean_speech.shape[0]]

    return reverb_speech

def mk_mixture(s1,s2,snr,eps = 1e-8):
    
    # norm_sig1 = s1 / np.sqrt(np.sum(s1 ** 2) + eps) 
    norm_sig1 = s1
    # norm_sig2 = s2 / np.sqrt(np.sum(s2 ** 2) + eps)
    norm_sig2 = s2 * np.math.sqrt(np.sum(s1 ** 2) + eps) / np.math.sqrt(np.sum(s2 ** 2) + eps)
    alpha = 10**(-snr/20)
    freq_prob = np.random.rand()
    sins = np.zeros(len(s1))
    if freq_prob < 0.1:
        freq = np.random.randint(50,20000)
        s_sin = np.sin(2*np.pi*freq*np.arange(len(s1))/48000)
        sins = (0.5*np.random.rand() + 0.5) * alpha * s_sin * np.math.sqrt(np.sum(s1 ** 2) + eps) / np.math.sqrt(np.sum(s_sin ** 2) + eps)
    mix = norm_sig1 + alpha * norm_sig2 + sins
    M = max(np.max(abs(mix)),np.max(abs(norm_sig1)),np.max(abs(alpha*norm_sig2))) + eps
    amp = 0.99*np.random.rand()+0.01
    mix = amp * mix / M
    norm_sig1 = amp * norm_sig1 / M

    return norm_sig1,mix

class Dataset(torch.utils.data.Dataset):
    def __init__(self, fs=48000, length_in_seconds=8, num_clip_per_epoch=4000, random_start_point=False, train=True):
        self.train_clean_list = pd.read_csv(TRAIN_CLEAN_CSV)['file_dir'].to_list()
        self.train_noise_list = pd.read_csv(TRAIN_NOISE_CSV)['file_dir'].to_list()
        self.valid_clean_list = pd.read_csv(VALID_CLEAN_CSV)['file_dir'].to_list()
        self.valid_noise_list = pd.read_csv(VALID_NOISE_CSV)['file_dir'].to_list()
        self.train_snr_list = pd.read_csv(TRAIN_CLEAN_CSV)['snr'].to_list()
        self.valid_snr_list = pd.read_csv(VALID_CLEAN_CSV)['snr'].to_list()
        self.L = int(length_in_seconds * fs)
        self.random_start_point = random_start_point
        self.fs = fs
        self.length_in_seconds = length_in_seconds
        self.train = train
        self.rir_list = librosa.util.find_files(RIR_DIR,ext = 'wav')
        self.num_clip_per_epoch = num_clip_per_epoch

        print('%s audios for training, %s for validation' %(len(self.train_clean_list), len(self.valid_clean_list)))

    def sample(self):
        self.train_clean_data = random.sample(self.train_clean_list, self.num_clip_per_epoch)
        self.train_snr_data = random.sample(self.train_snr_list, self.num_clip_per_epoch)
        self.train_noise_data = random.sample(self.train_noise_list, self.num_clip_per_epoch)
        
        
    
    def __getitem__(self, idx):
        if self.train:
            clean_list = self.train_clean_data
            noise_list = self.train_noise_data
            snr_list = self.train_snr_data
        else:
            clean_list = self.valid_clean_list
            noise_list = self.valid_noise_list 
            snr_list = self.valid_snr_list           
        # reverb_rate = 0 #np.random.rand()
        # clip_rate = 1.0 #np.random.rand()
        # clean_idx = np.random.randint(0,len(clean_list))
        reverb_rate = np.random.rand()
        lowpass_rate = np.random.rand()
        clip_rate = np.random.rand()

        if self.train:
            Begin_S = int(np.random.uniform(0,10 - self.length_in_seconds)) * self.fs
            clean, sr_s = sf.read(clean_list[idx], dtype='float32',start= Begin_S,stop = Begin_S + self.L)
            noise, sr_n = sf.read(noise_list[idx], dtype='float32',start= Begin_S,stop = Begin_S + self.L)

        else:
            clean, sr_s = sf.read(clean_list[idx], dtype='float32',start= 0,stop = self.L) 
            noise, sr_n = sf.read(noise_list[idx], dtype='float32',start= 0,stop = self.L)

            
        if reverb_rate < 0.1: # 模拟混响信号
            rir_idx = np.random.randint(0,len(self.rir_list) - 1)
            rir_f = self.rir_list[rir_idx]
            rir_s = sf.read(rir_f,dtype = 'float32')[0]
            if len(rir_s.shape)>1:
                rir_s = rir_s[:,0]

            rir_s = rir_s[:min(len(h),len(rir_s))] * h[:min(len(h),len(rir_s))] 
            reverb = add_pyreverb(clean, rir_s)
        else:
            reverb = clean
      

        if lowpass_rate < 0.1:           
            id = np.random.randint(0,len(FIR_LOW))
            fir = FIR_LOW[id]
            reverb = np.convolve(reverb, fir)[127:127+len(reverb)]
            noise = np.convolve(noise, fir)[127:127+len(noise)]
    
            
        reverb_s,noisy_s = mk_mixture(reverb,noise,snr_list[idx],eps = 1e-8)

        if clip_rate < 0.05:
            # clip_mag = 0.5 + 0.5 * np.random.rand() #[0.5, 1]
            # mag = np.random.rand()
            noisy_s = noisy_s /np.max(np.abs(noisy_s) + 1e-12)
            noisy_s = noisy_s * np.random.uniform(1.2,2)
            noisy_s[noisy_s > 1.0] = 1.0
            noisy_s[noisy_s < -1.0] = -1.0
            reverb_s = reverb_s /np.max(np.abs(reverb_s) + 1e-12)
        
        return noisy_s.astype(np.float32), reverb_s.astype(np.float32)

    def __len__(self):
        if self.train:
            return self.num_clip_per_epoch
        else:
            return len(self.valid_noise_list)

def collate_fn(batch):

    noisy, clean = zip(*batch)
    noisy = np.asarray(noisy)
    clean = np.asarray(clean)
    return noisy, clean 

