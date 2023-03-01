from typing import OrderedDict
from unicodedata import name
import torch
import torch.optim as optim
# from torch.utils.tensorboard import SummaryWriter
import numpy as np
from torch.optim.lr_scheduler import StepLR
from tqdm import tqdm
import soundfile as sf
from MTFAA_Net_full import MTFAA_Net as MTFAA
from MTFAA_Net_full_F_ASqbi import MTFAA_Net as MTFAA_ASqBi
from MTFAA_Net_full_local_atten import MTFAA_Net as MTFAA_LSA
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
torch.set_default_tensor_type(torch.FloatTensor)
from signal_processing import iSTFT_module_1_8
WINDOW = torch.sqrt(torch.hann_window(1536,device=device) + 1e-8)

import librosa
from collections import OrderedDict
import os

def main(args):
    noisy_dir = args.test_path
    noisy_list = librosa.util.find_files(noisy_dir, ext='wav')


    '''model'''
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    '''model'''
    if args.model == 'MTFAA':
        model = MTFAA()
    elif args.model == 'MTFAA_ASqBi'
        model = MTFAA_ASqBi()
    elif args.model == 'MTFAA_LSA':
        model = MTFAA_LSA()
    model = model.to(device)
    checkpoint = torch.load(args.chkpt_path,map_location=device)
    model.load_state_dict(checkpoint['state_dict'])
    model.eval()
    with torch.no_grad():
        for i in tqdm(range(len(noisy_list))):
                 
            noisy_s = sf.read(noisy_list[i])[0].astype('float32')
            noisy_s = torch.from_numpy(noisy_s.reshape((1,len(noisy_s)))).to(device)
            noisy_stft = torch.stft(noisy_s,1536,384,win_length=1536,window=WINDOW,center=True ,return_complex=True)
            enh_stft = model(noisy_stft)
            enh_s = iSTFT_module_1_8(n_fft=1536, hop_length=384, win_length=1536,window=WINDOW,center = True,length = noisy_s.shape[-1])(enh_stft.permute([0, 3, 2, 1]).contiguous())

            enh_s = enh_s[0,:].cpu().detach().numpy()
            sf.write(args.save_path+'/'+noisy_list[i].split('/')[-1], enh_s, 48000)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-m', "--model", default='MTFAA', 
                        help='Choose the model you wanna train: MTFAA, MTFAA_LSA or MTFAA_ASqBi')
    parser.add_argument('-c', "--chkpt_path", default=None, help='path to checkpoint to load')
    parser.add_argument('-t', "--test_path", default=None, 
                        help='path to folder containing noisy audios to enhance')
    parser.add_argument('-s', "--save_path", default=None, 
                        help='path to folder saving the enhanced clips')
    parser.add_argument('-d', "--device", default='cuda:0', 
                        help='Device used for inference')
    
    args = parser.parse_args()

    main(args)
