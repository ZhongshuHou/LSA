# Local Spectral Attention for Full-band Speech Enhancement  
![Visualization of local spectral attention](https://user-images.githubusercontent.com/103247057/222057044-e8cdc198-23c2-4fc7-b548-0ec98da8377d.png)

# Contents
- [Repository description](#repository-description)
- [Rquirements](#rquirements)
- [Network training](#network-training)
	- [Data preparation](#data-preparation)
	- [Start training](#start-training)
	- [Inference](#inference)
- [Ablation study and experiment results](#ablation-study-and-experiment-results)
	- [LSA on MTFAA and DPARN](#lsa-on-mtfaa-and-dparn)

## Repository description
This repository conduct ablation studies on local attention (a.k.a band attention) applied in full-band spectrum, namely local spectral attention (LSA). Two full-band speech enhancement (SE) models with spectral attention replace the conventional attention (a global manner) with LSA that only looks at adjacent bands at a certain frequency (a local manner). One model is our previous work called DPARN, whose source code can be found in https://github.com/Qinwen-Hu/dparn.   
The other model is the Multi-Scale Temporal Frequency with Axial Attention (MTFAA) network, which ranked 1st in the DNS-4 challenge for full-band SE, and its detailed description can be found in paper https://ieeexplore.ieee.org/document/9746610. Here we release an unofficial pytorch implementation of MTFAA as well as its modification.  

## Rquirements
soundfile: 0.10.3  
librosa:   0.8.1  
torch:     3.7.10  
numpy:     1.20.3  
scipy:     1.7.2  
pandas:    1.3.4  
tqdm:      4.62.3  

## Network training
### Data preparation
Split your speech and noise audios into ***10 seconds segments*** and generate the .csv files to manage your data. Prepare your RIR audios of .wav format in one folder. Edit the .csv path in [Dataloader.py](https://github.com/ZhongshuHou/LSA/blob/main/Dataloader.py):  
```TRAIN_CLEAN_CSV = './train_clean_data.csv'  
   TRAIN_NOISE_CSV = './train_noise_data.csv'  
   VALID_CLEAN_CSV = './valid_clean_data.csv'  
   VALID_NOISE_CSV = './valid_noise_data.csv'  
   RIR_DIR = 'direction to RIR .wav audios'
```
where the .csv files for clean speech are organized as  

|file_dir|snr|
| ---------- | :-----------:  | 
| ./clean_0001.wav | 4 |
| ./clean_0002.wav | -1 |
| ./clean_0003.wav | 0 |
| ... | ... |

and the .csv files for noise are organized as  

|file_dir|
| ---------- |
| ./noise_0001.wav |
| ./noise_0002.wav |
| ./noise_0003.wav |
| ... |

the 'file_dir' and 'snr' denote the absolute direction to audios and signal-to-noise ratio(SNR) respectively.

### Start training
After environment and data preparation, start to train the model by command:  
```
python Network_Training_MTFAA_full.py -m model_to_train(including MTFAA, MTFAA_LSA or MTFAA_ASqBi) -c Dir_to_save_the_checkpoint_files -e Epochs_for_training(default is 300) -d Device_used_for_training(cuda:0)
```
### Inference
Enhance noisy audios by command:  
```
python Infer.py -m model_to_train(including MTFAA, MTFAA_LSA or MTFAA_ASqBi) -c path_to_load_the_checkpoint_files -t path_to_folder_containing_noisy_audios -s path_to_folder_saving_the_enhanced_clips -d Device_used_for_training(cuda:0)
```
## Ablation study and experiment results
We demonstrate the effectiveness of our proposed method on the full-band dataset of the 4th DNS challenge. The total training dataset contains around 1000 hours of speech and 220 hours of noise. Room impulse responses are convolved with clean speech to generate simulated reverberant speech, which is preserved as training target. In the training stage, reverberant utterances are mixed with noise recordings with SNR ranging from -5 dB to 5 dB at1 dB intervals. For the test set, 800 clips of reverberant utterances are mixed with unseen noise types with SNR ranging from -5 dB to 15 dB. Each test clip is 5 seconds long. All utterances are sampled at 48 kHz in our experiments. We also conduct experiments on well-known VCTK-DEMAND dataset for comprehensive validation.

### LSA on MTFAA and DPARN
The visualization of LSA mechanism can be seen in the figure below:  
<!-- ![LSA](https://user-images.githubusercontent.com/103247057/222067488-191bf69f-238a-4616-96dd-582946f6473c.png) -->
<!-- <img src="https://user-images.githubusercontent.com/103247057/222067488-191bf69f-238a-4616-96dd-582946f6473c.png" width="600" height="300" /> -->
<div align=center><img src="https://user-images.githubusercontent.com/103247057/222067488-191bf69f-238a-4616-96dd-582946f6473c.png" width="600" height="250" /></div>  

The unofficial Pytorch implementation of MTFAA and its LSA-based model can be seen in [MTFAA_Net_full.py](https://github.com/ZhongshuHou/LSA/blob/main/MTFAA_Net_full.py) and [MTFAA_Net_full_local_atten.py](https://github.com/ZhongshuHou/LSA/blob/main/MTFAA_Net_full_local_atten.py) respectively. As for DPARN, readers may attend to https://github.com/Qinwen-Hu/dparn.  
Firstly, we conduct experiments on different setting of <sub>***N_l***</sub>

The training process can be seen in figures below, where both LSA-based models achieve better convergence compared with the original models. 
<!-- ![trainloss_dparn](https://user-images.githubusercontent.com/103247057/222072535-ab00598c-448d-47ff-9cf3-77e7da8d302c.jpeg) -->
<!-- ![Validation](https://user-images.githubusercontent.com/103247057/222072910-c3c7730d-b7be-45ca-9591-5706c395f1ad.jpeg) -->
<div align=center><img src="https://user-images.githubusercontent.com/103247057/222072535-ab00598c-448d-47ff-9cf3-77e7da8d302c.jpeg" width="320" height="240" /><img src="https://user-images.githubusercontent.com/103247057/222072910-c3c7730d-b7be-45ca-9591-5706c395f1ad.jpeg" width="320" height="240" /></div>
<!-- ![trainloss_mtfaa](https://user-images.githubusercontent.com/103247057/222074817-58680383-6d61-42c6-9f94-eb4c58ecefab.jpeg)
![Validation_mtfaa](https://user-images.githubusercontent.com/103247057/222074828-44bc05e4-c05e-40bc-b4b5-060cdb7030a7.jpeg) -->
<div align=center><img src="https://user-images.githubusercontent.com/103247057/222074817-58680383-6d61-42c6-9f94-eb4c58ecefab.jpeg" width="320" height="240" /><img src="https://user-images.githubusercontent.com/103247057/222074828-44bc05e4-c05e-40bc-b4b5-060cdb7030a7.jpeg" width="320" height="240" /></div>

