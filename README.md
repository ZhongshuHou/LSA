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
This repository conduct ablation studies on local attention (a.k.a band attention) applied in full-band spectrum, namely local spectral attention (LSA). Two full-band speech enhancement (SE) models with spectral attention replace the conventional attention (a global manner) with LSA that only looks at adjacent bands at a certain frequency (a local manner). One model is DPARN, whose source code can be found in https://github.com/Qinwen-Hu/dparn.   
The other model is the Multi-Scale Temporal Frequency with Axial Attention (MTFAA) network, which ranked 1st in the DNS-4 challenge for full-band SE, and its detailed description can be found in paper https://ieeexplore.ieee.org/document/9746610. Here we release an unofficial pytorch implementation of MTFAA as well as its modification. This work have been submitted to Interspeech2023.

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
Firstly, we conduct experiments on different setting of ***N<sub>l</sub>*** based on the VCTK-DEMAND dataset and the results can be seen in table below:  
<body lang=ZH-CN style='tab-interval:21.0pt;text-justify-trim:punctuation'>

<div class=WordSection1 style='layout-grid:15.6pt'>

<div align=center>

<table class=MsoNormalTable border=1 cellspacing=0 cellpadding=0 width=246
 style='width:184.3pt;border-collapse:collapse;border:none;mso-border-alt:solid windowtext .5pt;
 mso-yfti-tbllook:1184;mso-padding-alt:0cm 1.4pt 0cm 1.4pt;mso-border-insideh:
 .5pt solid windowtext;mso-border-insidev:.5pt solid windowtext'>
 <tr style='mso-yfti-irow:0;mso-yfti-firstrow:yes;height:11.35pt'>
  <td width=64 colspan=2 valign=bottom style='width:47.95pt;border:solid windowtext 1.0pt;
  mso-border-alt:solid windowtext .5pt;padding:0cm 1.4pt 0cm 1.4pt;height:11.35pt'>
  <p class=MsoNormal align=center style='text-align:center;mso-pagination:widow-orphan'><span
  lang=EN-US style='font-size:6.5pt;mso-bidi-font-size:7.5pt;font-family:"Times New Roman",serif;
  mso-fareast-font-family:等线;color:black;mso-bidi-font-weight:bold'>Config.<o:p></o:p></span></p>
  </td>
  <td width=106 nowrap colspan=4 valign=bottom style='width:79.65pt;border:
  solid windowtext 1.0pt;border-left:none;mso-border-left-alt:solid windowtext .5pt;
  mso-border-alt:solid windowtext .5pt;padding:0cm 1.4pt 0cm 1.4pt;height:11.35pt'>
  <p class=MsoNormal align=center style='text-align:center;mso-pagination:widow-orphan'><span
  lang=EN-US style='font-size:6.5pt;mso-bidi-font-size:7.5pt;font-family:"Times New Roman",serif;
  mso-fareast-font-family:等线;color:black'>Wideband Metrics<o:p></o:p></span></p>
  </td>
  <td width=76 nowrap colspan=2 valign=bottom style='width:2.0cm;border:solid windowtext 1.0pt;
  border-left:none;mso-border-left-alt:solid windowtext .5pt;mso-border-alt:
  solid windowtext .5pt;padding:0cm 1.4pt 0cm 1.4pt;height:11.35pt'>
  <p class=MsoNormal align=center style='text-align:center;mso-pagination:widow-orphan'><span
  lang=EN-US style='font-size:6.5pt;mso-bidi-font-size:7.5pt;font-family:"Times New Roman",serif;
  mso-fareast-font-family:等线;color:black'>Full-band Metrics<o:p></o:p></span></p>
  </td>
 </tr>
 <tr style='mso-yfti-irow:1;height:11.35pt'>
  <td width=28 valign=bottom style='width:21.1pt;border:solid windowtext 1.0pt;
  border-top:none;mso-border-top-alt:solid windowtext .5pt;mso-border-alt:solid windowtext .5pt;
  padding:0cm 1.4pt 0cm 1.4pt;height:11.35pt'>
  <p class=MsoNormal align=center style='text-align:center;mso-pagination:widow-orphan'><span
  lang=EN-US style='font-size:5.5pt;mso-bidi-font-size:6.5pt;font-family:"Times New Roman",serif;
  mso-fareast-font-family:等线;color:black;mso-bidi-font-weight:bold'>Model<o:p></o:p></span></p>
  </td>
  <td width=36 valign=bottom style='width:26.85pt;border-top:none;border-left:
  none;border-bottom:solid windowtext 1.0pt;border-right:solid windowtext 1.0pt;
  mso-border-top-alt:solid windowtext .5pt;mso-border-left-alt:solid windowtext .5pt;
  mso-border-alt:solid windowtext .5pt;padding:0cm 1.4pt 0cm 1.4pt;height:11.35pt'>
  <p class=MsoNormal align=center style='text-align:center;mso-pagination:widow-orphan'><span
  class=SpellE><i style='mso-bidi-font-style:normal'><span lang=EN-US
  style='font-size:7.5pt;mso-bidi-font-size:6.5pt;font-family:"Times New Roman",serif;
  mso-fareast-font-family:等线;color:black;mso-bidi-font-weight:bold'>N<sub>l</sub></span></i></span><i
  style='mso-bidi-font-style:normal'><sub><span lang=EN-US style='font-size:
  5.5pt;mso-bidi-font-size:6.5pt;font-family:"Times New Roman",serif;
  mso-fareast-font-family:等线;color:black;mso-bidi-font-weight:bold'><o:p></o:p></span></sub></i></p>
  </td>
  <td width=27 nowrap valign=bottom style='width:19.9pt;border-top:none;
  border-left:none;border-bottom:solid windowtext 1.0pt;border-right:solid windowtext 1.0pt;
  mso-border-top-alt:solid windowtext .5pt;mso-border-left-alt:solid windowtext .5pt;
  mso-border-alt:solid windowtext .5pt;padding:0cm 1.4pt 0cm 1.4pt;height:11.35pt'>
  <p class=MsoNormal align=center style='text-align:center;mso-pagination:widow-orphan'><span
  lang=EN-US style='font-size:5.5pt;mso-bidi-font-size:6.5pt;font-family:"Times New Roman",serif;
  mso-fareast-font-family:等线;color:black'>PESQ<o:p></o:p></span></p>
  </td>
  <td width=27 valign=bottom style='width:19.9pt;border-top:none;border-left:
  none;border-bottom:solid windowtext 1.0pt;border-right:solid windowtext 1.0pt;
  mso-border-top-alt:solid windowtext .5pt;mso-border-left-alt:solid windowtext .5pt;
  mso-border-alt:solid windowtext .5pt;padding:0cm 1.4pt 0cm 1.4pt;height:11.35pt'>
  <p class=MsoNormal align=center style='text-align:center;mso-pagination:widow-orphan'><span
  lang=EN-US style='font-size:5.5pt;mso-bidi-font-size:6.5pt;font-family:"Times New Roman",serif;
  mso-fareast-font-family:等线;color:black'>CSIG<o:p></o:p></span></p>
  </td>
  <td width=27 valign=bottom style='width:19.9pt;border-top:none;border-left:
  none;border-bottom:solid windowtext 1.0pt;border-right:solid windowtext 1.0pt;
  mso-border-top-alt:solid windowtext .5pt;mso-border-left-alt:solid windowtext .5pt;
  mso-border-alt:solid windowtext .5pt;padding:0cm 1.4pt 0cm 1.4pt;height:11.35pt'>
  <p class=MsoNormal align=center style='text-align:center;mso-pagination:widow-orphan'><span
  lang=EN-US style='font-size:5.5pt;mso-bidi-font-size:6.5pt;font-family:"Times New Roman",serif;
  mso-fareast-font-family:等线;color:black'>CBAK<o:p></o:p></span></p>
  </td>
  <td width=27 valign=bottom style='width:19.95pt;border-top:none;border-left:
  none;border-bottom:solid windowtext 1.0pt;border-right:solid windowtext 1.0pt;
  mso-border-top-alt:solid windowtext .5pt;mso-border-left-alt:solid windowtext .5pt;
  mso-border-alt:solid windowtext .5pt;padding:0cm 1.4pt 0cm 1.4pt;height:11.35pt'>
  <p class=MsoNormal align=center style='text-align:center;mso-pagination:widow-orphan'><span
  lang=EN-US style='font-size:5.5pt;mso-bidi-font-size:6.5pt;font-family:"Times New Roman",serif;
  mso-fareast-font-family:等线;color:black'>COVL<o:p></o:p></span></p>
  </td>
  <td width=38 nowrap valign=bottom style='width:1.0cm;border-top:none;
  border-left:none;border-bottom:solid windowtext 1.0pt;border-right:solid windowtext 1.0pt;
  mso-border-top-alt:solid windowtext .5pt;mso-border-left-alt:solid windowtext .5pt;
  mso-border-alt:solid windowtext .5pt;padding:0cm 1.4pt 0cm 1.4pt;height:11.35pt'>
  <p class=MsoNormal align=center style='text-align:center;mso-pagination:widow-orphan'><span
  class=GramE><span lang=EN-US style='font-size:5.5pt;mso-bidi-font-size:6.5pt;
  font-family:"Times New Roman",serif;mso-fareast-font-family:等线;color:black'>STOI(</span></span><span
  lang=EN-US style='font-size:5.5pt;mso-bidi-font-size:6.5pt;font-family:"Times New Roman",serif;
  mso-fareast-font-family:等线;color:black'>%)<o:p></o:p></span></p>
  </td>
  <td width=38 nowrap valign=bottom style='width:1.0cm;border-top:none;
  border-left:none;border-bottom:solid windowtext 1.0pt;border-right:solid windowtext 1.0pt;
  mso-border-top-alt:solid windowtext .5pt;mso-border-left-alt:solid windowtext .5pt;
  mso-border-alt:solid windowtext .5pt;padding:0cm 1.4pt 0cm 1.4pt;height:11.35pt'>
  <p class=MsoNormal align=center style='text-align:center;mso-pagination:widow-orphan'><span
  lang=EN-US style='font-size:5.5pt;mso-bidi-font-size:6.5pt;font-family:"Times New Roman",serif;
  mso-fareast-font-family:等线;color:black'>SiSDR(dB)<o:p></o:p></span></p>
  </td>
 </tr>
 <tr style='mso-yfti-irow:2;height:11.35pt'>
  <td width=28 rowspan=3 style='width:21.1pt;border:solid windowtext 1.0pt;
  border-top:none;mso-border-top-alt:solid windowtext .5pt;mso-border-alt:solid windowtext .5pt;
  padding:0cm 1.4pt 0cm 1.4pt;height:11.35pt'>
  <p class=MsoNormal align=center style='text-align:center;mso-pagination:widow-orphan'><span
  lang=EN-US style='font-size:5.0pt;mso-bidi-font-size:5.5pt;font-family:"Times New Roman",serif;
  mso-fareast-font-family:等线;color:black'>MTFAA<o:p></o:p></span></p>
  </td>
  <td width=36 style='width:26.85pt;border-top:none;border-left:none;
  border-bottom:solid windowtext 1.0pt;border-right:solid windowtext 1.0pt;
  mso-border-top-alt:solid windowtext .5pt;mso-border-left-alt:solid windowtext .5pt;
  mso-border-alt:solid windowtext .5pt;padding:0cm 1.4pt 0cm 1.4pt;height:11.35pt'>
  <p class=MsoNormal align=center style='text-align:center;mso-pagination:widow-orphan'><i
  style='mso-bidi-font-style:normal'><span lang=EN-US style='font-size:6.5pt;
  mso-bidi-font-size:7.5pt;font-family:"Times New Roman",serif;mso-fareast-font-family:
  等线;color:black'>F’/2<o:p></o:p></span></i></p>
  </td>
  <td width=27 nowrap style='width:19.9pt;border-top:none;border-left:none;
  border-bottom:solid windowtext 1.0pt;border-right:solid windowtext 1.0pt;
  mso-border-top-alt:solid windowtext .5pt;mso-border-left-alt:solid windowtext .5pt;
  mso-border-alt:solid windowtext .5pt;padding:0cm 1.4pt 0cm 1.4pt;height:11.35pt'>
  <p class=MsoNormal align=center style='text-align:center;mso-pagination:widow-orphan'><span
  lang=EN-US style='font-size:7.5pt;mso-bidi-font-size:8.0pt;font-family:"Times New Roman",serif;
  mso-fareast-font-family:等线;color:black'>3.16<o:p></o:p></span></p>
  </td>
  <td width=27 style='width:19.9pt;border-top:none;border-left:none;border-bottom:
  solid windowtext 1.0pt;border-right:solid windowtext 1.0pt;mso-border-top-alt:
  solid windowtext .5pt;mso-border-left-alt:solid windowtext .5pt;mso-border-alt:
  solid windowtext .5pt;padding:0cm 1.4pt 0cm 1.4pt;height:11.35pt'>
  <p class=MsoNormal align=center style='text-align:center;mso-pagination:widow-orphan'><span
  lang=EN-US style='font-size:7.5pt;mso-bidi-font-size:8.0pt;font-family:"Times New Roman",serif;
  mso-fareast-font-family:等线;color:black'>4.34<o:p></o:p></span></p>
  </td>
  <td width=27 style='width:19.9pt;border-top:none;border-left:none;border-bottom:
  solid windowtext 1.0pt;border-right:solid windowtext 1.0pt;mso-border-top-alt:
  solid windowtext .5pt;mso-border-left-alt:solid windowtext .5pt;mso-border-alt:
  solid windowtext .5pt;padding:0cm 1.4pt 0cm 1.4pt;height:11.35pt'>
  <p class=MsoNormal align=center style='text-align:center;mso-pagination:widow-orphan'><b><span
  lang=EN-US style='font-size:7.5pt;mso-bidi-font-size:8.0pt;font-family:"Times New Roman",serif;
  mso-fareast-font-family:等线;color:black'>3.63<o:p></o:p></span></b></p>
  </td>
  <td width=27 style='width:19.95pt;border-top:none;border-left:none;
  border-bottom:solid windowtext 1.0pt;border-right:solid windowtext 1.0pt;
  mso-border-top-alt:solid windowtext .5pt;mso-border-left-alt:solid windowtext .5pt;
  mso-border-alt:solid windowtext .5pt;padding:0cm 1.4pt 0cm 1.4pt;height:11.35pt'>
  <p class=MsoNormal align=center style='text-align:center;mso-pagination:widow-orphan'><span
  lang=EN-US style='font-size:7.5pt;mso-bidi-font-size:8.0pt;font-family:"Times New Roman",serif;
  mso-fareast-font-family:等线;color:black'>3.77<o:p></o:p></span></p>
  </td>
  <td width=38 nowrap style='width:1.0cm;border-top:none;border-left:none;
  border-bottom:solid windowtext 1.0pt;border-right:solid windowtext 1.0pt;
  mso-border-top-alt:solid windowtext .5pt;mso-border-left-alt:solid windowtext .5pt;
  mso-border-alt:solid windowtext .5pt;padding:0cm 1.4pt 0cm 1.4pt;height:11.35pt'>
  <p class=MsoNormal align=center style='text-align:center;mso-pagination:widow-orphan'><span
  lang=EN-US style='font-size:7.5pt;mso-bidi-font-size:8.0pt;font-family:"Times New Roman",serif;
  mso-fareast-font-family:等线;color:black'>94.7<o:p></o:p></span></p>
  </td>
  <td width=38 nowrap style='width:1.0cm;border-top:none;border-left:none;
  border-bottom:solid windowtext 1.0pt;border-right:solid windowtext 1.0pt;
  mso-border-top-alt:solid windowtext .5pt;mso-border-left-alt:solid windowtext .5pt;
  mso-border-alt:solid windowtext .5pt;padding:0cm 1.4pt 0cm 1.4pt;height:11.35pt'>
  <p class=MsoNormal align=center style='text-align:center;mso-pagination:widow-orphan'><span
  lang=EN-US style='font-size:7.5pt;mso-bidi-font-size:8.0pt;font-family:"Times New Roman",serif;
  mso-fareast-font-family:等线;color:black'>18.5<o:p></o:p></span></p>
  </td>
 </tr>
 <tr style='mso-yfti-irow:3;height:11.35pt'>
  <td width=36 style='width:26.85pt;border-top:none;border-left:none;
  border-bottom:solid windowtext 1.0pt;border-right:solid windowtext 1.0pt;
  mso-border-top-alt:solid windowtext .5pt;mso-border-left-alt:solid windowtext .5pt;
  mso-border-alt:solid windowtext .5pt;padding:0cm 1.4pt 0cm 1.4pt;height:11.35pt'>
  <p class=MsoNormal align=center style='text-align:center;mso-pagination:widow-orphan'><i
  style='mso-bidi-font-style:normal'><span lang=EN-US style='font-size:6.5pt;
  mso-bidi-font-size:7.5pt;font-family:"Times New Roman",serif;mso-fareast-font-family:
  等线;color:black'>F’/4</span></i><span lang=EN-US style='font-size:6.5pt;
  mso-bidi-font-size:7.5pt;font-family:"Times New Roman",serif;mso-fareast-font-family:
  等线'><o:p></o:p></span></p>
  </td>
  <td width=27 nowrap style='width:19.9pt;border-top:none;border-left:none;
  border-bottom:solid windowtext 1.0pt;border-right:solid windowtext 1.0pt;
  mso-border-top-alt:solid windowtext .5pt;mso-border-left-alt:solid windowtext .5pt;
  mso-border-alt:solid windowtext .5pt;padding:0cm 1.4pt 0cm 1.4pt;height:11.35pt'>
  <p class=MsoNormal align=center style='text-align:center;mso-pagination:widow-orphan'><span
  lang=EN-US style='font-size:7.5pt;mso-bidi-font-size:8.0pt;font-family:"Times New Roman",serif;
  mso-fareast-font-family:等线;color:black'>3.15<o:p></o:p></span></p>
  </td>
  <td width=27 style='width:19.9pt;border-top:none;border-left:none;border-bottom:
  solid windowtext 1.0pt;border-right:solid windowtext 1.0pt;mso-border-top-alt:
  solid windowtext .5pt;mso-border-left-alt:solid windowtext .5pt;mso-border-alt:
  solid windowtext .5pt;padding:0cm 1.4pt 0cm 1.4pt;height:11.35pt'>
  <p class=MsoNormal align=center style='text-align:center;mso-pagination:widow-orphan'><span
  lang=EN-US style='font-size:7.5pt;mso-bidi-font-size:8.0pt;font-family:"Times New Roman",serif;
  mso-fareast-font-family:等线;color:black'>4.32<o:p></o:p></span></p>
  </td>
  <td width=27 style='width:19.9pt;border-top:none;border-left:none;border-bottom:
  solid windowtext 1.0pt;border-right:solid windowtext 1.0pt;mso-border-top-alt:
  solid windowtext .5pt;mso-border-left-alt:solid windowtext .5pt;mso-border-alt:
  solid windowtext .5pt;padding:0cm 1.4pt 0cm 1.4pt;height:11.35pt'>
  <p class=MsoNormal align=center style='text-align:center;mso-pagination:widow-orphan'><span
  lang=EN-US style='font-size:7.5pt;mso-bidi-font-size:8.0pt;font-family:"Times New Roman",serif;
  mso-fareast-font-family:等线;color:black'>3.58<o:p></o:p></span></p>
  </td>
  <td width=27 style='width:19.95pt;border-top:none;border-left:none;
  border-bottom:solid windowtext 1.0pt;border-right:solid windowtext 1.0pt;
  mso-border-top-alt:solid windowtext .5pt;mso-border-left-alt:solid windowtext .5pt;
  mso-border-alt:solid windowtext .5pt;padding:0cm 1.4pt 0cm 1.4pt;height:11.35pt'>
  <p class=MsoNormal align=center style='text-align:center;mso-pagination:widow-orphan'><span
  lang=EN-US style='font-size:7.5pt;mso-bidi-font-size:8.0pt;font-family:"Times New Roman",serif;
  mso-fareast-font-family:等线;color:black'>3.76<o:p></o:p></span></p>
  </td>
  <td width=38 nowrap style='width:1.0cm;border-top:none;border-left:none;
  border-bottom:solid windowtext 1.0pt;border-right:solid windowtext 1.0pt;
  mso-border-top-alt:solid windowtext .5pt;mso-border-left-alt:solid windowtext .5pt;
  mso-border-alt:solid windowtext .5pt;padding:0cm 1.4pt 0cm 1.4pt;height:11.35pt'>
  <p class=MsoNormal align=center style='text-align:center;mso-pagination:widow-orphan'><span
  lang=EN-US style='font-size:7.5pt;mso-bidi-font-size:8.0pt;font-family:"Times New Roman",serif;
  mso-fareast-font-family:等线;color:black'>94.6<o:p></o:p></span></p>
  </td>
  <td width=38 nowrap style='width:1.0cm;border-top:none;border-left:none;
  border-bottom:solid windowtext 1.0pt;border-right:solid windowtext 1.0pt;
  mso-border-top-alt:solid windowtext .5pt;mso-border-left-alt:solid windowtext .5pt;
  mso-border-alt:solid windowtext .5pt;padding:0cm 1.4pt 0cm 1.4pt;height:11.35pt'>
  <p class=MsoNormal align=center style='text-align:center;mso-pagination:widow-orphan'><span
  lang=EN-US style='font-size:7.5pt;mso-bidi-font-size:8.0pt;font-family:"Times New Roman",serif;
  mso-fareast-font-family:等线;color:black'>18.1<o:p></o:p></span></p>
  </td>
 </tr>
 <tr style='mso-yfti-irow:4;height:11.35pt'>
  <td width=36 style='width:26.85pt;border-top:none;border-left:none;
  border-bottom:solid windowtext 1.0pt;border-right:solid windowtext 1.0pt;
  mso-border-top-alt:solid windowtext .5pt;mso-border-left-alt:solid windowtext .5pt;
  mso-border-alt:solid windowtext .5pt;padding:0cm 1.4pt 0cm 1.4pt;height:11.35pt'>
  <p class=MsoNormal align=center style='text-align:center;mso-pagination:widow-orphan'><span
  lang=EN-US style='font-size:6.5pt;mso-bidi-font-size:7.5pt;font-family:"Times New Roman",serif;
  color:black'>Sqrt(<i style='mso-bidi-font-style:normal'>F</i></span><span
  class=GramE><span style='font-size:6.5pt;mso-bidi-font-size:7.5pt;mso-ascii-font-family:
  "Times New Roman";mso-hansi-font-family:"Times New Roman";mso-bidi-font-family:
  "Times New Roman";color:black'>‘</span></span><span lang=EN-US
  style='font-size:6.5pt;mso-bidi-font-size:7.5pt;font-family:"Times New Roman",serif;
  color:black'>)<o:p></o:p></span></p>
  </td>
  <td width=27 nowrap style='width:19.9pt;border-top:none;border-left:none;
  border-bottom:solid windowtext 1.0pt;border-right:solid windowtext 1.0pt;
  mso-border-top-alt:solid windowtext .5pt;mso-border-left-alt:solid windowtext .5pt;
  mso-border-alt:solid windowtext .5pt;padding:0cm 1.4pt 0cm 1.4pt;height:11.35pt'>
  <p class=MsoNormal align=center style='text-align:center;mso-pagination:widow-orphan'><b
  style='mso-bidi-font-weight:normal'><span lang=EN-US style='font-size:7.5pt;
  mso-bidi-font-size:8.0pt;font-family:"Times New Roman",serif;mso-fareast-font-family:
  等线;color:black'>3.16<o:p></o:p></span></b></p>
  </td>
  <td width=27 style='width:19.9pt;border-top:none;border-left:none;border-bottom:
  solid windowtext 1.0pt;border-right:solid windowtext 1.0pt;mso-border-top-alt:
  solid windowtext .5pt;mso-border-left-alt:solid windowtext .5pt;mso-border-alt:
  solid windowtext .5pt;padding:0cm 1.4pt 0cm 1.4pt;height:11.35pt'>
  <p class=MsoNormal align=center style='text-align:center;mso-pagination:widow-orphan'><b
  style='mso-bidi-font-weight:normal'><span lang=EN-US style='font-size:7.5pt;
  mso-bidi-font-size:8.0pt;font-family:"Times New Roman",serif;mso-fareast-font-family:
  等线;color:black'>4.35<o:p></o:p></span></b></p>
  </td>
  <td width=27 style='width:19.9pt;border-top:none;border-left:none;border-bottom:
  solid windowtext 1.0pt;border-right:solid windowtext 1.0pt;mso-border-top-alt:
  solid windowtext .5pt;mso-border-left-alt:solid windowtext .5pt;mso-border-alt:
  solid windowtext .5pt;padding:0cm 1.4pt 0cm 1.4pt;height:11.35pt'>
  <p class=MsoNormal align=center style='text-align:center;mso-pagination:widow-orphan'><span
  lang=EN-US style='font-size:7.5pt;mso-bidi-font-size:8.0pt;font-family:"Times New Roman",serif;
  mso-fareast-font-family:等线;color:black;mso-bidi-font-weight:bold'>3.61<o:p></o:p></span></p>
  </td>
  <td width=27 style='width:19.95pt;border-top:none;border-left:none;
  border-bottom:solid windowtext 1.0pt;border-right:solid windowtext 1.0pt;
  mso-border-top-alt:solid windowtext .5pt;mso-border-left-alt:solid windowtext .5pt;
  mso-border-alt:solid windowtext .5pt;padding:0cm 1.4pt 0cm 1.4pt;height:11.35pt'>
  <p class=MsoNormal align=center style='text-align:center;mso-pagination:widow-orphan'><b
  style='mso-bidi-font-weight:normal'><span lang=EN-US style='font-size:7.5pt;
  mso-bidi-font-size:8.0pt;font-family:"Times New Roman",serif;mso-fareast-font-family:
  等线;color:black'>3.78<o:p></o:p></span></b></p>
  </td>
  <td width=38 nowrap style='width:1.0cm;border-top:none;border-left:none;
  border-bottom:solid windowtext 1.0pt;border-right:solid windowtext 1.0pt;
  mso-border-top-alt:solid windowtext .5pt;mso-border-left-alt:solid windowtext .5pt;
  mso-border-alt:solid windowtext .5pt;padding:0cm 1.4pt 0cm 1.4pt;height:11.35pt'>
  <p class=MsoNormal align=center style='text-align:center;mso-pagination:widow-orphan'><b
  style='mso-bidi-font-weight:normal'><span lang=EN-US style='font-size:7.5pt;
  mso-bidi-font-size:8.0pt;font-family:"Times New Roman",serif;mso-fareast-font-family:
  等线;color:black'>94.7<o:p></o:p></span></b></p>
  </td>
  <td width=38 nowrap style='width:1.0cm;border-top:none;border-left:none;
  border-bottom:solid windowtext 1.0pt;border-right:solid windowtext 1.0pt;
  mso-border-top-alt:solid windowtext .5pt;mso-border-left-alt:solid windowtext .5pt;
  mso-border-alt:solid windowtext .5pt;padding:0cm 1.4pt 0cm 1.4pt;height:11.35pt'>
  <p class=MsoNormal align=center style='text-align:center;mso-pagination:widow-orphan'><b
  style='mso-bidi-font-weight:normal'><span lang=EN-US style='font-size:7.5pt;
  mso-bidi-font-size:8.0pt;font-family:"Times New Roman",serif;mso-fareast-font-family:
  等线;color:black'>18.8<o:p></o:p></span></b></p>
  </td>
 </tr>
 <tr style='mso-yfti-irow:5;height:11.35pt'>
  <td width=28 rowspan=3 style='width:21.1pt;border:solid windowtext 1.0pt;
  border-top:none;mso-border-top-alt:solid windowtext .5pt;mso-border-alt:solid windowtext .5pt;
  padding:0cm 1.4pt 0cm 1.4pt;height:11.35pt'>
  <p class=MsoNormal align=center style='text-align:center'><span lang=EN-US
  style='font-size:5.0pt;mso-bidi-font-size:5.5pt;font-family:"Times New Roman",serif;
  color:black'>DPARN<o:p></o:p></span></p>
  </td>
  <td width=36 style='width:26.85pt;border-top:none;border-left:none;
  border-bottom:solid windowtext 1.0pt;border-right:solid windowtext 1.0pt;
  mso-border-top-alt:solid windowtext .5pt;mso-border-left-alt:solid windowtext .5pt;
  mso-border-alt:solid windowtext .5pt;padding:0cm 1.4pt 0cm 1.4pt;height:11.35pt'>
  <p class=MsoNormal align=center style='text-align:center'><i
  style='mso-bidi-font-style:normal'><span lang=EN-US style='font-size:6.5pt;
  mso-bidi-font-size:7.5pt;font-family:"Times New Roman",serif;mso-fareast-font-family:
  等线;color:black'>F’/2</span></i><span lang=EN-US style='font-size:6.5pt;
  mso-bidi-font-size:7.5pt;mso-fareast-font-family:等线;color:black'><o:p></o:p></span></p>
  </td>
  <td width=27 nowrap style='width:19.9pt;border-top:none;border-left:none;
  border-bottom:solid windowtext 1.0pt;border-right:solid windowtext 1.0pt;
  mso-border-top-alt:solid windowtext .5pt;mso-border-left-alt:solid windowtext .5pt;
  mso-border-alt:solid windowtext .5pt;padding:0cm 1.4pt 0cm 1.4pt;height:11.35pt'>
  <p class=MsoNormal align=center style='text-align:center'><b><span
  lang=EN-US style='font-size:7.5pt;mso-bidi-font-size:8.0pt;font-family:"Times New Roman",serif;
  color:black'>2.96</span></b><span lang=EN-US style='font-size:7.5pt;
  mso-bidi-font-size:8.0pt;font-family:"Times New Roman",serif;color:black'><o:p></o:p></span></p>
  </td>
  <td width=27 style='width:19.9pt;border-top:none;border-left:none;border-bottom:
  solid windowtext 1.0pt;border-right:solid windowtext 1.0pt;mso-border-top-alt:
  solid windowtext .5pt;mso-border-left-alt:solid windowtext .5pt;mso-border-alt:
  solid windowtext .5pt;padding:0cm 1.4pt 0cm 1.4pt;height:11.35pt'>
  <p class=MsoNormal align=center style='text-align:center'><b><span
  lang=EN-US style='font-size:7.5pt;mso-bidi-font-size:8.0pt;font-family:"Times New Roman",serif;
  color:black'>4.29</span></b><span lang=EN-US style='font-size:7.5pt;
  mso-bidi-font-size:8.0pt;font-family:"Times New Roman",serif;color:black;
  mso-fareast-language:EN-US'><o:p></o:p></span></p>
  </td>
  <td width=27 style='width:19.9pt;border-top:none;border-left:none;border-bottom:
  solid windowtext 1.0pt;border-right:solid windowtext 1.0pt;mso-border-top-alt:
  solid windowtext .5pt;mso-border-left-alt:solid windowtext .5pt;mso-border-alt:
  solid windowtext .5pt;padding:0cm 1.4pt 0cm 1.4pt;height:11.35pt'>
  <p class=MsoNormal align=center style='text-align:center'><span lang=EN-US
  style='font-size:7.5pt;mso-bidi-font-size:8.0pt;font-family:"Times New Roman",serif;
  color:black'>3.63<o:p></o:p></span></p>
  </td>
  <td width=27 style='width:19.95pt;border-top:none;border-left:none;
  border-bottom:solid windowtext 1.0pt;border-right:solid windowtext 1.0pt;
  mso-border-top-alt:solid windowtext .5pt;mso-border-left-alt:solid windowtext .5pt;
  mso-border-alt:solid windowtext .5pt;padding:0cm 1.4pt 0cm 1.4pt;height:11.35pt'>
  <p class=MsoNormal align=center style='text-align:center'><b><span
  lang=EN-US style='font-size:7.5pt;mso-bidi-font-size:8.0pt;font-family:"Times New Roman",serif;
  color:black'>3.68</span></b><span lang=EN-US style='font-size:7.5pt;
  mso-bidi-font-size:8.0pt;font-family:"Times New Roman",serif;color:black'><o:p></o:p></span></p>
  </td>
  <td width=38 nowrap style='width:1.0cm;border-top:none;border-left:none;
  border-bottom:solid windowtext 1.0pt;border-right:solid windowtext 1.0pt;
  mso-border-top-alt:solid windowtext .5pt;mso-border-left-alt:solid windowtext .5pt;
  mso-border-alt:solid windowtext .5pt;padding:0cm 1.4pt 0cm 1.4pt;height:11.35pt'>
  <p class=MsoNormal align=center style='text-align:center'><b><span
  lang=EN-US style='font-size:7.5pt;mso-bidi-font-size:8.0pt;font-family:"Times New Roman",serif;
  color:black'>94.2</span></b><span lang=EN-US style='font-size:7.5pt;
  mso-bidi-font-size:8.0pt;font-family:"Times New Roman",serif;color:black'><o:p></o:p></span></p>
  </td>
  <td width=38 nowrap style='width:1.0cm;border-top:none;border-left:none;
  border-bottom:solid windowtext 1.0pt;border-right:solid windowtext 1.0pt;
  mso-border-top-alt:solid windowtext .5pt;mso-border-left-alt:solid windowtext .5pt;
  mso-border-alt:solid windowtext .5pt;padding:0cm 1.4pt 0cm 1.4pt;height:11.35pt'>
  <p class=MsoNormal align=center style='text-align:center'><span lang=EN-US
  style='font-size:7.5pt;mso-bidi-font-size:8.0pt;font-family:"Times New Roman",serif;
  color:black'>18.7<o:p></o:p></span></p>
  </td>
 </tr>
 <tr style='mso-yfti-irow:6;height:11.35pt'>
  <td width=36 style='width:26.85pt;border-top:none;border-left:none;
  border-bottom:solid windowtext 1.0pt;border-right:solid windowtext 1.0pt;
  mso-border-top-alt:solid windowtext .5pt;mso-border-left-alt:solid windowtext .5pt;
  mso-border-alt:solid windowtext .5pt;padding:0cm 1.4pt 0cm 1.4pt;height:11.35pt'>
  <p class=MsoNormal align=center style='text-align:center'><i
  style='mso-bidi-font-style:normal'><span lang=EN-US style='font-size:6.5pt;
  mso-bidi-font-size:7.5pt;font-family:"Times New Roman",serif;mso-fareast-font-family:
  等线;color:black'>F’/4</span></i><span lang=EN-US style='font-size:6.5pt;
  mso-bidi-font-size:7.5pt;color:black'><o:p></o:p></span></p>
  </td>
  <td width=27 nowrap style='width:19.9pt;border-top:none;border-left:none;
  border-bottom:solid windowtext 1.0pt;border-right:solid windowtext 1.0pt;
  mso-border-top-alt:solid windowtext .5pt;mso-border-left-alt:solid windowtext .5pt;
  mso-border-alt:solid windowtext .5pt;padding:0cm 1.4pt 0cm 1.4pt;height:11.35pt'>
  <p class=MsoNormal align=center style='text-align:center'><span lang=EN-US
  style='font-size:7.5pt;mso-bidi-font-size:8.0pt;font-family:"Times New Roman",serif;
  color:black'>2.95<o:p></o:p></span></p>
  </td>
  <td width=27 style='width:19.9pt;border-top:none;border-left:none;border-bottom:
  solid windowtext 1.0pt;border-right:solid windowtext 1.0pt;mso-border-top-alt:
  solid windowtext .5pt;mso-border-left-alt:solid windowtext .5pt;mso-border-alt:
  solid windowtext .5pt;padding:0cm 1.4pt 0cm 1.4pt;height:11.35pt'>
  <p class=MsoNormal align=center style='text-align:center'><span lang=EN-US
  style='font-size:7.5pt;mso-bidi-font-size:8.0pt;font-family:"Times New Roman",serif;
  color:black'>4.27</span><span lang=EN-US style='font-size:7.5pt;mso-bidi-font-size:
  8.0pt;font-family:"Times New Roman",serif;color:black;mso-fareast-language:
  EN-US'><o:p></o:p></span></p>
  </td>
  <td width=27 style='width:19.9pt;border-top:none;border-left:none;border-bottom:
  solid windowtext 1.0pt;border-right:solid windowtext 1.0pt;mso-border-top-alt:
  solid windowtext .5pt;mso-border-left-alt:solid windowtext .5pt;mso-border-alt:
  solid windowtext .5pt;padding:0cm 1.4pt 0cm 1.4pt;height:11.35pt'>
  <p class=MsoNormal align=center style='text-align:center'><b><span
  lang=EN-US style='font-size:7.5pt;mso-bidi-font-size:8.0pt;font-family:"Times New Roman",serif;
  color:black'>3.65<o:p></o:p></span></b></p>
  </td>
  <td width=27 style='width:19.95pt;border-top:none;border-left:none;
  border-bottom:solid windowtext 1.0pt;border-right:solid windowtext 1.0pt;
  mso-border-top-alt:solid windowtext .5pt;mso-border-left-alt:solid windowtext .5pt;
  mso-border-alt:solid windowtext .5pt;padding:0cm 1.4pt 0cm 1.4pt;height:11.35pt'>
  <p class=MsoNormal align=center style='text-align:center'><span lang=EN-US
  style='font-size:7.5pt;mso-bidi-font-size:8.0pt;font-family:"Times New Roman",serif;
  color:black'>3.68<o:p></o:p></span></p>
  </td>
  <td width=38 nowrap style='width:1.0cm;border-top:none;border-left:none;
  border-bottom:solid windowtext 1.0pt;border-right:solid windowtext 1.0pt;
  mso-border-top-alt:solid windowtext .5pt;mso-border-left-alt:solid windowtext .5pt;
  mso-border-alt:solid windowtext .5pt;padding:0cm 1.4pt 0cm 1.4pt;height:11.35pt'>
  <p class=MsoNormal align=center style='text-align:center'><span lang=EN-US
  style='font-size:7.5pt;mso-bidi-font-size:8.0pt;font-family:"Times New Roman",serif;
  color:black'>94.2<o:p></o:p></span></p>
  </td>
  <td width=38 nowrap style='width:1.0cm;border-top:none;border-left:none;
  border-bottom:solid windowtext 1.0pt;border-right:solid windowtext 1.0pt;
  mso-border-top-alt:solid windowtext .5pt;mso-border-left-alt:solid windowtext .5pt;
  mso-border-alt:solid windowtext .5pt;padding:0cm 1.4pt 0cm 1.4pt;height:11.35pt'>
  <p class=MsoNormal align=center style='text-align:center'><b><span
  lang=EN-US style='font-size:7.5pt;mso-bidi-font-size:8.0pt;font-family:"Times New Roman",serif;
  color:black'>18.8<o:p></o:p></span></b></p>
  </td>
 </tr>
 <tr style='mso-yfti-irow:7;mso-yfti-lastrow:yes;height:11.35pt'>
  <td width=36 style='width:26.85pt;border-top:none;border-left:none;
  border-bottom:solid windowtext 1.0pt;border-right:solid windowtext 1.0pt;
  mso-border-top-alt:solid windowtext .5pt;mso-border-left-alt:solid windowtext .5pt;
  mso-border-alt:solid windowtext .5pt;padding:0cm 1.4pt 0cm 1.4pt;height:11.35pt'>
  <p class=MsoNormal align=center style='text-align:center'><span lang=EN-US
  style='font-size:6.5pt;mso-bidi-font-size:7.5pt;font-family:"Times New Roman",serif;
  color:black'>Sqrt(<i style='mso-bidi-font-style:normal'>F</i></span><span
  class=GramE><span style='font-size:6.5pt;mso-bidi-font-size:7.5pt;mso-ascii-font-family:
  "Times New Roman";mso-hansi-font-family:"Times New Roman";mso-bidi-font-family:
  "Times New Roman";color:black'>‘</span></span><span lang=EN-US
  style='font-size:6.5pt;mso-bidi-font-size:7.5pt;font-family:"Times New Roman",serif;
  color:black'>)</span><span lang=EN-US style='font-size:6.5pt;mso-bidi-font-size:
  7.5pt;font-family:"Times New Roman",serif;mso-fareast-font-family:等线'><o:p></o:p></span></p>
  </td>
  <td width=27 nowrap style='width:19.9pt;border-top:none;border-left:none;
  border-bottom:solid windowtext 1.0pt;border-right:solid windowtext 1.0pt;
  mso-border-top-alt:solid windowtext .5pt;mso-border-left-alt:solid windowtext .5pt;
  mso-border-alt:solid windowtext .5pt;padding:0cm 1.4pt 0cm 1.4pt;height:11.35pt'>
  <p class=MsoNormal align=center style='text-align:center'><span lang=EN-US
  style='font-size:7.5pt;mso-bidi-font-size:8.0pt;font-family:"Times New Roman",serif;
  color:black'>2.94<b><o:p></o:p></b></span></p>
  </td>
  <td width=27 style='width:19.9pt;border-top:none;border-left:none;border-bottom:
  solid windowtext 1.0pt;border-right:solid windowtext 1.0pt;mso-border-top-alt:
  solid windowtext .5pt;mso-border-left-alt:solid windowtext .5pt;mso-border-alt:
  solid windowtext .5pt;padding:0cm 1.4pt 0cm 1.4pt;height:11.35pt'>
  <p class=MsoNormal align=center style='text-align:center'><span lang=EN-US
  style='font-size:7.5pt;mso-bidi-font-size:8.0pt;font-family:"Times New Roman",serif;
  color:black'>4.27<b><o:p></o:p></b></span></p>
  </td>
  <td width=27 style='width:19.9pt;border-top:none;border-left:none;border-bottom:
  solid windowtext 1.0pt;border-right:solid windowtext 1.0pt;mso-border-top-alt:
  solid windowtext .5pt;mso-border-left-alt:solid windowtext .5pt;mso-border-alt:
  solid windowtext .5pt;padding:0cm 1.4pt 0cm 1.4pt;height:11.35pt'>
  <p class=MsoNormal align=center style='text-align:center'><span lang=EN-US
  style='font-size:7.5pt;mso-bidi-font-size:8.0pt;font-family:"Times New Roman",serif;
  color:black'>3.62<b><o:p></o:p></b></span></p>
  </td>
  <td width=27 style='width:19.95pt;border-top:none;border-left:none;
  border-bottom:solid windowtext 1.0pt;border-right:solid windowtext 1.0pt;
  mso-border-top-alt:solid windowtext .5pt;mso-border-left-alt:solid windowtext .5pt;
  mso-border-alt:solid windowtext .5pt;padding:0cm 1.4pt 0cm 1.4pt;height:11.35pt'>
  <p class=MsoNormal align=center style='text-align:center'><span lang=EN-US
  style='font-size:7.5pt;mso-bidi-font-size:8.0pt;font-family:"Times New Roman",serif;
  color:black'>3.67<b><o:p></o:p></b></span></p>
  </td>
  <td width=38 nowrap style='width:1.0cm;border-top:none;border-left:none;
  border-bottom:solid windowtext 1.0pt;border-right:solid windowtext 1.0pt;
  mso-border-top-alt:solid windowtext .5pt;mso-border-left-alt:solid windowtext .5pt;
  mso-border-alt:solid windowtext .5pt;padding:0cm 1.4pt 0cm 1.4pt;height:11.35pt'>
  <p class=MsoNormal align=center style='text-align:center'><span lang=EN-US
  style='font-size:7.5pt;mso-bidi-font-size:8.0pt;font-family:"Times New Roman",serif;
  color:black'>94.1<b><o:p></o:p></b></span></p>
  </td>
  <td width=38 nowrap style='width:1.0cm;border-top:none;border-left:none;
  border-bottom:solid windowtext 1.0pt;border-right:solid windowtext 1.0pt;
  mso-border-top-alt:solid windowtext .5pt;mso-border-left-alt:solid windowtext .5pt;
  mso-border-alt:solid windowtext .5pt;padding:0cm 1.4pt 0cm 1.4pt;height:11.35pt'>
  <p class=MsoNormal align=center style='text-align:center'><span lang=EN-US
  style='font-size:7.5pt;mso-bidi-font-size:8.0pt;font-family:"Times New Roman",serif;
  color:black'>18.5<b><o:p></o:p></b></span></p>
  </td>
 </tr>
</table>

</div>

<p class=MsoNormal><span lang=EN-US><o:p>&nbsp;</o:p></span></p>

</div>

</body>  

It can be seen that the setting of ***N<sub>l</sub>*** affects different models differently and we choose the setting achieving the best performance for each model, i.e. sqrt(***F'***) for MTFAA and ***F'/2*** for DPARN. Next, we train the models with the larger DNS4 dataset and the training process can be seen in figures below, where both LSA-based models achieve better convergence compared with the original models. 
<!-- ![trainloss_dparn](https://user-images.githubusercontent.com/103247057/222072535-ab00598c-448d-47ff-9cf3-77e7da8d302c.jpeg) -->
<!-- ![Validation](https://user-images.githubusercontent.com/103247057/222072910-c3c7730d-b7be-45ca-9591-5706c395f1ad.jpeg) -->
<div align=center><img src="https://user-images.githubusercontent.com/103247057/222072535-ab00598c-448d-47ff-9cf3-77e7da8d302c.jpeg" width="320" height="240" /><img src="https://user-images.githubusercontent.com/103247057/222072910-c3c7730d-b7be-45ca-9591-5706c395f1ad.jpeg" width="320" height="240" /></div>
<!-- ![trainloss_mtfaa](https://user-images.githubusercontent.com/103247057/222074817-58680383-6d61-42c6-9f94-eb4c58ecefab.jpeg)
![Validation_mtfaa](https://user-images.githubusercontent.com/103247057/222074828-44bc05e4-c05e-40bc-b4b5-060cdb7030a7.jpeg) -->
<div align=center><img src="https://user-images.githubusercontent.com/103247057/222074817-58680383-6d61-42c6-9f94-eb4c58ecefab.jpeg" width="320" height="240" /><img src="https://user-images.githubusercontent.com/103247057/222074828-44bc05e4-c05e-40bc-b4b5-060cdb7030a7.jpeg" width="320" height="240" /></div>

The objective test results can be seen in table below

<body lang=ZH-CN style='tab-interval:21.0pt;text-justify-trim:punctuation'>

<div class=WordSection1 style='layout-grid:15.6pt'>

<div align=center>

<table class=MsoTableGrid border=1 cellspacing=0 cellpadding=0 width=548
 style='border-collapse:collapse;mso-table-layout-alt:fixed;border:none;
 mso-border-alt:solid windowtext .5pt;mso-yfti-tbllook:1184;mso-padding-alt:
 0cm 5.4pt 0cm 5.4pt'>
 <tr style='mso-yfti-irow:0;mso-yfti-firstrow:yes;height:1.0pt'>
  <td width=75 nowrap style='width:56.45pt;border:solid windowtext 1.0pt;
  mso-border-alt:solid windowtext .5pt;padding:0cm 5.4pt 0cm 5.4pt;height:1.0pt'>
  <p class=MsoNormal align=center style='text-align:center;mso-pagination:widow-orphan'><span
  lang=EN-US style='font-size:5.5pt;mso-bidi-font-size:10.0pt;font-family:"Times New Roman",serif;
  mso-fareast-font-family:等线;mso-font-kerning:0pt'>Full-band Metrics</span><span
  lang=EN-US style='font-size:7.5pt;mso-bidi-font-size:10.0pt;font-family:"Times New Roman",serif;
  mso-fareast-font-family:等线;mso-font-kerning:0pt'><o:p></o:p></span></p>
  </td>
  <td width=118 nowrap colspan=3 style='width:88.6pt;border:solid windowtext 1.0pt;
  border-left:none;mso-border-left-alt:solid windowtext .5pt;mso-border-alt:
  solid windowtext .5pt;padding:0cm 5.4pt 0cm 5.4pt;height:1.0pt'>
  <p class=MsoNormal align=center style='text-align:center;mso-pagination:widow-orphan'><span
  lang=EN-US style='font-size:7.5pt;mso-bidi-font-size:10.0pt;font-family:"Times New Roman",serif;
  mso-fareast-font-family:等线;mso-font-kerning:0pt'>STOI<o:p></o:p></span></p>
  </td>
  <td width=118 nowrap colspan=3 style='width:88.6pt;border:solid windowtext 1.0pt;
  border-left:none;mso-border-left-alt:solid windowtext .5pt;mso-border-alt:
  solid windowtext .5pt;padding:0cm 5.4pt 0cm 5.4pt;height:1.0pt'>
  <p class=MsoNormal align=center style='text-align:center;mso-pagination:widow-orphan'><span
  lang=EN-US style='font-size:7.5pt;mso-bidi-font-size:10.0pt;font-family:"Times New Roman",serif;
  mso-fareast-font-family:等线;mso-font-kerning:0pt'>SiSDR (dB)<o:p></o:p></span></p>
  </td>
  <td width=39 nowrap rowspan=7 style='width:29.5pt;border:solid windowtext 1.0pt;
  border-left:none;mso-border-left-alt:solid windowtext .5pt;mso-border-alt:
  solid windowtext .5pt;padding:0cm 5.4pt 0cm 5.4pt;height:1.0pt'>
  <p class=MsoNormal align=center style='text-align:center;mso-pagination:widow-orphan'><span
  lang=EN-US style='font-size:7.5pt;mso-bidi-font-size:10.0pt;font-family:"Times New Roman",serif;
  mso-fareast-font-family:等线;mso-font-kerning:0pt'><o:p>&nbsp;</o:p></span></p>
  </td>
  <td width=197 colspan=5 style='width:147.7pt;border:solid windowtext 1.0pt;
  border-left:none;mso-border-left-alt:solid windowtext .5pt;mso-border-alt:
  solid windowtext .5pt;padding:0cm 5.4pt 0cm 5.4pt;height:1.0pt'>
  <p class=MsoNormal align=center style='text-align:center;mso-pagination:widow-orphan'><span
  lang=EN-US style='font-size:7.5pt;mso-bidi-font-size:10.0pt;font-family:"Times New Roman",serif;
  mso-fareast-font-family:等线;mso-font-kerning:0pt'>LSD (dB)<o:p></o:p></span></p>
  </td>
 </tr>
 <tr style='mso-yfti-irow:1;height:1.0pt'>
  <td width=75 nowrap style='width:56.45pt;border:solid windowtext 1.0pt;
  border-top:none;mso-border-top-alt:solid windowtext .5pt;mso-border-alt:solid windowtext .5pt;
  padding:0cm 5.4pt 0cm 5.4pt;height:1.0pt'>
  <p class=MsoNormal align=center style='text-align:center;mso-pagination:widow-orphan'><span
  class=GramE><span lang=EN-US style='font-size:6.5pt;mso-bidi-font-size:10.0pt;
  font-family:"Times New Roman",serif;mso-fareast-font-family:等线;mso-font-kerning:
  0pt'>SNR(</span></span><span lang=EN-US style='font-size:6.5pt;mso-bidi-font-size:
  10.0pt;font-family:"Times New Roman",serif;mso-fareast-font-family:等线;
  mso-font-kerning:0pt'>dB)</span><span lang=EN-US style='font-size:7.5pt;
  mso-bidi-font-size:10.0pt;font-family:"Times New Roman",serif;mso-fareast-font-family:
  等线;mso-font-kerning:0pt'><o:p></o:p></span></p>
  </td>
  <td width=39 nowrap style='width:29.5pt;border-top:none;border-left:none;
  border-bottom:solid windowtext 1.0pt;border-right:solid windowtext 1.0pt;
  mso-border-top-alt:solid windowtext .5pt;mso-border-left-alt:solid windowtext .5pt;
  mso-border-alt:solid windowtext .5pt;padding:0cm 5.4pt 0cm 5.4pt;height:1.0pt'>
  <p class=MsoNormal align=center style='text-align:center;mso-pagination:widow-orphan'><span
  lang=EN-US style='font-size:7.5pt;mso-bidi-font-size:10.0pt;font-family:"Times New Roman",serif;
  mso-fareast-font-family:等线;mso-font-kerning:0pt'>-5~0<o:p></o:p></span></p>
  </td>
  <td width=39 nowrap style='width:29.55pt;border-top:none;border-left:none;
  border-bottom:solid windowtext 1.0pt;border-right:solid windowtext 1.0pt;
  mso-border-top-alt:solid windowtext .5pt;mso-border-left-alt:solid windowtext .5pt;
  mso-border-alt:solid windowtext .5pt;padding:0cm 5.4pt 0cm 5.4pt;height:1.0pt'>
  <p class=MsoNormal align=center style='text-align:center;mso-pagination:widow-orphan'><span
  lang=EN-US style='font-size:7.5pt;mso-bidi-font-size:10.0pt;font-family:"Times New Roman",serif;
  mso-fareast-font-family:等线;mso-font-kerning:0pt'>0~15<o:p></o:p></span></p>
  </td>
  <td width=39 nowrap style='width:29.55pt;border-top:none;border-left:none;
  border-bottom:solid windowtext 1.0pt;border-right:solid windowtext 1.0pt;
  mso-border-top-alt:solid windowtext .5pt;mso-border-left-alt:solid windowtext .5pt;
  mso-border-alt:solid windowtext .5pt;padding:0cm 5.4pt 0cm 5.4pt;height:1.0pt'>
  <p class=MsoNormal align=center style='text-align:center;mso-pagination:widow-orphan'><span
  class=SpellE><span lang=EN-US style='font-size:7.5pt;mso-bidi-font-size:10.0pt;
  font-family:"Times New Roman",serif;mso-fareast-font-family:等线;mso-font-kerning:
  0pt'>Ovrl</span></span><span lang=EN-US style='font-size:7.5pt;mso-bidi-font-size:
  10.0pt;font-family:"Times New Roman",serif;mso-fareast-font-family:等线;
  mso-font-kerning:0pt'>.<o:p></o:p></span></p>
  </td>
  <td width=39 nowrap style='width:29.5pt;border-top:none;border-left:none;
  border-bottom:solid windowtext 1.0pt;border-right:solid windowtext 1.0pt;
  mso-border-top-alt:solid windowtext .5pt;mso-border-left-alt:solid windowtext .5pt;
  mso-border-alt:solid windowtext .5pt;padding:0cm 5.4pt 0cm 5.4pt;height:1.0pt'>
  <p class=MsoNormal align=center style='text-align:center;mso-pagination:widow-orphan'><span
  lang=EN-US style='font-size:7.5pt;mso-bidi-font-size:10.0pt;font-family:"Times New Roman",serif;
  mso-fareast-font-family:等线;mso-font-kerning:0pt'>-5~0<o:p></o:p></span></p>
  </td>
  <td width=39 nowrap style='width:29.55pt;border-top:none;border-left:none;
  border-bottom:solid windowtext 1.0pt;border-right:solid windowtext 1.0pt;
  mso-border-top-alt:solid windowtext .5pt;mso-border-left-alt:solid windowtext .5pt;
  mso-border-alt:solid windowtext .5pt;padding:0cm 5.4pt 0cm 5.4pt;height:1.0pt'>
  <p class=MsoNormal align=center style='text-align:center;mso-pagination:widow-orphan'><span
  lang=EN-US style='font-size:7.5pt;mso-bidi-font-size:10.0pt;font-family:"Times New Roman",serif;
  mso-fareast-font-family:等线;mso-font-kerning:0pt'>0~15<o:p></o:p></span></p>
  </td>
  <td width=39 nowrap style='width:29.55pt;border-top:none;border-left:none;
  border-bottom:solid windowtext 1.0pt;border-right:solid windowtext 1.0pt;
  mso-border-top-alt:solid windowtext .5pt;mso-border-left-alt:solid windowtext .5pt;
  mso-border-alt:solid windowtext .5pt;padding:0cm 5.4pt 0cm 5.4pt;height:1.0pt'>
  <p class=MsoNormal align=center style='text-align:center;mso-pagination:widow-orphan'><span
  class=SpellE><span lang=EN-US style='font-size:7.5pt;mso-bidi-font-size:10.0pt;
  font-family:"Times New Roman",serif;mso-fareast-font-family:等线;mso-font-kerning:
  0pt'>Ovrl</span></span><span lang=EN-US style='font-size:7.5pt;mso-bidi-font-size:
  10.0pt;font-family:"Times New Roman",serif;mso-fareast-font-family:等线;
  mso-font-kerning:0pt'>.<o:p></o:p></span></p>
  </td>
  <td width=79 nowrap colspan=2 style='width:59.1pt;border-top:none;border-left:
  none;border-bottom:solid windowtext 1.0pt;border-right:solid windowtext 1.0pt;
  mso-border-top-alt:solid windowtext .5pt;mso-border-left-alt:solid windowtext .5pt;
  mso-border-alt:solid windowtext .5pt;padding:0cm 5.4pt 0cm 5.4pt;height:1.0pt'>
  <p class=MsoNormal align=center style='text-align:center;mso-pagination:widow-orphan'><span
  lang=EN-US style='font-size:7.5pt;mso-bidi-font-size:10.0pt;font-family:"Times New Roman",serif;
  mso-fareast-font-family:等线;mso-font-kerning:0pt'>Band(kHz)</span><span
  lang=EN-US style='font-size:5.0pt;mso-bidi-font-size:10.0pt;font-family:"Times New Roman",serif;
  mso-fareast-font-family:等线;mso-font-kerning:0pt'><o:p></o:p></span></p>
  </td>
  <td width=39 style='width:29.5pt;border-top:none;border-left:none;border-bottom:
  solid windowtext 1.0pt;border-right:solid windowtext 1.0pt;mso-border-top-alt:
  solid windowtext .5pt;mso-border-left-alt:solid windowtext .5pt;mso-border-alt:
  solid windowtext .5pt;padding:0cm 5.4pt 0cm 5.4pt;height:1.0pt'>
  <p class=MsoNormal align=center style='text-align:center;mso-pagination:widow-orphan'><span
  lang=EN-US style='font-size:7.5pt;font-family:"Times New Roman",serif;
  mso-fareast-font-family:等线;mso-font-kerning:0pt'>0~8<o:p></o:p></span></p>
  </td>
  <td width=39 style='width:29.55pt;border-top:none;border-left:none;
  border-bottom:solid windowtext 1.0pt;border-right:solid windowtext 1.0pt;
  mso-border-top-alt:solid windowtext .5pt;mso-border-left-alt:solid windowtext .5pt;
  mso-border-alt:solid windowtext .5pt;padding:0cm 5.4pt 0cm 5.4pt;height:1.0pt'>
  <p class=MsoNormal align=center style='text-align:center;mso-pagination:widow-orphan'><span
  lang=EN-US style='font-size:7.5pt;font-family:"Times New Roman",serif;
  mso-fareast-font-family:等线;mso-font-kerning:0pt'>8~24<o:p></o:p></span></p>
  </td>
  <td width=39 style='width:29.55pt;border-top:none;border-left:none;
  border-bottom:solid windowtext 1.0pt;border-right:solid windowtext 1.0pt;
  mso-border-top-alt:solid windowtext .5pt;mso-border-left-alt:solid windowtext .5pt;
  mso-border-alt:solid windowtext .5pt;padding:0cm 5.4pt 0cm 5.4pt;height:1.0pt'>
  <p class=MsoNormal align=center style='text-align:center;mso-pagination:widow-orphan'><span
  lang=EN-US style='font-size:7.5pt;font-family:"Times New Roman",serif;
  mso-fareast-font-family:等线;mso-font-kerning:0pt'>Full.<o:p></o:p></span></p>
  </td>
 </tr>
 <tr style='mso-yfti-irow:2;height:1.0pt'>
  <td width=75 nowrap style='width:56.45pt;border:solid windowtext 1.0pt;
  border-top:none;mso-border-top-alt:solid windowtext .5pt;mso-border-alt:solid windowtext .5pt;
  padding:0cm 5.4pt 0cm 5.4pt;height:1.0pt'>
  <p class=MsoNormal align=center style='text-align:center;mso-pagination:widow-orphan'><span
  lang=EN-US style='font-size:6.5pt;mso-bidi-font-size:10.0pt;font-family:"Times New Roman",serif;
  mso-fareast-font-family:等线;mso-font-kerning:0pt'>Noisy</span><span
  lang=EN-US style='font-size:7.5pt;mso-bidi-font-size:10.0pt;font-family:"Times New Roman",serif;
  mso-fareast-font-family:等线;mso-font-kerning:0pt'><o:p></o:p></span></p>
  </td>
  <td width=39 nowrap style='width:29.5pt;border-top:none;border-left:none;
  border-bottom:solid windowtext 1.0pt;border-right:solid windowtext 1.0pt;
  mso-border-top-alt:solid windowtext .5pt;mso-border-left-alt:solid windowtext .5pt;
  mso-border-alt:solid windowtext .5pt;padding:0cm 5.4pt 0cm 5.4pt;height:1.0pt'>
  <p class=MsoNormal align=center style='text-align:center;mso-pagination:widow-orphan'><span
  lang=EN-US style='font-size:6.5pt;mso-bidi-font-size:10.0pt;font-family:"Times New Roman",serif;
  mso-fareast-font-family:等线;mso-font-kerning:0pt'>0.687<o:p></o:p></span></p>
  </td>
  <td width=39 nowrap style='width:29.55pt;border-top:none;border-left:none;
  border-bottom:solid windowtext 1.0pt;border-right:solid windowtext 1.0pt;
  mso-border-top-alt:solid windowtext .5pt;mso-border-left-alt:solid windowtext .5pt;
  mso-border-alt:solid windowtext .5pt;padding:0cm 5.4pt 0cm 5.4pt;height:1.0pt'>
  <p class=MsoNormal align=center style='text-align:center;mso-pagination:widow-orphan'><span
  lang=EN-US style='font-size:6.5pt;mso-bidi-font-size:10.0pt;font-family:"Times New Roman",serif;
  mso-fareast-font-family:等线;mso-font-kerning:0pt'>0.805<o:p></o:p></span></p>
  </td>
  <td width=39 nowrap style='width:29.55pt;border-top:none;border-left:none;
  border-bottom:solid windowtext 1.0pt;border-right:solid windowtext 1.0pt;
  mso-border-top-alt:solid windowtext .5pt;mso-border-left-alt:solid windowtext .5pt;
  mso-border-alt:solid windowtext .5pt;padding:0cm 5.4pt 0cm 5.4pt;height:1.0pt'>
  <p class=MsoNormal align=center style='text-align:center;mso-pagination:widow-orphan'><span
  lang=EN-US style='font-size:6.5pt;mso-bidi-font-size:10.0pt;font-family:"Times New Roman",serif;
  mso-fareast-font-family:等线;mso-font-kerning:0pt'>0.771<o:p></o:p></span></p>
  </td>
  <td width=39 nowrap valign=top style='width:29.5pt;border-top:none;
  border-left:none;border-bottom:solid windowtext 1.0pt;border-right:solid windowtext 1.0pt;
  mso-border-top-alt:solid windowtext .5pt;mso-border-left-alt:solid windowtext .5pt;
  mso-border-alt:solid windowtext .5pt;padding:0cm 5.4pt 0cm 5.4pt;height:1.0pt'>
  <p class=MsoNormal align=center style='text-align:center;mso-pagination:widow-orphan'><span
  lang=EN-US style='font-size:6.5pt;mso-bidi-font-size:10.0pt;font-family:"Times New Roman",serif;
  mso-fareast-font-family:等线;mso-font-kerning:0pt'>-2.515<o:p></o:p></span></p>
  </td>
  <td width=39 nowrap valign=top style='width:29.55pt;border-top:none;
  border-left:none;border-bottom:solid windowtext 1.0pt;border-right:solid windowtext 1.0pt;
  mso-border-top-alt:solid windowtext .5pt;mso-border-left-alt:solid windowtext .5pt;
  mso-border-alt:solid windowtext .5pt;padding:0cm 5.4pt 0cm 5.4pt;height:1.0pt'>
  <p class=MsoNormal align=center style='text-align:center;mso-pagination:widow-orphan'><span
  lang=EN-US style='font-size:6.5pt;mso-bidi-font-size:10.0pt;font-family:"Times New Roman",serif;
  mso-fareast-font-family:等线;mso-font-kerning:0pt'>7.971<o:p></o:p></span></p>
  </td>
  <td width=39 nowrap valign=top style='width:29.55pt;border-top:none;
  border-left:none;border-bottom:solid windowtext 1.0pt;border-right:solid windowtext 1.0pt;
  mso-border-top-alt:solid windowtext .5pt;mso-border-left-alt:solid windowtext .5pt;
  mso-border-alt:solid windowtext .5pt;padding:0cm 5.4pt 0cm 5.4pt;height:1.0pt'>
  <p class=MsoNormal align=center style='text-align:center;mso-pagination:widow-orphan'><span
  lang=EN-US style='font-size:6.5pt;mso-bidi-font-size:10.0pt;font-family:"Times New Roman",serif;
  mso-fareast-font-family:等线;mso-font-kerning:0pt'>5.166<o:p></o:p></span></p>
  </td>
  <td width=79 nowrap colspan=2 style='width:59.1pt;border-top:none;border-left:
  none;border-bottom:solid windowtext 1.0pt;border-right:solid windowtext 1.0pt;
  mso-border-top-alt:solid windowtext .5pt;mso-border-left-alt:solid windowtext .5pt;
  mso-border-alt:solid windowtext .5pt;padding:0cm 5.4pt 0cm 5.4pt;height:1.0pt'>
  <p class=MsoNormal align=center style='text-align:center;mso-pagination:widow-orphan'><span
  lang=EN-US style='font-size:6.5pt;mso-bidi-font-size:10.0pt;font-family:"Times New Roman",serif;
  mso-fareast-font-family:等线;mso-font-kerning:0pt'>Noisy<o:p></o:p></span></p>
  </td>
  <td width=39 style='width:29.5pt;border-top:none;border-left:none;border-bottom:
  solid windowtext 1.0pt;border-right:solid windowtext 1.0pt;mso-border-top-alt:
  solid windowtext .5pt;mso-border-left-alt:solid windowtext .5pt;mso-border-alt:
  solid windowtext .5pt;padding:0cm 5.4pt 0cm 5.4pt;height:1.0pt'>
  <p class=MsoNormal align=center style='text-align:center;mso-pagination:widow-orphan'><span
  lang=EN-US style='font-size:6.5pt;mso-bidi-font-size:10.0pt;font-family:"Times New Roman",serif;
  mso-fareast-font-family:等线;mso-font-kerning:0pt'>18.37<o:p></o:p></span></p>
  </td>
  <td width=39 style='width:29.55pt;border-top:none;border-left:none;
  border-bottom:solid windowtext 1.0pt;border-right:solid windowtext 1.0pt;
  mso-border-top-alt:solid windowtext .5pt;mso-border-left-alt:solid windowtext .5pt;
  mso-border-alt:solid windowtext .5pt;padding:0cm 5.4pt 0cm 5.4pt;height:1.0pt'>
  <p class=MsoNormal align=center style='text-align:center;mso-pagination:widow-orphan'><span
  lang=EN-US style='font-size:6.5pt;mso-bidi-font-size:10.0pt;font-family:"Times New Roman",serif;
  mso-fareast-font-family:等线;mso-font-kerning:0pt'>12.38<o:p></o:p></span></p>
  </td>
  <td width=39 valign=top style='width:29.55pt;border-top:none;border-left:
  none;border-bottom:solid windowtext 1.0pt;border-right:solid windowtext 1.0pt;
  mso-border-top-alt:solid windowtext .5pt;mso-border-left-alt:solid windowtext .5pt;
  mso-border-alt:solid windowtext .5pt;padding:0cm 5.4pt 0cm 5.4pt;height:1.0pt'>
  <p class=MsoNormal align=center style='text-align:center;mso-pagination:widow-orphan'><span
  lang=EN-US style='font-size:6.5pt;mso-bidi-font-size:10.0pt;font-family:"Times New Roman",serif;
  mso-fareast-font-family:等线;mso-font-kerning:0pt'>14.38<o:p></o:p></span></p>
  </td>
 </tr>
 <tr style='mso-yfti-irow:3;height:1.0pt'>
  <td width=75 nowrap style='width:56.45pt;border:solid windowtext 1.0pt;
  border-top:none;mso-border-top-alt:solid windowtext .5pt;mso-border-alt:solid windowtext .5pt;
  padding:0cm 5.4pt 0cm 5.4pt;height:1.0pt'>
  <p class=MsoNormal align=center style='text-align:center;mso-pagination:widow-orphan'><span
  lang=EN-US style='font-size:6.5pt;mso-bidi-font-size:10.0pt;font-family:"Times New Roman",serif;
  mso-fareast-font-family:等线;mso-font-kerning:0pt'>MTFAA<o:p></o:p></span></p>
  </td>
  <td width=39 nowrap style='width:29.5pt;border-top:none;border-left:none;
  border-bottom:solid windowtext 1.0pt;border-right:solid windowtext 1.0pt;
  mso-border-top-alt:solid windowtext .5pt;mso-border-left-alt:solid windowtext .5pt;
  mso-border-alt:solid windowtext .5pt;padding:0cm 5.4pt 0cm 5.4pt;height:1.0pt'>
  <p class=MsoNormal align=center style='text-align:center;mso-pagination:widow-orphan'><span
  lang=EN-US style='font-size:6.5pt;mso-bidi-font-size:10.0pt;font-family:"Times New Roman",serif;
  mso-fareast-font-family:等线;mso-font-kerning:0pt'>0.805<o:p></o:p></span></p>
  </td>
  <td width=39 nowrap style='width:29.55pt;border-top:none;border-left:none;
  border-bottom:solid windowtext 1.0pt;border-right:solid windowtext 1.0pt;
  mso-border-top-alt:solid windowtext .5pt;mso-border-left-alt:solid windowtext .5pt;
  mso-border-alt:solid windowtext .5pt;padding:0cm 5.4pt 0cm 5.4pt;height:1.0pt'>
  <p class=MsoNormal align=center style='text-align:center;mso-pagination:widow-orphan'><span
  lang=EN-US style='font-size:6.5pt;mso-bidi-font-size:10.0pt;font-family:"Times New Roman",serif;
  mso-fareast-font-family:等线;mso-font-kerning:0pt'>0.876<o:p></o:p></span></p>
  </td>
  <td width=39 nowrap style='width:29.55pt;border-top:none;border-left:none;
  border-bottom:solid windowtext 1.0pt;border-right:solid windowtext 1.0pt;
  mso-border-top-alt:solid windowtext .5pt;mso-border-left-alt:solid windowtext .5pt;
  mso-border-alt:solid windowtext .5pt;padding:0cm 5.4pt 0cm 5.4pt;height:1.0pt'>
  <p class=MsoNormal align=center style='text-align:center;mso-pagination:widow-orphan'><span
  lang=EN-US style='font-size:6.5pt;mso-bidi-font-size:10.0pt;font-family:"Times New Roman",serif;
  mso-fareast-font-family:等线;mso-font-kerning:0pt'>0.856<o:p></o:p></span></p>
  </td>
  <td width=39 nowrap valign=top style='width:29.5pt;border-top:none;
  border-left:none;border-bottom:solid windowtext 1.0pt;border-right:solid windowtext 1.0pt;
  mso-border-top-alt:solid windowtext .5pt;mso-border-left-alt:solid windowtext .5pt;
  mso-border-alt:solid windowtext .5pt;padding:0cm 5.4pt 0cm 5.4pt;height:1.0pt'>
  <p class=MsoNormal align=center style='text-align:center;mso-pagination:widow-orphan'><span
  lang=EN-US style='font-size:6.5pt;mso-bidi-font-size:10.0pt;font-family:"Times New Roman",serif;
  mso-fareast-font-family:等线;mso-font-kerning:0pt'>10.10<o:p></o:p></span></p>
  </td>
  <td width=39 nowrap valign=top style='width:29.55pt;border-top:none;
  border-left:none;border-bottom:solid windowtext 1.0pt;border-right:solid windowtext 1.0pt;
  mso-border-top-alt:solid windowtext .5pt;mso-border-left-alt:solid windowtext .5pt;
  mso-border-alt:solid windowtext .5pt;padding:0cm 5.4pt 0cm 5.4pt;height:1.0pt'>
  <p class=MsoNormal align=center style='text-align:center;mso-pagination:widow-orphan'><span
  lang=EN-US style='font-size:6.5pt;mso-bidi-font-size:10.0pt;font-family:"Times New Roman",serif;
  mso-fareast-font-family:等线;mso-font-kerning:0pt'>15.74<o:p></o:p></span></p>
  </td>
  <td width=39 nowrap valign=top style='width:29.55pt;border-top:none;
  border-left:none;border-bottom:solid windowtext 1.0pt;border-right:solid windowtext 1.0pt;
  mso-border-top-alt:solid windowtext .5pt;mso-border-left-alt:solid windowtext .5pt;
  mso-border-alt:solid windowtext .5pt;padding:0cm 5.4pt 0cm 5.4pt;height:1.0pt'>
  <p class=MsoNormal align=center style='text-align:center;mso-pagination:widow-orphan'><span
  lang=EN-US style='font-size:6.5pt;mso-bidi-font-size:10.0pt;font-family:"Times New Roman",serif;
  mso-fareast-font-family:等线;mso-font-kerning:0pt'>14.23<o:p></o:p></span></p>
  </td>
  <td width=79 nowrap colspan=2 style='width:59.1pt;border-top:none;border-left:
  none;border-bottom:solid windowtext 1.0pt;border-right:solid windowtext 1.0pt;
  mso-border-top-alt:solid windowtext .5pt;mso-border-left-alt:solid windowtext .5pt;
  mso-border-alt:solid windowtext .5pt;padding:0cm 5.4pt 0cm 5.4pt;height:1.0pt'>
  <p class=MsoNormal align=center style='text-align:center;mso-pagination:widow-orphan'><span
  lang=EN-US style='font-size:6.5pt;mso-bidi-font-size:10.0pt;font-family:"Times New Roman",serif;
  mso-fareast-font-family:等线;mso-font-kerning:0pt'>MTFAA<o:p></o:p></span></p>
  </td>
  <td width=39 style='width:29.5pt;border-top:none;border-left:none;border-bottom:
  solid windowtext 1.0pt;border-right:solid windowtext 1.0pt;mso-border-top-alt:
  solid windowtext .5pt;mso-border-left-alt:solid windowtext .5pt;mso-border-alt:
  solid windowtext .5pt;padding:0cm 5.4pt 0cm 5.4pt;height:1.0pt'>
  <p class=MsoNormal align=center style='text-align:center;mso-pagination:widow-orphan'><span
  lang=EN-US style='font-size:6.5pt;mso-bidi-font-size:10.0pt;font-family:"Times New Roman",serif;
  mso-fareast-font-family:等线;mso-font-kerning:0pt'>10.33<o:p></o:p></span></p>
  </td>
  <td width=39 style='width:29.55pt;border-top:none;border-left:none;
  border-bottom:solid windowtext 1.0pt;border-right:solid windowtext 1.0pt;
  mso-border-top-alt:solid windowtext .5pt;mso-border-left-alt:solid windowtext .5pt;
  mso-border-alt:solid windowtext .5pt;padding:0cm 5.4pt 0cm 5.4pt;height:1.0pt'>
  <p class=MsoNormal align=center style='text-align:center;mso-pagination:widow-orphan'><span
  lang=EN-US style='font-size:6.5pt;mso-bidi-font-size:10.0pt;font-family:"Times New Roman",serif;
  mso-fareast-font-family:等线;mso-font-kerning:0pt'>9.349<o:p></o:p></span></p>
  </td>
  <td width=39 valign=top style='width:29.55pt;border-top:none;border-left:
  none;border-bottom:solid windowtext 1.0pt;border-right:solid windowtext 1.0pt;
  mso-border-top-alt:solid windowtext .5pt;mso-border-left-alt:solid windowtext .5pt;
  mso-border-alt:solid windowtext .5pt;padding:0cm 5.4pt 0cm 5.4pt;height:1.0pt'>
  <p class=MsoNormal align=center style='text-align:center;mso-pagination:widow-orphan'><span
  lang=EN-US style='font-size:6.5pt;mso-bidi-font-size:10.0pt;font-family:"Times New Roman",serif;
  mso-fareast-font-family:等线;mso-font-kerning:0pt'>9.678<o:p></o:p></span></p>
  </td>
 </tr>
 <tr style='mso-yfti-irow:4;height:1.0pt'>
  <td width=75 nowrap style='width:56.45pt;border:solid windowtext 1.0pt;
  border-top:none;mso-border-top-alt:solid windowtext .5pt;mso-border-alt:solid windowtext .5pt;
  padding:0cm 5.4pt 0cm 5.4pt;height:1.0pt'>
  <p class=MsoNormal align=center style='text-align:center;mso-pagination:widow-orphan'><span
  lang=EN-US style='font-size:6.5pt;mso-bidi-font-size:10.0pt;font-family:"Times New Roman",serif;
  mso-fareast-font-family:等线;mso-font-kerning:0pt'>MTFAA-LSA</span><span
  lang=EN-US style='font-size:7.5pt;mso-bidi-font-size:10.0pt;font-family:"Times New Roman",serif;
  mso-fareast-font-family:等线;mso-font-kerning:0pt'><o:p></o:p></span></p>
  </td>
  <td width=39 nowrap style='width:29.5pt;border-top:none;border-left:none;
  border-bottom:solid windowtext 1.0pt;border-right:solid windowtext 1.0pt;
  mso-border-top-alt:solid windowtext .5pt;mso-border-left-alt:solid windowtext .5pt;
  mso-border-alt:solid windowtext .5pt;padding:0cm 5.4pt 0cm 5.4pt;height:1.0pt'>
  <p class=MsoNormal align=center style='text-align:center;mso-pagination:widow-orphan'><b
  style='mso-bidi-font-weight:normal'><span lang=EN-US style='font-size:6.5pt;
  mso-bidi-font-size:10.0pt;font-family:"Times New Roman",serif;mso-fareast-font-family:
  等线;mso-font-kerning:0pt'>0.809<o:p></o:p></span></b></p>
  </td>
  <td width=39 nowrap style='width:29.55pt;border-top:none;border-left:none;
  border-bottom:solid windowtext 1.0pt;border-right:solid windowtext 1.0pt;
  mso-border-top-alt:solid windowtext .5pt;mso-border-left-alt:solid windowtext .5pt;
  mso-border-alt:solid windowtext .5pt;padding:0cm 5.4pt 0cm 5.4pt;height:1.0pt'>
  <p class=MsoNormal align=center style='text-align:center;mso-pagination:widow-orphan'><b
  style='mso-bidi-font-weight:normal'><span lang=EN-US style='font-size:6.5pt;
  mso-bidi-font-size:10.0pt;font-family:"Times New Roman",serif;mso-fareast-font-family:
  等线;mso-font-kerning:0pt'>0.881<o:p></o:p></span></b></p>
  </td>
  <td width=39 nowrap style='width:29.55pt;border-top:none;border-left:none;
  border-bottom:solid windowtext 1.0pt;border-right:solid windowtext 1.0pt;
  mso-border-top-alt:solid windowtext .5pt;mso-border-left-alt:solid windowtext .5pt;
  mso-border-alt:solid windowtext .5pt;padding:0cm 5.4pt 0cm 5.4pt;height:1.0pt'>
  <p class=MsoNormal align=center style='text-align:center;mso-pagination:widow-orphan'><b
  style='mso-bidi-font-weight:normal'><span lang=EN-US style='font-size:6.5pt;
  mso-bidi-font-size:10.0pt;font-family:"Times New Roman",serif;mso-fareast-font-family:
  等线;mso-font-kerning:0pt'>0.860<o:p></o:p></span></b></p>
  </td>
  <td width=39 nowrap valign=top style='width:29.5pt;border-top:none;
  border-left:none;border-bottom:solid windowtext 1.0pt;border-right:solid windowtext 1.0pt;
  mso-border-top-alt:solid windowtext .5pt;mso-border-left-alt:solid windowtext .5pt;
  mso-border-alt:solid windowtext .5pt;padding:0cm 5.4pt 0cm 5.4pt;height:1.0pt'>
  <p class=MsoNormal align=center style='text-align:center;mso-pagination:widow-orphan'><b
  style='mso-bidi-font-weight:normal'><span lang=EN-US style='font-size:6.5pt;
  mso-bidi-font-size:10.0pt;font-family:"Times New Roman",serif;mso-fareast-font-family:
  等线;mso-font-kerning:0pt'>10.34<o:p></o:p></span></b></p>
  </td>
  <td width=39 nowrap valign=top style='width:29.55pt;border-top:none;
  border-left:none;border-bottom:solid windowtext 1.0pt;border-right:solid windowtext 1.0pt;
  mso-border-top-alt:solid windowtext .5pt;mso-border-left-alt:solid windowtext .5pt;
  mso-border-alt:solid windowtext .5pt;padding:0cm 5.4pt 0cm 5.4pt;height:1.0pt'>
  <p class=MsoNormal align=center style='text-align:center;mso-pagination:widow-orphan'><b
  style='mso-bidi-font-weight:normal'><span lang=EN-US style='font-size:6.5pt;
  mso-bidi-font-size:10.0pt;font-family:"Times New Roman",serif;mso-fareast-font-family:
  等线;mso-font-kerning:0pt'>16.20<o:p></o:p></span></b></p>
  </td>
  <td width=39 nowrap valign=top style='width:29.55pt;border-top:none;
  border-left:none;border-bottom:solid windowtext 1.0pt;border-right:solid windowtext 1.0pt;
  mso-border-top-alt:solid windowtext .5pt;mso-border-left-alt:solid windowtext .5pt;
  mso-border-alt:solid windowtext .5pt;padding:0cm 5.4pt 0cm 5.4pt;height:1.0pt'>
  <p class=MsoNormal align=center style='text-align:center;mso-pagination:widow-orphan'><b
  style='mso-bidi-font-weight:normal'><span lang=EN-US style='font-size:6.5pt;
  mso-bidi-font-size:10.0pt;font-family:"Times New Roman",serif;mso-fareast-font-family:
  等线;mso-font-kerning:0pt'>14.63</span></b><span lang=EN-US style='font-size:
  6.5pt;mso-bidi-font-size:10.0pt;font-family:"Times New Roman",serif;
  mso-fareast-font-family:等线;mso-font-kerning:0pt'><o:p></o:p></span></p>
  </td>
  <td width=79 nowrap colspan=2 style='width:59.1pt;border-top:none;border-left:
  none;border-bottom:solid windowtext 1.0pt;border-right:solid windowtext 1.0pt;
  mso-border-top-alt:solid windowtext .5pt;mso-border-left-alt:solid windowtext .5pt;
  mso-border-alt:solid windowtext .5pt;padding:0cm 5.4pt 0cm 5.4pt;height:1.0pt'>
  <p class=MsoNormal align=center style='text-align:center;mso-pagination:widow-orphan'><span
  lang=EN-US style='font-size:6.5pt;mso-bidi-font-size:10.0pt;font-family:"Times New Roman",serif;
  mso-fareast-font-family:等线;mso-font-kerning:0pt'>MTFAA-LSA<b
  style='mso-bidi-font-weight:normal'><o:p></o:p></b></span></p>
  </td>
  <td width=39 style='width:29.5pt;border-top:none;border-left:none;border-bottom:
  solid windowtext 1.0pt;border-right:solid windowtext 1.0pt;mso-border-top-alt:
  solid windowtext .5pt;mso-border-left-alt:solid windowtext .5pt;mso-border-alt:
  solid windowtext .5pt;padding:0cm 5.4pt 0cm 5.4pt;height:1.0pt'>
  <p class=MsoNormal align=center style='text-align:center;mso-pagination:widow-orphan'><b
  style='mso-bidi-font-weight:normal'><span lang=EN-US style='font-size:6.5pt;
  mso-bidi-font-size:10.0pt;font-family:"Times New Roman",serif;mso-fareast-font-family:
  等线;mso-font-kerning:0pt'>9.840<o:p></o:p></span></b></p>
  </td>
  <td width=39 style='width:29.55pt;border-top:none;border-left:none;
  border-bottom:solid windowtext 1.0pt;border-right:solid windowtext 1.0pt;
  mso-border-top-alt:solid windowtext .5pt;mso-border-left-alt:solid windowtext .5pt;
  mso-border-alt:solid windowtext .5pt;padding:0cm 5.4pt 0cm 5.4pt;height:1.0pt'>
  <p class=MsoNormal align=center style='text-align:center;mso-pagination:widow-orphan'><b
  style='mso-bidi-font-weight:normal'><span lang=EN-US style='font-size:6.5pt;
  mso-bidi-font-size:10.0pt;font-family:"Times New Roman",serif;mso-fareast-font-family:
  等线;mso-font-kerning:0pt'>8.636<o:p></o:p></span></b></p>
  </td>
  <td width=39 valign=top style='width:29.55pt;border-top:none;border-left:
  none;border-bottom:solid windowtext 1.0pt;border-right:solid windowtext 1.0pt;
  mso-border-top-alt:solid windowtext .5pt;mso-border-left-alt:solid windowtext .5pt;
  mso-border-alt:solid windowtext .5pt;padding:0cm 5.4pt 0cm 5.4pt;height:1.0pt'>
  <p class=MsoNormal align=center style='text-align:center;mso-pagination:widow-orphan'><b
  style='mso-bidi-font-weight:normal'><span lang=EN-US style='font-size:6.5pt;
  mso-bidi-font-size:10.0pt;font-family:"Times New Roman",serif;mso-fareast-font-family:
  等线;mso-font-kerning:0pt'>9.037<o:p></o:p></span></b></p>
  </td>
 </tr>
 <tr style='mso-yfti-irow:5;height:1.0pt'>
  <td width=75 nowrap style='width:56.45pt;border:solid windowtext 1.0pt;
  border-top:none;mso-border-top-alt:solid windowtext .5pt;mso-border-alt:solid windowtext .5pt;
  padding:0cm 5.4pt 0cm 5.4pt;height:1.0pt'>
  <p class=MsoNormal align=center style='text-align:center;mso-pagination:widow-orphan'><span
  lang=EN-US style='font-size:6.5pt;mso-bidi-font-size:10.0pt;font-family:"Times New Roman",serif;
  mso-fareast-font-family:等线;mso-font-kerning:0pt'>DPARN<o:p></o:p></span></p>
  </td>
  <td width=39 nowrap style='width:29.5pt;border-top:none;border-left:none;
  border-bottom:solid windowtext 1.0pt;border-right:solid windowtext 1.0pt;
  mso-border-top-alt:solid windowtext .5pt;mso-border-left-alt:solid windowtext .5pt;
  mso-border-alt:solid windowtext .5pt;padding:0cm 5.4pt 0cm 5.4pt;height:1.0pt'>
  <p class=MsoNormal align=center style='text-align:center;mso-pagination:widow-orphan'><span
  lang=EN-US style='font-size:6.5pt;mso-bidi-font-size:10.0pt;font-family:"Times New Roman",serif;
  mso-fareast-font-family:等线;mso-font-kerning:0pt'>0.752<o:p></o:p></span></p>
  </td>
  <td width=39 nowrap style='width:29.55pt;border-top:none;border-left:none;
  border-bottom:solid windowtext 1.0pt;border-right:solid windowtext 1.0pt;
  mso-border-top-alt:solid windowtext .5pt;mso-border-left-alt:solid windowtext .5pt;
  mso-border-alt:solid windowtext .5pt;padding:0cm 5.4pt 0cm 5.4pt;height:1.0pt'>
  <p class=MsoNormal align=center style='text-align:center;mso-pagination:widow-orphan'><span
  lang=EN-US style='font-size:6.5pt;mso-bidi-font-size:10.0pt;font-family:"Times New Roman",serif;
  mso-fareast-font-family:等线;mso-font-kerning:0pt'>0.858<o:p></o:p></span></p>
  </td>
  <td width=39 nowrap style='width:29.55pt;border-top:none;border-left:none;
  border-bottom:solid windowtext 1.0pt;border-right:solid windowtext 1.0pt;
  mso-border-top-alt:solid windowtext .5pt;mso-border-left-alt:solid windowtext .5pt;
  mso-border-alt:solid windowtext .5pt;padding:0cm 5.4pt 0cm 5.4pt;height:1.0pt'>
  <p class=MsoNormal align=center style='text-align:center;mso-pagination:widow-orphan'><span
  lang=EN-US style='font-size:6.5pt;mso-bidi-font-size:10.0pt;font-family:"Times New Roman",serif;
  mso-fareast-font-family:等线;mso-font-kerning:0pt'>0.828<o:p></o:p></span></p>
  </td>
  <td width=39 nowrap valign=top style='width:29.5pt;border-top:none;
  border-left:none;border-bottom:solid windowtext 1.0pt;border-right:solid windowtext 1.0pt;
  mso-border-top-alt:solid windowtext .5pt;mso-border-left-alt:solid windowtext .5pt;
  mso-border-alt:solid windowtext .5pt;padding:0cm 5.4pt 0cm 5.4pt;height:1.0pt'>
  <p class=MsoNormal align=center style='text-align:center;mso-pagination:widow-orphan'><span
  lang=EN-US style='font-size:6.5pt;mso-bidi-font-size:10.0pt;font-family:"Times New Roman",serif;
  mso-fareast-font-family:等线;mso-font-kerning:0pt'>8.461<o:p></o:p></span></p>
  </td>
  <td width=39 nowrap valign=top style='width:29.55pt;border-top:none;
  border-left:none;border-bottom:solid windowtext 1.0pt;border-right:solid windowtext 1.0pt;
  mso-border-top-alt:solid windowtext .5pt;mso-border-left-alt:solid windowtext .5pt;
  mso-border-alt:solid windowtext .5pt;padding:0cm 5.4pt 0cm 5.4pt;height:1.0pt'>
  <p class=MsoNormal align=center style='text-align:center;mso-pagination:widow-orphan'><span
  lang=EN-US style='font-size:6.5pt;mso-bidi-font-size:10.0pt;font-family:"Times New Roman",serif;
  mso-fareast-font-family:等线;mso-font-kerning:0pt'>13.71<o:p></o:p></span></p>
  </td>
  <td width=39 nowrap valign=top style='width:29.55pt;border-top:none;
  border-left:none;border-bottom:solid windowtext 1.0pt;border-right:solid windowtext 1.0pt;
  mso-border-top-alt:solid windowtext .5pt;mso-border-left-alt:solid windowtext .5pt;
  mso-border-alt:solid windowtext .5pt;padding:0cm 5.4pt 0cm 5.4pt;height:1.0pt'>
  <p class=MsoNormal align=center style='text-align:center;mso-pagination:widow-orphan'><span
  lang=EN-US style='font-size:6.5pt;mso-bidi-font-size:10.0pt;font-family:"Times New Roman",serif;
  mso-fareast-font-family:等线;mso-font-kerning:0pt'>12.31<o:p></o:p></span></p>
  </td>
  <td width=79 nowrap colspan=2 style='width:59.1pt;border-top:none;border-left:
  none;border-bottom:solid windowtext 1.0pt;border-right:solid windowtext 1.0pt;
  mso-border-top-alt:solid windowtext .5pt;mso-border-left-alt:solid windowtext .5pt;
  mso-border-alt:solid windowtext .5pt;padding:0cm 5.4pt 0cm 5.4pt;height:1.0pt'>
  <p class=MsoNormal align=center style='text-align:center;mso-pagination:widow-orphan'><span
  lang=EN-US style='font-size:6.5pt;mso-bidi-font-size:10.0pt;font-family:"Times New Roman",serif;
  mso-fareast-font-family:等线;mso-font-kerning:0pt'>DPARN<o:p></o:p></span></p>
  </td>
  <td width=39 style='width:29.5pt;border-top:none;border-left:none;border-bottom:
  solid windowtext 1.0pt;border-right:solid windowtext 1.0pt;mso-border-top-alt:
  solid windowtext .5pt;mso-border-left-alt:solid windowtext .5pt;mso-border-alt:
  solid windowtext .5pt;padding:0cm 5.4pt 0cm 5.4pt;height:1.0pt'>
  <p class=MsoNormal align=center style='text-align:center;mso-pagination:widow-orphan'><span
  lang=EN-US style='font-size:6.5pt;mso-bidi-font-size:10.0pt;font-family:"Times New Roman",serif;
  mso-fareast-font-family:等线;mso-font-kerning:0pt'>10.92<o:p></o:p></span></p>
  </td>
  <td width=39 style='width:29.55pt;border-top:none;border-left:none;
  border-bottom:solid windowtext 1.0pt;border-right:solid windowtext 1.0pt;
  mso-border-top-alt:solid windowtext .5pt;mso-border-left-alt:solid windowtext .5pt;
  mso-border-alt:solid windowtext .5pt;padding:0cm 5.4pt 0cm 5.4pt;height:1.0pt'>
  <p class=MsoNormal align=center style='text-align:center;mso-pagination:widow-orphan'><span
  lang=EN-US style='font-size:6.5pt;mso-bidi-font-size:10.0pt;font-family:"Times New Roman",serif;
  mso-fareast-font-family:等线;mso-font-kerning:0pt'>13.11<o:p></o:p></span></p>
  </td>
  <td width=39 valign=top style='width:29.55pt;border-top:none;border-left:
  none;border-bottom:solid windowtext 1.0pt;border-right:solid windowtext 1.0pt;
  mso-border-top-alt:solid windowtext .5pt;mso-border-left-alt:solid windowtext .5pt;
  mso-border-alt:solid windowtext .5pt;padding:0cm 5.4pt 0cm 5.4pt;height:1.0pt'>
  <p class=MsoNormal align=center style='text-align:center;mso-pagination:widow-orphan'><span
  lang=EN-US style='font-size:6.5pt;mso-bidi-font-size:10.0pt;font-family:"Times New Roman",serif;
  mso-fareast-font-family:等线;mso-font-kerning:0pt'>12.38<o:p></o:p></span></p>
  </td>
 </tr>
 <tr style='mso-yfti-irow:6;height:1.0pt'>
  <td width=75 nowrap style='width:56.45pt;border:solid windowtext 1.0pt;
  border-top:none;mso-border-top-alt:solid windowtext .5pt;mso-border-alt:solid windowtext .5pt;
  padding:0cm 5.4pt 0cm 5.4pt;height:1.0pt'>
  <p class=MsoNormal align=center style='text-align:center;mso-pagination:widow-orphan'><span
  lang=EN-US style='font-size:6.5pt;mso-bidi-font-size:10.0pt;font-family:"Times New Roman",serif;
  mso-fareast-font-family:等线;mso-font-kerning:0pt'>DPARN-LSA<o:p></o:p></span></p>
  </td>
  <td width=39 nowrap style='width:29.5pt;border-top:none;border-left:none;
  border-bottom:solid windowtext 1.0pt;border-right:solid windowtext 1.0pt;
  mso-border-top-alt:solid windowtext .5pt;mso-border-left-alt:solid windowtext .5pt;
  mso-border-alt:solid windowtext .5pt;padding:0cm 5.4pt 0cm 5.4pt;height:1.0pt'>
  <p class=MsoNormal align=center style='text-align:center;mso-pagination:widow-orphan'><b
  style='mso-bidi-font-weight:normal'><span lang=EN-US style='font-size:6.5pt;
  mso-bidi-font-size:10.0pt;font-family:"Times New Roman",serif;mso-fareast-font-family:
  等线;mso-font-kerning:0pt'>0.757<o:p></o:p></span></b></p>
  </td>
  <td width=39 nowrap style='width:29.55pt;border-top:none;border-left:none;
  border-bottom:solid windowtext 1.0pt;border-right:solid windowtext 1.0pt;
  mso-border-top-alt:solid windowtext .5pt;mso-border-left-alt:solid windowtext .5pt;
  mso-border-alt:solid windowtext .5pt;padding:0cm 5.4pt 0cm 5.4pt;height:1.0pt'>
  <p class=MsoNormal align=center style='text-align:center;mso-pagination:widow-orphan'><b
  style='mso-bidi-font-weight:normal'><span lang=EN-US style='font-size:6.5pt;
  mso-bidi-font-size:10.0pt;font-family:"Times New Roman",serif;mso-fareast-font-family:
  等线;mso-font-kerning:0pt'>0.861<o:p></o:p></span></b></p>
  </td>
  <td width=39 nowrap style='width:29.55pt;border-top:none;border-left:none;
  border-bottom:solid windowtext 1.0pt;border-right:solid windowtext 1.0pt;
  mso-border-top-alt:solid windowtext .5pt;mso-border-left-alt:solid windowtext .5pt;
  mso-border-alt:solid windowtext .5pt;padding:0cm 5.4pt 0cm 5.4pt;height:1.0pt'>
  <p class=MsoNormal align=center style='text-align:center;mso-pagination:widow-orphan'><b
  style='mso-bidi-font-weight:normal'><span lang=EN-US style='font-size:6.5pt;
  mso-bidi-font-size:10.0pt;font-family:"Times New Roman",serif;mso-fareast-font-family:
  等线;mso-font-kerning:0pt'>0.831<o:p></o:p></span></b></p>
  </td>
  <td width=39 nowrap valign=top style='width:29.5pt;border-top:none;
  border-left:none;border-bottom:solid windowtext 1.0pt;border-right:solid windowtext 1.0pt;
  mso-border-top-alt:solid windowtext .5pt;mso-border-left-alt:solid windowtext .5pt;
  mso-border-alt:solid windowtext .5pt;padding:0cm 5.4pt 0cm 5.4pt;height:1.0pt'>
  <p class=MsoNormal align=center style='text-align:center;mso-pagination:widow-orphan'><b
  style='mso-bidi-font-weight:normal'><span lang=EN-US style='font-size:6.5pt;
  mso-bidi-font-size:10.0pt;font-family:"Times New Roman",serif;mso-fareast-font-family:
  等线;mso-font-kerning:0pt'>8.617<o:p></o:p></span></b></p>
  </td>
  <td width=39 nowrap valign=top style='width:29.55pt;border-top:none;
  border-left:none;border-bottom:solid windowtext 1.0pt;border-right:solid windowtext 1.0pt;
  mso-border-top-alt:solid windowtext .5pt;mso-border-left-alt:solid windowtext .5pt;
  mso-border-alt:solid windowtext .5pt;padding:0cm 5.4pt 0cm 5.4pt;height:1.0pt'>
  <p class=MsoNormal align=center style='text-align:center;mso-pagination:widow-orphan'><b
  style='mso-bidi-font-weight:normal'><span lang=EN-US style='font-size:6.5pt;
  mso-bidi-font-size:10.0pt;font-family:"Times New Roman",serif;mso-fareast-font-family:
  等线;mso-font-kerning:0pt'>13.84<o:p></o:p></span></b></p>
  </td>
  <td width=39 nowrap valign=top style='width:29.55pt;border-top:none;
  border-left:none;border-bottom:solid windowtext 1.0pt;border-right:solid windowtext 1.0pt;
  mso-border-top-alt:solid windowtext .5pt;mso-border-left-alt:solid windowtext .5pt;
  mso-border-alt:solid windowtext .5pt;padding:0cm 5.4pt 0cm 5.4pt;height:1.0pt'>
  <p class=MsoNormal align=center style='text-align:center;mso-pagination:widow-orphan'><b
  style='mso-bidi-font-weight:normal'><span lang=EN-US style='font-size:6.5pt;
  mso-bidi-font-size:10.0pt;font-family:"Times New Roman",serif;mso-fareast-font-family:
  等线;mso-font-kerning:0pt'>12.47<o:p></o:p></span></b></p>
  </td>
  <td width=79 nowrap colspan=2 style='width:59.1pt;border-top:none;border-left:
  none;border-bottom:solid windowtext 1.0pt;border-right:solid windowtext 1.0pt;
  mso-border-top-alt:solid windowtext .5pt;mso-border-left-alt:solid windowtext .5pt;
  mso-border-alt:solid windowtext .5pt;padding:0cm 5.4pt 0cm 5.4pt;height:1.0pt'>
  <p class=MsoNormal align=center style='text-align:center;mso-pagination:widow-orphan'><span
  lang=EN-US style='font-size:6.5pt;mso-bidi-font-size:10.0pt;font-family:"Times New Roman",serif;
  mso-fareast-font-family:等线;mso-font-kerning:0pt'>DPARN-LSA<o:p></o:p></span></p>
  </td>
  <td width=39 style='width:29.5pt;border-top:none;border-left:none;border-bottom:
  solid windowtext 1.0pt;border-right:solid windowtext 1.0pt;mso-border-top-alt:
  solid windowtext .5pt;mso-border-left-alt:solid windowtext .5pt;mso-border-alt:
  solid windowtext .5pt;padding:0cm 5.4pt 0cm 5.4pt;height:1.0pt'>
  <p class=MsoNormal align=center style='text-align:center;mso-pagination:widow-orphan'><b
  style='mso-bidi-font-weight:normal'><span lang=EN-US style='font-size:6.5pt;
  mso-bidi-font-size:10.0pt;font-family:"Times New Roman",serif;mso-fareast-font-family:
  等线;mso-font-kerning:0pt'>10.76<o:p></o:p></span></b></p>
  </td>
  <td width=39 style='width:29.55pt;border-top:none;border-left:none;
  border-bottom:solid windowtext 1.0pt;border-right:solid windowtext 1.0pt;
  mso-border-top-alt:solid windowtext .5pt;mso-border-left-alt:solid windowtext .5pt;
  mso-border-alt:solid windowtext .5pt;padding:0cm 5.4pt 0cm 5.4pt;height:1.0pt'>
  <p class=MsoNormal align=center style='text-align:center;mso-pagination:widow-orphan'><b
  style='mso-bidi-font-weight:normal'><span lang=EN-US style='font-size:6.5pt;
  mso-bidi-font-size:10.0pt;font-family:"Times New Roman",serif;mso-fareast-font-family:
  等线;mso-font-kerning:0pt'>12.99<o:p></o:p></span></b></p>
  </td>
  <td width=39 valign=top style='width:29.55pt;border-top:none;border-left:
  none;border-bottom:solid windowtext 1.0pt;border-right:solid windowtext 1.0pt;
  mso-border-top-alt:solid windowtext .5pt;mso-border-left-alt:solid windowtext .5pt;
  mso-border-alt:solid windowtext .5pt;padding:0cm 5.4pt 0cm 5.4pt;height:1.0pt'>
  <p class=MsoNormal align=center style='text-align:center;mso-pagination:widow-orphan'><b
  style='mso-bidi-font-weight:normal'><span lang=EN-US style='font-size:6.5pt;
  mso-bidi-font-size:10.0pt;font-family:"Times New Roman",serif;mso-fareast-font-family:
  等线;mso-font-kerning:0pt'>12.25<o:p></o:p></span></b></p>
  </td>
 </tr>
 <tr style='mso-yfti-irow:7;height:15.0pt'>
  <td width=548 nowrap colspan=13 style='width:410.85pt;border:solid windowtext 1.0pt;
  border-top:none;mso-border-top-alt:solid windowtext .5pt;mso-border-alt:solid windowtext .5pt;
  padding:0cm 5.4pt 0cm 5.4pt;height:15.0pt'></td>
 </tr>
 <tr style='mso-yfti-irow:8;height:15.0pt'>
  <td width=75 nowrap style='width:56.45pt;border:solid windowtext 1.0pt;
  border-top:none;mso-border-top-alt:solid windowtext .5pt;mso-border-alt:solid windowtext .5pt;
  padding:0cm 5.4pt 0cm 5.4pt;height:15.0pt'>
  <p class=MsoNormal align=center style='text-align:center;mso-pagination:widow-orphan'><span
  lang=EN-US style='font-size:5.5pt;mso-bidi-font-size:10.0pt;font-family:"Times New Roman",serif;
  mso-fareast-font-family:等线;mso-font-kerning:0pt'>Wideband Metrics</span><span
  lang=EN-US style='font-size:7.5pt;mso-bidi-font-size:10.0pt;font-family:"Times New Roman",serif;
  mso-fareast-font-family:等线;mso-font-kerning:0pt'><o:p></o:p></span></p>
  </td>
  <td width=118 nowrap colspan=3 style='width:88.6pt;border-top:none;
  border-left:none;border-bottom:solid windowtext 1.0pt;border-right:solid windowtext 1.0pt;
  mso-border-top-alt:solid windowtext .5pt;mso-border-left-alt:solid windowtext .5pt;
  mso-border-alt:solid windowtext .5pt;padding:0cm 5.4pt 0cm 5.4pt;height:15.0pt'>
  <p class=MsoNormal align=center style='text-align:center;mso-pagination:widow-orphan'><span
  lang=EN-US style='font-size:7.5pt;mso-bidi-font-size:10.0pt;font-family:"Times New Roman",serif;
  mso-fareast-font-family:等线;mso-font-kerning:0pt'>PESQ<o:p></o:p></span></p>
  </td>
  <td width=118 nowrap colspan=3 style='width:88.6pt;border-top:none;
  border-left:none;border-bottom:solid windowtext 1.0pt;border-right:solid windowtext 1.0pt;
  mso-border-top-alt:solid windowtext .5pt;mso-border-left-alt:solid windowtext .5pt;
  mso-border-alt:solid windowtext .5pt;padding:0cm 5.4pt 0cm 5.4pt;height:15.0pt'>
  <p class=MsoNormal align=center style='text-align:center;mso-pagination:widow-orphan'><span
  lang=EN-US style='font-size:7.5pt;mso-bidi-font-size:10.0pt;font-family:"Times New Roman",serif;
  mso-fareast-font-family:等线;mso-font-kerning:0pt'>CSIG<o:p></o:p></span></p>
  </td>
  <td width=118 nowrap colspan=3 style='width:88.6pt;border-top:none;
  border-left:none;border-bottom:solid windowtext 1.0pt;border-right:solid windowtext 1.0pt;
  mso-border-top-alt:solid windowtext .5pt;mso-border-left-alt:solid windowtext .5pt;
  mso-border-alt:solid windowtext .5pt;padding:0cm 5.4pt 0cm 5.4pt;height:15.0pt'>
  <p class=MsoNormal align=center style='text-align:center;mso-pagination:widow-orphan'><span
  lang=EN-US style='font-size:7.5pt;mso-bidi-font-size:10.0pt;font-family:"Times New Roman",serif;
  mso-fareast-font-family:等线;mso-font-kerning:0pt'>CBAK<o:p></o:p></span></p>
  </td>
  <td width=118 colspan=3 valign=top style='width:88.6pt;border-top:none;
  border-left:none;border-bottom:solid windowtext 1.0pt;border-right:solid windowtext 1.0pt;
  mso-border-top-alt:solid windowtext .5pt;mso-border-left-alt:solid windowtext .5pt;
  mso-border-alt:solid windowtext .5pt;padding:0cm 5.4pt 0cm 5.4pt;height:15.0pt'>
  <p class=MsoNormal align=center style='text-align:center;mso-pagination:widow-orphan'><span
  lang=EN-US style='font-size:7.5pt;mso-bidi-font-size:10.0pt;font-family:"Times New Roman",serif;
  mso-fareast-font-family:等线;mso-font-kerning:0pt'>COVL<o:p></o:p></span></p>
  </td>
 </tr>
 <tr style='mso-yfti-irow:9;height:15.0pt'>
  <td width=75 nowrap style='width:56.45pt;border:solid windowtext 1.0pt;
  border-top:none;mso-border-top-alt:solid windowtext .5pt;mso-border-alt:solid windowtext .5pt;
  padding:0cm 5.4pt 0cm 5.4pt;height:15.0pt'>
  <p class=MsoNormal align=center style='text-align:center;mso-pagination:widow-orphan'><span
  class=GramE><span lang=EN-US style='font-size:6.5pt;mso-bidi-font-size:10.0pt;
  font-family:"Times New Roman",serif;mso-fareast-font-family:等线;mso-font-kerning:
  0pt'>SNR(</span></span><span lang=EN-US style='font-size:6.5pt;mso-bidi-font-size:
  10.0pt;font-family:"Times New Roman",serif;mso-fareast-font-family:等线;
  mso-font-kerning:0pt'>dB)<o:p></o:p></span></p>
  </td>
  <td width=39 nowrap style='width:29.5pt;border-top:none;border-left:none;
  border-bottom:solid windowtext 1.0pt;border-right:solid windowtext 1.0pt;
  mso-border-top-alt:solid windowtext .5pt;mso-border-left-alt:solid windowtext .5pt;
  mso-border-alt:solid windowtext .5pt;padding:0cm 5.4pt 0cm 5.4pt;height:15.0pt'>
  <p class=MsoNormal align=center style='text-align:center;mso-pagination:widow-orphan'><span
  lang=EN-US style='font-size:7.5pt;font-family:"Times New Roman",serif;
  mso-fareast-font-family:等线;mso-font-kerning:0pt'>-5~0<o:p></o:p></span></p>
  </td>
  <td width=39 nowrap style='width:29.55pt;border-top:none;border-left:none;
  border-bottom:solid windowtext 1.0pt;border-right:solid windowtext 1.0pt;
  mso-border-top-alt:solid windowtext .5pt;mso-border-left-alt:solid windowtext .5pt;
  mso-border-alt:solid windowtext .5pt;padding:0cm 5.4pt 0cm 5.4pt;height:15.0pt'>
  <p class=MsoNormal align=center style='text-align:center;mso-pagination:widow-orphan'><span
  lang=EN-US style='font-size:7.5pt;font-family:"Times New Roman",serif;
  mso-fareast-font-family:等线;mso-font-kerning:0pt'>0~15<o:p></o:p></span></p>
  </td>
  <td width=39 nowrap style='width:29.55pt;border-top:none;border-left:none;
  border-bottom:solid windowtext 1.0pt;border-right:solid windowtext 1.0pt;
  mso-border-top-alt:solid windowtext .5pt;mso-border-left-alt:solid windowtext .5pt;
  mso-border-alt:solid windowtext .5pt;padding:0cm 5.4pt 0cm 5.4pt;height:15.0pt'>
  <p class=MsoNormal align=center style='text-align:center;mso-pagination:widow-orphan'><span
  class=SpellE><span lang=EN-US style='font-size:7.5pt;font-family:"Times New Roman",serif;
  mso-fareast-font-family:等线;mso-font-kerning:0pt'>Ovrl</span></span><span
  lang=EN-US style='font-size:7.5pt;font-family:"Times New Roman",serif;
  mso-fareast-font-family:等线;mso-font-kerning:0pt'>.<o:p></o:p></span></p>
  </td>
  <td width=39 nowrap style='width:29.5pt;border-top:none;border-left:none;
  border-bottom:solid windowtext 1.0pt;border-right:solid windowtext 1.0pt;
  mso-border-top-alt:solid windowtext .5pt;mso-border-left-alt:solid windowtext .5pt;
  mso-border-alt:solid windowtext .5pt;padding:0cm 5.4pt 0cm 5.4pt;height:15.0pt'>
  <p class=MsoNormal align=center style='text-align:center;mso-pagination:widow-orphan'><span
  lang=EN-US style='font-size:7.5pt;font-family:"Times New Roman",serif;
  mso-fareast-font-family:等线;mso-font-kerning:0pt'>-5~0<o:p></o:p></span></p>
  </td>
  <td width=39 nowrap style='width:29.55pt;border-top:none;border-left:none;
  border-bottom:solid windowtext 1.0pt;border-right:solid windowtext 1.0pt;
  mso-border-top-alt:solid windowtext .5pt;mso-border-left-alt:solid windowtext .5pt;
  mso-border-alt:solid windowtext .5pt;padding:0cm 5.4pt 0cm 5.4pt;height:15.0pt'>
  <p class=MsoNormal align=center style='text-align:center;mso-pagination:widow-orphan'><span
  lang=EN-US style='font-size:7.5pt;font-family:"Times New Roman",serif;
  mso-fareast-font-family:等线;mso-font-kerning:0pt'>0~15<o:p></o:p></span></p>
  </td>
  <td width=39 nowrap style='width:29.55pt;border-top:none;border-left:none;
  border-bottom:solid windowtext 1.0pt;border-right:solid windowtext 1.0pt;
  mso-border-top-alt:solid windowtext .5pt;mso-border-left-alt:solid windowtext .5pt;
  mso-border-alt:solid windowtext .5pt;padding:0cm 5.4pt 0cm 5.4pt;height:15.0pt'>
  <p class=MsoNormal align=center style='text-align:center;mso-pagination:widow-orphan'><span
  class=SpellE><span lang=EN-US style='font-size:7.5pt;font-family:"Times New Roman",serif;
  mso-fareast-font-family:等线;mso-font-kerning:0pt'>Ovrl</span></span><span
  lang=EN-US style='font-size:7.5pt;font-family:"Times New Roman",serif;
  mso-fareast-font-family:等线;mso-font-kerning:0pt'>.<o:p></o:p></span></p>
  </td>
  <td width=39 nowrap style='width:29.5pt;border-top:none;border-left:none;
  border-bottom:solid windowtext 1.0pt;border-right:solid windowtext 1.0pt;
  mso-border-top-alt:solid windowtext .5pt;mso-border-left-alt:solid windowtext .5pt;
  mso-border-alt:solid windowtext .5pt;padding:0cm 5.4pt 0cm 5.4pt;height:15.0pt'>
  <p class=MsoNormal align=center style='text-align:center;mso-pagination:widow-orphan'><span
  lang=EN-US style='font-size:7.5pt;font-family:"Times New Roman",serif;
  mso-fareast-font-family:等线;mso-font-kerning:0pt'>-5~0<o:p></o:p></span></p>
  </td>
  <td width=39 nowrap style='width:29.55pt;border-top:none;border-left:none;
  border-bottom:solid windowtext 1.0pt;border-right:solid windowtext 1.0pt;
  mso-border-top-alt:solid windowtext .5pt;mso-border-left-alt:solid windowtext .5pt;
  mso-border-alt:solid windowtext .5pt;padding:0cm 5.4pt 0cm 5.4pt;height:15.0pt'>
  <p class=MsoNormal align=center style='text-align:center;mso-pagination:widow-orphan'><span
  lang=EN-US style='font-size:7.5pt;font-family:"Times New Roman",serif;
  mso-fareast-font-family:等线;mso-font-kerning:0pt'>0~15<o:p></o:p></span></p>
  </td>
  <td width=39 nowrap style='width:29.55pt;border-top:none;border-left:none;
  border-bottom:solid windowtext 1.0pt;border-right:solid windowtext 1.0pt;
  mso-border-top-alt:solid windowtext .5pt;mso-border-left-alt:solid windowtext .5pt;
  mso-border-alt:solid windowtext .5pt;padding:0cm 5.4pt 0cm 5.4pt;height:15.0pt'>
  <p class=MsoNormal align=center style='text-align:center;mso-pagination:widow-orphan'><span
  class=SpellE><span lang=EN-US style='font-size:7.5pt;font-family:"Times New Roman",serif;
  mso-fareast-font-family:等线;mso-font-kerning:0pt'>Ovrl</span></span><span
  lang=EN-US style='font-size:7.5pt;font-family:"Times New Roman",serif;
  mso-fareast-font-family:等线;mso-font-kerning:0pt'>.<o:p></o:p></span></p>
  </td>
  <td width=39 style='width:29.5pt;border-top:none;border-left:none;border-bottom:
  solid windowtext 1.0pt;border-right:solid windowtext 1.0pt;mso-border-top-alt:
  solid windowtext .5pt;mso-border-left-alt:solid windowtext .5pt;mso-border-alt:
  solid windowtext .5pt;padding:0cm 5.4pt 0cm 5.4pt;height:15.0pt'>
  <p class=MsoNormal align=center style='text-align:center;mso-pagination:widow-orphan'><span
  lang=EN-US style='font-size:7.5pt;font-family:"Times New Roman",serif;
  mso-fareast-font-family:等线;mso-font-kerning:0pt'>-5~0<o:p></o:p></span></p>
  </td>
  <td width=39 style='width:29.55pt;border-top:none;border-left:none;
  border-bottom:solid windowtext 1.0pt;border-right:solid windowtext 1.0pt;
  mso-border-top-alt:solid windowtext .5pt;mso-border-left-alt:solid windowtext .5pt;
  mso-border-alt:solid windowtext .5pt;padding:0cm 5.4pt 0cm 5.4pt;height:15.0pt'>
  <p class=MsoNormal align=center style='text-align:center;mso-pagination:widow-orphan'><span
  lang=EN-US style='font-size:7.5pt;font-family:"Times New Roman",serif;
  mso-fareast-font-family:等线;mso-font-kerning:0pt'>0~15<o:p></o:p></span></p>
  </td>
  <td width=39 style='width:29.55pt;border-top:none;border-left:none;
  border-bottom:solid windowtext 1.0pt;border-right:solid windowtext 1.0pt;
  mso-border-top-alt:solid windowtext .5pt;mso-border-left-alt:solid windowtext .5pt;
  mso-border-alt:solid windowtext .5pt;padding:0cm 5.4pt 0cm 5.4pt;height:15.0pt'>
  <p class=MsoNormal align=center style='text-align:center;mso-pagination:widow-orphan'><span
  class=SpellE><span lang=EN-US style='font-size:7.5pt;font-family:"Times New Roman",serif;
  mso-fareast-font-family:等线;mso-font-kerning:0pt'>Ovrl</span></span><span
  lang=EN-US style='font-size:7.5pt;font-family:"Times New Roman",serif;
  mso-fareast-font-family:等线;mso-font-kerning:0pt'>.<o:p></o:p></span></p>
  </td>
 </tr>
 <tr style='mso-yfti-irow:10;height:15.6pt'>
  <td width=75 nowrap style='width:56.45pt;border:solid windowtext 1.0pt;
  border-top:none;mso-border-top-alt:solid windowtext .5pt;mso-border-alt:solid windowtext .5pt;
  padding:0cm 5.4pt 0cm 5.4pt;height:15.6pt'>
  <p class=MsoNormal align=center style='text-align:center;mso-pagination:widow-orphan'><span
  lang=EN-US style='font-size:6.5pt;mso-bidi-font-size:10.0pt;font-family:"Times New Roman",serif;
  mso-fareast-font-family:等线;mso-font-kerning:0pt'>Noisy<o:p></o:p></span></p>
  </td>
  <td width=39 nowrap style='width:29.5pt;border-top:none;border-left:none;
  border-bottom:solid windowtext 1.0pt;border-right:solid windowtext 1.0pt;
  mso-border-top-alt:solid windowtext .5pt;mso-border-left-alt:solid windowtext .5pt;
  mso-border-alt:solid windowtext .5pt;padding:0cm 5.4pt 0cm 5.4pt;height:15.6pt'>
  <p class=MsoNormal align=center style='text-align:center;mso-pagination:widow-orphan'><span
  lang=EN-US style='font-size:6.5pt;mso-bidi-font-size:10.0pt;font-family:"Times New Roman",serif;
  mso-fareast-font-family:等线;mso-font-kerning:0pt'>1.160</span><span
  lang=EN-US style='font-size:6.5pt;font-family:"Times New Roman",serif;
  mso-fareast-font-family:等线;mso-font-kerning:0pt'><o:p></o:p></span></p>
  </td>
  <td width=39 nowrap style='width:29.55pt;border-top:none;border-left:none;
  border-bottom:solid windowtext 1.0pt;border-right:solid windowtext 1.0pt;
  mso-border-top-alt:solid windowtext .5pt;mso-border-left-alt:solid windowtext .5pt;
  mso-border-alt:solid windowtext .5pt;padding:0cm 5.4pt 0cm 5.4pt;height:15.6pt'>
  <p class=MsoNormal align=center style='text-align:center;mso-pagination:widow-orphan'><span
  lang=EN-US style='font-size:6.5pt;mso-bidi-font-size:10.0pt;font-family:"Times New Roman",serif;
  mso-fareast-font-family:等线;mso-font-kerning:0pt'>1.446</span><span
  lang=EN-US style='font-size:6.5pt;font-family:"Times New Roman",serif;
  mso-fareast-font-family:等线;mso-font-kerning:0pt'><o:p></o:p></span></p>
  </td>
  <td width=39 nowrap style='width:29.55pt;border-top:none;border-left:none;
  border-bottom:solid windowtext 1.0pt;border-right:solid windowtext 1.0pt;
  mso-border-top-alt:solid windowtext .5pt;mso-border-left-alt:solid windowtext .5pt;
  mso-border-alt:solid windowtext .5pt;padding:0cm 5.4pt 0cm 5.4pt;height:15.6pt'>
  <p class=MsoNormal align=center style='text-align:center;mso-pagination:widow-orphan'><span
  lang=EN-US style='font-size:6.5pt;mso-bidi-font-size:10.0pt;font-family:"Times New Roman",serif;
  mso-fareast-font-family:等线;mso-font-kerning:0pt'>1.364</span><span
  lang=EN-US style='font-size:6.5pt;font-family:"Times New Roman",serif;
  mso-fareast-font-family:等线;mso-font-kerning:0pt'><o:p></o:p></span></p>
  </td>
  <td width=39 nowrap style='width:29.5pt;border-top:none;border-left:none;
  border-bottom:solid windowtext 1.0pt;border-right:solid windowtext 1.0pt;
  mso-border-top-alt:solid windowtext .5pt;mso-border-left-alt:solid windowtext .5pt;
  mso-border-alt:solid windowtext .5pt;padding:0cm 5.4pt 0cm 5.4pt;height:15.6pt'>
  <p class=MsoNormal align=center style='text-align:center;mso-pagination:widow-orphan'><span
  lang=EN-US style='font-size:6.5pt;font-family:"Times New Roman",serif;
  mso-fareast-font-family:等线;mso-font-kerning:0pt'>2.023<o:p></o:p></span></p>
  </td>
  <td width=39 nowrap style='width:29.55pt;border-top:none;border-left:none;
  border-bottom:solid windowtext 1.0pt;border-right:solid windowtext 1.0pt;
  mso-border-top-alt:solid windowtext .5pt;mso-border-left-alt:solid windowtext .5pt;
  mso-border-alt:solid windowtext .5pt;padding:0cm 5.4pt 0cm 5.4pt;height:15.6pt'>
  <p class=MsoNormal align=center style='text-align:center;mso-pagination:widow-orphan'><span
  lang=EN-US style='font-size:6.5pt;font-family:"Times New Roman",serif;
  mso-fareast-font-family:等线;mso-font-kerning:0pt'>2.719<o:p></o:p></span></p>
  </td>
  <td width=39 nowrap style='width:29.55pt;border-top:none;border-left:none;
  border-bottom:solid windowtext 1.0pt;border-right:solid windowtext 1.0pt;
  mso-border-top-alt:solid windowtext .5pt;mso-border-left-alt:solid windowtext .5pt;
  mso-border-alt:solid windowtext .5pt;padding:0cm 5.4pt 0cm 5.4pt;height:15.6pt'>
  <p class=MsoNormal align=center style='text-align:center;mso-pagination:widow-orphan'><span
  lang=EN-US style='font-size:6.5pt;font-family:"Times New Roman",serif;
  mso-fareast-font-family:等线;mso-font-kerning:0pt'>2.517<o:p></o:p></span></p>
  </td>
  <td width=39 nowrap style='width:29.5pt;border-top:none;border-left:none;
  border-bottom:solid windowtext 1.0pt;border-right:solid windowtext 1.0pt;
  mso-border-top-alt:solid windowtext .5pt;mso-border-left-alt:solid windowtext .5pt;
  mso-border-alt:solid windowtext .5pt;padding:0cm 5.4pt 0cm 5.4pt;height:15.6pt'>
  <p class=MsoNormal align=center style='text-align:center;mso-pagination:widow-orphan'><span
  lang=EN-US style='font-size:6.5pt;font-family:"Times New Roman",serif;
  mso-fareast-font-family:等线;mso-font-kerning:0pt'>1.833<o:p></o:p></span></p>
  </td>
  <td width=39 nowrap style='width:29.55pt;border-top:none;border-left:none;
  border-bottom:solid windowtext 1.0pt;border-right:solid windowtext 1.0pt;
  mso-border-top-alt:solid windowtext .5pt;mso-border-left-alt:solid windowtext .5pt;
  mso-border-alt:solid windowtext .5pt;padding:0cm 5.4pt 0cm 5.4pt;height:15.6pt'>
  <p class=MsoNormal align=center style='text-align:center;mso-pagination:widow-orphan'><span
  lang=EN-US style='font-size:6.5pt;font-family:"Times New Roman",serif;
  mso-fareast-font-family:等线;mso-font-kerning:0pt'>2.481<o:p></o:p></span></p>
  </td>
  <td width=39 nowrap style='width:29.55pt;border-top:none;border-left:none;
  border-bottom:solid windowtext 1.0pt;border-right:solid windowtext 1.0pt;
  mso-border-top-alt:solid windowtext .5pt;mso-border-left-alt:solid windowtext .5pt;
  mso-border-alt:solid windowtext .5pt;padding:0cm 5.4pt 0cm 5.4pt;height:15.6pt'>
  <p class=MsoNormal align=center style='text-align:center;mso-pagination:widow-orphan'><span
  lang=EN-US style='font-size:6.5pt;font-family:"Times New Roman",serif;
  mso-fareast-font-family:等线;mso-font-kerning:0pt'>2.293<o:p></o:p></span></p>
  </td>
  <td width=39 style='width:29.5pt;border-top:none;border-left:none;border-bottom:
  solid windowtext 1.0pt;border-right:solid windowtext 1.0pt;mso-border-top-alt:
  solid windowtext .5pt;mso-border-left-alt:solid windowtext .5pt;mso-border-alt:
  solid windowtext .5pt;padding:0cm 5.4pt 0cm 5.4pt;height:15.6pt'>
  <p class=MsoNormal align=center style='text-align:center;mso-pagination:widow-orphan'><span
  lang=EN-US style='font-size:6.5pt;font-family:"Times New Roman",serif;
  mso-fareast-font-family:等线;mso-font-kerning:0pt'>1.571<o:p></o:p></span></p>
  </td>
  <td width=39 style='width:29.55pt;border-top:none;border-left:none;
  border-bottom:solid windowtext 1.0pt;border-right:solid windowtext 1.0pt;
  mso-border-top-alt:solid windowtext .5pt;mso-border-left-alt:solid windowtext .5pt;
  mso-border-alt:solid windowtext .5pt;padding:0cm 5.4pt 0cm 5.4pt;height:15.6pt'>
  <p class=MsoNormal align=center style='text-align:center;mso-pagination:widow-orphan'><span
  lang=EN-US style='font-size:6.5pt;font-family:"Times New Roman",serif;
  mso-fareast-font-family:等线;mso-font-kerning:0pt'>2.095<o:p></o:p></span></p>
  </td>
  <td width=39 style='width:29.55pt;border-top:none;border-left:none;
  border-bottom:solid windowtext 1.0pt;border-right:solid windowtext 1.0pt;
  mso-border-top-alt:solid windowtext .5pt;mso-border-left-alt:solid windowtext .5pt;
  mso-border-alt:solid windowtext .5pt;padding:0cm 5.4pt 0cm 5.4pt;height:15.6pt'>
  <p class=MsoNormal align=center style='text-align:center;mso-pagination:widow-orphan'><span
  lang=EN-US style='font-size:6.5pt;font-family:"Times New Roman",serif;
  mso-fareast-font-family:等线;mso-font-kerning:0pt'>1.943<o:p></o:p></span></p>
  </td>
 </tr>
 <tr style='mso-yfti-irow:11;height:15.6pt'>
  <td width=75 nowrap style='width:56.45pt;border:solid windowtext 1.0pt;
  border-top:none;mso-border-top-alt:solid windowtext .5pt;mso-border-alt:solid windowtext .5pt;
  padding:0cm 5.4pt 0cm 5.4pt;height:15.6pt'>
  <p class=MsoNormal align=center style='text-align:center;mso-pagination:widow-orphan'><span
  lang=EN-US style='font-size:6.5pt;mso-bidi-font-size:10.0pt;font-family:"Times New Roman",serif;
  mso-fareast-font-family:等线;mso-font-kerning:0pt'>MTFAA<o:p></o:p></span></p>
  </td>
  <td width=39 nowrap style='width:29.5pt;border-top:none;border-left:none;
  border-bottom:solid windowtext 1.0pt;border-right:solid windowtext 1.0pt;
  mso-border-top-alt:solid windowtext .5pt;mso-border-left-alt:solid windowtext .5pt;
  mso-border-alt:solid windowtext .5pt;padding:0cm 5.4pt 0cm 5.4pt;height:15.6pt'>
  <p class=MsoNormal align=center style='text-align:center;mso-pagination:widow-orphan'><span
  lang=EN-US style='font-size:6.5pt;mso-bidi-font-size:10.0pt;font-family:"Times New Roman",serif;
  mso-fareast-font-family:等线;mso-font-kerning:0pt'>1.981</span><span
  lang=EN-US style='font-size:6.5pt;font-family:"Times New Roman",serif;
  mso-fareast-font-family:等线;mso-font-kerning:0pt'><o:p></o:p></span></p>
  </td>
  <td width=39 nowrap style='width:29.55pt;border-top:none;border-left:none;
  border-bottom:solid windowtext 1.0pt;border-right:solid windowtext 1.0pt;
  mso-border-top-alt:solid windowtext .5pt;mso-border-left-alt:solid windowtext .5pt;
  mso-border-alt:solid windowtext .5pt;padding:0cm 5.4pt 0cm 5.4pt;height:15.6pt'>
  <p class=MsoNormal align=center style='text-align:center;mso-pagination:widow-orphan'><span
  lang=EN-US style='font-size:6.5pt;mso-bidi-font-size:10.0pt;font-family:"Times New Roman",serif;
  mso-fareast-font-family:等线;mso-font-kerning:0pt'>2.669</span><span
  lang=EN-US style='font-size:6.5pt;font-family:"Times New Roman",serif;
  mso-fareast-font-family:等线;mso-font-kerning:0pt'><o:p></o:p></span></p>
  </td>
  <td width=39 nowrap style='width:29.55pt;border-top:none;border-left:none;
  border-bottom:solid windowtext 1.0pt;border-right:solid windowtext 1.0pt;
  mso-border-top-alt:solid windowtext .5pt;mso-border-left-alt:solid windowtext .5pt;
  mso-border-alt:solid windowtext .5pt;padding:0cm 5.4pt 0cm 5.4pt;height:15.6pt'>
  <p class=MsoNormal align=center style='text-align:center;mso-pagination:widow-orphan'><span
  lang=EN-US style='font-size:6.5pt;mso-bidi-font-size:10.0pt;font-family:"Times New Roman",serif;
  mso-fareast-font-family:等线;mso-font-kerning:0pt'>2.470</span><span
  lang=EN-US style='font-size:6.5pt;font-family:"Times New Roman",serif;
  mso-fareast-font-family:等线;mso-font-kerning:0pt'><o:p></o:p></span></p>
  </td>
  <td width=39 nowrap style='width:29.5pt;border-top:none;border-left:none;
  border-bottom:solid windowtext 1.0pt;border-right:solid windowtext 1.0pt;
  mso-border-top-alt:solid windowtext .5pt;mso-border-left-alt:solid windowtext .5pt;
  mso-border-alt:solid windowtext .5pt;padding:0cm 5.4pt 0cm 5.4pt;height:15.6pt'>
  <p class=MsoNormal align=center style='text-align:center;mso-pagination:widow-orphan'><span
  lang=EN-US style='font-size:6.5pt;font-family:"Times New Roman",serif;
  mso-fareast-font-family:等线;mso-font-kerning:0pt'>3.465<o:p></o:p></span></p>
  </td>
  <td width=39 nowrap style='width:29.55pt;border-top:none;border-left:none;
  border-bottom:solid windowtext 1.0pt;border-right:solid windowtext 1.0pt;
  mso-border-top-alt:solid windowtext .5pt;mso-border-left-alt:solid windowtext .5pt;
  mso-border-alt:solid windowtext .5pt;padding:0cm 5.4pt 0cm 5.4pt;height:15.6pt'>
  <p class=MsoNormal align=center style='text-align:center;mso-pagination:widow-orphan'><span
  lang=EN-US style='font-size:6.5pt;font-family:"Times New Roman",serif;
  mso-fareast-font-family:等线;mso-font-kerning:0pt'>4.113<o:p></o:p></span></p>
  </td>
  <td width=39 nowrap style='width:29.55pt;border-top:none;border-left:none;
  border-bottom:solid windowtext 1.0pt;border-right:solid windowtext 1.0pt;
  mso-border-top-alt:solid windowtext .5pt;mso-border-left-alt:solid windowtext .5pt;
  mso-border-alt:solid windowtext .5pt;padding:0cm 5.4pt 0cm 5.4pt;height:15.6pt'>
  <p class=MsoNormal align=center style='text-align:center;mso-pagination:widow-orphan'><span
  lang=EN-US style='font-size:6.5pt;font-family:"Times New Roman",serif;
  mso-fareast-font-family:等线;mso-font-kerning:0pt'>3.925<o:p></o:p></span></p>
  </td>
  <td width=39 nowrap style='width:29.5pt;border-top:none;border-left:none;
  border-bottom:solid windowtext 1.0pt;border-right:solid windowtext 1.0pt;
  mso-border-top-alt:solid windowtext .5pt;mso-border-left-alt:solid windowtext .5pt;
  mso-border-alt:solid windowtext .5pt;padding:0cm 5.4pt 0cm 5.4pt;height:15.6pt'>
  <p class=MsoNormal align=center style='text-align:center;mso-pagination:widow-orphan'><span
  lang=EN-US style='font-size:6.5pt;font-family:"Times New Roman",serif;
  mso-fareast-font-family:等线;mso-font-kerning:0pt'>2.951<o:p></o:p></span></p>
  </td>
  <td width=39 nowrap style='width:29.55pt;border-top:none;border-left:none;
  border-bottom:solid windowtext 1.0pt;border-right:solid windowtext 1.0pt;
  mso-border-top-alt:solid windowtext .5pt;mso-border-left-alt:solid windowtext .5pt;
  mso-border-alt:solid windowtext .5pt;padding:0cm 5.4pt 0cm 5.4pt;height:15.6pt'>
  <p class=MsoNormal align=center style='text-align:center;mso-pagination:widow-orphan'><span
  lang=EN-US style='font-size:6.5pt;font-family:"Times New Roman",serif;
  mso-fareast-font-family:等线;mso-font-kerning:0pt'>3.523<o:p></o:p></span></p>
  </td>
  <td width=39 nowrap style='width:29.55pt;border-top:none;border-left:none;
  border-bottom:solid windowtext 1.0pt;border-right:solid windowtext 1.0pt;
  mso-border-top-alt:solid windowtext .5pt;mso-border-left-alt:solid windowtext .5pt;
  mso-border-alt:solid windowtext .5pt;padding:0cm 5.4pt 0cm 5.4pt;height:15.6pt'>
  <p class=MsoNormal align=center style='text-align:center;mso-pagination:widow-orphan'><span
  lang=EN-US style='font-size:6.5pt;font-family:"Times New Roman",serif;
  mso-fareast-font-family:等线;mso-font-kerning:0pt'>3.357<o:p></o:p></span></p>
  </td>
  <td width=39 style='width:29.5pt;border-top:none;border-left:none;border-bottom:
  solid windowtext 1.0pt;border-right:solid windowtext 1.0pt;mso-border-top-alt:
  solid windowtext .5pt;mso-border-left-alt:solid windowtext .5pt;mso-border-alt:
  solid windowtext .5pt;padding:0cm 5.4pt 0cm 5.4pt;height:15.6pt'>
  <p class=MsoNormal align=center style='text-align:center;mso-pagination:widow-orphan'><span
  lang=EN-US style='font-size:6.5pt;font-family:"Times New Roman",serif;
  mso-fareast-font-family:等线;mso-font-kerning:0pt'>2.754<o:p></o:p></span></p>
  </td>
  <td width=39 style='width:29.55pt;border-top:none;border-left:none;
  border-bottom:solid windowtext 1.0pt;border-right:solid windowtext 1.0pt;
  mso-border-top-alt:solid windowtext .5pt;mso-border-left-alt:solid windowtext .5pt;
  mso-border-alt:solid windowtext .5pt;padding:0cm 5.4pt 0cm 5.4pt;height:15.6pt'>
  <p class=MsoNormal align=center style='text-align:center;mso-pagination:widow-orphan'><span
  lang=EN-US style='font-size:6.5pt;font-family:"Times New Roman",serif;
  mso-fareast-font-family:等线;mso-font-kerning:0pt'>3.436<o:p></o:p></span></p>
  </td>
  <td width=39 style='width:29.55pt;border-top:none;border-left:none;
  border-bottom:solid windowtext 1.0pt;border-right:solid windowtext 1.0pt;
  mso-border-top-alt:solid windowtext .5pt;mso-border-left-alt:solid windowtext .5pt;
  mso-border-alt:solid windowtext .5pt;padding:0cm 5.4pt 0cm 5.4pt;height:15.6pt'>
  <p class=MsoNormal align=center style='text-align:center;mso-pagination:widow-orphan'><span
  lang=EN-US style='font-size:6.5pt;font-family:"Times New Roman",serif;
  mso-fareast-font-family:等线;mso-font-kerning:0pt'>3.238<o:p></o:p></span></p>
  </td>
 </tr>
 <tr style='mso-yfti-irow:12;height:15.6pt'>
  <td width=75 nowrap style='width:56.45pt;border:solid windowtext 1.0pt;
  border-top:none;mso-border-top-alt:solid windowtext .5pt;mso-border-alt:solid windowtext .5pt;
  padding:0cm 5.4pt 0cm 5.4pt;height:15.6pt'>
  <p class=MsoNormal align=center style='text-align:center;mso-pagination:widow-orphan'><span
  lang=EN-US style='font-size:6.5pt;mso-bidi-font-size:10.0pt;font-family:"Times New Roman",serif;
  mso-fareast-font-family:等线;mso-font-kerning:0pt'>MTFAA-LSA<o:p></o:p></span></p>
  </td>
  <td width=39 nowrap style='width:29.5pt;border-top:none;border-left:none;
  border-bottom:solid windowtext 1.0pt;border-right:solid windowtext 1.0pt;
  mso-border-top-alt:solid windowtext .5pt;mso-border-left-alt:solid windowtext .5pt;
  mso-border-alt:solid windowtext .5pt;padding:0cm 5.4pt 0cm 5.4pt;height:15.6pt'>
  <p class=MsoNormal align=center style='text-align:center;mso-pagination:widow-orphan'><b
  style='mso-bidi-font-weight:normal'><span lang=EN-US style='font-size:6.5pt;
  mso-bidi-font-size:10.0pt;font-family:"Times New Roman",serif;mso-fareast-font-family:
  等线;mso-font-kerning:0pt'>2.084</span></b><b style='mso-bidi-font-weight:normal'><span
  lang=EN-US style='font-size:6.5pt;font-family:"Times New Roman",serif;
  mso-fareast-font-family:等线;mso-font-kerning:0pt'><o:p></o:p></span></b></p>
  </td>
  <td width=39 nowrap style='width:29.55pt;border-top:none;border-left:none;
  border-bottom:solid windowtext 1.0pt;border-right:solid windowtext 1.0pt;
  mso-border-top-alt:solid windowtext .5pt;mso-border-left-alt:solid windowtext .5pt;
  mso-border-alt:solid windowtext .5pt;padding:0cm 5.4pt 0cm 5.4pt;height:15.6pt'>
  <p class=MsoNormal align=center style='text-align:center;mso-pagination:widow-orphan'><b
  style='mso-bidi-font-weight:normal'><span lang=EN-US style='font-size:6.5pt;
  mso-bidi-font-size:10.0pt;font-family:"Times New Roman",serif;mso-fareast-font-family:
  等线;mso-font-kerning:0pt'>2.795</span></b><b style='mso-bidi-font-weight:normal'><span
  lang=EN-US style='font-size:6.5pt;font-family:"Times New Roman",serif;
  mso-fareast-font-family:等线;mso-font-kerning:0pt'><o:p></o:p></span></b></p>
  </td>
  <td width=39 nowrap style='width:29.55pt;border-top:none;border-left:none;
  border-bottom:solid windowtext 1.0pt;border-right:solid windowtext 1.0pt;
  mso-border-top-alt:solid windowtext .5pt;mso-border-left-alt:solid windowtext .5pt;
  mso-border-alt:solid windowtext .5pt;padding:0cm 5.4pt 0cm 5.4pt;height:15.6pt'>
  <p class=MsoNormal align=center style='text-align:center;mso-pagination:widow-orphan'><b
  style='mso-bidi-font-weight:normal'><span lang=EN-US style='font-size:6.5pt;
  mso-bidi-font-size:10.0pt;font-family:"Times New Roman",serif;mso-fareast-font-family:
  等线;mso-font-kerning:0pt'>2.589</span></b><b style='mso-bidi-font-weight:normal'><span
  lang=EN-US style='font-size:6.5pt;font-family:"Times New Roman",serif;
  mso-fareast-font-family:等线;mso-font-kerning:0pt'><o:p></o:p></span></b></p>
  </td>
  <td width=39 nowrap style='width:29.5pt;border-top:none;border-left:none;
  border-bottom:solid windowtext 1.0pt;border-right:solid windowtext 1.0pt;
  mso-border-top-alt:solid windowtext .5pt;mso-border-left-alt:solid windowtext .5pt;
  mso-border-alt:solid windowtext .5pt;padding:0cm 5.4pt 0cm 5.4pt;height:15.6pt'>
  <p class=MsoNormal align=center style='text-align:center;mso-pagination:widow-orphan'><b
  style='mso-bidi-font-weight:normal'><span lang=EN-US style='font-size:6.5pt;
  font-family:"Times New Roman",serif;mso-fareast-font-family:等线;mso-font-kerning:
  0pt'>3.517<o:p></o:p></span></b></p>
  </td>
  <td width=39 nowrap style='width:29.55pt;border-top:none;border-left:none;
  border-bottom:solid windowtext 1.0pt;border-right:solid windowtext 1.0pt;
  mso-border-top-alt:solid windowtext .5pt;mso-border-left-alt:solid windowtext .5pt;
  mso-border-alt:solid windowtext .5pt;padding:0cm 5.4pt 0cm 5.4pt;height:15.6pt'>
  <p class=MsoNormal align=center style='text-align:center;mso-pagination:widow-orphan'><b
  style='mso-bidi-font-weight:normal'><span lang=EN-US style='font-size:6.5pt;
  font-family:"Times New Roman",serif;mso-fareast-font-family:等线;mso-font-kerning:
  0pt'>4.203<o:p></o:p></span></b></p>
  </td>
  <td width=39 nowrap style='width:29.55pt;border-top:none;border-left:none;
  border-bottom:solid windowtext 1.0pt;border-right:solid windowtext 1.0pt;
  mso-border-top-alt:solid windowtext .5pt;mso-border-left-alt:solid windowtext .5pt;
  mso-border-alt:solid windowtext .5pt;padding:0cm 5.4pt 0cm 5.4pt;height:15.6pt'>
  <p class=MsoNormal align=center style='text-align:center;mso-pagination:widow-orphan'><b
  style='mso-bidi-font-weight:normal'><span lang=EN-US style='font-size:6.5pt;
  font-family:"Times New Roman",serif;mso-fareast-font-family:等线;mso-font-kerning:
  0pt'>4.004<o:p></o:p></span></b></p>
  </td>
  <td width=39 nowrap style='width:29.5pt;border-top:none;border-left:none;
  border-bottom:solid windowtext 1.0pt;border-right:solid windowtext 1.0pt;
  mso-border-top-alt:solid windowtext .5pt;mso-border-left-alt:solid windowtext .5pt;
  mso-border-alt:solid windowtext .5pt;padding:0cm 5.4pt 0cm 5.4pt;height:15.6pt'>
  <p class=MsoNormal align=center style='text-align:center;mso-pagination:widow-orphan'><b
  style='mso-bidi-font-weight:normal'><span lang=EN-US style='font-size:6.5pt;
  font-family:"Times New Roman",serif;mso-fareast-font-family:等线;mso-font-kerning:
  0pt'>3.006<o:p></o:p></span></b></p>
  </td>
  <td width=39 nowrap style='width:29.55pt;border-top:none;border-left:none;
  border-bottom:solid windowtext 1.0pt;border-right:solid windowtext 1.0pt;
  mso-border-top-alt:solid windowtext .5pt;mso-border-left-alt:solid windowtext .5pt;
  mso-border-alt:solid windowtext .5pt;padding:0cm 5.4pt 0cm 5.4pt;height:15.6pt'>
  <p class=MsoNormal align=center style='text-align:center;mso-pagination:widow-orphan'><b
  style='mso-bidi-font-weight:normal'><span lang=EN-US style='font-size:6.5pt;
  font-family:"Times New Roman",serif;mso-fareast-font-family:等线;mso-font-kerning:
  0pt'>3.593<o:p></o:p></span></b></p>
  </td>
  <td width=39 nowrap style='width:29.55pt;border-top:none;border-left:none;
  border-bottom:solid windowtext 1.0pt;border-right:solid windowtext 1.0pt;
  mso-border-top-alt:solid windowtext .5pt;mso-border-left-alt:solid windowtext .5pt;
  mso-border-alt:solid windowtext .5pt;padding:0cm 5.4pt 0cm 5.4pt;height:15.6pt'>
  <p class=MsoNormal align=center style='text-align:center;mso-pagination:widow-orphan'><b
  style='mso-bidi-font-weight:normal'><span lang=EN-US style='font-size:6.5pt;
  font-family:"Times New Roman",serif;mso-fareast-font-family:等线;mso-font-kerning:
  0pt'>3.423<o:p></o:p></span></b></p>
  </td>
  <td width=39 style='width:29.5pt;border-top:none;border-left:none;border-bottom:
  solid windowtext 1.0pt;border-right:solid windowtext 1.0pt;mso-border-top-alt:
  solid windowtext .5pt;mso-border-left-alt:solid windowtext .5pt;mso-border-alt:
  solid windowtext .5pt;padding:0cm 5.4pt 0cm 5.4pt;height:15.6pt'>
  <p class=MsoNormal align=center style='text-align:center;mso-pagination:widow-orphan'><b
  style='mso-bidi-font-weight:normal'><span lang=EN-US style='font-size:6.5pt;
  font-family:"Times New Roman",serif;mso-fareast-font-family:等线;mso-font-kerning:
  0pt'>2.829<o:p></o:p></span></b></p>
  </td>
  <td width=39 style='width:29.55pt;border-top:none;border-left:none;
  border-bottom:solid windowtext 1.0pt;border-right:solid windowtext 1.0pt;
  mso-border-top-alt:solid windowtext .5pt;mso-border-left-alt:solid windowtext .5pt;
  mso-border-alt:solid windowtext .5pt;padding:0cm 5.4pt 0cm 5.4pt;height:15.6pt'>
  <p class=MsoNormal align=center style='text-align:center;mso-pagination:widow-orphan'><b
  style='mso-bidi-font-weight:normal'><span lang=EN-US style='font-size:6.5pt;
  font-family:"Times New Roman",serif;mso-fareast-font-family:等线;mso-font-kerning:
  0pt'>3.547<o:p></o:p></span></b></p>
  </td>
  <td width=39 style='width:29.55pt;border-top:none;border-left:none;
  border-bottom:solid windowtext 1.0pt;border-right:solid windowtext 1.0pt;
  mso-border-top-alt:solid windowtext .5pt;mso-border-left-alt:solid windowtext .5pt;
  mso-border-alt:solid windowtext .5pt;padding:0cm 5.4pt 0cm 5.4pt;height:15.6pt'>
  <p class=MsoNormal align=center style='text-align:center;mso-pagination:widow-orphan'><b
  style='mso-bidi-font-weight:normal'><span lang=EN-US style='font-size:6.5pt;
  font-family:"Times New Roman",serif;mso-fareast-font-family:等线;mso-font-kerning:
  0pt'>3.339<o:p></o:p></span></b></p>
  </td>
 </tr>
 <tr style='mso-yfti-irow:13;height:15.6pt'>
  <td width=75 nowrap style='width:56.45pt;border:solid windowtext 1.0pt;
  border-top:none;mso-border-top-alt:solid windowtext .5pt;mso-border-alt:solid windowtext .5pt;
  padding:0cm 5.4pt 0cm 5.4pt;height:15.6pt'>
  <p class=MsoNormal align=center style='text-align:center;mso-pagination:widow-orphan'><span
  lang=EN-US style='font-size:6.5pt;mso-bidi-font-size:10.0pt;font-family:"Times New Roman",serif;
  mso-fareast-font-family:等线;mso-font-kerning:0pt'>DPARN<o:p></o:p></span></p>
  </td>
  <td width=39 nowrap style='width:29.5pt;border-top:none;border-left:none;
  border-bottom:solid windowtext 1.0pt;border-right:solid windowtext 1.0pt;
  mso-border-top-alt:solid windowtext .5pt;mso-border-left-alt:solid windowtext .5pt;
  mso-border-alt:solid windowtext .5pt;padding:0cm 5.4pt 0cm 5.4pt;height:15.6pt'>
  <p class=MsoNormal align=center style='text-align:center;mso-pagination:widow-orphan'><span
  lang=EN-US style='font-size:6.5pt;mso-bidi-font-size:10.0pt;font-family:"Times New Roman",serif;
  mso-fareast-font-family:等线;mso-font-kerning:0pt'>1.702<o:p></o:p></span></p>
  </td>
  <td width=39 nowrap style='width:29.55pt;border-top:none;border-left:none;
  border-bottom:solid windowtext 1.0pt;border-right:solid windowtext 1.0pt;
  mso-border-top-alt:solid windowtext .5pt;mso-border-left-alt:solid windowtext .5pt;
  mso-border-alt:solid windowtext .5pt;padding:0cm 5.4pt 0cm 5.4pt;height:15.6pt'>
  <p class=MsoNormal align=center style='text-align:center;mso-pagination:widow-orphan'><span
  lang=EN-US style='font-size:6.5pt;mso-bidi-font-size:10.0pt;font-family:"Times New Roman",serif;
  mso-fareast-font-family:等线;mso-font-kerning:0pt'>2.309<o:p></o:p></span></p>
  </td>
  <td width=39 nowrap style='width:29.55pt;border-top:none;border-left:none;
  border-bottom:solid windowtext 1.0pt;border-right:solid windowtext 1.0pt;
  mso-border-top-alt:solid windowtext .5pt;mso-border-left-alt:solid windowtext .5pt;
  mso-border-alt:solid windowtext .5pt;padding:0cm 5.4pt 0cm 5.4pt;height:15.6pt'>
  <p class=MsoNormal align=center style='text-align:center;mso-pagination:widow-orphan'><span
  lang=EN-US style='font-size:6.5pt;mso-bidi-font-size:10.0pt;font-family:"Times New Roman",serif;
  mso-fareast-font-family:等线;mso-font-kerning:0pt'>2.134<o:p></o:p></span></p>
  </td>
  <td width=39 nowrap style='width:29.5pt;border-top:none;border-left:none;
  border-bottom:solid windowtext 1.0pt;border-right:solid windowtext 1.0pt;
  mso-border-top-alt:solid windowtext .5pt;mso-border-left-alt:solid windowtext .5pt;
  mso-border-alt:solid windowtext .5pt;padding:0cm 5.4pt 0cm 5.4pt;height:15.6pt'>
  <p class=MsoNormal align=center style='text-align:center;mso-pagination:widow-orphan'><span
  lang=EN-US style='font-size:6.5pt;font-family:"Times New Roman",serif;
  mso-fareast-font-family:等线;mso-font-kerning:0pt'>3.136<o:p></o:p></span></p>
  </td>
  <td width=39 nowrap style='width:29.55pt;border-top:none;border-left:none;
  border-bottom:solid windowtext 1.0pt;border-right:solid windowtext 1.0pt;
  mso-border-top-alt:solid windowtext .5pt;mso-border-left-alt:solid windowtext .5pt;
  mso-border-alt:solid windowtext .5pt;padding:0cm 5.4pt 0cm 5.4pt;height:15.6pt'>
  <p class=MsoNormal align=center style='text-align:center;mso-pagination:widow-orphan'><span
  lang=EN-US style='font-size:6.5pt;font-family:"Times New Roman",serif;
  mso-fareast-font-family:等线;mso-font-kerning:0pt'>3.759<o:p></o:p></span></p>
  </td>
  <td width=39 nowrap style='width:29.55pt;border-top:none;border-left:none;
  border-bottom:solid windowtext 1.0pt;border-right:solid windowtext 1.0pt;
  mso-border-top-alt:solid windowtext .5pt;mso-border-left-alt:solid windowtext .5pt;
  mso-border-alt:solid windowtext .5pt;padding:0cm 5.4pt 0cm 5.4pt;height:15.6pt'>
  <p class=MsoNormal align=center style='text-align:center;mso-pagination:widow-orphan'><span
  lang=EN-US style='font-size:6.5pt;font-family:"Times New Roman",serif;
  mso-fareast-font-family:等线;mso-font-kerning:0pt'>3.580<o:p></o:p></span></p>
  </td>
  <td width=39 nowrap style='width:29.5pt;border-top:none;border-left:none;
  border-bottom:solid windowtext 1.0pt;border-right:solid windowtext 1.0pt;
  mso-border-top-alt:solid windowtext .5pt;mso-border-left-alt:solid windowtext .5pt;
  mso-border-alt:solid windowtext .5pt;padding:0cm 5.4pt 0cm 5.4pt;height:15.6pt'>
  <p class=MsoNormal align=center style='text-align:center;mso-pagination:widow-orphan'><span
  lang=EN-US style='font-size:6.5pt;font-family:"Times New Roman",serif;
  mso-fareast-font-family:等线;mso-font-kerning:0pt'>2.505<o:p></o:p></span></p>
  </td>
  <td width=39 nowrap style='width:29.55pt;border-top:none;border-left:none;
  border-bottom:solid windowtext 1.0pt;border-right:solid windowtext 1.0pt;
  mso-border-top-alt:solid windowtext .5pt;mso-border-left-alt:solid windowtext .5pt;
  mso-border-alt:solid windowtext .5pt;padding:0cm 5.4pt 0cm 5.4pt;height:15.6pt'>
  <p class=MsoNormal align=center style='text-align:center;mso-pagination:widow-orphan'><span
  lang=EN-US style='font-size:6.5pt;font-family:"Times New Roman",serif;
  mso-fareast-font-family:等线;mso-font-kerning:0pt'>2.859<o:p></o:p></span></p>
  </td>
  <td width=39 nowrap style='width:29.55pt;border-top:none;border-left:none;
  border-bottom:solid windowtext 1.0pt;border-right:solid windowtext 1.0pt;
  mso-border-top-alt:solid windowtext .5pt;mso-border-left-alt:solid windowtext .5pt;
  mso-border-alt:solid windowtext .5pt;padding:0cm 5.4pt 0cm 5.4pt;height:15.6pt'>
  <p class=MsoNormal align=center style='text-align:center;mso-pagination:widow-orphan'><span
  lang=EN-US style='font-size:6.5pt;font-family:"Times New Roman",serif;
  mso-fareast-font-family:等线;mso-font-kerning:0pt'>2.757<o:p></o:p></span></p>
  </td>
  <td width=39 style='width:29.5pt;border-top:none;border-left:none;border-bottom:
  solid windowtext 1.0pt;border-right:solid windowtext 1.0pt;mso-border-top-alt:
  solid windowtext .5pt;mso-border-left-alt:solid windowtext .5pt;mso-border-alt:
  solid windowtext .5pt;padding:0cm 5.4pt 0cm 5.4pt;height:15.6pt'>
  <p class=MsoNormal align=center style='text-align:center;mso-pagination:widow-orphan'><span
  lang=EN-US style='font-size:6.5pt;font-family:"Times New Roman",serif;
  mso-fareast-font-family:等线;mso-font-kerning:0pt'>2.447<o:p></o:p></span></p>
  </td>
  <td width=39 style='width:29.55pt;border-top:none;border-left:none;
  border-bottom:solid windowtext 1.0pt;border-right:solid windowtext 1.0pt;
  mso-border-top-alt:solid windowtext .5pt;mso-border-left-alt:solid windowtext .5pt;
  mso-border-alt:solid windowtext .5pt;padding:0cm 5.4pt 0cm 5.4pt;height:15.6pt'>
  <p class=MsoNormal align=center style='text-align:center;mso-pagination:widow-orphan'><span
  lang=EN-US style='font-size:6.5pt;font-family:"Times New Roman",serif;
  mso-fareast-font-family:等线;mso-font-kerning:0pt'>3.069<o:p></o:p></span></p>
  </td>
  <td width=39 style='width:29.55pt;border-top:none;border-left:none;
  border-bottom:solid windowtext 1.0pt;border-right:solid windowtext 1.0pt;
  mso-border-top-alt:solid windowtext .5pt;mso-border-left-alt:solid windowtext .5pt;
  mso-border-alt:solid windowtext .5pt;padding:0cm 5.4pt 0cm 5.4pt;height:15.6pt'>
  <p class=MsoNormal align=center style='text-align:center;mso-pagination:widow-orphan'><span
  lang=EN-US style='font-size:6.5pt;font-family:"Times New Roman",serif;
  mso-fareast-font-family:等线;mso-font-kerning:0pt'>2.890<o:p></o:p></span></p>
  </td>
 </tr>
 <tr style='mso-yfti-irow:14;mso-yfti-lastrow:yes;height:15.6pt'>
  <td width=75 nowrap style='width:56.45pt;border:solid windowtext 1.0pt;
  border-top:none;mso-border-top-alt:solid windowtext .5pt;mso-border-alt:solid windowtext .5pt;
  padding:0cm 5.4pt 0cm 5.4pt;height:15.6pt'>
  <p class=MsoNormal align=center style='text-align:center;mso-pagination:widow-orphan'><span
  lang=EN-US style='font-size:6.5pt;mso-bidi-font-size:10.0pt;font-family:"Times New Roman",serif;
  mso-fareast-font-family:等线;mso-font-kerning:0pt'>DPARN-LSA<o:p></o:p></span></p>
  </td>
  <td width=39 nowrap style='width:29.5pt;border-top:none;border-left:none;
  border-bottom:solid windowtext 1.0pt;border-right:solid windowtext 1.0pt;
  mso-border-top-alt:solid windowtext .5pt;mso-border-left-alt:solid windowtext .5pt;
  mso-border-alt:solid windowtext .5pt;padding:0cm 5.4pt 0cm 5.4pt;height:15.6pt'>
  <p class=MsoNormal align=center style='text-align:center;mso-pagination:widow-orphan'><b
  style='mso-bidi-font-weight:normal'><span lang=EN-US style='font-size:6.5pt;
  mso-bidi-font-size:10.0pt;font-family:"Times New Roman",serif;mso-fareast-font-family:
  等线;mso-font-kerning:0pt'>1.776<o:p></o:p></span></b></p>
  </td>
  <td width=39 nowrap style='width:29.55pt;border-top:none;border-left:none;
  border-bottom:solid windowtext 1.0pt;border-right:solid windowtext 1.0pt;
  mso-border-top-alt:solid windowtext .5pt;mso-border-left-alt:solid windowtext .5pt;
  mso-border-alt:solid windowtext .5pt;padding:0cm 5.4pt 0cm 5.4pt;height:15.6pt'>
  <p class=MsoNormal align=center style='text-align:center;mso-pagination:widow-orphan'><b
  style='mso-bidi-font-weight:normal'><span lang=EN-US style='font-size:6.5pt;
  mso-bidi-font-size:10.0pt;font-family:"Times New Roman",serif;mso-fareast-font-family:
  等线;mso-font-kerning:0pt'>2.423<o:p></o:p></span></b></p>
  </td>
  <td width=39 nowrap style='width:29.55pt;border-top:none;border-left:none;
  border-bottom:solid windowtext 1.0pt;border-right:solid windowtext 1.0pt;
  mso-border-top-alt:solid windowtext .5pt;mso-border-left-alt:solid windowtext .5pt;
  mso-border-alt:solid windowtext .5pt;padding:0cm 5.4pt 0cm 5.4pt;height:15.6pt'>
  <p class=MsoNormal align=center style='text-align:center;mso-pagination:widow-orphan'><b
  style='mso-bidi-font-weight:normal'><span lang=EN-US style='font-size:6.5pt;
  mso-bidi-font-size:10.0pt;font-family:"Times New Roman",serif;mso-fareast-font-family:
  等线;mso-font-kerning:0pt'>2.237<o:p></o:p></span></b></p>
  </td>
  <td width=39 nowrap style='width:29.5pt;border-top:none;border-left:none;
  border-bottom:solid windowtext 1.0pt;border-right:solid windowtext 1.0pt;
  mso-border-top-alt:solid windowtext .5pt;mso-border-left-alt:solid windowtext .5pt;
  mso-border-alt:solid windowtext .5pt;padding:0cm 5.4pt 0cm 5.4pt;height:15.6pt'>
  <p class=MsoNormal align=center style='text-align:center;mso-pagination:widow-orphan'><b
  style='mso-bidi-font-weight:normal'><span lang=EN-US style='font-size:6.5pt;
  font-family:"Times New Roman",serif;mso-fareast-font-family:等线;mso-font-kerning:
  0pt'>3.179<o:p></o:p></span></b></p>
  </td>
  <td width=39 nowrap style='width:29.55pt;border-top:none;border-left:none;
  border-bottom:solid windowtext 1.0pt;border-right:solid windowtext 1.0pt;
  mso-border-top-alt:solid windowtext .5pt;mso-border-left-alt:solid windowtext .5pt;
  mso-border-alt:solid windowtext .5pt;padding:0cm 5.4pt 0cm 5.4pt;height:15.6pt'>
  <p class=MsoNormal align=center style='text-align:center;mso-pagination:widow-orphan'><b
  style='mso-bidi-font-weight:normal'><span lang=EN-US style='font-size:6.5pt;
  font-family:"Times New Roman",serif;mso-fareast-font-family:等线;mso-font-kerning:
  0pt'>3.829<o:p></o:p></span></b></p>
  </td>
  <td width=39 nowrap style='width:29.55pt;border-top:none;border-left:none;
  border-bottom:solid windowtext 1.0pt;border-right:solid windowtext 1.0pt;
  mso-border-top-alt:solid windowtext .5pt;mso-border-left-alt:solid windowtext .5pt;
  mso-border-alt:solid windowtext .5pt;padding:0cm 5.4pt 0cm 5.4pt;height:15.6pt'>
  <p class=MsoNormal align=center style='text-align:center;mso-pagination:widow-orphan'><b
  style='mso-bidi-font-weight:normal'><span lang=EN-US style='font-size:6.5pt;
  font-family:"Times New Roman",serif;mso-fareast-font-family:等线;mso-font-kerning:
  0pt'>3.642<o:p></o:p></span></b></p>
  </td>
  <td width=39 nowrap style='width:29.5pt;border-top:none;border-left:none;
  border-bottom:solid windowtext 1.0pt;border-right:solid windowtext 1.0pt;
  mso-border-top-alt:solid windowtext .5pt;mso-border-left-alt:solid windowtext .5pt;
  mso-border-alt:solid windowtext .5pt;padding:0cm 5.4pt 0cm 5.4pt;height:15.6pt'>
  <p class=MsoNormal align=center style='text-align:center;mso-pagination:widow-orphan'><b
  style='mso-bidi-font-weight:normal'><span lang=EN-US style='font-size:6.5pt;
  font-family:"Times New Roman",serif;mso-fareast-font-family:等线;mso-font-kerning:
  0pt'>2.619<o:p></o:p></span></b></p>
  </td>
  <td width=39 nowrap style='width:29.55pt;border-top:none;border-left:none;
  border-bottom:solid windowtext 1.0pt;border-right:solid windowtext 1.0pt;
  mso-border-top-alt:solid windowtext .5pt;mso-border-left-alt:solid windowtext .5pt;
  mso-border-alt:solid windowtext .5pt;padding:0cm 5.4pt 0cm 5.4pt;height:15.6pt'>
  <p class=MsoNormal align=center style='text-align:center;mso-pagination:widow-orphan'><b
  style='mso-bidi-font-weight:normal'><span lang=EN-US style='font-size:6.5pt;
  font-family:"Times New Roman",serif;mso-fareast-font-family:等线;mso-font-kerning:
  0pt'>3.030<o:p></o:p></span></b></p>
  </td>
  <td width=39 nowrap style='width:29.55pt;border-top:none;border-left:none;
  border-bottom:solid windowtext 1.0pt;border-right:solid windowtext 1.0pt;
  mso-border-top-alt:solid windowtext .5pt;mso-border-left-alt:solid windowtext .5pt;
  mso-border-alt:solid windowtext .5pt;padding:0cm 5.4pt 0cm 5.4pt;height:15.6pt'>
  <p class=MsoNormal align=center style='text-align:center;mso-pagination:widow-orphan'><b
  style='mso-bidi-font-weight:normal'><span lang=EN-US style='font-size:6.5pt;
  font-family:"Times New Roman",serif;mso-fareast-font-family:等线;mso-font-kerning:
  0pt'>2.912<o:p></o:p></span></b></p>
  </td>
  <td width=39 style='width:29.5pt;border-top:none;border-left:none;border-bottom:
  solid windowtext 1.0pt;border-right:solid windowtext 1.0pt;mso-border-top-alt:
  solid windowtext .5pt;mso-border-left-alt:solid windowtext .5pt;mso-border-alt:
  solid windowtext .5pt;padding:0cm 5.4pt 0cm 5.4pt;height:15.6pt'>
  <p class=MsoNormal align=center style='text-align:center;mso-pagination:widow-orphan'><b
  style='mso-bidi-font-weight:normal'><span lang=EN-US style='font-size:6.5pt;
  font-family:"Times New Roman",serif;mso-fareast-font-family:等线;mso-font-kerning:
  0pt'>2.507<o:p></o:p></span></b></p>
  </td>
  <td width=39 style='width:29.55pt;border-top:none;border-left:none;
  border-bottom:solid windowtext 1.0pt;border-right:solid windowtext 1.0pt;
  mso-border-top-alt:solid windowtext .5pt;mso-border-left-alt:solid windowtext .5pt;
  mso-border-alt:solid windowtext .5pt;padding:0cm 5.4pt 0cm 5.4pt;height:15.6pt'>
  <p class=MsoNormal align=center style='text-align:center;mso-pagination:widow-orphan'><b
  style='mso-bidi-font-weight:normal'><span lang=EN-US style='font-size:6.5pt;
  font-family:"Times New Roman",serif;mso-fareast-font-family:等线;mso-font-kerning:
  0pt'>3.166<o:p></o:p></span></b></p>
  </td>
  <td width=39 style='width:29.55pt;border-top:none;border-left:none;
  border-bottom:solid windowtext 1.0pt;border-right:solid windowtext 1.0pt;
  mso-border-top-alt:solid windowtext .5pt;mso-border-left-alt:solid windowtext .5pt;
  mso-border-alt:solid windowtext .5pt;padding:0cm 5.4pt 0cm 5.4pt;height:15.6pt'>
  <p class=MsoNormal align=center style='text-align:center;mso-pagination:widow-orphan'><b
  style='mso-bidi-font-weight:normal'><span lang=EN-US style='font-size:6.5pt;
  font-family:"Times New Roman",serif;mso-fareast-font-family:等线;mso-font-kerning:
  0pt'>2.977<o:p></o:p></span></b></p>
  </td>
 </tr>
</table>

</div>

<p class=MsoNormal><span lang=EN-US><o:p>&nbsp;</o:p></span></p>

</div>

</body>

The proposed LSA improves the enhancement performance of both the casual DPARN and MTFAA models in terms of all objective metrics. To reveal the benefit of LSA mechanism, we visualize the normalized average spectral attention plots, generated from audios in the test set, of attention blocks in both original MTFAA and LSA-based MTFAA, as shown in figures below  
<!-- ![attn_vis](https://user-images.githubusercontent.com/103247057/222086124-5cc5f250-322b-4ffa-bcd1-11b3ae31047e.JPG) -->
<div align=center><img src="https://user-images.githubusercontent.com/103247057/222086124-5cc5f250-322b-4ffa-bcd1-11b3ae31047e.JPG"/></div>

It can be seen from the fifth layer of attention that the LSA-based model more effectively emphasizes the structural features of harmonics in low bands (marked with red boxes) and the almost randomly distributed components in high bands (marked with black boxes). Furthermore, it can be seen from the blue boxes that LSA can also effectively alleviate the modeling of the invalid correlation between the low bands and the high bands. Hence, the speech pattern in spectrum can be better modeled by LSA. Further investigation of the enhanced signals reveals that the global attention in frequency domain is more likely to inflict distortion to speech components or produce excessive residual noise in non-speech segments, while this problem can be effectively alleviated by the proposed LSA. Two typical examples are shown in Figure 3, where the benefit of LSA can be clearly seen. A possible explanation is that the better exploitation to speech pattern helps LSA-based model more effectively discriminate speech and noise components especially in low-SNR environments.
<!-- ![samples](https://user-images.githubusercontent.com/103247057/222087161-bf9ce54d-23d6-45c6-8e1d-c7693cbee839.JPG) -->
<div align=center><img src="https://user-images.githubusercontent.com/103247057/222087161-bf9ce54d-23d6-45c6-8e1d-c7693cbee839.JPG"/></div>


To further demonstrate the importance of modeling local correlation in spectrum for full-band SE tasks, we also compare local attention with a recently proposed biased attention method, namely Attention with Linear Biases (ALiBi), which negatively biases attention scores with a linearly decreasing penalty proportional to the distance between the relevant key and query for efficient extrapolation. Its application on spectral attention can be seen in the figure below  

<!-- ![Abi](https://user-images.githubusercontent.com/103247057/222088489-44d05699-41df-4f0e-9bc9-d7f8a84947b7.png) -->
<div align=center><img src="https://user-images.githubusercontent.com/103247057/222088489-44d05699-41df-4f0e-9bc9-d7f8a84947b7.png" width="600" height="250" /></div>  
We modify the penalty bias to decrease in a square manner for better performance and name the method as ASqBi, indicated in the figures below.  
<!-- ![biasmode](https://user-images.githubusercontent.com/103247057/222089742-d042e36c-f9cb-4747-9bab-700b891c25c1.JPG) -->
<div align=center><img src="https://user-images.githubusercontent.com/103247057/222089742-d042e36c-f9cb-4747-9bab-700b891c25c1.JPG" width="400" height="250" /></div>  

The modified method is combined with MTFAA [MTFAA_Net_full_F_ASqbi.py](https://github.com/ZhongshuHou/LSA/blob/main/MTFAA_Net_full_F_ASqbi.py) 
and the ablation test results can be found in the last row of table below. It can be seen that the overall performance degrades compared with LSA. It may be explained that the negative bias added to local attention region weakens the model capability to extract local intercorrelation.
 

<body lang=ZH-CN style='tab-interval:21.0pt;text-justify-trim:punctuation'>

<div class=WordSection1 style='layout-grid:15.6pt'>

<div align=center>

<table class=MsoTableGrid border=1 cellspacing=0 cellpadding=0 width=548
 style='border-collapse:collapse;mso-table-layout-alt:fixed;border:none;
 mso-border-alt:solid windowtext .5pt;mso-yfti-tbllook:1184;mso-padding-alt:
 0cm 5.4pt 0cm 5.4pt'>
 <tr style='mso-yfti-irow:0;mso-yfti-firstrow:yes;height:15.0pt'>
  <td width=75 nowrap style='width:56.45pt;border:solid windowtext 1.0pt;
  mso-border-alt:solid windowtext .5pt;padding:0cm 5.4pt 0cm 5.4pt;height:15.0pt'>
  <p class=MsoNormal align=center style='text-align:center;mso-pagination:widow-orphan'><a
  name="_Hlk127719777"><span lang=EN-US style='font-size:5.5pt;mso-bidi-font-size:
  10.0pt;font-family:"Times New Roman",serif;mso-fareast-font-family:等线;
  mso-font-kerning:0pt'>Full-band Metrics</span></a><span style='mso-bookmark:
  _Hlk127719777'><span lang=EN-US style='font-size:7.5pt;mso-bidi-font-size:
  10.0pt;font-family:"Times New Roman",serif;mso-fareast-font-family:等线;
  mso-font-kerning:0pt'><o:p></o:p></span></span></p>
  </td>
  <span style='mso-bookmark:_Hlk127719777'></span>
  <td width=118 nowrap colspan=3 style='width:88.6pt;border:solid windowtext 1.0pt;
  border-left:none;mso-border-left-alt:solid windowtext .5pt;mso-border-alt:
  solid windowtext .5pt;padding:0cm 5.4pt 0cm 5.4pt;height:15.0pt'>
  <p class=MsoNormal align=center style='text-align:center;mso-pagination:widow-orphan'><span
  style='mso-bookmark:_Hlk127719777'><span lang=EN-US style='font-size:7.5pt;
  mso-bidi-font-size:10.0pt;font-family:"Times New Roman",serif;mso-fareast-font-family:
  等线;mso-font-kerning:0pt'>STOI<o:p></o:p></span></span></p>
  </td>
  <span style='mso-bookmark:_Hlk127719777'></span>
  <td width=118 nowrap colspan=3 style='width:88.6pt;border:solid windowtext 1.0pt;
  border-left:none;mso-border-left-alt:solid windowtext .5pt;mso-border-alt:
  solid windowtext .5pt;padding:0cm 5.4pt 0cm 5.4pt;height:15.0pt'>
  <p class=MsoNormal align=center style='text-align:center;mso-pagination:widow-orphan'><span
  style='mso-bookmark:_Hlk127719777'><span lang=EN-US style='font-size:7.5pt;
  mso-bidi-font-size:10.0pt;font-family:"Times New Roman",serif;mso-fareast-font-family:
  等线;mso-font-kerning:0pt'>SiSDR(dB)<o:p></o:p></span></span></p>
  </td>
  <span style='mso-bookmark:_Hlk127719777'></span>
  <td width=39 nowrap rowspan=4 style='width:29.5pt;border:solid windowtext 1.0pt;
  border-left:none;mso-border-left-alt:solid windowtext .5pt;mso-border-alt:
  solid windowtext .5pt;padding:0cm 5.4pt 0cm 5.4pt;height:15.0pt'><span
  style='mso-bookmark:_Hlk127719777'></span>
  <p class=MsoNormal align=center style='text-align:center;mso-pagination:widow-orphan'><span
  style='mso-bookmark:_Hlk127719777'><span lang=EN-US style='font-size:7.5pt;
  mso-bidi-font-size:10.0pt;font-family:"Times New Roman",serif;mso-fareast-font-family:
  等线;mso-font-kerning:0pt'><o:p>&nbsp;</o:p></span></span></p>
  </td>
  <span style='mso-bookmark:_Hlk127719777'></span>
  <td width=197 colspan=5 style='width:147.7pt;border:solid windowtext 1.0pt;
  border-left:none;mso-border-left-alt:solid windowtext .5pt;mso-border-alt:
  solid windowtext .5pt;padding:0cm 5.4pt 0cm 5.4pt;height:15.0pt'>
  <p class=MsoNormal align=center style='text-align:center;mso-pagination:widow-orphan'><span
  style='mso-bookmark:_Hlk127719777'><span class=GramE><span lang=EN-US
  style='font-size:7.5pt;mso-bidi-font-size:10.0pt;font-family:"Times New Roman",serif;
  mso-fareast-font-family:等线;mso-font-kerning:0pt'>LSD(</span></span></span><span
  style='mso-bookmark:_Hlk127719777'><span lang=EN-US style='font-size:7.5pt;
  mso-bidi-font-size:10.0pt;font-family:"Times New Roman",serif;mso-fareast-font-family:
  等线;mso-font-kerning:0pt'>dB)<o:p></o:p></span></span></p>
  </td>
  <span style='mso-bookmark:_Hlk127719777'></span>
 </tr>
 <tr style='mso-yfti-irow:1;height:15.0pt'>
  <td width=75 nowrap style='width:56.45pt;border:solid windowtext 1.0pt;
  border-top:none;mso-border-top-alt:solid windowtext .5pt;mso-border-alt:solid windowtext .5pt;
  padding:0cm 5.4pt 0cm 5.4pt;height:15.0pt'>
  <p class=MsoNormal align=center style='text-align:center;mso-pagination:widow-orphan'><span
  style='mso-bookmark:_Hlk127719777'><span class=GramE><span lang=EN-US
  style='font-size:6.5pt;mso-bidi-font-size:10.0pt;font-family:"Times New Roman",serif;
  mso-fareast-font-family:等线;mso-font-kerning:0pt'>SNR(</span></span></span><span
  style='mso-bookmark:_Hlk127719777'><span lang=EN-US style='font-size:6.5pt;
  mso-bidi-font-size:10.0pt;font-family:"Times New Roman",serif;mso-fareast-font-family:
  等线;mso-font-kerning:0pt'>dB)</span></span><span style='mso-bookmark:_Hlk127719777'><span
  lang=EN-US style='font-size:7.5pt;mso-bidi-font-size:10.0pt;font-family:"Times New Roman",serif;
  mso-fareast-font-family:等线;mso-font-kerning:0pt'><o:p></o:p></span></span></p>
  </td>
  <span style='mso-bookmark:_Hlk127719777'></span>
  <td width=39 nowrap style='width:29.5pt;border-top:none;border-left:none;
  border-bottom:solid windowtext 1.0pt;border-right:solid windowtext 1.0pt;
  mso-border-top-alt:solid windowtext .5pt;mso-border-left-alt:solid windowtext .5pt;
  mso-border-alt:solid windowtext .5pt;padding:0cm 5.4pt 0cm 5.4pt;height:15.0pt'>
  <p class=MsoNormal align=center style='text-align:center;mso-pagination:widow-orphan'><span
  style='mso-bookmark:_Hlk127719777'><span lang=EN-US style='font-size:7.5pt;
  mso-bidi-font-size:10.0pt;font-family:"Times New Roman",serif;mso-fareast-font-family:
  等线;mso-font-kerning:0pt'>-5~0<o:p></o:p></span></span></p>
  </td>
  <span style='mso-bookmark:_Hlk127719777'></span>
  <td width=39 nowrap style='width:29.55pt;border-top:none;border-left:none;
  border-bottom:solid windowtext 1.0pt;border-right:solid windowtext 1.0pt;
  mso-border-top-alt:solid windowtext .5pt;mso-border-left-alt:solid windowtext .5pt;
  mso-border-alt:solid windowtext .5pt;padding:0cm 5.4pt 0cm 5.4pt;height:15.0pt'>
  <p class=MsoNormal align=center style='text-align:center;mso-pagination:widow-orphan'><span
  style='mso-bookmark:_Hlk127719777'><span lang=EN-US style='font-size:7.5pt;
  mso-bidi-font-size:10.0pt;font-family:"Times New Roman",serif;mso-fareast-font-family:
  等线;mso-font-kerning:0pt'>0~15<o:p></o:p></span></span></p>
  </td>
  <span style='mso-bookmark:_Hlk127719777'></span>
  <td width=39 nowrap style='width:29.55pt;border-top:none;border-left:none;
  border-bottom:solid windowtext 1.0pt;border-right:solid windowtext 1.0pt;
  mso-border-top-alt:solid windowtext .5pt;mso-border-left-alt:solid windowtext .5pt;
  mso-border-alt:solid windowtext .5pt;padding:0cm 5.4pt 0cm 5.4pt;height:15.0pt'>
  <p class=MsoNormal align=center style='text-align:center;mso-pagination:widow-orphan'><span
  style='mso-bookmark:_Hlk127719777'><span class=SpellE><span lang=EN-US
  style='font-size:7.5pt;mso-bidi-font-size:10.0pt;font-family:"Times New Roman",serif;
  mso-fareast-font-family:等线;mso-font-kerning:0pt'>Ovrl</span></span></span><span
  style='mso-bookmark:_Hlk127719777'><span lang=EN-US style='font-size:7.5pt;
  mso-bidi-font-size:10.0pt;font-family:"Times New Roman",serif;mso-fareast-font-family:
  等线;mso-font-kerning:0pt'>.<o:p></o:p></span></span></p>
  </td>
  <span style='mso-bookmark:_Hlk127719777'></span>
  <td width=39 nowrap style='width:29.5pt;border-top:none;border-left:none;
  border-bottom:solid windowtext 1.0pt;border-right:solid windowtext 1.0pt;
  mso-border-top-alt:solid windowtext .5pt;mso-border-left-alt:solid windowtext .5pt;
  mso-border-alt:solid windowtext .5pt;padding:0cm 5.4pt 0cm 5.4pt;height:15.0pt'>
  <p class=MsoNormal align=center style='text-align:center;mso-pagination:widow-orphan'><span
  style='mso-bookmark:_Hlk127719777'><span lang=EN-US style='font-size:7.5pt;
  mso-bidi-font-size:10.0pt;font-family:"Times New Roman",serif;mso-fareast-font-family:
  等线;mso-font-kerning:0pt'>-5~0<o:p></o:p></span></span></p>
  </td>
  <span style='mso-bookmark:_Hlk127719777'></span>
  <td width=39 nowrap style='width:29.55pt;border-top:none;border-left:none;
  border-bottom:solid windowtext 1.0pt;border-right:solid windowtext 1.0pt;
  mso-border-top-alt:solid windowtext .5pt;mso-border-left-alt:solid windowtext .5pt;
  mso-border-alt:solid windowtext .5pt;padding:0cm 5.4pt 0cm 5.4pt;height:15.0pt'>
  <p class=MsoNormal align=center style='text-align:center;mso-pagination:widow-orphan'><span
  style='mso-bookmark:_Hlk127719777'><span lang=EN-US style='font-size:7.5pt;
  mso-bidi-font-size:10.0pt;font-family:"Times New Roman",serif;mso-fareast-font-family:
  等线;mso-font-kerning:0pt'>0~15<o:p></o:p></span></span></p>
  </td>
  <span style='mso-bookmark:_Hlk127719777'></span>
  <td width=39 nowrap style='width:29.55pt;border-top:none;border-left:none;
  border-bottom:solid windowtext 1.0pt;border-right:solid windowtext 1.0pt;
  mso-border-top-alt:solid windowtext .5pt;mso-border-left-alt:solid windowtext .5pt;
  mso-border-alt:solid windowtext .5pt;padding:0cm 5.4pt 0cm 5.4pt;height:15.0pt'>
  <p class=MsoNormal align=center style='text-align:center;mso-pagination:widow-orphan'><span
  style='mso-bookmark:_Hlk127719777'><span class=SpellE><span lang=EN-US
  style='font-size:7.5pt;mso-bidi-font-size:10.0pt;font-family:"Times New Roman",serif;
  mso-fareast-font-family:等线;mso-font-kerning:0pt'>Ovrl</span></span></span><span
  style='mso-bookmark:_Hlk127719777'><span lang=EN-US style='font-size:7.5pt;
  mso-bidi-font-size:10.0pt;font-family:"Times New Roman",serif;mso-fareast-font-family:
  等线;mso-font-kerning:0pt'>.<o:p></o:p></span></span></p>
  </td>
  <span style='mso-bookmark:_Hlk127719777'></span>
  <td width=79 nowrap colspan=2 style='width:59.1pt;border-top:none;border-left:
  none;border-bottom:solid windowtext 1.0pt;border-right:solid windowtext 1.0pt;
  mso-border-top-alt:solid windowtext .5pt;mso-border-left-alt:solid windowtext .5pt;
  mso-border-alt:solid windowtext .5pt;padding:0cm 5.4pt 0cm 5.4pt;height:15.0pt'>
  <p class=MsoNormal align=center style='text-align:center;mso-pagination:widow-orphan'><span
  style='mso-bookmark:_Hlk127719777'><span lang=EN-US style='font-size:7.5pt;
  mso-bidi-font-size:10.0pt;font-family:"Times New Roman",serif;mso-fareast-font-family:
  等线;mso-font-kerning:0pt'>Band(kHz)</span></span><span style='mso-bookmark:
  _Hlk127719777'><span lang=EN-US style='font-size:5.0pt;mso-bidi-font-size:
  10.0pt;font-family:"Times New Roman",serif;mso-fareast-font-family:等线;
  mso-font-kerning:0pt'><o:p></o:p></span></span></p>
  </td>
  <span style='mso-bookmark:_Hlk127719777'></span>
  <td width=39 style='width:29.5pt;border-top:none;border-left:none;border-bottom:
  solid windowtext 1.0pt;border-right:solid windowtext 1.0pt;mso-border-top-alt:
  solid windowtext .5pt;mso-border-left-alt:solid windowtext .5pt;mso-border-alt:
  solid windowtext .5pt;padding:0cm 5.4pt 0cm 5.4pt;height:15.0pt'>
  <p class=MsoNormal align=center style='text-align:center;mso-pagination:widow-orphan'><span
  style='mso-bookmark:_Hlk127719777'><span lang=EN-US style='font-size:6.5pt;
  mso-bidi-font-size:10.0pt;font-family:"Times New Roman",serif;mso-fareast-font-family:
  等线;mso-font-kerning:0pt'>0~8<o:p></o:p></span></span></p>
  </td>
  <span style='mso-bookmark:_Hlk127719777'></span>
  <td width=39 style='width:29.55pt;border-top:none;border-left:none;
  border-bottom:solid windowtext 1.0pt;border-right:solid windowtext 1.0pt;
  mso-border-top-alt:solid windowtext .5pt;mso-border-left-alt:solid windowtext .5pt;
  mso-border-alt:solid windowtext .5pt;padding:0cm 5.4pt 0cm 5.4pt;height:15.0pt'>
  <p class=MsoNormal align=center style='text-align:center;mso-pagination:widow-orphan'><span
  style='mso-bookmark:_Hlk127719777'><span lang=EN-US style='font-size:6.5pt;
  mso-bidi-font-size:10.0pt;font-family:"Times New Roman",serif;mso-fareast-font-family:
  等线;mso-font-kerning:0pt'>8~24<o:p></o:p></span></span></p>
  </td>
  <span style='mso-bookmark:_Hlk127719777'></span>
  <td width=39 style='width:29.55pt;border-top:none;border-left:none;
  border-bottom:solid windowtext 1.0pt;border-right:solid windowtext 1.0pt;
  mso-border-top-alt:solid windowtext .5pt;mso-border-left-alt:solid windowtext .5pt;
  mso-border-alt:solid windowtext .5pt;padding:0cm 5.4pt 0cm 5.4pt;height:15.0pt'>
  <p class=MsoNormal align=center style='text-align:center;mso-pagination:widow-orphan'><span
  style='mso-bookmark:_Hlk127719777'><span lang=EN-US style='font-size:7.5pt;
  mso-bidi-font-size:10.0pt;font-family:"Times New Roman",serif;mso-fareast-font-family:
  等线;mso-font-kerning:0pt'>Full.<o:p></o:p></span></span></p>
  </td>
  <span style='mso-bookmark:_Hlk127719777'></span>
 </tr>
 <tr style='mso-yfti-irow:2;height:15.6pt'>
  <td width=75 nowrap style='width:56.45pt;border:solid windowtext 1.0pt;
  border-top:none;mso-border-top-alt:solid windowtext .5pt;mso-border-alt:solid windowtext .5pt;
  padding:0cm 5.4pt 0cm 5.4pt;height:15.6pt'>
  <p class=MsoNormal align=center style='text-align:center;mso-pagination:widow-orphan'><span
  style='mso-bookmark:_Hlk127719777'><span lang=EN-US style='font-size:6.5pt;
  mso-bidi-font-size:10.0pt;font-family:"Times New Roman",serif;mso-fareast-font-family:
  等线;mso-font-kerning:0pt'>MTFAA-<span class=SpellE>ASqBi</span><o:p></o:p></span></span></p>
  </td>
  <span style='mso-bookmark:_Hlk127719777'></span>
  <td width=39 nowrap style='width:29.5pt;border-top:none;border-left:none;
  border-bottom:solid windowtext 1.0pt;border-right:solid windowtext 1.0pt;
  mso-border-top-alt:solid windowtext .5pt;mso-border-left-alt:solid windowtext .5pt;
  mso-border-alt:solid windowtext .5pt;padding:0cm 5.4pt 0cm 5.4pt;height:15.6pt'>
  <p class=MsoNormal align=center style='text-align:center;mso-pagination:widow-orphan'><span
  style='mso-bookmark:_Hlk127719777'><b><span lang=EN-US style='font-size:6.5pt;
  mso-bidi-font-size:10.0pt;font-family:"Times New Roman",serif;mso-fareast-font-family:
  等线;mso-font-kerning:0pt'>0.811</span></b></span><span style='mso-bookmark:
  _Hlk127719777'><span lang=EN-US style='font-size:6.5pt;mso-bidi-font-size:
  10.0pt;font-family:"Times New Roman",serif;mso-fareast-font-family:等线;
  mso-font-kerning:0pt'><o:p></o:p></span></span></p>
  </td>
  <span style='mso-bookmark:_Hlk127719777'></span>
  <td width=39 nowrap style='width:29.55pt;border-top:none;border-left:none;
  border-bottom:solid windowtext 1.0pt;border-right:solid windowtext 1.0pt;
  mso-border-top-alt:solid windowtext .5pt;mso-border-left-alt:solid windowtext .5pt;
  mso-border-alt:solid windowtext .5pt;padding:0cm 5.4pt 0cm 5.4pt;height:15.6pt'>
  <p class=MsoNormal align=center style='text-align:center;mso-pagination:widow-orphan'><span
  style='mso-bookmark:_Hlk127719777'><span lang=EN-US style='font-size:6.5pt;
  mso-bidi-font-size:10.0pt;font-family:"Times New Roman",serif;mso-fareast-font-family:
  等线;mso-font-kerning:0pt'>0.881<o:p></o:p></span></span></p>
  </td>
  <span style='mso-bookmark:_Hlk127719777'></span>
  <td width=39 nowrap style='width:29.55pt;border-top:none;border-left:none;
  border-bottom:solid windowtext 1.0pt;border-right:solid windowtext 1.0pt;
  mso-border-top-alt:solid windowtext .5pt;mso-border-left-alt:solid windowtext .5pt;
  mso-border-alt:solid windowtext .5pt;padding:0cm 5.4pt 0cm 5.4pt;height:15.6pt'>
  <p class=MsoNormal align=center style='text-align:center;mso-pagination:widow-orphan'><span
  style='mso-bookmark:_Hlk127719777'><b style='mso-bidi-font-weight:normal'><span
  lang=EN-US style='font-size:6.5pt;mso-bidi-font-size:10.0pt;font-family:"Times New Roman",serif;
  mso-fareast-font-family:等线;mso-font-kerning:0pt'>0.860</span></b></span><span
  style='mso-bookmark:_Hlk127719777'><span lang=EN-US style='font-size:6.5pt;
  mso-bidi-font-size:10.0pt;font-family:"Times New Roman",serif;mso-fareast-font-family:
  等线;mso-font-kerning:0pt'><o:p></o:p></span></span></p>
  </td>
  <span style='mso-bookmark:_Hlk127719777'></span>
  <td width=39 nowrap valign=top style='width:29.5pt;border-top:none;
  border-left:none;border-bottom:solid windowtext 1.0pt;border-right:solid windowtext 1.0pt;
  mso-border-top-alt:solid windowtext .5pt;mso-border-left-alt:solid windowtext .5pt;
  mso-border-alt:solid windowtext .5pt;padding:0cm 5.4pt 0cm 5.4pt;height:15.6pt'>
  <p class=MsoNormal align=center style='text-align:center;mso-pagination:widow-orphan'><span
  style='mso-bookmark:_Hlk127719777'><b><span lang=EN-US style='font-size:6.5pt;
  mso-bidi-font-size:10.0pt;font-family:"Times New Roman",serif;mso-fareast-font-family:
  等线;mso-font-kerning:0pt'>10.425<o:p></o:p></span></b></span></p>
  </td>
  <span style='mso-bookmark:_Hlk127719777'></span>
  <td width=39 nowrap valign=top style='width:29.55pt;border-top:none;
  border-left:none;border-bottom:solid windowtext 1.0pt;border-right:solid windowtext 1.0pt;
  mso-border-top-alt:solid windowtext .5pt;mso-border-left-alt:solid windowtext .5pt;
  mso-border-alt:solid windowtext .5pt;padding:0cm 5.4pt 0cm 5.4pt;height:15.6pt'>
  <p class=MsoNormal align=center style='text-align:center;mso-pagination:widow-orphan'><span
  style='mso-bookmark:_Hlk127719777'><span lang=EN-US style='font-size:6.5pt;
  mso-bidi-font-size:10.0pt;font-family:"Times New Roman",serif;mso-fareast-font-family:
  等线;mso-font-kerning:0pt;mso-bidi-font-weight:bold'>15.944</span></span><span
  style='mso-bookmark:_Hlk127719777'><span lang=EN-US style='font-size:6.5pt;
  mso-bidi-font-size:10.0pt;font-family:"Times New Roman",serif;mso-fareast-font-family:
  等线;mso-font-kerning:0pt'><o:p></o:p></span></span></p>
  </td>
  <span style='mso-bookmark:_Hlk127719777'></span>
  <td width=39 nowrap valign=top style='width:29.55pt;border-top:none;
  border-left:none;border-bottom:solid windowtext 1.0pt;border-right:solid windowtext 1.0pt;
  mso-border-top-alt:solid windowtext .5pt;mso-border-left-alt:solid windowtext .5pt;
  mso-border-alt:solid windowtext .5pt;padding:0cm 5.4pt 0cm 5.4pt;height:15.6pt'>
  <p class=MsoNormal align=center style='text-align:center;mso-pagination:widow-orphan'><span
  style='mso-bookmark:_Hlk127719777'><span lang=EN-US style='font-size:6.5pt;
  mso-bidi-font-size:10.0pt;font-family:"Times New Roman",serif;mso-fareast-font-family:
  等线;mso-font-kerning:0pt;mso-bidi-font-weight:bold'>14.468</span></span><span
  style='mso-bookmark:_Hlk127719777'><b style='mso-bidi-font-weight:normal'><span
  lang=EN-US style='font-size:6.5pt;mso-bidi-font-size:10.0pt;font-family:"Times New Roman",serif;
  mso-fareast-font-family:等线;mso-font-kerning:0pt'><o:p></o:p></span></b></span></p>
  </td>
  <span style='mso-bookmark:_Hlk127719777'></span>
  <td width=79 nowrap colspan=2 style='width:59.1pt;border-top:none;border-left:
  none;border-bottom:solid windowtext 1.0pt;border-right:solid windowtext 1.0pt;
  mso-border-top-alt:solid windowtext .5pt;mso-border-left-alt:solid windowtext .5pt;
  mso-border-alt:solid windowtext .5pt;padding:0cm 5.4pt 0cm 5.4pt;height:15.6pt'>
  <p class=MsoNormal align=center style='text-align:center;mso-pagination:widow-orphan'><span
  style='mso-bookmark:_Hlk127719777'><span lang=EN-US style='font-size:6.5pt;
  mso-bidi-font-size:10.0pt;font-family:"Times New Roman",serif;mso-fareast-font-family:
  等线;mso-font-kerning:0pt'>MTFAA-<span class=SpellE>ASqaBi</span><span
  style='mso-bidi-font-weight:bold'><o:p></o:p></span></span></span></p>
  </td>
  <span style='mso-bookmark:_Hlk127719777'></span>
  <td width=39 style='width:29.5pt;border-top:none;border-left:none;border-bottom:
  solid windowtext 1.0pt;border-right:solid windowtext 1.0pt;mso-border-top-alt:
  solid windowtext .5pt;mso-border-left-alt:solid windowtext .5pt;mso-border-alt:
  solid windowtext .5pt;padding:0cm 5.4pt 0cm 5.4pt;height:15.6pt'>
  <p class=MsoNormal align=center style='text-align:center;mso-pagination:widow-orphan'><span
  style='mso-bookmark:_Hlk127719777'><span lang=EN-US style='font-size:6.5pt;
  mso-bidi-font-size:10.0pt;font-family:"Times New Roman",serif;mso-fareast-font-family:
  等线;mso-font-kerning:0pt'>10.307<b><o:p></o:p></b></span></span></p>
  </td>
  <span style='mso-bookmark:_Hlk127719777'></span>
  <td width=39 style='width:29.55pt;border-top:none;border-left:none;
  border-bottom:solid windowtext 1.0pt;border-right:solid windowtext 1.0pt;
  mso-border-top-alt:solid windowtext .5pt;mso-border-left-alt:solid windowtext .5pt;
  mso-border-alt:solid windowtext .5pt;padding:0cm 5.4pt 0cm 5.4pt;height:15.6pt'>
  <p class=MsoNormal align=center style='text-align:center;mso-pagination:widow-orphan'><span
  style='mso-bookmark:_Hlk127719777'><span lang=EN-US style='font-size:6.5pt;
  mso-bidi-font-size:10.0pt;font-family:"Times New Roman",serif;mso-fareast-font-family:
  等线;mso-font-kerning:0pt'>9.495<span style='mso-bidi-font-weight:bold'><o:p></o:p></span></span></span></p>
  </td>
  <span style='mso-bookmark:_Hlk127719777'></span>
  <td width=39 valign=top style='width:29.55pt;border-top:none;border-left:
  none;border-bottom:solid windowtext 1.0pt;border-right:solid windowtext 1.0pt;
  mso-border-top-alt:solid windowtext .5pt;mso-border-left-alt:solid windowtext .5pt;
  mso-border-alt:solid windowtext .5pt;padding:0cm 5.4pt 0cm 5.4pt;height:15.6pt'>
  <p class=MsoNormal align=center style='text-align:center;mso-pagination:widow-orphan'><span
  style='mso-bookmark:_Hlk127719777'><span lang=EN-US style='font-size:6.5pt;
  mso-bidi-font-size:10.0pt;font-family:"Times New Roman",serif;mso-fareast-font-family:
  等线;mso-font-kerning:0pt;mso-bidi-font-weight:bold'>9.766<o:p></o:p></span></span></p>
  </td>
  <span style='mso-bookmark:_Hlk127719777'></span>
 </tr>
 <tr style='mso-yfti-irow:3;height:15.6pt'>
  <td width=75 nowrap style='width:56.45pt;border:solid windowtext 1.0pt;
  border-top:none;mso-border-top-alt:solid windowtext .5pt;mso-border-alt:solid windowtext .5pt;
  padding:0cm 5.4pt 0cm 5.4pt;height:15.6pt'>
  <p class=MsoNormal align=center style='text-align:center;mso-pagination:widow-orphan'><span
  style='mso-bookmark:_Hlk127719777'><span lang=EN-US style='font-size:6.5pt;
  mso-bidi-font-size:10.0pt;font-family:"Times New Roman",serif;mso-fareast-font-family:
  等线;mso-font-kerning:0pt'>MTFAA-LSA</span></span><span style='mso-bookmark:
  _Hlk127719777'><span lang=EN-US style='font-size:7.5pt;mso-bidi-font-size:
  10.0pt;font-family:"Times New Roman",serif;mso-fareast-font-family:等线;
  mso-font-kerning:0pt'><o:p></o:p></span></span></p>
  </td>
  <span style='mso-bookmark:_Hlk127719777'></span>
  <td width=39 nowrap style='width:29.5pt;border-top:none;border-left:none;
  border-bottom:solid windowtext 1.0pt;border-right:solid windowtext 1.0pt;
  mso-border-top-alt:solid windowtext .5pt;mso-border-left-alt:solid windowtext .5pt;
  mso-border-alt:solid windowtext .5pt;padding:0cm 5.4pt 0cm 5.4pt;height:15.6pt'>
  <p class=MsoNormal align=center style='text-align:center;mso-pagination:widow-orphan'><span
  style='mso-bookmark:_Hlk127719777'><span lang=EN-US style='font-size:6.5pt;
  mso-bidi-font-size:10.0pt;font-family:"Times New Roman",serif;mso-fareast-font-family:
  等线;mso-font-kerning:0pt'>0.809<b style='mso-bidi-font-weight:normal'><o:p></o:p></b></span></span></p>
  </td>
  <span style='mso-bookmark:_Hlk127719777'></span>
  <td width=39 nowrap style='width:29.55pt;border-top:none;border-left:none;
  border-bottom:solid windowtext 1.0pt;border-right:solid windowtext 1.0pt;
  mso-border-top-alt:solid windowtext .5pt;mso-border-left-alt:solid windowtext .5pt;
  mso-border-alt:solid windowtext .5pt;padding:0cm 5.4pt 0cm 5.4pt;height:15.6pt'>
  <p class=MsoNormal align=center style='text-align:center;mso-pagination:widow-orphan'><span
  style='mso-bookmark:_Hlk127719777'><b style='mso-bidi-font-weight:normal'><span
  lang=EN-US style='font-size:6.5pt;mso-bidi-font-size:10.0pt;font-family:"Times New Roman",serif;
  mso-fareast-font-family:等线;mso-font-kerning:0pt'>0.881<o:p></o:p></span></b></span></p>
  </td>
  <span style='mso-bookmark:_Hlk127719777'></span>
  <td width=39 nowrap style='width:29.55pt;border-top:none;border-left:none;
  border-bottom:solid windowtext 1.0pt;border-right:solid windowtext 1.0pt;
  mso-border-top-alt:solid windowtext .5pt;mso-border-left-alt:solid windowtext .5pt;
  mso-border-alt:solid windowtext .5pt;padding:0cm 5.4pt 0cm 5.4pt;height:15.6pt'>
  <p class=MsoNormal align=center style='text-align:center;mso-pagination:widow-orphan'><span
  style='mso-bookmark:_Hlk127719777'><span lang=EN-US style='font-size:6.5pt;
  mso-bidi-font-size:10.0pt;font-family:"Times New Roman",serif;mso-fareast-font-family:
  等线;mso-font-kerning:0pt'>0.860<b style='mso-bidi-font-weight:normal'><o:p></o:p></b></span></span></p>
  </td>
  <span style='mso-bookmark:_Hlk127719777'></span>
  <td width=39 nowrap valign=top style='width:29.5pt;border-top:none;
  border-left:none;border-bottom:solid windowtext 1.0pt;border-right:solid windowtext 1.0pt;
  mso-border-top-alt:solid windowtext .5pt;mso-border-left-alt:solid windowtext .5pt;
  mso-border-alt:solid windowtext .5pt;padding:0cm 5.4pt 0cm 5.4pt;height:15.6pt'>
  <p class=MsoNormal align=center style='text-align:center;mso-pagination:widow-orphan'><span
  style='mso-bookmark:_Hlk127719777'><span lang=EN-US style='font-size:6.5pt;
  mso-bidi-font-size:10.0pt;font-family:"Times New Roman",serif;mso-fareast-font-family:
  等线;mso-font-kerning:0pt'>10.347<o:p></o:p></span></span></p>
  </td>
  <span style='mso-bookmark:_Hlk127719777'></span>
  <td width=39 nowrap valign=top style='width:29.55pt;border-top:none;
  border-left:none;border-bottom:solid windowtext 1.0pt;border-right:solid windowtext 1.0pt;
  mso-border-top-alt:solid windowtext .5pt;mso-border-left-alt:solid windowtext .5pt;
  mso-border-alt:solid windowtext .5pt;padding:0cm 5.4pt 0cm 5.4pt;height:15.6pt'>
  <p class=MsoNormal align=center style='text-align:center;mso-pagination:widow-orphan'><span
  style='mso-bookmark:_Hlk127719777'><b style='mso-bidi-font-weight:normal'><span
  lang=EN-US style='font-size:6.5pt;mso-bidi-font-size:10.0pt;font-family:"Times New Roman",serif;
  mso-fareast-font-family:等线;mso-font-kerning:0pt'>16.201<o:p></o:p></span></b></span></p>
  </td>
  <span style='mso-bookmark:_Hlk127719777'></span>
  <td width=39 nowrap valign=top style='width:29.55pt;border-top:none;
  border-left:none;border-bottom:solid windowtext 1.0pt;border-right:solid windowtext 1.0pt;
  mso-border-top-alt:solid windowtext .5pt;mso-border-left-alt:solid windowtext .5pt;
  mso-border-alt:solid windowtext .5pt;padding:0cm 5.4pt 0cm 5.4pt;height:15.6pt'>
  <p class=MsoNormal align=center style='text-align:center;mso-pagination:widow-orphan'><span
  style='mso-bookmark:_Hlk127719777'><b style='mso-bidi-font-weight:normal'><span
  lang=EN-US style='font-size:6.5pt;mso-bidi-font-size:10.0pt;font-family:"Times New Roman",serif;
  mso-fareast-font-family:等线;mso-font-kerning:0pt'>14.635</span></b></span><span
  style='mso-bookmark:_Hlk127719777'><span lang=EN-US style='font-size:6.5pt;
  mso-bidi-font-size:10.0pt;font-family:"Times New Roman",serif;mso-fareast-font-family:
  等线;mso-font-kerning:0pt'><o:p></o:p></span></span></p>
  </td>
  <span style='mso-bookmark:_Hlk127719777'></span>
  <td width=79 nowrap colspan=2 style='width:59.1pt;border-top:none;border-left:
  none;border-bottom:solid windowtext 1.0pt;border-right:solid windowtext 1.0pt;
  mso-border-top-alt:solid windowtext .5pt;mso-border-left-alt:solid windowtext .5pt;
  mso-border-alt:solid windowtext .5pt;padding:0cm 5.4pt 0cm 5.4pt;height:15.6pt'>
  <p class=MsoNormal align=center style='text-align:center;mso-pagination:widow-orphan'><span
  style='mso-bookmark:_Hlk127719777'><span lang=EN-US style='font-size:6.5pt;
  mso-bidi-font-size:10.0pt;font-family:"Times New Roman",serif;mso-fareast-font-family:
  等线;mso-font-kerning:0pt'>MTFAA-LSA<b style='mso-bidi-font-weight:normal'><o:p></o:p></b></span></span></p>
  </td>
  <span style='mso-bookmark:_Hlk127719777'></span>
  <td width=39 style='width:29.5pt;border-top:none;border-left:none;border-bottom:
  solid windowtext 1.0pt;border-right:solid windowtext 1.0pt;mso-border-top-alt:
  solid windowtext .5pt;mso-border-left-alt:solid windowtext .5pt;mso-border-alt:
  solid windowtext .5pt;padding:0cm 5.4pt 0cm 5.4pt;height:15.6pt'>
  <p class=MsoNormal align=center style='text-align:center;mso-pagination:widow-orphan'><span
  style='mso-bookmark:_Hlk127719777'><b style='mso-bidi-font-weight:normal'><span
  lang=EN-US style='font-size:6.5pt;mso-bidi-font-size:10.0pt;font-family:"Times New Roman",serif;
  mso-fareast-font-family:等线;mso-font-kerning:0pt'>9.840<o:p></o:p></span></b></span></p>
  </td>
  <span style='mso-bookmark:_Hlk127719777'></span>
  <td width=39 style='width:29.55pt;border-top:none;border-left:none;
  border-bottom:solid windowtext 1.0pt;border-right:solid windowtext 1.0pt;
  mso-border-top-alt:solid windowtext .5pt;mso-border-left-alt:solid windowtext .5pt;
  mso-border-alt:solid windowtext .5pt;padding:0cm 5.4pt 0cm 5.4pt;height:15.6pt'>
  <p class=MsoNormal align=center style='text-align:center;mso-pagination:widow-orphan'><span
  style='mso-bookmark:_Hlk127719777'><b style='mso-bidi-font-weight:normal'><span
  lang=EN-US style='font-size:6.5pt;mso-bidi-font-size:10.0pt;font-family:"Times New Roman",serif;
  mso-fareast-font-family:等线;mso-font-kerning:0pt'>8.636<o:p></o:p></span></b></span></p>
  </td>
  <span style='mso-bookmark:_Hlk127719777'></span>
  <td width=39 valign=top style='width:29.55pt;border-top:none;border-left:
  none;border-bottom:solid windowtext 1.0pt;border-right:solid windowtext 1.0pt;
  mso-border-top-alt:solid windowtext .5pt;mso-border-left-alt:solid windowtext .5pt;
  mso-border-alt:solid windowtext .5pt;padding:0cm 5.4pt 0cm 5.4pt;height:15.6pt'>
  <p class=MsoNormal align=center style='text-align:center;mso-pagination:widow-orphan'><span
  style='mso-bookmark:_Hlk127719777'><b style='mso-bidi-font-weight:normal'><span
  lang=EN-US style='font-size:6.5pt;mso-bidi-font-size:10.0pt;font-family:"Times New Roman",serif;
  mso-fareast-font-family:等线;mso-font-kerning:0pt'>9.037<o:p></o:p></span></b></span></p>
  </td>
  <span style='mso-bookmark:_Hlk127719777'></span>
 </tr>
 <tr style='mso-yfti-irow:4;height:15.0pt'>
  <td width=548 nowrap colspan=13 style='width:410.85pt;border:solid windowtext 1.0pt;
  border-top:none;mso-border-top-alt:solid windowtext .5pt;mso-border-alt:solid windowtext .5pt;
  padding:0cm 5.4pt 0cm 5.4pt;height:15.0pt'><span style='mso-bookmark:_Hlk127719777'></span></td>
  <span style='mso-bookmark:_Hlk127719777'></span>
 </tr>
 <tr style='mso-yfti-irow:5;height:15.0pt'>
  <td width=75 nowrap style='width:56.45pt;border:solid windowtext 1.0pt;
  border-top:none;mso-border-top-alt:solid windowtext .5pt;mso-border-alt:solid windowtext .5pt;
  padding:0cm 5.4pt 0cm 5.4pt;height:15.0pt'>
  <p class=MsoNormal align=center style='text-align:center;mso-pagination:widow-orphan'><span
  lang=EN-US style='font-size:5.5pt;mso-bidi-font-size:10.0pt;font-family:"Times New Roman",serif;
  mso-fareast-font-family:等线;mso-font-kerning:0pt'>Wideband Metrics</span><span
  lang=EN-US style='font-size:7.5pt;mso-bidi-font-size:10.0pt;font-family:"Times New Roman",serif;
  mso-fareast-font-family:等线;mso-font-kerning:0pt'><o:p></o:p></span></p>
  </td>
  <td width=118 nowrap colspan=3 style='width:88.6pt;border-top:none;
  border-left:none;border-bottom:solid windowtext 1.0pt;border-right:solid windowtext 1.0pt;
  mso-border-top-alt:solid windowtext .5pt;mso-border-left-alt:solid windowtext .5pt;
  mso-border-alt:solid windowtext .5pt;padding:0cm 5.4pt 0cm 5.4pt;height:15.0pt'>
  <p class=MsoNormal align=center style='text-align:center;mso-pagination:widow-orphan'><span
  lang=EN-US style='font-size:7.5pt;mso-bidi-font-size:10.0pt;font-family:"Times New Roman",serif;
  mso-fareast-font-family:等线;mso-font-kerning:0pt'>PESQ<o:p></o:p></span></p>
  </td>
  <td width=118 nowrap colspan=3 style='width:88.6pt;border-top:none;
  border-left:none;border-bottom:solid windowtext 1.0pt;border-right:solid windowtext 1.0pt;
  mso-border-top-alt:solid windowtext .5pt;mso-border-left-alt:solid windowtext .5pt;
  mso-border-alt:solid windowtext .5pt;padding:0cm 5.4pt 0cm 5.4pt;height:15.0pt'>
  <p class=MsoNormal align=center style='text-align:center;mso-pagination:widow-orphan'><span
  lang=EN-US style='font-size:7.5pt;mso-bidi-font-size:10.0pt;font-family:"Times New Roman",serif;
  mso-fareast-font-family:等线;mso-font-kerning:0pt'>CSIG<o:p></o:p></span></p>
  </td>
  <td width=118 nowrap colspan=3 style='width:88.6pt;border-top:none;
  border-left:none;border-bottom:solid windowtext 1.0pt;border-right:solid windowtext 1.0pt;
  mso-border-top-alt:solid windowtext .5pt;mso-border-left-alt:solid windowtext .5pt;
  mso-border-alt:solid windowtext .5pt;padding:0cm 5.4pt 0cm 5.4pt;height:15.0pt'>
  <p class=MsoNormal align=center style='text-align:center;mso-pagination:widow-orphan'><span
  lang=EN-US style='font-size:7.5pt;mso-bidi-font-size:10.0pt;font-family:"Times New Roman",serif;
  mso-fareast-font-family:等线;mso-font-kerning:0pt'>CBAK<o:p></o:p></span></p>
  </td>
  <td width=118 colspan=3 valign=top style='width:88.6pt;border-top:none;
  border-left:none;border-bottom:solid windowtext 1.0pt;border-right:solid windowtext 1.0pt;
  mso-border-top-alt:solid windowtext .5pt;mso-border-left-alt:solid windowtext .5pt;
  mso-border-alt:solid windowtext .5pt;padding:0cm 5.4pt 0cm 5.4pt;height:15.0pt'>
  <p class=MsoNormal align=center style='text-align:center;mso-pagination:widow-orphan'><span
  lang=EN-US style='font-size:7.5pt;mso-bidi-font-size:10.0pt;font-family:"Times New Roman",serif;
  mso-fareast-font-family:等线;mso-font-kerning:0pt'>COVL<o:p></o:p></span></p>
  </td>
 </tr>
 <tr style='mso-yfti-irow:6;height:15.0pt'>
  <td width=75 nowrap style='width:56.45pt;border:solid windowtext 1.0pt;
  border-top:none;mso-border-top-alt:solid windowtext .5pt;mso-border-alt:solid windowtext .5pt;
  padding:0cm 5.4pt 0cm 5.4pt;height:15.0pt'>
  <p class=MsoNormal align=center style='text-align:center;mso-pagination:widow-orphan'><span
  class=GramE><span lang=EN-US style='font-size:6.5pt;mso-bidi-font-size:10.0pt;
  font-family:"Times New Roman",serif;mso-fareast-font-family:等线;mso-font-kerning:
  0pt'>SNR(</span></span><span lang=EN-US style='font-size:6.5pt;mso-bidi-font-size:
  10.0pt;font-family:"Times New Roman",serif;mso-fareast-font-family:等线;
  mso-font-kerning:0pt'>dB)<o:p></o:p></span></p>
  </td>
  <td width=39 nowrap style='width:29.5pt;border-top:none;border-left:none;
  border-bottom:solid windowtext 1.0pt;border-right:solid windowtext 1.0pt;
  mso-border-top-alt:solid windowtext .5pt;mso-border-left-alt:solid windowtext .5pt;
  mso-border-alt:solid windowtext .5pt;padding:0cm 5.4pt 0cm 5.4pt;height:15.0pt'>
  <p class=MsoNormal align=center style='text-align:center;mso-pagination:widow-orphan'><span
  lang=EN-US style='font-size:7.5pt;mso-bidi-font-size:10.0pt;font-family:"Times New Roman",serif;
  mso-fareast-font-family:等线;mso-font-kerning:0pt'>-5~0</span><span lang=EN-US
  style='font-size:6.5pt;font-family:"Times New Roman",serif;mso-fareast-font-family:
  等线;mso-font-kerning:0pt'><o:p></o:p></span></p>
  </td>
  <td width=39 nowrap style='width:29.55pt;border-top:none;border-left:none;
  border-bottom:solid windowtext 1.0pt;border-right:solid windowtext 1.0pt;
  mso-border-top-alt:solid windowtext .5pt;mso-border-left-alt:solid windowtext .5pt;
  mso-border-alt:solid windowtext .5pt;padding:0cm 5.4pt 0cm 5.4pt;height:15.0pt'>
  <p class=MsoNormal align=center style='text-align:center;mso-pagination:widow-orphan'><span
  lang=EN-US style='font-size:7.5pt;mso-bidi-font-size:10.0pt;font-family:"Times New Roman",serif;
  mso-fareast-font-family:等线;mso-font-kerning:0pt'>0~15</span><span lang=EN-US
  style='font-size:6.5pt;font-family:"Times New Roman",serif;mso-fareast-font-family:
  等线;mso-font-kerning:0pt'><o:p></o:p></span></p>
  </td>
  <td width=39 nowrap style='width:29.55pt;border-top:none;border-left:none;
  border-bottom:solid windowtext 1.0pt;border-right:solid windowtext 1.0pt;
  mso-border-top-alt:solid windowtext .5pt;mso-border-left-alt:solid windowtext .5pt;
  mso-border-alt:solid windowtext .5pt;padding:0cm 5.4pt 0cm 5.4pt;height:15.0pt'>
  <p class=MsoNormal align=center style='text-align:center;mso-pagination:widow-orphan'><span
  class=SpellE><span lang=EN-US style='font-size:7.5pt;mso-bidi-font-size:10.0pt;
  font-family:"Times New Roman",serif;mso-fareast-font-family:等线;mso-font-kerning:
  0pt'>Ovrl</span></span><span lang=EN-US style='font-size:7.5pt;mso-bidi-font-size:
  10.0pt;font-family:"Times New Roman",serif;mso-fareast-font-family:等线;
  mso-font-kerning:0pt'>.</span><span lang=EN-US style='font-size:6.5pt;
  font-family:"Times New Roman",serif;mso-fareast-font-family:等线;mso-font-kerning:
  0pt'><o:p></o:p></span></p>
  </td>
  <td width=39 nowrap style='width:29.5pt;border-top:none;border-left:none;
  border-bottom:solid windowtext 1.0pt;border-right:solid windowtext 1.0pt;
  mso-border-top-alt:solid windowtext .5pt;mso-border-left-alt:solid windowtext .5pt;
  mso-border-alt:solid windowtext .5pt;padding:0cm 5.4pt 0cm 5.4pt;height:15.0pt'>
  <p class=MsoNormal align=center style='text-align:center;mso-pagination:widow-orphan'><span
  lang=EN-US style='font-size:6.5pt;font-family:"Times New Roman",serif;
  mso-fareast-font-family:等线;mso-font-kerning:0pt'>-5~0<o:p></o:p></span></p>
  </td>
  <td width=39 nowrap style='width:29.55pt;border-top:none;border-left:none;
  border-bottom:solid windowtext 1.0pt;border-right:solid windowtext 1.0pt;
  mso-border-top-alt:solid windowtext .5pt;mso-border-left-alt:solid windowtext .5pt;
  mso-border-alt:solid windowtext .5pt;padding:0cm 5.4pt 0cm 5.4pt;height:15.0pt'>
  <p class=MsoNormal align=center style='text-align:center;mso-pagination:widow-orphan'><span
  lang=EN-US style='font-size:6.5pt;font-family:"Times New Roman",serif;
  mso-fareast-font-family:等线;mso-font-kerning:0pt'>0~15<o:p></o:p></span></p>
  </td>
  <td width=39 nowrap style='width:29.55pt;border-top:none;border-left:none;
  border-bottom:solid windowtext 1.0pt;border-right:solid windowtext 1.0pt;
  mso-border-top-alt:solid windowtext .5pt;mso-border-left-alt:solid windowtext .5pt;
  mso-border-alt:solid windowtext .5pt;padding:0cm 5.4pt 0cm 5.4pt;height:15.0pt'>
  <p class=MsoNormal align=center style='text-align:center;mso-pagination:widow-orphan'><span
  class=SpellE><span lang=EN-US style='font-size:6.5pt;font-family:"Times New Roman",serif;
  mso-fareast-font-family:等线;mso-font-kerning:0pt'>Ovrl</span></span><span
  lang=EN-US style='font-size:6.5pt;font-family:"Times New Roman",serif;
  mso-fareast-font-family:等线;mso-font-kerning:0pt'>.<o:p></o:p></span></p>
  </td>
  <td width=39 nowrap style='width:29.5pt;border-top:none;border-left:none;
  border-bottom:solid windowtext 1.0pt;border-right:solid windowtext 1.0pt;
  mso-border-top-alt:solid windowtext .5pt;mso-border-left-alt:solid windowtext .5pt;
  mso-border-alt:solid windowtext .5pt;padding:0cm 5.4pt 0cm 5.4pt;height:15.0pt'>
  <p class=MsoNormal align=center style='text-align:center;mso-pagination:widow-orphan'><span
  lang=EN-US style='font-size:6.5pt;font-family:"Times New Roman",serif;
  mso-fareast-font-family:等线;mso-font-kerning:0pt'>-5~0<o:p></o:p></span></p>
  </td>
  <td width=39 nowrap style='width:29.55pt;border-top:none;border-left:none;
  border-bottom:solid windowtext 1.0pt;border-right:solid windowtext 1.0pt;
  mso-border-top-alt:solid windowtext .5pt;mso-border-left-alt:solid windowtext .5pt;
  mso-border-alt:solid windowtext .5pt;padding:0cm 5.4pt 0cm 5.4pt;height:15.0pt'>
  <p class=MsoNormal align=center style='text-align:center;mso-pagination:widow-orphan'><span
  lang=EN-US style='font-size:6.5pt;font-family:"Times New Roman",serif;
  mso-fareast-font-family:等线;mso-font-kerning:0pt'>0~15<o:p></o:p></span></p>
  </td>
  <td width=39 nowrap style='width:29.55pt;border-top:none;border-left:none;
  border-bottom:solid windowtext 1.0pt;border-right:solid windowtext 1.0pt;
  mso-border-top-alt:solid windowtext .5pt;mso-border-left-alt:solid windowtext .5pt;
  mso-border-alt:solid windowtext .5pt;padding:0cm 5.4pt 0cm 5.4pt;height:15.0pt'>
  <p class=MsoNormal align=center style='text-align:center;mso-pagination:widow-orphan'><span
  class=SpellE><span lang=EN-US style='font-size:6.5pt;font-family:"Times New Roman",serif;
  mso-fareast-font-family:等线;mso-font-kerning:0pt'>Ovrl</span></span><span
  lang=EN-US style='font-size:6.5pt;font-family:"Times New Roman",serif;
  mso-fareast-font-family:等线;mso-font-kerning:0pt'>.<o:p></o:p></span></p>
  </td>
  <td width=39 style='width:29.5pt;border-top:none;border-left:none;border-bottom:
  solid windowtext 1.0pt;border-right:solid windowtext 1.0pt;mso-border-top-alt:
  solid windowtext .5pt;mso-border-left-alt:solid windowtext .5pt;mso-border-alt:
  solid windowtext .5pt;padding:0cm 5.4pt 0cm 5.4pt;height:15.0pt'>
  <p class=MsoNormal align=center style='text-align:center;mso-pagination:widow-orphan'><span
  lang=EN-US style='font-size:6.5pt;font-family:"Times New Roman",serif;
  mso-fareast-font-family:等线;mso-font-kerning:0pt'>-5~0<o:p></o:p></span></p>
  </td>
  <td width=39 style='width:29.55pt;border-top:none;border-left:none;
  border-bottom:solid windowtext 1.0pt;border-right:solid windowtext 1.0pt;
  mso-border-top-alt:solid windowtext .5pt;mso-border-left-alt:solid windowtext .5pt;
  mso-border-alt:solid windowtext .5pt;padding:0cm 5.4pt 0cm 5.4pt;height:15.0pt'>
  <p class=MsoNormal align=center style='text-align:center;mso-pagination:widow-orphan'><span
  lang=EN-US style='font-size:6.5pt;font-family:"Times New Roman",serif;
  mso-fareast-font-family:等线;mso-font-kerning:0pt'>0~15<o:p></o:p></span></p>
  </td>
  <td width=39 style='width:29.55pt;border-top:none;border-left:none;
  border-bottom:solid windowtext 1.0pt;border-right:solid windowtext 1.0pt;
  mso-border-top-alt:solid windowtext .5pt;mso-border-left-alt:solid windowtext .5pt;
  mso-border-alt:solid windowtext .5pt;padding:0cm 5.4pt 0cm 5.4pt;height:15.0pt'>
  <p class=MsoNormal align=center style='text-align:center;mso-pagination:widow-orphan'><span
  class=SpellE><span lang=EN-US style='font-size:6.5pt;font-family:"Times New Roman",serif;
  mso-fareast-font-family:等线;mso-font-kerning:0pt'>Ovrl</span></span><span
  lang=EN-US style='font-size:6.5pt;font-family:"Times New Roman",serif;
  mso-fareast-font-family:等线;mso-font-kerning:0pt'>.<o:p></o:p></span></p>
  </td>
 </tr>
 <tr style='mso-yfti-irow:7;height:15.6pt'>
  <td width=75 nowrap style='width:56.45pt;border:solid windowtext 1.0pt;
  border-top:none;mso-border-top-alt:solid windowtext .5pt;mso-border-alt:solid windowtext .5pt;
  padding:0cm 5.4pt 0cm 5.4pt;height:15.6pt'>
  <p class=MsoNormal align=center style='text-align:center;mso-pagination:widow-orphan'><span
  lang=EN-US style='font-size:6.5pt;mso-bidi-font-size:10.0pt;font-family:"Times New Roman",serif;
  mso-fareast-font-family:等线;mso-font-kerning:0pt'>MTFAA-<span class=SpellE>ASqBi</span><o:p></o:p></span></p>
  </td>
  <td width=39 nowrap style='width:29.5pt;border-top:none;border-left:none;
  border-bottom:solid windowtext 1.0pt;border-right:solid windowtext 1.0pt;
  mso-border-top-alt:solid windowtext .5pt;mso-border-left-alt:solid windowtext .5pt;
  mso-border-alt:solid windowtext .5pt;padding:0cm 5.4pt 0cm 5.4pt;height:15.6pt'>
  <p class=MsoNormal align=center style='text-align:center;mso-pagination:widow-orphan'><span
  lang=EN-US style='font-size:6.5pt;mso-bidi-font-size:10.0pt;font-family:"Times New Roman",serif;
  mso-fareast-font-family:等线;mso-font-kerning:0pt'>2.064</span><span
  lang=EN-US style='font-size:6.5pt;font-family:"Times New Roman",serif;
  mso-fareast-font-family:等线;mso-font-kerning:0pt'><o:p></o:p></span></p>
  </td>
  <td width=39 nowrap style='width:29.55pt;border-top:none;border-left:none;
  border-bottom:solid windowtext 1.0pt;border-right:solid windowtext 1.0pt;
  mso-border-top-alt:solid windowtext .5pt;mso-border-left-alt:solid windowtext .5pt;
  mso-border-alt:solid windowtext .5pt;padding:0cm 5.4pt 0cm 5.4pt;height:15.6pt'>
  <p class=MsoNormal align=center style='text-align:center;mso-pagination:widow-orphan'><span
  lang=EN-US style='font-size:6.5pt;mso-bidi-font-size:10.0pt;font-family:"Times New Roman",serif;
  mso-fareast-font-family:等线;mso-font-kerning:0pt'>2.769</span><span
  lang=EN-US style='font-size:6.5pt;font-family:"Times New Roman",serif;
  mso-fareast-font-family:等线;mso-font-kerning:0pt'><o:p></o:p></span></p>
  </td>
  <td width=39 nowrap style='width:29.55pt;border-top:none;border-left:none;
  border-bottom:solid windowtext 1.0pt;border-right:solid windowtext 1.0pt;
  mso-border-top-alt:solid windowtext .5pt;mso-border-left-alt:solid windowtext .5pt;
  mso-border-alt:solid windowtext .5pt;padding:0cm 5.4pt 0cm 5.4pt;height:15.6pt'>
  <p class=MsoNormal align=center style='text-align:center;mso-pagination:widow-orphan'><span
  lang=EN-US style='font-size:6.5pt;mso-bidi-font-size:10.0pt;font-family:"Times New Roman",serif;
  mso-fareast-font-family:等线;mso-font-kerning:0pt'>2.564</span><span
  lang=EN-US style='font-size:6.5pt;font-family:"Times New Roman",serif;
  mso-fareast-font-family:等线;mso-font-kerning:0pt'><o:p></o:p></span></p>
  </td>
  <td width=39 nowrap style='width:29.5pt;border-top:none;border-left:none;
  border-bottom:solid windowtext 1.0pt;border-right:solid windowtext 1.0pt;
  mso-border-top-alt:solid windowtext .5pt;mso-border-left-alt:solid windowtext .5pt;
  mso-border-alt:solid windowtext .5pt;padding:0cm 5.4pt 0cm 5.4pt;height:15.6pt'>
  <p class=MsoNormal align=center style='text-align:center;mso-pagination:widow-orphan'><span
  lang=EN-US style='font-size:6.5pt;font-family:"Times New Roman",serif;
  mso-fareast-font-family:等线;mso-font-kerning:0pt'>3.487<span style='mso-bidi-font-weight:
  bold'><o:p></o:p></span></span></p>
  </td>
  <td width=39 nowrap style='width:29.55pt;border-top:none;border-left:none;
  border-bottom:solid windowtext 1.0pt;border-right:solid windowtext 1.0pt;
  mso-border-top-alt:solid windowtext .5pt;mso-border-left-alt:solid windowtext .5pt;
  mso-border-alt:solid windowtext .5pt;padding:0cm 5.4pt 0cm 5.4pt;height:15.6pt'>
  <p class=MsoNormal align=center style='text-align:center;mso-pagination:widow-orphan'><span
  lang=EN-US style='font-size:6.5pt;font-family:"Times New Roman",serif;
  mso-fareast-font-family:等线;mso-font-kerning:0pt'>4.165<o:p></o:p></span></p>
  </td>
  <td width=39 nowrap style='width:29.55pt;border-top:none;border-left:none;
  border-bottom:solid windowtext 1.0pt;border-right:solid windowtext 1.0pt;
  mso-border-top-alt:solid windowtext .5pt;mso-border-left-alt:solid windowtext .5pt;
  mso-border-alt:solid windowtext .5pt;padding:0cm 5.4pt 0cm 5.4pt;height:15.6pt'>
  <p class=MsoNormal align=center style='text-align:center;mso-pagination:widow-orphan'><span
  lang=EN-US style='font-size:6.5pt;font-family:"Times New Roman",serif;
  mso-fareast-font-family:等线;mso-font-kerning:0pt'>3.968<o:p></o:p></span></p>
  </td>
  <td width=39 nowrap style='width:29.5pt;border-top:none;border-left:none;
  border-bottom:solid windowtext 1.0pt;border-right:solid windowtext 1.0pt;
  mso-border-top-alt:solid windowtext .5pt;mso-border-left-alt:solid windowtext .5pt;
  mso-border-alt:solid windowtext .5pt;padding:0cm 5.4pt 0cm 5.4pt;height:15.6pt'>
  <p class=MsoNormal align=center style='text-align:center;mso-pagination:widow-orphan'><span
  lang=EN-US style='font-size:6.5pt;font-family:"Times New Roman",serif;
  mso-fareast-font-family:等线;mso-font-kerning:0pt;mso-bidi-font-weight:bold'>2.987<o:p></o:p></span></p>
  </td>
  <td width=39 nowrap style='width:29.55pt;border-top:none;border-left:none;
  border-bottom:solid windowtext 1.0pt;border-right:solid windowtext 1.0pt;
  mso-border-top-alt:solid windowtext .5pt;mso-border-left-alt:solid windowtext .5pt;
  mso-border-alt:solid windowtext .5pt;padding:0cm 5.4pt 0cm 5.4pt;height:15.6pt'>
  <p class=MsoNormal align=center style='text-align:center;mso-pagination:widow-orphan'><span
  lang=EN-US style='font-size:6.5pt;font-family:"Times New Roman",serif;
  mso-fareast-font-family:等线;mso-font-kerning:0pt'>3.558<span style='mso-bidi-font-weight:
  bold'><o:p></o:p></span></span></p>
  </td>
  <td width=39 nowrap style='width:29.55pt;border-top:none;border-left:none;
  border-bottom:solid windowtext 1.0pt;border-right:solid windowtext 1.0pt;
  mso-border-top-alt:solid windowtext .5pt;mso-border-left-alt:solid windowtext .5pt;
  mso-border-alt:solid windowtext .5pt;padding:0cm 5.4pt 0cm 5.4pt;height:15.6pt'>
  <p class=MsoNormal align=center style='text-align:center;mso-pagination:widow-orphan'><span
  lang=EN-US style='font-size:6.5pt;font-family:"Times New Roman",serif;
  mso-fareast-font-family:等线;mso-font-kerning:0pt'>3.392<span style='mso-bidi-font-weight:
  bold'><o:p></o:p></span></span></p>
  </td>
  <td width=39 style='width:29.5pt;border-top:none;border-left:none;border-bottom:
  solid windowtext 1.0pt;border-right:solid windowtext 1.0pt;mso-border-top-alt:
  solid windowtext .5pt;mso-border-left-alt:solid windowtext .5pt;mso-border-alt:
  solid windowtext .5pt;padding:0cm 5.4pt 0cm 5.4pt;height:15.6pt'>
  <p class=MsoNormal align=center style='text-align:center;mso-pagination:widow-orphan'><span
  lang=EN-US style='font-size:6.5pt;font-family:"Times New Roman",serif;
  mso-fareast-font-family:等线;mso-font-kerning:0pt;mso-bidi-font-weight:bold'>2.804<o:p></o:p></span></p>
  </td>
  <td width=39 style='width:29.55pt;border-top:none;border-left:none;
  border-bottom:solid windowtext 1.0pt;border-right:solid windowtext 1.0pt;
  mso-border-top-alt:solid windowtext .5pt;mso-border-left-alt:solid windowtext .5pt;
  mso-border-alt:solid windowtext .5pt;padding:0cm 5.4pt 0cm 5.4pt;height:15.6pt'>
  <p class=MsoNormal align=center style='text-align:center;mso-pagination:widow-orphan'><span
  lang=EN-US style='font-size:6.5pt;font-family:"Times New Roman",serif;
  mso-fareast-font-family:等线;mso-font-kerning:0pt;mso-bidi-font-weight:bold'>3.513<o:p></o:p></span></p>
  </td>
  <td width=39 style='width:29.55pt;border-top:none;border-left:none;
  border-bottom:solid windowtext 1.0pt;border-right:solid windowtext 1.0pt;
  mso-border-top-alt:solid windowtext .5pt;mso-border-left-alt:solid windowtext .5pt;
  mso-border-alt:solid windowtext .5pt;padding:0cm 5.4pt 0cm 5.4pt;height:15.6pt'>
  <p class=MsoNormal align=center style='text-align:center;mso-pagination:widow-orphan'><span
  lang=EN-US style='font-size:6.5pt;font-family:"Times New Roman",serif;
  mso-fareast-font-family:等线;mso-font-kerning:0pt;mso-bidi-font-weight:bold'>3.308<o:p></o:p></span></p>
  </td>
 </tr>
 <tr style='mso-yfti-irow:8;mso-yfti-lastrow:yes;height:15.6pt'>
  <td width=75 nowrap style='width:56.45pt;border:solid windowtext 1.0pt;
  border-top:none;mso-border-top-alt:solid windowtext .5pt;mso-border-alt:solid windowtext .5pt;
  padding:0cm 5.4pt 0cm 5.4pt;height:15.6pt'>
  <p class=MsoNormal align=center style='text-align:center;mso-pagination:widow-orphan'><span
  lang=EN-US style='font-size:6.5pt;mso-bidi-font-size:10.0pt;font-family:"Times New Roman",serif;
  mso-fareast-font-family:等线;mso-font-kerning:0pt'>MTFAA-LSA<o:p></o:p></span></p>
  </td>
  <td width=39 nowrap style='width:29.5pt;border-top:none;border-left:none;
  border-bottom:solid windowtext 1.0pt;border-right:solid windowtext 1.0pt;
  mso-border-top-alt:solid windowtext .5pt;mso-border-left-alt:solid windowtext .5pt;
  mso-border-alt:solid windowtext .5pt;padding:0cm 5.4pt 0cm 5.4pt;height:15.6pt'>
  <p class=MsoNormal align=center style='text-align:center;mso-pagination:widow-orphan'><b
  style='mso-bidi-font-weight:normal'><span lang=EN-US style='font-size:6.5pt;
  mso-bidi-font-size:10.0pt;font-family:"Times New Roman",serif;mso-fareast-font-family:
  等线;mso-font-kerning:0pt'>2.084</span></b><b style='mso-bidi-font-weight:normal'><span
  lang=EN-US style='font-size:6.5pt;font-family:"Times New Roman",serif;
  mso-fareast-font-family:等线;mso-font-kerning:0pt'><o:p></o:p></span></b></p>
  </td>
  <td width=39 nowrap style='width:29.55pt;border-top:none;border-left:none;
  border-bottom:solid windowtext 1.0pt;border-right:solid windowtext 1.0pt;
  mso-border-top-alt:solid windowtext .5pt;mso-border-left-alt:solid windowtext .5pt;
  mso-border-alt:solid windowtext .5pt;padding:0cm 5.4pt 0cm 5.4pt;height:15.6pt'>
  <p class=MsoNormal align=center style='text-align:center;mso-pagination:widow-orphan'><b
  style='mso-bidi-font-weight:normal'><span lang=EN-US style='font-size:6.5pt;
  mso-bidi-font-size:10.0pt;font-family:"Times New Roman",serif;mso-fareast-font-family:
  等线;mso-font-kerning:0pt'>2.795</span></b><b style='mso-bidi-font-weight:normal'><span
  lang=EN-US style='font-size:6.5pt;font-family:"Times New Roman",serif;
  mso-fareast-font-family:等线;mso-font-kerning:0pt'><o:p></o:p></span></b></p>
  </td>
  <td width=39 nowrap style='width:29.55pt;border-top:none;border-left:none;
  border-bottom:solid windowtext 1.0pt;border-right:solid windowtext 1.0pt;
  mso-border-top-alt:solid windowtext .5pt;mso-border-left-alt:solid windowtext .5pt;
  mso-border-alt:solid windowtext .5pt;padding:0cm 5.4pt 0cm 5.4pt;height:15.6pt'>
  <p class=MsoNormal align=center style='text-align:center;mso-pagination:widow-orphan'><b
  style='mso-bidi-font-weight:normal'><span lang=EN-US style='font-size:6.5pt;
  mso-bidi-font-size:10.0pt;font-family:"Times New Roman",serif;mso-fareast-font-family:
  等线;mso-font-kerning:0pt'>2.589</span></b><b style='mso-bidi-font-weight:normal'><span
  lang=EN-US style='font-size:6.5pt;font-family:"Times New Roman",serif;
  mso-fareast-font-family:等线;mso-font-kerning:0pt'><o:p></o:p></span></b></p>
  </td>
  <td width=39 nowrap style='width:29.5pt;border-top:none;border-left:none;
  border-bottom:solid windowtext 1.0pt;border-right:solid windowtext 1.0pt;
  mso-border-top-alt:solid windowtext .5pt;mso-border-left-alt:solid windowtext .5pt;
  mso-border-alt:solid windowtext .5pt;padding:0cm 5.4pt 0cm 5.4pt;height:15.6pt'>
  <p class=MsoNormal align=center style='text-align:center;mso-pagination:widow-orphan'><b
  style='mso-bidi-font-weight:normal'><span lang=EN-US style='font-size:6.5pt;
  font-family:"Times New Roman",serif;mso-fareast-font-family:等线;mso-font-kerning:
  0pt'>3.517<o:p></o:p></span></b></p>
  </td>
  <td width=39 nowrap style='width:29.55pt;border-top:none;border-left:none;
  border-bottom:solid windowtext 1.0pt;border-right:solid windowtext 1.0pt;
  mso-border-top-alt:solid windowtext .5pt;mso-border-left-alt:solid windowtext .5pt;
  mso-border-alt:solid windowtext .5pt;padding:0cm 5.4pt 0cm 5.4pt;height:15.6pt'>
  <p class=MsoNormal align=center style='text-align:center;mso-pagination:widow-orphan'><b
  style='mso-bidi-font-weight:normal'><span lang=EN-US style='font-size:6.5pt;
  font-family:"Times New Roman",serif;mso-fareast-font-family:等线;mso-font-kerning:
  0pt'>4.203<o:p></o:p></span></b></p>
  </td>
  <td width=39 nowrap style='width:29.55pt;border-top:none;border-left:none;
  border-bottom:solid windowtext 1.0pt;border-right:solid windowtext 1.0pt;
  mso-border-top-alt:solid windowtext .5pt;mso-border-left-alt:solid windowtext .5pt;
  mso-border-alt:solid windowtext .5pt;padding:0cm 5.4pt 0cm 5.4pt;height:15.6pt'>
  <p class=MsoNormal align=center style='text-align:center;mso-pagination:widow-orphan'><b
  style='mso-bidi-font-weight:normal'><span lang=EN-US style='font-size:6.5pt;
  font-family:"Times New Roman",serif;mso-fareast-font-family:等线;mso-font-kerning:
  0pt'>4.004<o:p></o:p></span></b></p>
  </td>
  <td width=39 nowrap style='width:29.5pt;border-top:none;border-left:none;
  border-bottom:solid windowtext 1.0pt;border-right:solid windowtext 1.0pt;
  mso-border-top-alt:solid windowtext .5pt;mso-border-left-alt:solid windowtext .5pt;
  mso-border-alt:solid windowtext .5pt;padding:0cm 5.4pt 0cm 5.4pt;height:15.6pt'>
  <p class=MsoNormal align=center style='text-align:center;mso-pagination:widow-orphan'><b
  style='mso-bidi-font-weight:normal'><span lang=EN-US style='font-size:6.5pt;
  font-family:"Times New Roman",serif;mso-fareast-font-family:等线;mso-font-kerning:
  0pt'>3.006<o:p></o:p></span></b></p>
  </td>
  <td width=39 nowrap style='width:29.55pt;border-top:none;border-left:none;
  border-bottom:solid windowtext 1.0pt;border-right:solid windowtext 1.0pt;
  mso-border-top-alt:solid windowtext .5pt;mso-border-left-alt:solid windowtext .5pt;
  mso-border-alt:solid windowtext .5pt;padding:0cm 5.4pt 0cm 5.4pt;height:15.6pt'>
  <p class=MsoNormal align=center style='text-align:center;mso-pagination:widow-orphan'><b
  style='mso-bidi-font-weight:normal'><span lang=EN-US style='font-size:6.5pt;
  font-family:"Times New Roman",serif;mso-fareast-font-family:等线;mso-font-kerning:
  0pt'>3.593<o:p></o:p></span></b></p>
  </td>
  <td width=39 nowrap style='width:29.55pt;border-top:none;border-left:none;
  border-bottom:solid windowtext 1.0pt;border-right:solid windowtext 1.0pt;
  mso-border-top-alt:solid windowtext .5pt;mso-border-left-alt:solid windowtext .5pt;
  mso-border-alt:solid windowtext .5pt;padding:0cm 5.4pt 0cm 5.4pt;height:15.6pt'>
  <p class=MsoNormal align=center style='text-align:center;mso-pagination:widow-orphan'><b
  style='mso-bidi-font-weight:normal'><span lang=EN-US style='font-size:6.5pt;
  font-family:"Times New Roman",serif;mso-fareast-font-family:等线;mso-font-kerning:
  0pt'>3.423<o:p></o:p></span></b></p>
  </td>
  <td width=39 style='width:29.5pt;border-top:none;border-left:none;border-bottom:
  solid windowtext 1.0pt;border-right:solid windowtext 1.0pt;mso-border-top-alt:
  solid windowtext .5pt;mso-border-left-alt:solid windowtext .5pt;mso-border-alt:
  solid windowtext .5pt;padding:0cm 5.4pt 0cm 5.4pt;height:15.6pt'>
  <p class=MsoNormal align=center style='text-align:center;mso-pagination:widow-orphan'><b
  style='mso-bidi-font-weight:normal'><span lang=EN-US style='font-size:6.5pt;
  font-family:"Times New Roman",serif;mso-fareast-font-family:等线;mso-font-kerning:
  0pt'>2.829<o:p></o:p></span></b></p>
  </td>
  <td width=39 style='width:29.55pt;border-top:none;border-left:none;
  border-bottom:solid windowtext 1.0pt;border-right:solid windowtext 1.0pt;
  mso-border-top-alt:solid windowtext .5pt;mso-border-left-alt:solid windowtext .5pt;
  mso-border-alt:solid windowtext .5pt;padding:0cm 5.4pt 0cm 5.4pt;height:15.6pt'>
  <p class=MsoNormal align=center style='text-align:center;mso-pagination:widow-orphan'><b
  style='mso-bidi-font-weight:normal'><span lang=EN-US style='font-size:6.5pt;
  font-family:"Times New Roman",serif;mso-fareast-font-family:等线;mso-font-kerning:
  0pt'>3.547<o:p></o:p></span></b></p>
  </td>
  <td width=39 style='width:29.55pt;border-top:none;border-left:none;
  border-bottom:solid windowtext 1.0pt;border-right:solid windowtext 1.0pt;
  mso-border-top-alt:solid windowtext .5pt;mso-border-left-alt:solid windowtext .5pt;
  mso-border-alt:solid windowtext .5pt;padding:0cm 5.4pt 0cm 5.4pt;height:15.6pt'>
  <p class=MsoNormal align=center style='text-align:center;mso-pagination:widow-orphan'><b
  style='mso-bidi-font-weight:normal'><span lang=EN-US style='font-size:6.5pt;
  font-family:"Times New Roman",serif;mso-fareast-font-family:等线;mso-font-kerning:
  0pt'>3.339<o:p></o:p></span></b></p>
  </td>
 </tr>
</table>

</div>

<p class=MsoNormal><span lang=EN-US><o:p>&nbsp;</o:p></span></p>

</div>

</body>

We also conduct subjective listening preference test on MTFAA model to validate the benifit of LSA mechanism. 50 enhanced samples as well as their reference target speech are randomly selected from the test set. 15 listeners with normal hearing compare the enhanced results based on the reference speech and choose the preferred result. Each sample is evaluated by at least 3 listeners. The subjective listening preference test results can be seen in table below. Over 60% samples enhanced by LSA-based MTFAA are considered to have better perceptual quality and lower noise levels, which demonstrates the efficiency of LSA in full-band SE tasks.


<body lang=ZH-CN style='tab-interval:21.0pt;text-justify-trim:punctuation'>

<div class=WordSection1 style='layout-grid:15.6pt'>

<div align=center>

<table class=PlainTable22 border=1 cellspacing=0 cellpadding=0
 style='border-collapse:collapse;border:none;mso-border-top-alt:solid #7F7F7F .5pt;
 mso-border-top-themecolor:text1;mso-border-top-themetint:128;mso-border-bottom-alt:
 solid #7F7F7F .5pt;mso-border-bottom-themecolor:text1;mso-border-bottom-themetint:
 128;mso-yfti-tbllook:1568;mso-padding-alt:0cm 5.4pt 0cm 5.4pt'>
 <tr style='mso-yfti-irow:-1;mso-yfti-firstrow:yes;mso-yfti-lastfirstrow:yes;
  height:10.5pt'>
  <td style='border-top:solid #7F7F7F 1.0pt;mso-border-top-themecolor:text1;
  mso-border-top-themetint:128;border-left:none;border-bottom:solid #7F7F7F 1.0pt;
  mso-border-bottom-themecolor:text1;mso-border-bottom-themetint:128;
  border-right:solid windowtext 1.0pt;mso-border-top-alt:solid #7F7F7F .5pt;
  mso-border-top-themecolor:text1;mso-border-top-themetint:128;mso-border-bottom-alt:
  solid #7F7F7F .5pt;mso-border-bottom-themecolor:text1;mso-border-bottom-themetint:
  128;mso-border-right-alt:solid windowtext .5pt;padding:0cm 5.4pt 0cm 5.4pt;
  height:10.5pt'>
  <p class=MsoNormal align=center style='text-align:center;mso-yfti-cnfc:1'><span
  lang=IT style='font-size:8.0pt;mso-bidi-font-size:9.0pt;font-family:"Times New Roman",serif;
  mso-fareast-font-family:宋体;mso-font-kerning:0pt;mso-ansi-language:IT;
  mso-fareast-language:IT;mso-bidi-font-weight:bold'>Model<o:p></o:p></span></p>
  </td>
  <td style='border-top:solid #7F7F7F 1.0pt;mso-border-top-themecolor:text1;
  mso-border-top-themetint:128;border-left:none;border-bottom:solid #7F7F7F 1.0pt;
  mso-border-bottom-themecolor:text1;mso-border-bottom-themetint:128;
  border-right:solid windowtext 1.0pt;mso-border-left-alt:solid windowtext .5pt;
  mso-border-top-alt:#7F7F7F;mso-border-top-themecolor:text1;mso-border-top-themetint:
  128;mso-border-left-alt:windowtext;mso-border-bottom-alt:#7F7F7F;mso-border-bottom-themecolor:
  text1;mso-border-bottom-themetint:128;mso-border-right-alt:windowtext;
  mso-border-style-alt:solid;mso-border-width-alt:.5pt;padding:0cm 5.4pt 0cm 5.4pt;
  height:10.5pt'>
  <p class=MsoNormal align=center style='text-align:center;mso-yfti-cnfc:1'><span
  lang=IT style='font-size:8.0pt;mso-bidi-font-size:9.0pt;font-family:"Times New Roman",serif;
  mso-fareast-font-family:宋体;mso-font-kerning:0pt;mso-ansi-language:IT;
  mso-bidi-font-weight:bold'>LSA<o:p></o:p></span></p>
  </td>
  <td style='border-top:solid #7F7F7F 1.0pt;mso-border-top-themecolor:text1;
  mso-border-top-themetint:128;border-left:none;border-bottom:solid #7F7F7F 1.0pt;
  mso-border-bottom-themecolor:text1;mso-border-bottom-themetint:128;
  border-right:none;mso-border-left-alt:solid windowtext .5pt;mso-border-top-alt:
  solid #7F7F7F .5pt;mso-border-top-themecolor:text1;mso-border-top-themetint:
  128;mso-border-left-alt:solid windowtext .5pt;mso-border-bottom-alt:solid #7F7F7F .5pt;
  mso-border-bottom-themecolor:text1;mso-border-bottom-themetint:128;
  padding:0cm 5.4pt 0cm 5.4pt;height:10.5pt'>
  <p class=MsoNormal align=center style='text-align:center;mso-yfti-cnfc:1'><span
  lang=IT style='font-size:8.0pt;mso-bidi-font-size:9.0pt;font-family:"Times New Roman",serif;
  mso-fareast-font-family:宋体;mso-font-kerning:0pt;mso-ansi-language:IT;
  mso-fareast-language:IT;mso-bidi-font-weight:bold'>Preference (%)<o:p></o:p></span></p>
  </td>
 </tr>
 <tr style='mso-yfti-irow:0;height:10.5pt'>
  <td rowspan=2 style='border-top:none;border-left:none;border-bottom:solid windowtext 1.0pt;
  border-right:solid windowtext 1.0pt;mso-border-top-alt:solid #7F7F7F .5pt;
  mso-border-top-themecolor:text1;mso-border-top-themetint:128;mso-border-top-alt:
  solid #7F7F7F .5pt;mso-border-top-themecolor:text1;mso-border-top-themetint:
  128;mso-border-bottom-alt:solid windowtext .5pt;mso-border-right-alt:solid windowtext .5pt;
  padding:0cm 5.4pt 0cm 5.4pt;height:10.5pt'>
  <p class=MsoNormal align=center style='text-align:center'><span lang=IT
  style='font-size:8.0pt;mso-bidi-font-size:9.0pt;font-family:"Times New Roman",serif;
  mso-fareast-font-family:等线;mso-font-kerning:0pt;mso-ansi-language:IT;
  mso-fareast-language:IT'>MTFAA</span><span lang=IT style='font-size:8.0pt;
  mso-bidi-font-size:9.0pt;font-family:"Times New Roman",serif;mso-fareast-font-family:
  宋体;mso-font-kerning:0pt;mso-ansi-language:IT;mso-fareast-language:IT'><o:p></o:p></span></p>
  </td>
  <td style='border:none;border-right:solid windowtext 1.0pt;mso-border-left-alt:
  solid windowtext .5pt;mso-border-left-alt:solid windowtext .5pt;mso-border-right-alt:
  solid windowtext .5pt;padding:0cm 5.4pt 0cm 5.4pt;height:10.5pt'>
  <p class=MsoNormal align=center style='text-align:center'><span
  style='font-size:8.0pt;font-family:"Malgun Gothic",sans-serif;mso-bidi-font-family:
  "Times New Roman";color:black;mso-font-kerning:0pt;mso-ansi-language:IT'>Ⅹ</span><span
  lang=IT style='font-size:8.0pt;mso-bidi-font-size:9.0pt;font-family:"Times New Roman",serif;
  mso-fareast-font-family:宋体;mso-font-kerning:0pt;mso-ansi-language:IT;
  mso-fareast-language:IT'><o:p></o:p></span></p>
  </td>
  <td style='border:none;mso-border-left-alt:solid windowtext .5pt;padding:
  0cm 5.4pt 0cm 5.4pt;height:10.5pt'>
  <p class=MsoNormal align=center style='text-align:center'><span lang=IT
  style='font-size:8.0pt;mso-bidi-font-size:9.0pt;font-family:"Times New Roman",serif;
  mso-fareast-font-family:宋体;mso-font-kerning:0pt;mso-ansi-language:IT;
  mso-fareast-language:IT'>38.0<o:p></o:p></span></p>
  </td>
 </tr>
 <tr style='mso-yfti-irow:1;mso-yfti-lastrow:yes;height:10.5pt'>
  <td style='border-top:none;border-left:none;border-bottom:solid #7F7F7F 1.0pt;
  mso-border-bottom-themecolor:text1;mso-border-bottom-themetint:128;
  border-right:solid windowtext 1.0pt;mso-border-left-alt:solid windowtext .5pt;
  mso-border-left-alt:solid windowtext .5pt;mso-border-bottom-alt:solid #7F7F7F .5pt;
  mso-border-bottom-themecolor:text1;mso-border-bottom-themetint:128;
  mso-border-right-alt:solid windowtext .5pt;padding:0cm 5.4pt 0cm 5.4pt;
  height:10.5pt'>
  <p class=MsoNormal align=center style='text-align:center'><span
  style='font-size:8.0pt;font-family:"Malgun Gothic",sans-serif;mso-bidi-font-family:
  "Times New Roman";color:black;mso-font-kerning:0pt;mso-ansi-language:IT'>√</span><span
  lang=IT style='font-size:8.0pt;mso-bidi-font-size:9.0pt;font-family:"Times New Roman",serif;
  mso-fareast-font-family:宋体;mso-font-kerning:0pt;mso-ansi-language:IT;
  mso-fareast-language:IT'><o:p></o:p></span></p>
  </td>
  <td style='border:none;border-bottom:solid #7F7F7F 1.0pt;mso-border-bottom-themecolor:
  text1;mso-border-bottom-themetint:128;mso-border-left-alt:solid windowtext .5pt;
  mso-border-left-alt:solid windowtext .5pt;mso-border-bottom-alt:solid #7F7F7F .5pt;
  mso-border-bottom-themecolor:text1;mso-border-bottom-themetint:128;
  padding:0cm 5.4pt 0cm 5.4pt;height:10.5pt'>
  <p class=MsoNormal align=center style='text-align:center'><b
  style='mso-bidi-font-weight:normal'><span lang=IT style='font-size:8.0pt;
  mso-bidi-font-size:9.0pt;font-family:"Times New Roman",serif;mso-fareast-font-family:
  宋体;mso-font-kerning:0pt;mso-ansi-language:IT;mso-fareast-language:IT'>62.0<o:p></o:p></span></b></p>
  </td>
 </tr>
</table>

</div>

<p class=MsoNormal><span lang=EN-US><o:p>&nbsp;</o:p></span></p>

</div>

</body>

The proposed method also reduces computational complexity in spectral attention and the statistics are given in Table below


<body lang=ZH-CN style='tab-interval:21.0pt;text-justify-trim:punctuation'>

<div class=WordSection1 style='layout-grid:15.6pt'>

<div align=center>

<table class=PlainTable22 border=1 cellspacing=0 cellpadding=0
 style='border-collapse:collapse;border:none;mso-border-top-alt:solid #7F7F7F .5pt;
 mso-border-top-themecolor:text1;mso-border-top-themetint:128;mso-border-bottom-alt:
 solid #7F7F7F .5pt;mso-border-bottom-themecolor:text1;mso-border-bottom-themetint:
 128;mso-yfti-tbllook:1568;mso-padding-alt:0cm 5.4pt 0cm 5.4pt'>
 <tr style='mso-yfti-irow:-1;mso-yfti-firstrow:yes;mso-yfti-lastfirstrow:yes;
  height:10.5pt'>
  <td style='border-top:solid #7F7F7F 1.0pt;mso-border-top-themecolor:text1;
  mso-border-top-themetint:128;border-left:none;border-bottom:solid #7F7F7F 1.0pt;
  mso-border-bottom-themecolor:text1;mso-border-bottom-themetint:128;
  border-right:solid windowtext 1.0pt;mso-border-top-alt:solid #7F7F7F .5pt;
  mso-border-top-themecolor:text1;mso-border-top-themetint:128;mso-border-bottom-alt:
  solid #7F7F7F .5pt;mso-border-bottom-themecolor:text1;mso-border-bottom-themetint:
  128;mso-border-right-alt:solid windowtext .5pt;padding:0cm 5.4pt 0cm 5.4pt;
  height:10.5pt'>
  <p class=MsoNormal align=center style='text-align:center;mso-yfti-cnfc:1'><span
  lang=IT style='font-size:8.0pt;mso-bidi-font-size:9.0pt;font-family:"Times New Roman",serif;
  mso-fareast-font-family:宋体;mso-font-kerning:0pt;mso-ansi-language:IT;
  mso-fareast-language:IT;mso-bidi-font-weight:bold'>Model<o:p></o:p></span></p>
  </td>
  <td style='border-top:solid #7F7F7F 1.0pt;mso-border-top-themecolor:text1;
  mso-border-top-themetint:128;border-left:none;border-bottom:solid #7F7F7F 1.0pt;
  mso-border-bottom-themecolor:text1;mso-border-bottom-themetint:128;
  border-right:none;mso-border-left-alt:solid windowtext .5pt;mso-border-top-alt:
  solid #7F7F7F .5pt;mso-border-top-themecolor:text1;mso-border-top-themetint:
  128;mso-border-left-alt:solid windowtext .5pt;mso-border-bottom-alt:solid #7F7F7F .5pt;
  mso-border-bottom-themecolor:text1;mso-border-bottom-themetint:128;
  padding:0cm 5.4pt 0cm 5.4pt;height:10.5pt'>
  <p class=MsoNormal align=center style='text-align:center;mso-yfti-cnfc:1'><span
  lang=IT style='font-size:8.0pt;mso-bidi-font-size:9.0pt;font-family:"Times New Roman",serif;
  mso-fareast-font-family:宋体;mso-font-kerning:0pt;mso-ansi-language:IT;
  mso-fareast-language:IT;mso-bidi-font-weight:bold'>Percentage of complexity
  reduction </span><b style='mso-bidi-font-weight:normal'><span lang=IT
  style='font-size:8.0pt;mso-bidi-font-size:9.0pt;font-family:"Times New Roman",serif;
  mso-fareast-font-family:宋体;mso-font-kerning:0pt;mso-ansi-language:IT;
  mso-fareast-language:IT'><o:p></o:p></span></b></p>
  <p class=MsoNormal align=center style='text-align:center;mso-yfti-cnfc:1'><span
  lang=IT style='font-size:8.0pt;mso-bidi-font-size:9.0pt;font-family:"Times New Roman",serif;
  mso-fareast-font-family:宋体;mso-font-kerning:0pt;mso-ansi-language:IT;
  mso-fareast-language:IT;mso-bidi-font-weight:bold'>in spectral attention (%)<o:p></o:p></span></p>
  </td>
 </tr>
 <tr style='mso-yfti-irow:0;height:10.5pt'>
  <td style='border:none;border-right:solid windowtext 1.0pt;mso-border-top-alt:
  solid #7F7F7F .5pt;mso-border-top-themecolor:text1;mso-border-top-themetint:
  128;mso-border-top-alt:solid #7F7F7F .5pt;mso-border-top-themecolor:text1;
  mso-border-top-themetint:128;mso-border-right-alt:solid windowtext .5pt;
  padding:0cm 5.4pt 0cm 5.4pt;height:10.5pt'>
  <p class=MsoNormal align=center style='text-align:center'><span lang=IT
  style='font-size:8.0pt;mso-bidi-font-size:9.0pt;font-family:"Times New Roman",serif;
  mso-fareast-font-family:等线;mso-font-kerning:0pt;mso-ansi-language:IT;
  mso-fareast-language:IT'>MTFAA</span><span lang=IT style='font-size:8.0pt;
  mso-bidi-font-size:9.0pt;font-family:"Times New Roman",serif;mso-fareast-font-family:
  宋体;mso-font-kerning:0pt;mso-ansi-language:IT;mso-fareast-language:IT'><o:p></o:p></span></p>
  </td>
  <td style='border:none;mso-border-left-alt:solid windowtext .5pt;padding:
  0cm 5.4pt 0cm 5.4pt;height:10.5pt'>
  <p class=MsoNormal align=center style='text-align:center'><span lang=IT
  style='font-size:8.0pt;mso-bidi-font-size:9.0pt;font-family:"Times New Roman",serif;
  mso-fareast-font-family:宋体;mso-font-kerning:0pt;mso-ansi-language:IT;
  mso-fareast-language:IT'>63.2<o:p></o:p></span></p>
  </td>
 </tr>
 <tr style='mso-yfti-irow:1;mso-yfti-lastrow:yes;height:10.5pt'>
  <td style='border-top:none;border-left:none;border-bottom:solid windowtext 1.0pt;
  border-right:solid windowtext 1.0pt;mso-border-bottom-alt:solid windowtext .5pt;
  mso-border-right-alt:solid windowtext .5pt;padding:0cm 5.4pt 0cm 5.4pt;
  height:10.5pt'>
  <p class=MsoNormal align=center style='text-align:center'><span lang=IT
  style='font-size:8.0pt;mso-bidi-font-size:9.0pt;font-family:"Times New Roman",serif;
  mso-fareast-font-family:宋体;mso-font-kerning:0pt;mso-ansi-language:IT'>DPARN<o:p></o:p></span></p>
  </td>
  <td style='border:none;border-bottom:solid #7F7F7F 1.0pt;mso-border-bottom-themecolor:
  text1;mso-border-bottom-themetint:128;mso-border-left-alt:solid windowtext .5pt;
  mso-border-left-alt:solid windowtext .5pt;mso-border-bottom-alt:solid #7F7F7F .5pt;
  mso-border-bottom-themecolor:text1;mso-border-bottom-themetint:128;
  padding:0cm 5.4pt 0cm 5.4pt;height:10.5pt'>
  <p class=MsoNormal align=center style='text-align:center'><span lang=IT
  style='font-size:8.0pt;mso-bidi-font-size:9.0pt;font-family:"Times New Roman",serif;
  mso-fareast-font-family:宋体;mso-font-kerning:0pt;mso-ansi-language:IT;
  mso-fareast-language:IT'>25.4<o:p></o:p></span></p>
  </td>
 </tr>
</table>

</div>

<p class=MsoNormal><span lang=EN-US><o:p>&nbsp;</o:p></span></p>

</div>

</body>

We compare the modified MTFAA model with previous full-band SOTA methods on VCTK-DEMAND dataset and the results are listed in table below


<body lang=ZH-CN style='tab-interval:21.0pt;text-justify-trim:punctuation'>

<div class=WordSection1 style='layout-grid:15.6pt'>

<div align=center>

<table class=1 border=1 cellspacing=0 cellpadding=0 width=447 style='border-collapse:
 collapse;mso-table-layout-alt:fixed;border:none;mso-border-alt:solid windowtext .5pt;
 mso-yfti-tbllook:1184;mso-padding-alt:0cm 2.85pt 0cm 2.85pt'>
 <tr style='mso-yfti-irow:0;mso-yfti-firstrow:yes;height:12.75pt'>
  <td width=104 nowrap valign=top style='width:78.0pt;border-top:solid windowtext 1.0pt;
  border-left:none;border-bottom:solid windowtext 1.0pt;border-right:none;
  mso-border-top-alt:solid windowtext .5pt;mso-border-bottom-alt:solid windowtext .5pt;
  padding:0cm 2.85pt 0cm 2.85pt;height:12.75pt'>
  <p class=MsoNormal align=left style='text-align:left'><span lang=EN-US
  style='font-size:8.0pt;font-family:"Times New Roman",serif'>Models<o:p></o:p></span></p>
  </td>
  <td width=38 valign=top style='width:1.0cm;border-top:solid windowtext 1.0pt;
  border-left:none;border-bottom:solid windowtext 1.0pt;border-right:none;
  mso-border-top-alt:solid windowtext .5pt;mso-border-bottom-alt:solid windowtext .5pt;
  padding:0cm 2.85pt 0cm 2.85pt;height:12.75pt'>
  <p class=MsoNormal align=center style='text-align:center'><span lang=EN-US
  style='font-size:8.0pt;font-family:"Times New Roman",serif'>Year<o:p></o:p></span></p>
  </td>
  <td width=60 valign=top style='width:45.1pt;border-top:solid windowtext 1.0pt;
  border-left:none;border-bottom:solid windowtext 1.0pt;border-right:none;
  mso-border-top-alt:solid windowtext .5pt;mso-border-bottom-alt:solid windowtext .5pt;
  padding:0cm 2.85pt 0cm 2.85pt;height:12.75pt'>
  <p class=MsoNormal align=center style='text-align:center'><span class=GramE><span
  lang=EN-US style='font-size:8.0pt;font-family:"Times New Roman",serif'>Param.(</span></span><span
  lang=EN-US style='font-size:8.0pt;font-family:"Times New Roman",serif'>M)<o:p></o:p></span></p>
  </td>
  <td width=49 nowrap valign=top style='width:36.7pt;border-top:solid windowtext 1.0pt;
  border-left:none;border-bottom:solid windowtext 1.0pt;border-right:none;
  mso-border-top-alt:solid windowtext .5pt;mso-border-bottom-alt:solid windowtext .5pt;
  padding:0cm 2.85pt 0cm 2.85pt;height:12.75pt'>
  <p class=MsoNormal align=center style='text-align:center'><span lang=EN-US
  style='font-size:8.0pt;font-family:"Times New Roman",serif'>PESQ<o:p></o:p></span></p>
  </td>
  <td width=49 nowrap valign=top style='width:36.75pt;border-top:solid windowtext 1.0pt;
  border-left:none;border-bottom:solid windowtext 1.0pt;border-right:none;
  mso-border-top-alt:solid windowtext .5pt;mso-border-bottom-alt:solid windowtext .5pt;
  padding:0cm 2.85pt 0cm 2.85pt;height:12.75pt'>
  <p class=MsoNormal align=center style='text-align:center'><span class=GramE><span
  lang=EN-US style='font-size:8.0pt;font-family:"Times New Roman",serif'>STOI(</span></span><span
  lang=EN-US style='font-size:8.0pt;font-family:"Times New Roman",serif'>%)<o:p></o:p></span></p>
  </td>
  <td width=49 nowrap valign=top style='width:36.7pt;border-top:solid windowtext 1.0pt;
  border-left:none;border-bottom:solid windowtext 1.0pt;border-right:none;
  mso-border-top-alt:solid windowtext .5pt;mso-border-bottom-alt:solid windowtext .5pt;
  padding:0cm 2.85pt 0cm 2.85pt;height:12.75pt'>
  <p class=MsoNormal align=center style='text-align:center'><span lang=EN-US
  style='font-size:8.0pt;font-family:"Times New Roman",serif'>CSIG<o:p></o:p></span></p>
  </td>
  <td width=49 nowrap valign=top style='width:36.75pt;border-top:solid windowtext 1.0pt;
  border-left:none;border-bottom:solid windowtext 1.0pt;border-right:none;
  mso-border-top-alt:solid windowtext .5pt;mso-border-bottom-alt:solid windowtext .5pt;
  padding:0cm 2.85pt 0cm 2.85pt;height:12.75pt'>
  <p class=MsoNormal align=center style='text-align:center'><span lang=EN-US
  style='font-size:8.0pt;font-family:"Times New Roman",serif'>CBAK<o:p></o:p></span></p>
  </td>
  <td width=49 nowrap valign=top style='width:36.75pt;border-top:solid windowtext 1.0pt;
  border-left:none;border-bottom:solid windowtext 1.0pt;border-right:none;
  mso-border-top-alt:solid windowtext .5pt;mso-border-bottom-alt:solid windowtext .5pt;
  padding:0cm 2.85pt 0cm 2.85pt;height:12.75pt'>
  <p class=MsoNormal align=center style='text-align:center'><span lang=EN-US
  style='font-size:8.0pt;font-family:"Times New Roman",serif'>COVL<o:p></o:p></span></p>
  </td>
 </tr>
 <tr style='mso-yfti-irow:1;height:12.75pt'>
  <td width=104 nowrap valign=top style='width:78.0pt;border:none;mso-border-top-alt:
  solid windowtext .5pt;padding:0cm 2.85pt 0cm 2.85pt;height:12.75pt'>
  <p class=MsoNormal align=left style='text-align:left'><span lang=EN-US
  style='font-size:8.0pt;font-family:"Times New Roman",serif'>Noisy<o:p></o:p></span></p>
  </td>
  <td width=38 valign=top style='width:1.0cm;border:none;mso-border-top-alt:
  solid windowtext .5pt;padding:0cm 2.85pt 0cm 2.85pt;height:12.75pt'>
  <p class=MsoNormal align=center style='text-align:center'><span lang=EN-US
  style='font-size:8.0pt;font-family:"Times New Roman",serif'>-<o:p></o:p></span></p>
  </td>
  <td width=60 valign=top style='width:45.1pt;border:none;mso-border-top-alt:
  solid windowtext .5pt;padding:0cm 2.85pt 0cm 2.85pt;height:12.75pt'>
  <p class=MsoNormal align=center style='text-align:center'><span lang=EN-US
  style='font-size:8.0pt;font-family:"Times New Roman",serif'>-<o:p></o:p></span></p>
  </td>
  <td width=49 nowrap valign=top style='width:36.7pt;border:none;mso-border-top-alt:
  solid windowtext .5pt;padding:0cm 2.85pt 0cm 2.85pt;height:12.75pt'>
  <p class=MsoNormal align=center style='text-align:center'><span lang=EN-US
  style='font-size:8.0pt;font-family:"Times New Roman",serif'>1.97<o:p></o:p></span></p>
  </td>
  <td width=49 nowrap valign=top style='width:36.75pt;border:none;mso-border-top-alt:
  solid windowtext .5pt;padding:0cm 2.85pt 0cm 2.85pt;height:12.75pt'>
  <p class=MsoNormal align=center style='text-align:center'><span lang=EN-US
  style='font-size:8.0pt;font-family:"Times New Roman",serif'>92.1<o:p></o:p></span></p>
  </td>
  <td width=49 nowrap valign=top style='width:36.7pt;border:none;mso-border-top-alt:
  solid windowtext .5pt;padding:0cm 2.85pt 0cm 2.85pt;height:12.75pt'>
  <p class=MsoNormal align=center style='text-align:center'><span lang=EN-US
  style='font-size:8.0pt;font-family:"Times New Roman",serif'>3.34<o:p></o:p></span></p>
  </td>
  <td width=49 nowrap valign=top style='width:36.75pt;border:none;mso-border-top-alt:
  solid windowtext .5pt;padding:0cm 2.85pt 0cm 2.85pt;height:12.75pt'>
  <p class=MsoNormal align=center style='text-align:center'><span lang=EN-US
  style='font-size:8.0pt;font-family:"Times New Roman",serif'>2.44<o:p></o:p></span></p>
  </td>
  <td width=49 nowrap valign=top style='width:36.75pt;border:none;mso-border-top-alt:
  solid windowtext .5pt;padding:0cm 2.85pt 0cm 2.85pt;height:12.75pt'>
  <p class=MsoNormal align=center style='text-align:center'><span lang=EN-US
  style='font-size:8.0pt;font-family:"Times New Roman",serif'>2.63<o:p></o:p></span></p>
  </td>
 </tr>
 <tr style='mso-yfti-irow:2;height:12.75pt'>
  <td width=104 nowrap valign=top style='width:78.0pt;border:none;padding:0cm 2.85pt 0cm 2.85pt;
  height:12.75pt'>
  <p class=MsoNormal align=left style='text-align:left'><span class=SpellE><span
  lang=EN-US style='font-size:8.0pt;font-family:"Times New Roman",serif'>RNNoise</span></span><span
  lang=EN-US style='font-size:8.0pt;font-family:"Times New Roman",serif'><o:p></o:p></span></p>
  </td>
  <td width=38 valign=top style='width:1.0cm;border:none;padding:0cm 2.85pt 0cm 2.85pt;
  height:12.75pt'>
  <p class=MsoNormal align=center style='text-align:center'><span lang=EN-US
  style='font-size:8.0pt;font-family:"Times New Roman",serif'>2020<o:p></o:p></span></p>
  </td>
  <td width=60 valign=top style='width:45.1pt;border:none;padding:0cm 2.85pt 0cm 2.85pt;
  height:12.75pt'>
  <p class=MsoNormal align=center style='text-align:center'><span lang=EN-US
  style='font-size:8.0pt;font-family:"Times New Roman",serif'>0.06<o:p></o:p></span></p>
  </td>
  <td width=49 nowrap valign=top style='width:36.7pt;border:none;padding:0cm 2.85pt 0cm 2.85pt;
  height:12.75pt'>
  <p class=MsoNormal align=center style='text-align:center'><span lang=EN-US
  style='font-size:8.0pt;font-family:"Times New Roman",serif'>2.33<o:p></o:p></span></p>
  </td>
  <td width=49 nowrap valign=top style='width:36.75pt;border:none;padding:0cm 2.85pt 0cm 2.85pt;
  height:12.75pt'>
  <p class=MsoNormal align=center style='text-align:center'><span lang=EN-US
  style='font-size:8.0pt;font-family:"Times New Roman",serif'>92.2<o:p></o:p></span></p>
  </td>
  <td width=49 nowrap valign=top style='width:36.7pt;border:none;padding:0cm 2.85pt 0cm 2.85pt;
  height:12.75pt'>
  <p class=MsoNormal align=center style='text-align:center'><span lang=EN-US
  style='font-size:8.0pt;font-family:"Times New Roman",serif;mso-bidi-font-weight:
  bold'>3.40<o:p></o:p></span></p>
  </td>
  <td width=49 nowrap valign=top style='width:36.75pt;border:none;padding:0cm 2.85pt 0cm 2.85pt;
  height:12.75pt'>
  <p class=MsoNormal align=center style='text-align:center'><span lang=EN-US
  style='font-size:8.0pt;font-family:"Times New Roman",serif;mso-bidi-font-weight:
  bold'>2.51<o:p></o:p></span></p>
  </td>
  <td width=49 nowrap valign=top style='width:36.75pt;border:none;padding:0cm 2.85pt 0cm 2.85pt;
  height:12.75pt'>
  <p class=MsoNormal align=center style='text-align:center'><span lang=EN-US
  style='font-size:8.0pt;font-family:"Times New Roman",serif'>2.84<o:p></o:p></span></p>
  </td>
 </tr>
 <tr style='mso-yfti-irow:3;height:12.75pt'>
  <td width=104 nowrap valign=top style='width:78.0pt;border:none;padding:0cm 2.85pt 0cm 2.85pt;
  height:12.75pt'>
  <p class=MsoNormal align=left style='text-align:left'><span lang=EN-US
  style='font-size:8.0pt;font-family:"Times New Roman",serif;mso-bidi-font-weight:
  bold'>PercepNet</span><span lang=EN-US style='font-size:8.0pt;font-family:
  "Times New Roman",serif'><o:p></o:p></span></p>
  </td>
  <td width=38 valign=top style='width:1.0cm;border:none;padding:0cm 2.85pt 0cm 2.85pt;
  height:12.75pt'>
  <p class=MsoNormal align=center style='text-align:center'><span lang=EN-US
  style='font-size:8.0pt;font-family:"Times New Roman",serif'>2020<o:p></o:p></span></p>
  </td>
  <td width=60 valign=top style='width:45.1pt;border:none;padding:0cm 2.85pt 0cm 2.85pt;
  height:12.75pt'>
  <p class=MsoNormal align=center style='text-align:center'><span lang=EN-US
  style='font-size:8.0pt;font-family:"Times New Roman",serif'>8.00<o:p></o:p></span></p>
  </td>
  <td width=49 nowrap valign=top style='width:36.7pt;border:none;padding:0cm 2.85pt 0cm 2.85pt;
  height:12.75pt'>
  <p class=MsoNormal align=center style='text-align:center'><span lang=EN-US
  style='font-size:8.0pt;font-family:"Times New Roman",serif'>2.73<o:p></o:p></span></p>
  </td>
  <td width=49 nowrap valign=top style='width:36.75pt;border:none;padding:0cm 2.85pt 0cm 2.85pt;
  height:12.75pt'>
  <p class=MsoNormal align=center style='text-align:center'><span lang=EN-US
  style='font-size:8.0pt;font-family:"Times New Roman",serif'>-<o:p></o:p></span></p>
  </td>
  <td width=49 nowrap valign=top style='width:36.7pt;border:none;padding:0cm 2.85pt 0cm 2.85pt;
  height:12.75pt'>
  <p class=MsoNormal align=center style='text-align:center'><span lang=EN-US
  style='font-size:8.0pt;font-family:"Times New Roman",serif'>-<span
  style='mso-bidi-font-weight:bold'><o:p></o:p></span></span></p>
  </td>
  <td width=49 nowrap valign=top style='width:36.75pt;border:none;padding:0cm 2.85pt 0cm 2.85pt;
  height:12.75pt'>
  <p class=MsoNormal align=center style='text-align:center'><span lang=EN-US
  style='font-size:8.0pt;font-family:"Times New Roman",serif'>-<span
  style='mso-bidi-font-weight:bold'><o:p></o:p></span></span></p>
  </td>
  <td width=49 nowrap valign=top style='width:36.75pt;border:none;padding:0cm 2.85pt 0cm 2.85pt;
  height:12.75pt'>
  <p class=MsoNormal align=center style='text-align:center'><span lang=EN-US
  style='font-size:8.0pt;font-family:"Times New Roman",serif'>-<o:p></o:p></span></p>
  </td>
 </tr>
 <tr style='mso-yfti-irow:4;height:12.75pt'>
  <td width=104 nowrap valign=top style='width:78.0pt;border:none;padding:0cm 2.85pt 0cm 2.85pt;
  height:12.75pt'>
  <p class=MsoNormal align=left style='text-align:left'><span lang=EN-US
  style='font-size:8.0pt;font-family:"Times New Roman",serif;mso-bidi-font-weight:
  bold'>CTS-Net(full) <o:p></o:p></span></p>
  </td>
  <td width=38 valign=top style='width:1.0cm;border:none;padding:0cm 2.85pt 0cm 2.85pt;
  height:12.75pt'>
  <p class=MsoNormal align=center style='text-align:center'><span lang=EN-US
  style='font-size:8.0pt;font-family:"Times New Roman",serif;mso-bidi-font-weight:
  bold'>2020<o:p></o:p></span></p>
  </td>
  <td width=60 valign=top style='width:45.1pt;border:none;padding:0cm 2.85pt 0cm 2.85pt;
  height:12.75pt'>
  <p class=MsoNormal align=center style='text-align:center'><span lang=EN-US
  style='font-size:8.0pt;font-family:"Times New Roman",serif;mso-bidi-font-weight:
  bold'>7.09<o:p></o:p></span></p>
  </td>
  <td width=49 nowrap valign=top style='width:36.7pt;border:none;padding:0cm 2.85pt 0cm 2.85pt;
  height:12.75pt'>
  <p class=MsoNormal align=center style='text-align:center'><span lang=EN-US
  style='font-size:8.0pt;font-family:"Times New Roman",serif;mso-bidi-font-weight:
  bold'>2.92<o:p></o:p></span></p>
  </td>
  <td width=49 nowrap valign=top style='width:36.75pt;border:none;padding:0cm 2.85pt 0cm 2.85pt;
  height:12.75pt'>
  <p class=MsoNormal align=center style='text-align:center'><span lang=EN-US
  style='font-size:8.0pt;font-family:"Times New Roman",serif'>94.3<span
  style='mso-bidi-font-weight:bold'><o:p></o:p></span></span></p>
  </td>
  <td width=49 nowrap valign=top style='width:36.7pt;border:none;padding:0cm 2.85pt 0cm 2.85pt;
  height:12.75pt'>
  <p class=MsoNormal align=center style='text-align:center'><span lang=EN-US
  style='font-size:8.0pt;font-family:"Times New Roman",serif'>4.22<o:p></o:p></span></p>
  </td>
  <td width=49 nowrap valign=top style='width:36.75pt;border:none;padding:0cm 2.85pt 0cm 2.85pt;
  height:12.75pt'>
  <p class=MsoNormal align=center style='text-align:center'><span lang=EN-US
  style='font-size:8.0pt;font-family:"Times New Roman",serif'>3.43<o:p></o:p></span></p>
  </td>
  <td width=49 nowrap valign=top style='width:36.75pt;border:none;padding:0cm 2.85pt 0cm 2.85pt;
  height:12.75pt'>
  <p class=MsoNormal align=center style='text-align:center'><span lang=EN-US
  style='font-size:8.0pt;font-family:"Times New Roman",serif'>3.62<span
  style='mso-bidi-font-weight:bold'><o:p></o:p></span></span></p>
  </td>
 </tr>
 <tr style='mso-yfti-irow:5;height:12.75pt'>
  <td width=104 nowrap valign=top style='width:78.0pt;border:none;padding:0cm 2.85pt 0cm 2.85pt;
  height:12.75pt'>
  <p class=MsoNormal align=left style='text-align:left'><span lang=EN-US
  style='font-size:8.0pt;font-family:"Times New Roman",serif'>DCCRN<span
  style='mso-bidi-font-weight:bold'><o:p></o:p></span></span></p>
  </td>
  <td width=38 valign=top style='width:1.0cm;border:none;padding:0cm 2.85pt 0cm 2.85pt;
  height:12.75pt'>
  <p class=MsoNormal align=center style='text-align:center'><span lang=EN-US
  style='font-size:8.0pt;font-family:"Times New Roman",serif'>2020<o:p></o:p></span></p>
  </td>
  <td width=60 valign=top style='width:45.1pt;border:none;padding:0cm 2.85pt 0cm 2.85pt;
  height:12.75pt'>
  <p class=MsoNormal align=center style='text-align:center'><span lang=EN-US
  style='font-size:8.0pt;font-family:"Times New Roman",serif'>3.70<o:p></o:p></span></p>
  </td>
  <td width=49 nowrap valign=top style='width:36.7pt;border:none;padding:0cm 2.85pt 0cm 2.85pt;
  height:12.75pt'>
  <p class=MsoNormal align=center style='text-align:center'><span lang=EN-US
  style='font-size:8.0pt;font-family:"Times New Roman",serif'>2.54<o:p></o:p></span></p>
  </td>
  <td width=49 nowrap valign=top style='width:36.75pt;border:none;padding:0cm 2.85pt 0cm 2.85pt;
  height:12.75pt'>
  <p class=MsoNormal align=center style='text-align:center'><span lang=EN-US
  style='font-size:8.0pt;font-family:"Times New Roman",serif;mso-bidi-font-weight:
  bold'>93.8</span><span lang=EN-US style='font-size:8.0pt;font-family:"Times New Roman",serif'><o:p></o:p></span></p>
  </td>
  <td width=49 nowrap valign=top style='width:36.7pt;border:none;padding:0cm 2.85pt 0cm 2.85pt;
  height:12.75pt'>
  <p class=MsoNormal align=center style='text-align:center'><span lang=EN-US
  style='font-size:8.0pt;font-family:"Times New Roman",serif;mso-bidi-font-weight:
  bold'>3.74</span><span lang=EN-US style='font-size:8.0pt;font-family:"Times New Roman",serif'><o:p></o:p></span></p>
  </td>
  <td width=49 nowrap valign=top style='width:36.75pt;border:none;padding:0cm 2.85pt 0cm 2.85pt;
  height:12.75pt'>
  <p class=MsoNormal align=center style='text-align:center'><span lang=EN-US
  style='font-size:8.0pt;font-family:"Times New Roman",serif;mso-bidi-font-weight:
  bold'>3.13</span><span lang=EN-US style='font-size:8.0pt;font-family:"Times New Roman",serif'><o:p></o:p></span></p>
  </td>
  <td width=49 nowrap valign=top style='width:36.75pt;border:none;padding:0cm 2.85pt 0cm 2.85pt;
  height:12.75pt'>
  <p class=MsoNormal align=center style='text-align:center'><span lang=EN-US
  style='font-size:8.0pt;font-family:"Times New Roman",serif;mso-bidi-font-weight:
  bold'>2.75</span><span lang=EN-US style='font-size:8.0pt;font-family:"Times New Roman",serif'><o:p></o:p></span></p>
  </td>
 </tr>
 <tr style='mso-yfti-irow:6;height:12.75pt'>
  <td width=104 nowrap valign=top style='width:78.0pt;border:none;padding:0cm 2.85pt 0cm 2.85pt;
  height:12.75pt'>
  <p class=MsoNormal align=left style='text-align:left'><span lang=EN-US
  style='font-size:8.0pt;font-family:"Times New Roman",serif;mso-bidi-font-weight:
  bold'>NSNet2</span><span lang=EN-US style='font-size:8.0pt;font-family:"Times New Roman",serif'><o:p></o:p></span></p>
  </td>
  <td width=38 valign=top style='width:1.0cm;border:none;padding:0cm 2.85pt 0cm 2.85pt;
  height:12.75pt'>
  <p class=MsoNormal align=center style='text-align:center'><span lang=EN-US
  style='font-size:8.0pt;font-family:"Times New Roman",serif'>2021<o:p></o:p></span></p>
  </td>
  <td width=60 valign=top style='width:45.1pt;border:none;padding:0cm 2.85pt 0cm 2.85pt;
  height:12.75pt'>
  <p class=MsoNormal align=center style='text-align:center'><span lang=EN-US
  style='font-size:8.0pt;font-family:"Times New Roman",serif;mso-bidi-font-weight:
  bold'>6.17</span><span lang=EN-US style='font-size:8.0pt;font-family:"Times New Roman",serif'><o:p></o:p></span></p>
  </td>
  <td width=49 nowrap valign=top style='width:36.7pt;border:none;padding:0cm 2.85pt 0cm 2.85pt;
  height:12.75pt'>
  <p class=MsoNormal align=center style='text-align:center'><span lang=EN-US
  style='font-size:8.0pt;font-family:"Times New Roman",serif;mso-bidi-font-weight:
  bold'>2.47</span><span lang=EN-US style='font-size:8.0pt;font-family:"Times New Roman",serif'><o:p></o:p></span></p>
  </td>
  <td width=49 nowrap valign=top style='width:36.75pt;border:none;padding:0cm 2.85pt 0cm 2.85pt;
  height:12.75pt'>
  <p class=MsoNormal align=center style='text-align:center'><span lang=EN-US
  style='font-size:8.0pt;font-family:"Times New Roman",serif;mso-bidi-font-weight:
  bold'>90.3<o:p></o:p></span></p>
  </td>
  <td width=49 nowrap valign=top style='width:36.7pt;border:none;padding:0cm 2.85pt 0cm 2.85pt;
  height:12.75pt'>
  <p class=MsoNormal align=center style='text-align:center'><span lang=EN-US
  style='font-size:8.0pt;font-family:"Times New Roman",serif'>3.23<span
  style='mso-bidi-font-weight:bold'><o:p></o:p></span></span></p>
  </td>
  <td width=49 nowrap valign=top style='width:36.75pt;border:none;padding:0cm 2.85pt 0cm 2.85pt;
  height:12.75pt'>
  <p class=MsoNormal align=center style='text-align:center'><span lang=EN-US
  style='font-size:8.0pt;font-family:"Times New Roman",serif'>2.99<span
  style='mso-bidi-font-weight:bold'><o:p></o:p></span></span></p>
  </td>
  <td width=49 nowrap valign=top style='width:36.75pt;border:none;padding:0cm 2.85pt 0cm 2.85pt;
  height:12.75pt'>
  <p class=MsoNormal align=center style='text-align:center'><span lang=EN-US
  style='font-size:8.0pt;font-family:"Times New Roman",serif;mso-bidi-font-weight:
  bold'>2.90<o:p></o:p></span></p>
  </td>
 </tr>
 <tr style='mso-yfti-irow:7;height:12.75pt'>
  <td width=104 nowrap valign=top style='width:78.0pt;border:none;padding:0cm 2.85pt 0cm 2.85pt;
  height:12.75pt'>
  <p class=MsoNormal align=left style='text-align:left'><span lang=EN-US
  style='font-size:8.0pt;font-family:"Times New Roman",serif;mso-bidi-font-weight:
  bold'>S-DCCRN<o:p></o:p></span></p>
  </td>
  <td width=38 valign=top style='width:1.0cm;border:none;padding:0cm 2.85pt 0cm 2.85pt;
  height:12.75pt'>
  <p class=MsoNormal align=center style='text-align:center'><span lang=EN-US
  style='font-size:8.0pt;font-family:"Times New Roman",serif;mso-bidi-font-weight:
  bold'>2022<o:p></o:p></span></p>
  </td>
  <td width=60 valign=top style='width:45.1pt;border:none;padding:0cm 2.85pt 0cm 2.85pt;
  height:12.75pt'>
  <p class=MsoNormal align=center style='text-align:center'><span lang=EN-US
  style='font-size:8.0pt;font-family:"Times New Roman",serif;mso-bidi-font-weight:
  bold'>2.34<o:p></o:p></span></p>
  </td>
  <td width=49 nowrap valign=top style='width:36.7pt;border:none;padding:0cm 2.85pt 0cm 2.85pt;
  height:12.75pt'>
  <p class=MsoNormal align=center style='text-align:center'><span lang=EN-US
  style='font-size:8.0pt;font-family:"Times New Roman",serif;mso-bidi-font-weight:
  bold'>2.84<o:p></o:p></span></p>
  </td>
  <td width=49 nowrap valign=top style='width:36.75pt;border:none;padding:0cm 2.85pt 0cm 2.85pt;
  height:12.75pt'>
  <p class=MsoNormal align=center style='text-align:center'><span lang=EN-US
  style='font-size:8.0pt;font-family:"Times New Roman",serif'>94.0<o:p></o:p></span></p>
  </td>
  <td width=49 nowrap valign=top style='width:36.7pt;border:none;padding:0cm 2.85pt 0cm 2.85pt;
  height:12.75pt'>
  <p class=MsoNormal align=center style='text-align:center'><span lang=EN-US
  style='font-size:8.0pt;font-family:"Times New Roman",serif'>4.03<o:p></o:p></span></p>
  </td>
  <td width=49 nowrap valign=top style='width:36.75pt;border:none;padding:0cm 2.85pt 0cm 2.85pt;
  height:12.75pt'>
  <p class=MsoNormal align=center style='text-align:center'><span lang=EN-US
  style='font-size:8.0pt;font-family:"Times New Roman",serif'>3.43<o:p></o:p></span></p>
  </td>
  <td width=49 nowrap valign=top style='width:36.75pt;border:none;padding:0cm 2.85pt 0cm 2.85pt;
  height:12.75pt'>
  <p class=MsoNormal align=center style='text-align:center'><span lang=EN-US
  style='font-size:8.0pt;font-family:"Times New Roman",serif'>2.97<o:p></o:p></span></p>
  </td>
 </tr>
 <tr style='mso-yfti-irow:8;height:12.75pt'>
  <td width=104 nowrap valign=top style='width:78.0pt;border:none;padding:0cm 2.85pt 0cm 2.85pt;
  height:12.75pt'>
  <p class=MsoNormal align=left style='text-align:left'><span class=SpellE><span
  lang=EN-US style='font-size:8.0pt;font-family:"Times New Roman",serif;
  mso-bidi-font-weight:bold'>FullSubNet</span></span><span lang=EN-US
  style='font-size:8.0pt;font-family:"Times New Roman",serif;mso-bidi-font-weight:
  bold'>+<o:p></o:p></span></p>
  </td>
  <td width=38 valign=top style='width:1.0cm;border:none;padding:0cm 2.85pt 0cm 2.85pt;
  height:12.75pt'>
  <p class=MsoNormal align=center style='text-align:center'><span lang=EN-US
  style='font-size:8.0pt;font-family:"Times New Roman",serif;mso-bidi-font-weight:
  bold'>2022<o:p></o:p></span></p>
  </td>
  <td width=60 valign=top style='width:45.1pt;border:none;padding:0cm 2.85pt 0cm 2.85pt;
  height:12.75pt'>
  <p class=MsoNormal align=center style='text-align:center'><span lang=EN-US
  style='font-size:8.0pt;font-family:"Times New Roman",serif;mso-bidi-font-weight:
  bold'>8.67<o:p></o:p></span></p>
  </td>
  <td width=49 nowrap valign=top style='width:36.7pt;border:none;padding:0cm 2.85pt 0cm 2.85pt;
  height:12.75pt'>
  <p class=MsoNormal align=center style='text-align:center'><span lang=EN-US
  style='font-size:8.0pt;font-family:"Times New Roman",serif;mso-bidi-font-weight:
  bold'>2.88<o:p></o:p></span></p>
  </td>
  <td width=49 nowrap valign=top style='width:36.75pt;border:none;padding:0cm 2.85pt 0cm 2.85pt;
  height:12.75pt'>
  <p class=MsoNormal align=center style='text-align:center'><span lang=EN-US
  style='font-size:8.0pt;font-family:"Times New Roman",serif'>94.0<o:p></o:p></span></p>
  </td>
  <td width=49 nowrap valign=top style='width:36.7pt;border:none;padding:0cm 2.85pt 0cm 2.85pt;
  height:12.75pt'>
  <p class=MsoNormal align=center style='text-align:center'><span lang=EN-US
  style='font-size:8.0pt;font-family:"Times New Roman",serif'>3.86<o:p></o:p></span></p>
  </td>
  <td width=49 nowrap valign=top style='width:36.75pt;border:none;padding:0cm 2.85pt 0cm 2.85pt;
  height:12.75pt'>
  <p class=MsoNormal align=center style='text-align:center'><span lang=EN-US
  style='font-size:8.0pt;font-family:"Times New Roman",serif'>3.42<o:p></o:p></span></p>
  </td>
  <td width=49 nowrap valign=top style='width:36.75pt;border:none;padding:0cm 2.85pt 0cm 2.85pt;
  height:12.75pt'>
  <p class=MsoNormal align=center style='text-align:center'><span lang=EN-US
  style='font-size:8.0pt;font-family:"Times New Roman",serif'>3.57<o:p></o:p></span></p>
  </td>
 </tr>
 <tr style='mso-yfti-irow:9;height:12.75pt'>
  <td width=104 nowrap valign=top style='width:78.0pt;border:none;padding:0cm 2.85pt 0cm 2.85pt;
  height:12.75pt'>
  <p class=MsoNormal align=left style='text-align:left'><span class=SpellE><span
  lang=EN-US style='font-size:8.0pt;font-family:"Times New Roman",serif;
  mso-bidi-font-weight:bold'>GaGNet</span></span><span lang=EN-US
  style='font-size:8.0pt;font-family:"Times New Roman",serif;mso-bidi-font-weight:
  bold'><o:p></o:p></span></p>
  </td>
  <td width=38 valign=top style='width:1.0cm;border:none;padding:0cm 2.85pt 0cm 2.85pt;
  height:12.75pt'>
  <p class=MsoNormal align=center style='text-align:center'><span lang=EN-US
  style='font-size:8.0pt;font-family:"Times New Roman",serif;mso-bidi-font-weight:
  bold'>2022<o:p></o:p></span></p>
  </td>
  <td width=60 valign=top style='width:45.1pt;border:none;padding:0cm 2.85pt 0cm 2.85pt;
  height:12.75pt'>
  <p class=MsoNormal align=center style='text-align:center'><span lang=EN-US
  style='font-size:8.0pt;font-family:"Times New Roman",serif;mso-bidi-font-weight:
  bold'>5.95<o:p></o:p></span></p>
  </td>
  <td width=49 nowrap valign=top style='width:36.7pt;border:none;padding:0cm 2.85pt 0cm 2.85pt;
  height:12.75pt'>
  <p class=MsoNormal align=center style='text-align:center'><span lang=EN-US
  style='font-size:8.0pt;font-family:"Times New Roman",serif;mso-bidi-font-weight:
  bold'>2.94<o:p></o:p></span></p>
  </td>
  <td width=49 nowrap valign=top style='width:36.75pt;border:none;padding:0cm 2.85pt 0cm 2.85pt;
  height:12.75pt'>
  <p class=MsoNormal align=center style='text-align:center'><span lang=EN-US
  style='font-size:8.0pt;font-family:"Times New Roman",serif'>-<o:p></o:p></span></p>
  </td>
  <td width=49 nowrap valign=top style='width:36.7pt;border:none;padding:0cm 2.85pt 0cm 2.85pt;
  height:12.75pt'>
  <p class=MsoNormal align=center style='text-align:center'><span lang=EN-US
  style='font-size:8.0pt;font-family:"Times New Roman",serif'>4.26<o:p></o:p></span></p>
  </td>
  <td width=49 nowrap valign=top style='width:36.75pt;border:none;padding:0cm 2.85pt 0cm 2.85pt;
  height:12.75pt'>
  <p class=MsoNormal align=center style='text-align:center'><span lang=EN-US
  style='font-size:8.0pt;font-family:"Times New Roman",serif'>3.45<o:p></o:p></span></p>
  </td>
  <td width=49 nowrap valign=top style='width:36.75pt;border:none;padding:0cm 2.85pt 0cm 2.85pt;
  height:12.75pt'>
  <p class=MsoNormal align=center style='text-align:center'><span lang=EN-US
  style='font-size:8.0pt;font-family:"Times New Roman",serif'>3.59<o:p></o:p></span></p>
  </td>
 </tr>
 <tr style='mso-yfti-irow:10;height:12.75pt'>
  <td width=104 nowrap valign=top style='width:78.0pt;border:none;padding:0cm 2.85pt 0cm 2.85pt;
  height:12.75pt'>
  <p class=MsoNormal align=left style='text-align:left'><span lang=EN-US
  style='font-size:8.0pt;font-family:"Times New Roman",serif;mso-bidi-font-weight:
  bold'>DMF-Net<o:p></o:p></span></p>
  </td>
  <td width=38 valign=top style='width:1.0cm;border:none;padding:0cm 2.85pt 0cm 2.85pt;
  height:12.75pt'>
  <p class=MsoNormal align=center style='text-align:center'><span lang=EN-US
  style='font-size:8.0pt;font-family:"Times New Roman",serif;mso-bidi-font-weight:
  bold'>2022<o:p></o:p></span></p>
  </td>
  <td width=60 valign=top style='width:45.1pt;border:none;padding:0cm 2.85pt 0cm 2.85pt;
  height:12.75pt'>
  <p class=MsoNormal align=center style='text-align:center'><span lang=EN-US
  style='font-size:8.0pt;font-family:"Times New Roman",serif;mso-bidi-font-weight:
  bold'>7.84<o:p></o:p></span></p>
  </td>
  <td width=49 nowrap valign=top style='width:36.7pt;border:none;padding:0cm 2.85pt 0cm 2.85pt;
  height:12.75pt'>
  <p class=MsoNormal align=center style='text-align:center'><span lang=EN-US
  style='font-size:8.0pt;font-family:"Times New Roman",serif;mso-bidi-font-weight:
  bold'>2.97<o:p></o:p></span></p>
  </td>
  <td width=49 nowrap valign=top style='width:36.75pt;border:none;padding:0cm 2.85pt 0cm 2.85pt;
  height:12.75pt'>
  <p class=MsoNormal align=center style='text-align:center'><span lang=EN-US
  style='font-size:8.0pt;font-family:"Times New Roman",serif'>94.4<o:p></o:p></span></p>
  </td>
  <td width=49 nowrap valign=top style='width:36.7pt;border:none;padding:0cm 2.85pt 0cm 2.85pt;
  height:12.75pt'>
  <p class=MsoNormal align=center style='text-align:center'><span lang=EN-US
  style='font-size:8.0pt;font-family:"Times New Roman",serif'>4.26<o:p></o:p></span></p>
  </td>
  <td width=49 nowrap valign=top style='width:36.75pt;border:none;padding:0cm 2.85pt 0cm 2.85pt;
  height:12.75pt'>
  <p class=MsoNormal align=center style='text-align:center'><span lang=EN-US
  style='font-size:8.0pt;font-family:"Times New Roman",serif'>3.52<o:p></o:p></span></p>
  </td>
  <td width=49 nowrap valign=top style='width:36.75pt;border:none;padding:0cm 2.85pt 0cm 2.85pt;
  height:12.75pt'>
  <p class=MsoNormal align=center style='text-align:center'><span lang=EN-US
  style='font-size:8.0pt;font-family:"Times New Roman",serif'>3.62<o:p></o:p></span></p>
  </td>
 </tr>
 <tr style='mso-yfti-irow:11;height:12.75pt'>
  <td width=104 nowrap valign=top style='width:78.0pt;border:none;padding:0cm 2.85pt 0cm 2.85pt;
  height:12.75pt'>
  <p class=MsoNormal align=left style='text-align:left'><span lang=EN-US
  style='font-size:8.0pt;font-family:"Times New Roman",serif;mso-bidi-font-weight:
  bold'>DS-Net<o:p></o:p></span></p>
  </td>
  <td width=38 valign=top style='width:1.0cm;border:none;padding:0cm 2.85pt 0cm 2.85pt;
  height:12.75pt'>
  <p class=MsoNormal align=center style='text-align:center'><span lang=EN-US
  style='font-size:8.0pt;font-family:"Times New Roman",serif;mso-bidi-font-weight:
  bold'>2022<o:p></o:p></span></p>
  </td>
  <td width=60 valign=top style='width:45.1pt;border:none;padding:0cm 2.85pt 0cm 2.85pt;
  height:12.75pt'>
  <p class=MsoNormal align=center style='text-align:center'><span lang=EN-US
  style='font-size:8.0pt;font-family:"Times New Roman",serif;mso-bidi-font-weight:
  bold'>3.30<o:p></o:p></span></p>
  </td>
  <td width=49 nowrap valign=top style='width:36.7pt;border:none;padding:0cm 2.85pt 0cm 2.85pt;
  height:12.75pt'>
  <p class=MsoNormal align=center style='text-align:center'><span lang=EN-US
  style='font-size:8.0pt;font-family:"Times New Roman",serif;mso-bidi-font-weight:
  bold'>2.78<o:p></o:p></span></p>
  </td>
  <td width=49 nowrap valign=top style='width:36.75pt;border:none;padding:0cm 2.85pt 0cm 2.85pt;
  height:12.75pt'>
  <p class=MsoNormal align=center style='text-align:center'><span lang=EN-US
  style='font-size:8.0pt;font-family:"Times New Roman",serif'>94.3<o:p></o:p></span></p>
  </td>
  <td width=49 nowrap valign=top style='width:36.7pt;border:none;padding:0cm 2.85pt 0cm 2.85pt;
  height:12.75pt'>
  <p class=MsoNormal align=center style='text-align:center'><span lang=EN-US
  style='font-size:8.0pt;font-family:"Times New Roman",serif'>4.20<o:p></o:p></span></p>
  </td>
  <td width=49 nowrap valign=top style='width:36.75pt;border:none;padding:0cm 2.85pt 0cm 2.85pt;
  height:12.75pt'>
  <p class=MsoNormal align=center style='text-align:center'><span lang=EN-US
  style='font-size:8.0pt;font-family:"Times New Roman",serif'>3.34<o:p></o:p></span></p>
  </td>
  <td width=49 nowrap valign=top style='width:36.75pt;border:none;padding:0cm 2.85pt 0cm 2.85pt;
  height:12.75pt'>
  <p class=MsoNormal align=center style='text-align:center'><span lang=EN-US
  style='font-size:8.0pt;font-family:"Times New Roman",serif'>3.48<o:p></o:p></span></p>
  </td>
 </tr>
 <tr style='mso-yfti-irow:12;height:12.75pt'>
  <td width=104 nowrap valign=top style='width:78.0pt;border:none;padding:0cm 2.85pt 0cm 2.85pt;
  height:12.75pt'>
  <p class=MsoNormal align=left style='text-align:left'><span lang=EN-US
  style='font-size:8.0pt;font-family:"Times New Roman",serif;mso-bidi-font-weight:
  bold'>SF-Net<o:p></o:p></span></p>
  </td>
  <td width=38 valign=top style='width:1.0cm;border:none;padding:0cm 2.85pt 0cm 2.85pt;
  height:12.75pt'>
  <p class=MsoNormal align=center style='text-align:center'><span lang=EN-US
  style='font-size:8.0pt;font-family:"Times New Roman",serif;mso-bidi-font-weight:
  bold'>2022<o:p></o:p></span></p>
  </td>
  <td width=60 valign=top style='width:45.1pt;border:none;padding:0cm 2.85pt 0cm 2.85pt;
  height:12.75pt'>
  <p class=MsoNormal align=center style='text-align:center'><span lang=EN-US
  style='font-size:8.0pt;font-family:"Times New Roman",serif;mso-bidi-font-weight:
  bold'>6.98<o:p></o:p></span></p>
  </td>
  <td width=49 nowrap valign=top style='width:36.7pt;border:none;padding:0cm 2.85pt 0cm 2.85pt;
  height:12.75pt'>
  <p class=MsoNormal align=center style='text-align:center'><span lang=EN-US
  style='font-size:8.0pt;font-family:"Times New Roman",serif;mso-bidi-font-weight:
  bold'>3.02<o:p></o:p></span></p>
  </td>
  <td width=49 nowrap valign=top style='width:36.75pt;border:none;padding:0cm 2.85pt 0cm 2.85pt;
  height:12.75pt'>
  <p class=MsoNormal align=center style='text-align:center'><span lang=EN-US
  style='font-size:8.0pt;font-family:"Times New Roman",serif'>94.5<o:p></o:p></span></p>
  </td>
  <td width=49 nowrap valign=top style='width:36.7pt;border:none;padding:0cm 2.85pt 0cm 2.85pt;
  height:12.75pt'>
  <p class=MsoNormal align=center style='text-align:center'><span lang=EN-US
  style='font-size:8.0pt;font-family:"Times New Roman",serif'>4.36<o:p></o:p></span></p>
  </td>
  <td width=49 nowrap valign=top style='width:36.75pt;border:none;padding:0cm 2.85pt 0cm 2.85pt;
  height:12.75pt'>
  <p class=MsoNormal align=center style='text-align:center'><span lang=EN-US
  style='font-size:8.0pt;font-family:"Times New Roman",serif'>3.54<o:p></o:p></span></p>
  </td>
  <td width=49 nowrap valign=top style='width:36.75pt;border:none;padding:0cm 2.85pt 0cm 2.85pt;
  height:12.75pt'>
  <p class=MsoNormal align=center style='text-align:center'><span lang=EN-US
  style='font-size:8.0pt;font-family:"Times New Roman",serif'>3.67<o:p></o:p></span></p>
  </td>
 </tr>
 <tr style='mso-yfti-irow:13;height:12.75pt'>
  <td width=104 nowrap valign=top style='width:78.0pt;border:none;border-bottom:
  solid windowtext 1.0pt;mso-border-bottom-alt:solid windowtext .5pt;
  padding:0cm 2.85pt 0cm 2.85pt;height:12.75pt'>
  <p class=MsoNormal align=left style='text-align:left'><span lang=EN-US
  style='font-size:8.0pt;font-family:"Times New Roman",serif;mso-bidi-font-weight:
  bold'>DeepFilterNet2<o:p></o:p></span></p>
  </td>
  <td width=38 valign=top style='width:1.0cm;border:none;border-bottom:solid windowtext 1.0pt;
  mso-border-bottom-alt:solid windowtext .5pt;padding:0cm 2.85pt 0cm 2.85pt;
  height:12.75pt'>
  <p class=MsoNormal align=center style='text-align:center'><span lang=EN-US
  style='font-size:8.0pt;font-family:"Times New Roman",serif;mso-bidi-font-weight:
  bold'>2022<o:p></o:p></span></p>
  </td>
  <td width=60 valign=top style='width:45.1pt;border:none;border-bottom:solid windowtext 1.0pt;
  mso-border-bottom-alt:solid windowtext .5pt;padding:0cm 2.85pt 0cm 2.85pt;
  height:12.75pt'>
  <p class=MsoNormal align=center style='text-align:center'><span lang=EN-US
  style='font-size:8.0pt;font-family:"Times New Roman",serif;mso-bidi-font-weight:
  bold'>2.31<o:p></o:p></span></p>
  </td>
  <td width=49 nowrap valign=top style='width:36.7pt;border:none;border-bottom:
  solid windowtext 1.0pt;mso-border-bottom-alt:solid windowtext .5pt;
  padding:0cm 2.85pt 0cm 2.85pt;height:12.75pt'>
  <p class=MsoNormal align=center style='text-align:center'><span lang=EN-US
  style='font-size:8.0pt;font-family:"Times New Roman",serif;mso-bidi-font-weight:
  bold'>3.08<o:p></o:p></span></p>
  </td>
  <td width=49 nowrap valign=top style='width:36.75pt;border:none;border-bottom:
  solid windowtext 1.0pt;mso-border-bottom-alt:solid windowtext .5pt;
  padding:0cm 2.85pt 0cm 2.85pt;height:12.75pt'>
  <p class=MsoNormal align=center style='text-align:center'><span lang=EN-US
  style='font-size:8.0pt;font-family:"Times New Roman",serif'>94.3<o:p></o:p></span></p>
  </td>
  <td width=49 nowrap valign=top style='width:36.7pt;border:none;border-bottom:
  solid windowtext 1.0pt;mso-border-bottom-alt:solid windowtext .5pt;
  padding:0cm 2.85pt 0cm 2.85pt;height:12.75pt'>
  <p class=MsoNormal align=center style='text-align:center'><span lang=EN-US
  style='font-size:8.0pt;font-family:"Times New Roman",serif'>4.30<o:p></o:p></span></p>
  </td>
  <td width=49 nowrap valign=top style='width:36.75pt;border:none;border-bottom:
  solid windowtext 1.0pt;mso-border-bottom-alt:solid windowtext .5pt;
  padding:0cm 2.85pt 0cm 2.85pt;height:12.75pt'>
  <p class=MsoNormal align=center style='text-align:center'><span lang=EN-US
  style='font-size:8.0pt;font-family:"Times New Roman",serif'>3.40<o:p></o:p></span></p>
  </td>
  <td width=49 nowrap valign=top style='width:36.75pt;border:none;border-bottom:
  solid windowtext 1.0pt;mso-border-bottom-alt:solid windowtext .5pt;
  padding:0cm 2.85pt 0cm 2.85pt;height:12.75pt'>
  <p class=MsoNormal align=center style='text-align:center'><span lang=EN-US
  style='font-size:8.0pt;font-family:"Times New Roman",serif'>3.70<o:p></o:p></span></p>
  </td>
 </tr>
 <tr style='mso-yfti-irow:14;height:12.75pt'>
  <td width=104 nowrap style='width:78.0pt;border:none;border-bottom:dashed windowtext 1.0pt;
  mso-border-top-alt:solid windowtext .5pt;mso-border-top-alt:solid windowtext .5pt;
  mso-border-bottom-alt:dashed windowtext .5pt;padding:0cm 2.85pt 0cm 2.85pt;
  height:12.75pt'>
  <p class=MsoNormal align=left style='text-align:left'><b><span lang=EN-US
  style='font-size:6.5pt;mso-bidi-font-size:7.0pt;font-family:"Times New Roman",serif'>MTFAA
  (<span class=SpellE>Cau</span>., LSA)<o:p></o:p></span></b></p>
  </td>
  <td width=38 style='width:1.0cm;border:none;border-bottom:dashed windowtext 1.0pt;
  mso-border-top-alt:solid windowtext .5pt;mso-border-top-alt:solid windowtext .5pt;
  mso-border-bottom-alt:dashed windowtext .5pt;padding:0cm 2.85pt 0cm 2.85pt;
  height:12.75pt'>
  <p class=MsoNormal align=center style='text-align:center'><span lang=EN-US
  style='font-size:8.0pt;font-family:"Times New Roman",serif;mso-bidi-font-weight:
  bold'>2023<o:p></o:p></span></p>
  </td>
  <td width=60 style='width:45.1pt;border:none;border-bottom:dashed windowtext 1.0pt;
  mso-border-top-alt:solid windowtext .5pt;mso-border-top-alt:solid windowtext .5pt;
  mso-border-bottom-alt:dashed windowtext .5pt;padding:0cm 2.85pt 0cm 2.85pt;
  height:12.75pt'>
  <p class=MsoNormal align=center style='text-align:center'><span lang=EN-US
  style='font-size:8.0pt;font-family:"Times New Roman",serif;mso-bidi-font-weight:
  bold'>1.5<o:p></o:p></span></p>
  </td>
  <td width=49 nowrap style='width:36.7pt;border:none;border-bottom:dashed windowtext 1.0pt;
  mso-border-top-alt:solid windowtext .5pt;mso-border-top-alt:solid windowtext .5pt;
  mso-border-bottom-alt:dashed windowtext .5pt;padding:0cm 2.85pt 0cm 2.85pt;
  height:12.75pt'>
  <p class=MsoNormal align=center style='text-align:center'><b
  style='mso-bidi-font-weight:normal'><span lang=EN-US style='font-size:8.0pt;
  mso-bidi-font-size:7.5pt;font-family:"Times New Roman",serif;mso-fareast-font-family:
  等线;color:black;mso-font-kerning:0pt'>3.16</span></b><span lang=EN-US
  style='font-size:8.0pt;font-family:"Times New Roman",serif;mso-bidi-font-weight:
  bold'><o:p></o:p></span></p>
  </td>
  <td width=49 nowrap style='width:36.75pt;border:none;border-bottom:dashed windowtext 1.0pt;
  mso-border-top-alt:solid windowtext .5pt;mso-border-top-alt:solid windowtext .5pt;
  mso-border-bottom-alt:dashed windowtext .5pt;padding:0cm 2.85pt 0cm 2.85pt;
  height:12.75pt'>
  <p class=MsoNormal align=center style='text-align:center'><b
  style='mso-bidi-font-weight:normal'><span lang=EN-US style='font-size:8.0pt;
  mso-bidi-font-size:7.5pt;font-family:"Times New Roman",serif;mso-fareast-font-family:
  等线;color:black;mso-font-kerning:0pt'>94.7</span></b><span lang=EN-US
  style='font-size:8.0pt;font-family:"Times New Roman",serif'><o:p></o:p></span></p>
  </td>
  <td width=49 nowrap style='width:36.7pt;border:none;border-bottom:dashed windowtext 1.0pt;
  mso-border-top-alt:solid windowtext .5pt;mso-border-top-alt:solid windowtext .5pt;
  mso-border-bottom-alt:dashed windowtext .5pt;padding:0cm 2.85pt 0cm 2.85pt;
  height:12.75pt'>
  <p class=MsoNormal align=center style='text-align:center'><b
  style='mso-bidi-font-weight:normal'><span lang=EN-US style='font-size:8.0pt;
  mso-bidi-font-size:7.5pt;font-family:"Times New Roman",serif;mso-fareast-font-family:
  等线;color:black;mso-font-kerning:0pt'>4.35</span></b><span lang=EN-US
  style='font-size:8.0pt;font-family:"Times New Roman",serif'><o:p></o:p></span></p>
  </td>
  <td width=49 nowrap style='width:36.75pt;border:none;border-bottom:dashed windowtext 1.0pt;
  mso-border-top-alt:solid windowtext .5pt;mso-border-top-alt:solid windowtext .5pt;
  mso-border-bottom-alt:dashed windowtext .5pt;padding:0cm 2.85pt 0cm 2.85pt;
  height:12.75pt'>
  <p class=MsoNormal align=center style='text-align:center'><b
  style='mso-bidi-font-weight:normal'><span lang=EN-US style='font-size:8.0pt;
  mso-bidi-font-size:7.5pt;font-family:"Times New Roman",serif;mso-fareast-font-family:
  等线;color:black;mso-font-kerning:0pt'>3.61</span></b><span lang=EN-US
  style='font-size:8.0pt;font-family:"Times New Roman",serif'><o:p></o:p></span></p>
  </td>
  <td width=49 nowrap style='width:36.75pt;border:none;border-bottom:dashed windowtext 1.0pt;
  mso-border-top-alt:solid windowtext .5pt;mso-border-top-alt:solid windowtext .5pt;
  mso-border-bottom-alt:dashed windowtext .5pt;padding:0cm 2.85pt 0cm 2.85pt;
  height:12.75pt'>
  <p class=MsoNormal align=center style='text-align:center'><b
  style='mso-bidi-font-weight:normal'><span lang=EN-US style='font-size:8.0pt;
  mso-bidi-font-size:7.5pt;font-family:"Times New Roman",serif;mso-fareast-font-family:
  等线;color:black;mso-font-kerning:0pt'>3.78</span></b><span lang=EN-US
  style='font-size:8.0pt;font-family:"Times New Roman",serif'><o:p></o:p></span></p>
  </td>
 </tr>
 <tr style='mso-yfti-irow:15;mso-yfti-lastrow:yes;height:12.75pt'>
  <td width=104 nowrap style='width:78.0pt;border:none;border-bottom:solid windowtext 1.0pt;
  mso-border-top-alt:dashed windowtext .5pt;mso-border-top-alt:dashed windowtext .5pt;
  mso-border-bottom-alt:solid windowtext .5pt;padding:0cm 2.85pt 0cm 2.85pt;
  height:12.75pt'>
  <p class=MsoNormal align=left style='text-align:left'><b><span lang=EN-US
  style='font-size:6.5pt;mso-bidi-font-size:7.0pt;font-family:"Times New Roman",serif'>MTFAA
  (Non-<span class=SpellE>cau</span>., LSA)<o:p></o:p></span></b></p>
  </td>
  <td width=38 style='width:1.0cm;border:none;border-bottom:solid windowtext 1.0pt;
  mso-border-top-alt:dashed windowtext .5pt;mso-border-top-alt:dashed windowtext .5pt;
  mso-border-bottom-alt:solid windowtext .5pt;padding:0cm 2.85pt 0cm 2.85pt;
  height:12.75pt'>
  <p class=MsoNormal align=center style='text-align:center'><span lang=EN-US
  style='font-size:8.0pt;font-family:"Times New Roman",serif;mso-bidi-font-weight:
  bold'>2023<o:p></o:p></span></p>
  </td>
  <td width=60 style='width:45.1pt;border:none;border-bottom:solid windowtext 1.0pt;
  mso-border-top-alt:dashed windowtext .5pt;mso-border-top-alt:dashed windowtext .5pt;
  mso-border-bottom-alt:solid windowtext .5pt;padding:0cm 2.85pt 0cm 2.85pt;
  height:12.75pt'>
  <p class=MsoNormal align=center style='text-align:center'><span lang=EN-US
  style='font-size:8.0pt;font-family:"Times New Roman",serif;mso-bidi-font-weight:
  bold'>1.5<o:p></o:p></span></p>
  </td>
  <td width=49 nowrap style='width:36.7pt;border:none;border-bottom:solid windowtext 1.0pt;
  mso-border-top-alt:dashed windowtext .5pt;mso-border-top-alt:dashed windowtext .5pt;
  mso-border-bottom-alt:solid windowtext .5pt;padding:0cm 2.85pt 0cm 2.85pt;
  height:12.75pt'>
  <p class=MsoNormal align=center style='text-align:center'><b><span
  lang=EN-US style='font-size:8.0pt;mso-bidi-font-size:7.5pt;font-family:"Times New Roman",serif;
  mso-fareast-font-family:等线;color:black;mso-font-kerning:0pt;mso-fareast-language:
  EN-US'>3.30</span></b><span lang=EN-US style='font-size:8.0pt;font-family:
  "Times New Roman",serif;mso-bidi-font-weight:bold'><o:p></o:p></span></p>
  </td>
  <td width=49 nowrap style='width:36.75pt;border:none;border-bottom:solid windowtext 1.0pt;
  mso-border-top-alt:dashed windowtext .5pt;mso-border-top-alt:dashed windowtext .5pt;
  mso-border-bottom-alt:solid windowtext .5pt;padding:0cm 2.85pt 0cm 2.85pt;
  height:12.75pt'>
  <p class=MsoNormal align=center style='text-align:center'><b><span
  lang=EN-US style='font-size:8.0pt;mso-bidi-font-size:7.5pt;font-family:"Times New Roman",serif;
  mso-fareast-font-family:等线;color:black;mso-font-kerning:0pt;mso-fareast-language:
  EN-US'>95.3</span></b><span lang=EN-US style='font-size:8.0pt;font-family:
  "Times New Roman",serif'><o:p></o:p></span></p>
  </td>
  <td width=49 nowrap style='width:36.7pt;border:none;border-bottom:solid windowtext 1.0pt;
  mso-border-top-alt:dashed windowtext .5pt;mso-border-top-alt:dashed windowtext .5pt;
  mso-border-bottom-alt:solid windowtext .5pt;padding:0cm 2.85pt 0cm 2.85pt;
  height:12.75pt'>
  <p class=MsoNormal align=center style='text-align:center'><b><span
  lang=EN-US style='font-size:8.0pt;mso-bidi-font-size:7.5pt;font-family:"Times New Roman",serif;
  mso-fareast-font-family:等线;color:black;mso-font-kerning:0pt;mso-fareast-language:
  EN-US'>4.45</span></b><span lang=EN-US style='font-size:8.0pt;font-family:
  "Times New Roman",serif'><o:p></o:p></span></p>
  </td>
  <td width=49 nowrap style='width:36.75pt;border:none;border-bottom:solid windowtext 1.0pt;
  mso-border-top-alt:dashed windowtext .5pt;mso-border-top-alt:dashed windowtext .5pt;
  mso-border-bottom-alt:solid windowtext .5pt;padding:0cm 2.85pt 0cm 2.85pt;
  height:12.75pt'>
  <p class=MsoNormal align=center style='text-align:center'><b><span
  lang=EN-US style='font-size:8.0pt;mso-bidi-font-size:7.5pt;font-family:"Times New Roman",serif;
  mso-fareast-font-family:等线;color:black;mso-font-kerning:0pt;mso-fareast-language:
  EN-US'>3.73</span></b><span lang=EN-US style='font-size:8.0pt;font-family:
  "Times New Roman",serif'><o:p></o:p></span></p>
  </td>
  <td width=49 nowrap style='width:36.75pt;border:none;border-bottom:solid windowtext 1.0pt;
  mso-border-top-alt:dashed windowtext .5pt;mso-border-top-alt:dashed windowtext .5pt;
  mso-border-bottom-alt:solid windowtext .5pt;padding:0cm 2.85pt 0cm 2.85pt;
  height:12.75pt'>
  <p class=MsoNormal align=center style='text-align:center'><b><span
  lang=EN-US style='font-size:8.0pt;mso-bidi-font-size:7.5pt;font-family:"Times New Roman",serif;
  mso-fareast-font-family:等线;color:black;mso-font-kerning:0pt;mso-fareast-language:
  EN-US'>3.90</span></b><span lang=EN-US style='font-size:8.0pt;font-family:
  "Times New Roman",serif'><o:p></o:p></span></p>
  </td>
 </tr>
</table>

</div>

<p class=MsoNormal><span lang=EN-US><o:p>&nbsp;</o:p></span></p>

</div>

</body>

Ignited by ALiBi, we also conduct experiment on the multi-scale local spectral attention (MSLSA) as shown in the figure below,
<!-- ![MSLSA](https://user-images.githubusercontent.com/103247057/223663488-83df65ad-deb8-49fb-9ab0-af1b001e63b8.jpg) -->
<div align=center><img src="https://user-images.githubusercontent.com/103247057/223663488-83df65ad-deb8-49fb-9ab0-af1b001e63b8.jpg" width="600" height="400" /></div>  
the performance of MSLSA can be seen in the table below, 


<body lang=ZH-CN style='tab-interval:21.0pt;text-justify-trim:punctuation'>

<div class=WordSection1 style='layout-grid:15.6pt'>

<div align=center>

<table class=MsoTableGrid border=1 cellspacing=0 cellpadding=0 width=548
 style='width:410.85pt;border-collapse:collapse;border:none;mso-border-alt:
 solid windowtext .5pt;mso-yfti-tbllook:1184;mso-padding-alt:0cm 5.4pt 0cm 5.4pt'>
 <tr style='mso-yfti-irow:0;mso-yfti-firstrow:yes;height:1.0pt'>
  <td width=75 nowrap style='width:56.45pt;border:solid windowtext 1.0pt;
  mso-border-alt:solid windowtext .5pt;padding:0cm 5.4pt 0cm 5.4pt;height:1.0pt'>
  <p class=MsoNormal align=center style='text-align:center;mso-pagination:widow-orphan'><span
  lang=EN-US style='font-size:5.5pt;mso-bidi-font-size:10.0pt;font-family:"Times New Roman",serif;
  mso-fareast-font-family:等线;mso-font-kerning:0pt'>Full-band Metrics</span><span
  lang=EN-US style='font-size:7.5pt;mso-bidi-font-size:10.0pt;font-family:"Times New Roman",serif;
  mso-fareast-font-family:等线;mso-font-kerning:0pt'><o:p></o:p></span></p>
  </td>
  <td width=118 nowrap colspan=3 style='width:88.6pt;border:solid windowtext 1.0pt;
  border-left:none;mso-border-left-alt:solid windowtext .5pt;mso-border-alt:
  solid windowtext .5pt;padding:0cm 5.4pt 0cm 5.4pt;height:1.0pt'>
  <p class=MsoNormal align=center style='text-align:center;mso-pagination:widow-orphan'><span
  lang=EN-US style='font-size:7.5pt;mso-bidi-font-size:10.0pt;font-family:"Times New Roman",serif;
  mso-fareast-font-family:等线;mso-font-kerning:0pt'>STOI<o:p></o:p></span></p>
  </td>
  <td width=118 nowrap colspan=3 style='width:88.6pt;border:solid windowtext 1.0pt;
  border-left:none;mso-border-left-alt:solid windowtext .5pt;mso-border-alt:
  solid windowtext .5pt;padding:0cm 5.4pt 0cm 5.4pt;height:1.0pt'>
  <p class=MsoNormal align=center style='text-align:center;mso-pagination:widow-orphan'><span
  lang=EN-US style='font-size:7.5pt;mso-bidi-font-size:10.0pt;font-family:"Times New Roman",serif;
  mso-fareast-font-family:等线;mso-font-kerning:0pt'>SiSDR (dB)<o:p></o:p></span></p>
  </td>
  <td width=39 nowrap rowspan=5 style='width:29.5pt;border:solid windowtext 1.0pt;
  border-left:none;mso-border-left-alt:solid windowtext .5pt;mso-border-alt:
  solid windowtext .5pt;padding:0cm 5.4pt 0cm 5.4pt;height:1.0pt'>
  <p class=MsoNormal align=center style='text-align:center;mso-pagination:widow-orphan'><span
  lang=EN-US style='font-size:7.5pt;mso-bidi-font-size:10.0pt;font-family:"Times New Roman",serif;
  mso-fareast-font-family:等线;mso-font-kerning:0pt'><o:p>&nbsp;</o:p></span></p>
  </td>
  <td width=197 colspan=5 style='width:147.7pt;border:solid windowtext 1.0pt;
  border-left:none;mso-border-left-alt:solid windowtext .5pt;mso-border-alt:
  solid windowtext .5pt;padding:0cm 5.4pt 0cm 5.4pt;height:1.0pt'>
  <p class=MsoNormal align=center style='text-align:center;mso-pagination:widow-orphan'><span
  lang=EN-US style='font-size:7.5pt;mso-bidi-font-size:10.0pt;font-family:"Times New Roman",serif;
  mso-fareast-font-family:等线;mso-font-kerning:0pt'>LSD (dB)<o:p></o:p></span></p>
  </td>
 </tr>
 <tr style='mso-yfti-irow:1;height:1.0pt'>
  <td width=75 nowrap style='width:56.45pt;border:solid windowtext 1.0pt;
  border-top:none;mso-border-top-alt:solid windowtext .5pt;mso-border-alt:solid windowtext .5pt;
  padding:0cm 5.4pt 0cm 5.4pt;height:1.0pt'>
  <p class=MsoNormal align=center style='text-align:center;mso-pagination:widow-orphan'><span
  class=GramE><span lang=EN-US style='font-size:6.5pt;mso-bidi-font-size:10.0pt;
  font-family:"Times New Roman",serif;mso-fareast-font-family:等线;mso-font-kerning:
  0pt'>SNR(</span></span><span lang=EN-US style='font-size:6.5pt;mso-bidi-font-size:
  10.0pt;font-family:"Times New Roman",serif;mso-fareast-font-family:等线;
  mso-font-kerning:0pt'>dB)</span><span lang=EN-US style='font-size:7.5pt;
  mso-bidi-font-size:10.0pt;font-family:"Times New Roman",serif;mso-fareast-font-family:
  等线;mso-font-kerning:0pt'><o:p></o:p></span></p>
  </td>
  <td width=39 nowrap style='width:29.5pt;border-top:none;border-left:none;
  border-bottom:solid windowtext 1.0pt;border-right:solid windowtext 1.0pt;
  mso-border-top-alt:solid windowtext .5pt;mso-border-left-alt:solid windowtext .5pt;
  mso-border-alt:solid windowtext .5pt;padding:0cm 5.4pt 0cm 5.4pt;height:1.0pt'>
  <p class=MsoNormal align=center style='text-align:center;mso-pagination:widow-orphan'><span
  lang=EN-US style='font-size:7.5pt;mso-bidi-font-size:10.0pt;font-family:"Times New Roman",serif;
  mso-fareast-font-family:等线;mso-font-kerning:0pt'>-5~0<o:p></o:p></span></p>
  </td>
  <td width=39 nowrap style='width:29.55pt;border-top:none;border-left:none;
  border-bottom:solid windowtext 1.0pt;border-right:solid windowtext 1.0pt;
  mso-border-top-alt:solid windowtext .5pt;mso-border-left-alt:solid windowtext .5pt;
  mso-border-alt:solid windowtext .5pt;padding:0cm 5.4pt 0cm 5.4pt;height:1.0pt'>
  <p class=MsoNormal align=center style='text-align:center;mso-pagination:widow-orphan'><span
  lang=EN-US style='font-size:7.5pt;mso-bidi-font-size:10.0pt;font-family:"Times New Roman",serif;
  mso-fareast-font-family:等线;mso-font-kerning:0pt'>0~15<o:p></o:p></span></p>
  </td>
  <td width=39 nowrap style='width:29.55pt;border-top:none;border-left:none;
  border-bottom:solid windowtext 1.0pt;border-right:solid windowtext 1.0pt;
  mso-border-top-alt:solid windowtext .5pt;mso-border-left-alt:solid windowtext .5pt;
  mso-border-alt:solid windowtext .5pt;padding:0cm 5.4pt 0cm 5.4pt;height:1.0pt'>
  <p class=MsoNormal align=center style='text-align:center;mso-pagination:widow-orphan'><span
  class=SpellE><span lang=EN-US style='font-size:7.5pt;mso-bidi-font-size:10.0pt;
  font-family:"Times New Roman",serif;mso-fareast-font-family:等线;mso-font-kerning:
  0pt'>Ovrl</span></span><span lang=EN-US style='font-size:7.5pt;mso-bidi-font-size:
  10.0pt;font-family:"Times New Roman",serif;mso-fareast-font-family:等线;
  mso-font-kerning:0pt'>.<o:p></o:p></span></p>
  </td>
  <td width=39 nowrap style='width:29.5pt;border-top:none;border-left:none;
  border-bottom:solid windowtext 1.0pt;border-right:solid windowtext 1.0pt;
  mso-border-top-alt:solid windowtext .5pt;mso-border-left-alt:solid windowtext .5pt;
  mso-border-alt:solid windowtext .5pt;padding:0cm 5.4pt 0cm 5.4pt;height:1.0pt'>
  <p class=MsoNormal align=center style='text-align:center;mso-pagination:widow-orphan'><span
  lang=EN-US style='font-size:7.5pt;mso-bidi-font-size:10.0pt;font-family:"Times New Roman",serif;
  mso-fareast-font-family:等线;mso-font-kerning:0pt'>-5~0<o:p></o:p></span></p>
  </td>
  <td width=39 nowrap style='width:29.55pt;border-top:none;border-left:none;
  border-bottom:solid windowtext 1.0pt;border-right:solid windowtext 1.0pt;
  mso-border-top-alt:solid windowtext .5pt;mso-border-left-alt:solid windowtext .5pt;
  mso-border-alt:solid windowtext .5pt;padding:0cm 5.4pt 0cm 5.4pt;height:1.0pt'>
  <p class=MsoNormal align=center style='text-align:center;mso-pagination:widow-orphan'><span
  lang=EN-US style='font-size:7.5pt;mso-bidi-font-size:10.0pt;font-family:"Times New Roman",serif;
  mso-fareast-font-family:等线;mso-font-kerning:0pt'>0~15<o:p></o:p></span></p>
  </td>
  <td width=39 nowrap style='width:29.55pt;border-top:none;border-left:none;
  border-bottom:solid windowtext 1.0pt;border-right:solid windowtext 1.0pt;
  mso-border-top-alt:solid windowtext .5pt;mso-border-left-alt:solid windowtext .5pt;
  mso-border-alt:solid windowtext .5pt;padding:0cm 5.4pt 0cm 5.4pt;height:1.0pt'>
  <p class=MsoNormal align=center style='text-align:center;mso-pagination:widow-orphan'><span
  class=SpellE><span lang=EN-US style='font-size:7.5pt;mso-bidi-font-size:10.0pt;
  font-family:"Times New Roman",serif;mso-fareast-font-family:等线;mso-font-kerning:
  0pt'>Ovrl</span></span><span lang=EN-US style='font-size:7.5pt;mso-bidi-font-size:
  10.0pt;font-family:"Times New Roman",serif;mso-fareast-font-family:等线;
  mso-font-kerning:0pt'>.<o:p></o:p></span></p>
  </td>
  <td width=79 nowrap colspan=2 style='width:59.1pt;border-top:none;border-left:
  none;border-bottom:solid windowtext 1.0pt;border-right:solid windowtext 1.0pt;
  mso-border-top-alt:solid windowtext .5pt;mso-border-left-alt:solid windowtext .5pt;
  mso-border-alt:solid windowtext .5pt;padding:0cm 5.4pt 0cm 5.4pt;height:1.0pt'>
  <p class=MsoNormal align=center style='text-align:center;mso-pagination:widow-orphan'><span
  lang=EN-US style='font-size:7.5pt;mso-bidi-font-size:10.0pt;font-family:"Times New Roman",serif;
  mso-fareast-font-family:等线;mso-font-kerning:0pt'>Band(kHz)</span><span
  lang=EN-US style='font-size:5.0pt;mso-bidi-font-size:10.0pt;font-family:"Times New Roman",serif;
  mso-fareast-font-family:等线;mso-font-kerning:0pt'><o:p></o:p></span></p>
  </td>
  <td width=39 style='width:29.5pt;border-top:none;border-left:none;border-bottom:
  solid windowtext 1.0pt;border-right:solid windowtext 1.0pt;mso-border-top-alt:
  solid windowtext .5pt;mso-border-left-alt:solid windowtext .5pt;mso-border-alt:
  solid windowtext .5pt;padding:0cm 5.4pt 0cm 5.4pt;height:1.0pt'>
  <p class=MsoNormal align=center style='text-align:center;mso-pagination:widow-orphan'><span
  lang=EN-US style='font-size:7.5pt;font-family:"Times New Roman",serif;
  mso-fareast-font-family:等线;mso-font-kerning:0pt'>0~8<o:p></o:p></span></p>
  </td>
  <td width=39 style='width:29.55pt;border-top:none;border-left:none;
  border-bottom:solid windowtext 1.0pt;border-right:solid windowtext 1.0pt;
  mso-border-top-alt:solid windowtext .5pt;mso-border-left-alt:solid windowtext .5pt;
  mso-border-alt:solid windowtext .5pt;padding:0cm 5.4pt 0cm 5.4pt;height:1.0pt'>
  <p class=MsoNormal align=center style='text-align:center;mso-pagination:widow-orphan'><span
  lang=EN-US style='font-size:7.5pt;font-family:"Times New Roman",serif;
  mso-fareast-font-family:等线;mso-font-kerning:0pt'>8~24<o:p></o:p></span></p>
  </td>
  <td width=39 style='width:29.55pt;border-top:none;border-left:none;
  border-bottom:solid windowtext 1.0pt;border-right:solid windowtext 1.0pt;
  mso-border-top-alt:solid windowtext .5pt;mso-border-left-alt:solid windowtext .5pt;
  mso-border-alt:solid windowtext .5pt;padding:0cm 5.4pt 0cm 5.4pt;height:1.0pt'>
  <p class=MsoNormal align=center style='text-align:center;mso-pagination:widow-orphan'><span
  lang=EN-US style='font-size:7.5pt;font-family:"Times New Roman",serif;
  mso-fareast-font-family:等线;mso-font-kerning:0pt'>Full.<o:p></o:p></span></p>
  </td>
 </tr>
 <tr style='mso-yfti-irow:2;height:1.0pt'>
  <td width=75 nowrap style='width:56.45pt;border:solid windowtext 1.0pt;
  border-top:none;mso-border-top-alt:solid windowtext .5pt;mso-border-alt:solid windowtext .5pt;
  padding:0cm 5.4pt 0cm 5.4pt;height:1.0pt'>
  <p class=MsoNormal align=center style='text-align:center;mso-pagination:widow-orphan'><span
  lang=EN-US style='font-size:6.5pt;mso-bidi-font-size:10.0pt;font-family:"Times New Roman",serif;
  mso-fareast-font-family:等线;mso-font-kerning:0pt'>Noisy</span><span
  lang=EN-US style='font-size:7.5pt;mso-bidi-font-size:10.0pt;font-family:"Times New Roman",serif;
  mso-fareast-font-family:等线;mso-font-kerning:0pt'><o:p></o:p></span></p>
  </td>
  <td width=39 nowrap style='width:29.5pt;border-top:none;border-left:none;
  border-bottom:solid windowtext 1.0pt;border-right:solid windowtext 1.0pt;
  mso-border-top-alt:solid windowtext .5pt;mso-border-left-alt:solid windowtext .5pt;
  mso-border-alt:solid windowtext .5pt;padding:0cm 5.4pt 0cm 5.4pt;height:1.0pt'>
  <p class=MsoNormal align=center style='text-align:center;mso-pagination:widow-orphan'><span
  lang=EN-US style='font-size:6.5pt;mso-bidi-font-size:10.0pt;font-family:"Times New Roman",serif;
  mso-fareast-font-family:等线;mso-font-kerning:0pt'>0.687<o:p></o:p></span></p>
  </td>
  <td width=39 nowrap style='width:29.55pt;border-top:none;border-left:none;
  border-bottom:solid windowtext 1.0pt;border-right:solid windowtext 1.0pt;
  mso-border-top-alt:solid windowtext .5pt;mso-border-left-alt:solid windowtext .5pt;
  mso-border-alt:solid windowtext .5pt;padding:0cm 5.4pt 0cm 5.4pt;height:1.0pt'>
  <p class=MsoNormal align=center style='text-align:center;mso-pagination:widow-orphan'><span
  lang=EN-US style='font-size:6.5pt;mso-bidi-font-size:10.0pt;font-family:"Times New Roman",serif;
  mso-fareast-font-family:等线;mso-font-kerning:0pt'>0.805<o:p></o:p></span></p>
  </td>
  <td width=39 nowrap style='width:29.55pt;border-top:none;border-left:none;
  border-bottom:solid windowtext 1.0pt;border-right:solid windowtext 1.0pt;
  mso-border-top-alt:solid windowtext .5pt;mso-border-left-alt:solid windowtext .5pt;
  mso-border-alt:solid windowtext .5pt;padding:0cm 5.4pt 0cm 5.4pt;height:1.0pt'>
  <p class=MsoNormal align=center style='text-align:center;mso-pagination:widow-orphan'><span
  lang=EN-US style='font-size:6.5pt;mso-bidi-font-size:10.0pt;font-family:"Times New Roman",serif;
  mso-fareast-font-family:等线;mso-font-kerning:0pt'>0.771<o:p></o:p></span></p>
  </td>
  <td width=39 nowrap valign=top style='width:29.5pt;border-top:none;
  border-left:none;border-bottom:solid windowtext 1.0pt;border-right:solid windowtext 1.0pt;
  mso-border-top-alt:solid windowtext .5pt;mso-border-left-alt:solid windowtext .5pt;
  mso-border-alt:solid windowtext .5pt;padding:0cm 5.4pt 0cm 5.4pt;height:1.0pt'>
  <p class=MsoNormal align=center style='text-align:center;mso-pagination:widow-orphan'><span
  lang=EN-US style='font-size:6.5pt;mso-bidi-font-size:10.0pt;font-family:"Times New Roman",serif;
  mso-fareast-font-family:等线;mso-font-kerning:0pt'>-2.515<o:p></o:p></span></p>
  </td>
  <td width=39 nowrap valign=top style='width:29.55pt;border-top:none;
  border-left:none;border-bottom:solid windowtext 1.0pt;border-right:solid windowtext 1.0pt;
  mso-border-top-alt:solid windowtext .5pt;mso-border-left-alt:solid windowtext .5pt;
  mso-border-alt:solid windowtext .5pt;padding:0cm 5.4pt 0cm 5.4pt;height:1.0pt'>
  <p class=MsoNormal align=center style='text-align:center;mso-pagination:widow-orphan'><span
  lang=EN-US style='font-size:6.5pt;mso-bidi-font-size:10.0pt;font-family:"Times New Roman",serif;
  mso-fareast-font-family:等线;mso-font-kerning:0pt'>7.971<o:p></o:p></span></p>
  </td>
  <td width=39 nowrap valign=top style='width:29.55pt;border-top:none;
  border-left:none;border-bottom:solid windowtext 1.0pt;border-right:solid windowtext 1.0pt;
  mso-border-top-alt:solid windowtext .5pt;mso-border-left-alt:solid windowtext .5pt;
  mso-border-alt:solid windowtext .5pt;padding:0cm 5.4pt 0cm 5.4pt;height:1.0pt'>
  <p class=MsoNormal align=center style='text-align:center;mso-pagination:widow-orphan'><span
  lang=EN-US style='font-size:6.5pt;mso-bidi-font-size:10.0pt;font-family:"Times New Roman",serif;
  mso-fareast-font-family:等线;mso-font-kerning:0pt'>5.166<o:p></o:p></span></p>
  </td>
  <td width=79 nowrap colspan=2 style='width:59.1pt;border-top:none;border-left:
  none;border-bottom:solid windowtext 1.0pt;border-right:solid windowtext 1.0pt;
  mso-border-top-alt:solid windowtext .5pt;mso-border-left-alt:solid windowtext .5pt;
  mso-border-alt:solid windowtext .5pt;padding:0cm 5.4pt 0cm 5.4pt;height:1.0pt'>
  <p class=MsoNormal align=center style='text-align:center;mso-pagination:widow-orphan'><span
  lang=EN-US style='font-size:6.5pt;mso-bidi-font-size:10.0pt;font-family:"Times New Roman",serif;
  mso-fareast-font-family:等线;mso-font-kerning:0pt'>Noisy<o:p></o:p></span></p>
  </td>
  <td width=39 style='width:29.5pt;border-top:none;border-left:none;border-bottom:
  solid windowtext 1.0pt;border-right:solid windowtext 1.0pt;mso-border-top-alt:
  solid windowtext .5pt;mso-border-left-alt:solid windowtext .5pt;mso-border-alt:
  solid windowtext .5pt;padding:0cm 5.4pt 0cm 5.4pt;height:1.0pt'>
  <p class=MsoNormal align=center style='text-align:center;mso-pagination:widow-orphan'><span
  lang=EN-US style='font-size:6.5pt;mso-bidi-font-size:10.0pt;font-family:"Times New Roman",serif;
  mso-fareast-font-family:等线;mso-font-kerning:0pt'>18.37<o:p></o:p></span></p>
  </td>
  <td width=39 style='width:29.55pt;border-top:none;border-left:none;
  border-bottom:solid windowtext 1.0pt;border-right:solid windowtext 1.0pt;
  mso-border-top-alt:solid windowtext .5pt;mso-border-left-alt:solid windowtext .5pt;
  mso-border-alt:solid windowtext .5pt;padding:0cm 5.4pt 0cm 5.4pt;height:1.0pt'>
  <p class=MsoNormal align=center style='text-align:center;mso-pagination:widow-orphan'><span
  lang=EN-US style='font-size:6.5pt;mso-bidi-font-size:10.0pt;font-family:"Times New Roman",serif;
  mso-fareast-font-family:等线;mso-font-kerning:0pt'>12.38<o:p></o:p></span></p>
  </td>
  <td width=39 valign=top style='width:29.55pt;border-top:none;border-left:
  none;border-bottom:solid windowtext 1.0pt;border-right:solid windowtext 1.0pt;
  mso-border-top-alt:solid windowtext .5pt;mso-border-left-alt:solid windowtext .5pt;
  mso-border-alt:solid windowtext .5pt;padding:0cm 5.4pt 0cm 5.4pt;height:1.0pt'>
  <p class=MsoNormal align=center style='text-align:center;mso-pagination:widow-orphan'><span
  lang=EN-US style='font-size:6.5pt;mso-bidi-font-size:10.0pt;font-family:"Times New Roman",serif;
  mso-fareast-font-family:等线;mso-font-kerning:0pt'>14.38<o:p></o:p></span></p>
  </td>
 </tr>
 <tr style='mso-yfti-irow:3;height:1.0pt'>
  <td width=75 nowrap style='width:56.45pt;border:solid windowtext 1.0pt;
  border-top:none;mso-border-top-alt:solid windowtext .5pt;mso-border-alt:solid windowtext .5pt;
  padding:0cm 5.4pt 0cm 5.4pt;height:1.0pt'>
  <p class=MsoNormal align=center style='text-align:center;mso-pagination:widow-orphan'><span
  lang=EN-US style='font-size:6.5pt;mso-bidi-font-size:10.0pt;font-family:"Times New Roman",serif;
  mso-fareast-font-family:等线;mso-font-kerning:0pt'>MTFAA<o:p></o:p></span></p>
  </td>
  <td width=39 nowrap style='width:29.5pt;border-top:none;border-left:none;
  border-bottom:solid windowtext 1.0pt;border-right:solid windowtext 1.0pt;
  mso-border-top-alt:solid windowtext .5pt;mso-border-left-alt:solid windowtext .5pt;
  mso-border-alt:solid windowtext .5pt;padding:0cm 5.4pt 0cm 5.4pt;height:1.0pt'>
  <p class=MsoNormal align=center style='text-align:center;mso-pagination:widow-orphan'><span
  lang=EN-US style='font-size:6.5pt;mso-bidi-font-size:10.0pt;font-family:"Times New Roman",serif;
  mso-fareast-font-family:等线;mso-font-kerning:0pt'>0.805<o:p></o:p></span></p>
  </td>
  <td width=39 nowrap style='width:29.55pt;border-top:none;border-left:none;
  border-bottom:solid windowtext 1.0pt;border-right:solid windowtext 1.0pt;
  mso-border-top-alt:solid windowtext .5pt;mso-border-left-alt:solid windowtext .5pt;
  mso-border-alt:solid windowtext .5pt;padding:0cm 5.4pt 0cm 5.4pt;height:1.0pt'>
  <p class=MsoNormal align=center style='text-align:center;mso-pagination:widow-orphan'><span
  lang=EN-US style='font-size:6.5pt;mso-bidi-font-size:10.0pt;font-family:"Times New Roman",serif;
  mso-fareast-font-family:等线;mso-font-kerning:0pt'>0.876<o:p></o:p></span></p>
  </td>
  <td width=39 nowrap style='width:29.55pt;border-top:none;border-left:none;
  border-bottom:solid windowtext 1.0pt;border-right:solid windowtext 1.0pt;
  mso-border-top-alt:solid windowtext .5pt;mso-border-left-alt:solid windowtext .5pt;
  mso-border-alt:solid windowtext .5pt;padding:0cm 5.4pt 0cm 5.4pt;height:1.0pt'>
  <p class=MsoNormal align=center style='text-align:center;mso-pagination:widow-orphan'><span
  lang=EN-US style='font-size:6.5pt;mso-bidi-font-size:10.0pt;font-family:"Times New Roman",serif;
  mso-fareast-font-family:等线;mso-font-kerning:0pt'>0.856<o:p></o:p></span></p>
  </td>
  <td width=39 nowrap valign=top style='width:29.5pt;border-top:none;
  border-left:none;border-bottom:solid windowtext 1.0pt;border-right:solid windowtext 1.0pt;
  mso-border-top-alt:solid windowtext .5pt;mso-border-left-alt:solid windowtext .5pt;
  mso-border-alt:solid windowtext .5pt;padding:0cm 5.4pt 0cm 5.4pt;height:1.0pt'>
  <p class=MsoNormal align=center style='text-align:center;mso-pagination:widow-orphan'><span
  lang=EN-US style='font-size:6.5pt;mso-bidi-font-size:10.0pt;font-family:"Times New Roman",serif;
  mso-fareast-font-family:等线;mso-font-kerning:0pt'>10.10<o:p></o:p></span></p>
  </td>
  <td width=39 nowrap valign=top style='width:29.55pt;border-top:none;
  border-left:none;border-bottom:solid windowtext 1.0pt;border-right:solid windowtext 1.0pt;
  mso-border-top-alt:solid windowtext .5pt;mso-border-left-alt:solid windowtext .5pt;
  mso-border-alt:solid windowtext .5pt;padding:0cm 5.4pt 0cm 5.4pt;height:1.0pt'>
  <p class=MsoNormal align=center style='text-align:center;mso-pagination:widow-orphan'><span
  lang=EN-US style='font-size:6.5pt;mso-bidi-font-size:10.0pt;font-family:"Times New Roman",serif;
  mso-fareast-font-family:等线;mso-font-kerning:0pt'>15.74<o:p></o:p></span></p>
  </td>
  <td width=39 nowrap valign=top style='width:29.55pt;border-top:none;
  border-left:none;border-bottom:solid windowtext 1.0pt;border-right:solid windowtext 1.0pt;
  mso-border-top-alt:solid windowtext .5pt;mso-border-left-alt:solid windowtext .5pt;
  mso-border-alt:solid windowtext .5pt;padding:0cm 5.4pt 0cm 5.4pt;height:1.0pt'>
  <p class=MsoNormal align=center style='text-align:center;mso-pagination:widow-orphan'><span
  lang=EN-US style='font-size:6.5pt;mso-bidi-font-size:10.0pt;font-family:"Times New Roman",serif;
  mso-fareast-font-family:等线;mso-font-kerning:0pt'>14.23<o:p></o:p></span></p>
  </td>
  <td width=79 nowrap colspan=2 style='width:59.1pt;border-top:none;border-left:
  none;border-bottom:solid windowtext 1.0pt;border-right:solid windowtext 1.0pt;
  mso-border-top-alt:solid windowtext .5pt;mso-border-left-alt:solid windowtext .5pt;
  mso-border-alt:solid windowtext .5pt;padding:0cm 5.4pt 0cm 5.4pt;height:1.0pt'>
  <p class=MsoNormal align=center style='text-align:center;mso-pagination:widow-orphan'><span
  lang=EN-US style='font-size:6.5pt;mso-bidi-font-size:10.0pt;font-family:"Times New Roman",serif;
  mso-fareast-font-family:等线;mso-font-kerning:0pt'>MTFAA<o:p></o:p></span></p>
  </td>
  <td width=39 style='width:29.5pt;border-top:none;border-left:none;border-bottom:
  solid windowtext 1.0pt;border-right:solid windowtext 1.0pt;mso-border-top-alt:
  solid windowtext .5pt;mso-border-left-alt:solid windowtext .5pt;mso-border-alt:
  solid windowtext .5pt;padding:0cm 5.4pt 0cm 5.4pt;height:1.0pt'>
  <p class=MsoNormal align=center style='text-align:center;mso-pagination:widow-orphan'><span
  lang=EN-US style='font-size:6.5pt;mso-bidi-font-size:10.0pt;font-family:"Times New Roman",serif;
  mso-fareast-font-family:等线;mso-font-kerning:0pt'>10.33<o:p></o:p></span></p>
  </td>
  <td width=39 style='width:29.55pt;border-top:none;border-left:none;
  border-bottom:solid windowtext 1.0pt;border-right:solid windowtext 1.0pt;
  mso-border-top-alt:solid windowtext .5pt;mso-border-left-alt:solid windowtext .5pt;
  mso-border-alt:solid windowtext .5pt;padding:0cm 5.4pt 0cm 5.4pt;height:1.0pt'>
  <p class=MsoNormal align=center style='text-align:center;mso-pagination:widow-orphan'><span
  lang=EN-US style='font-size:6.5pt;mso-bidi-font-size:10.0pt;font-family:"Times New Roman",serif;
  mso-fareast-font-family:等线;mso-font-kerning:0pt'>9.349<o:p></o:p></span></p>
  </td>
  <td width=39 valign=top style='width:29.55pt;border-top:none;border-left:
  none;border-bottom:solid windowtext 1.0pt;border-right:solid windowtext 1.0pt;
  mso-border-top-alt:solid windowtext .5pt;mso-border-left-alt:solid windowtext .5pt;
  mso-border-alt:solid windowtext .5pt;padding:0cm 5.4pt 0cm 5.4pt;height:1.0pt'>
  <p class=MsoNormal align=center style='text-align:center;mso-pagination:widow-orphan'><span
  lang=EN-US style='font-size:6.5pt;mso-bidi-font-size:10.0pt;font-family:"Times New Roman",serif;
  mso-fareast-font-family:等线;mso-font-kerning:0pt'>9.678<o:p></o:p></span></p>
  </td>
 </tr>
 <tr style='mso-yfti-irow:4;height:1.0pt'>
  <td width=75 nowrap style='width:56.45pt;border:solid windowtext 1.0pt;
  border-top:none;mso-border-top-alt:solid windowtext .5pt;mso-border-alt:solid windowtext .5pt;
  padding:0cm 5.4pt 0cm 5.4pt;height:1.0pt'>
  <p class=MsoNormal align=center style='text-align:center;mso-pagination:widow-orphan'><span
  lang=EN-US style='font-size:6.5pt;mso-bidi-font-size:10.0pt;font-family:"Times New Roman",serif;
  mso-fareast-font-family:等线;mso-font-kerning:0pt'>MTFAA-LSA</span><span
  lang=EN-US style='font-size:7.5pt;mso-bidi-font-size:10.0pt;font-family:"Times New Roman",serif;
  mso-fareast-font-family:等线;mso-font-kerning:0pt'><o:p></o:p></span></p>
  </td>
  <td width=39 nowrap style='width:29.5pt;border-top:none;border-left:none;
  border-bottom:solid windowtext 1.0pt;border-right:solid windowtext 1.0pt;
  mso-border-top-alt:solid windowtext .5pt;mso-border-left-alt:solid windowtext .5pt;
  mso-border-alt:solid windowtext .5pt;padding:0cm 5.4pt 0cm 5.4pt;height:1.0pt'>
  <p class=MsoNormal align=center style='text-align:center;mso-pagination:widow-orphan'><b
  style='mso-bidi-font-weight:normal'><span lang=EN-US style='font-size:6.5pt;
  mso-bidi-font-size:10.0pt;font-family:"Times New Roman",serif;mso-fareast-font-family:
  等线;mso-font-kerning:0pt'>0.809<o:p></o:p></span></b></p>
  </td>
  <td width=39 nowrap style='width:29.55pt;border-top:none;border-left:none;
  border-bottom:solid windowtext 1.0pt;border-right:solid windowtext 1.0pt;
  mso-border-top-alt:solid windowtext .5pt;mso-border-left-alt:solid windowtext .5pt;
  mso-border-alt:solid windowtext .5pt;padding:0cm 5.4pt 0cm 5.4pt;height:1.0pt'>
  <p class=MsoNormal align=center style='text-align:center;mso-pagination:widow-orphan'><b
  style='mso-bidi-font-weight:normal'><span lang=EN-US style='font-size:6.5pt;
  mso-bidi-font-size:10.0pt;font-family:"Times New Roman",serif;mso-fareast-font-family:
  等线;mso-font-kerning:0pt'>0.881<o:p></o:p></span></b></p>
  </td>
  <td width=39 nowrap style='width:29.55pt;border-top:none;border-left:none;
  border-bottom:solid windowtext 1.0pt;border-right:solid windowtext 1.0pt;
  mso-border-top-alt:solid windowtext .5pt;mso-border-left-alt:solid windowtext .5pt;
  mso-border-alt:solid windowtext .5pt;padding:0cm 5.4pt 0cm 5.4pt;height:1.0pt'>
  <p class=MsoNormal align=center style='text-align:center;mso-pagination:widow-orphan'><b
  style='mso-bidi-font-weight:normal'><span lang=EN-US style='font-size:6.5pt;
  mso-bidi-font-size:10.0pt;font-family:"Times New Roman",serif;mso-fareast-font-family:
  等线;mso-font-kerning:0pt'>0.860<o:p></o:p></span></b></p>
  </td>
  <td width=39 nowrap valign=top style='width:29.5pt;border-top:none;
  border-left:none;border-bottom:solid windowtext 1.0pt;border-right:solid windowtext 1.0pt;
  mso-border-top-alt:solid windowtext .5pt;mso-border-left-alt:solid windowtext .5pt;
  mso-border-alt:solid windowtext .5pt;padding:0cm 5.4pt 0cm 5.4pt;height:1.0pt'>
  <p class=MsoNormal align=center style='text-align:center;mso-pagination:widow-orphan'><span
  lang=EN-US style='font-size:6.5pt;mso-bidi-font-size:10.0pt;font-family:"Times New Roman",serif;
  mso-fareast-font-family:等线;mso-font-kerning:0pt'>10.34<o:p></o:p></span></p>
  </td>
  <td width=39 nowrap valign=top style='width:29.55pt;border-top:none;
  border-left:none;border-bottom:solid windowtext 1.0pt;border-right:solid windowtext 1.0pt;
  mso-border-top-alt:solid windowtext .5pt;mso-border-left-alt:solid windowtext .5pt;
  mso-border-alt:solid windowtext .5pt;padding:0cm 5.4pt 0cm 5.4pt;height:1.0pt'>
  <p class=MsoNormal align=center style='text-align:center;mso-pagination:widow-orphan'><b
  style='mso-bidi-font-weight:normal'><span lang=EN-US style='font-size:6.5pt;
  mso-bidi-font-size:10.0pt;font-family:"Times New Roman",serif;mso-fareast-font-family:
  等线;mso-font-kerning:0pt'>16.20<o:p></o:p></span></b></p>
  </td>
  <td width=39 nowrap valign=top style='width:29.55pt;border-top:none;
  border-left:none;border-bottom:solid windowtext 1.0pt;border-right:solid windowtext 1.0pt;
  mso-border-top-alt:solid windowtext .5pt;mso-border-left-alt:solid windowtext .5pt;
  mso-border-alt:solid windowtext .5pt;padding:0cm 5.4pt 0cm 5.4pt;height:1.0pt'>
  <p class=MsoNormal align=center style='text-align:center;mso-pagination:widow-orphan'><b
  style='mso-bidi-font-weight:normal'><span lang=EN-US style='font-size:6.5pt;
  mso-bidi-font-size:10.0pt;font-family:"Times New Roman",serif;mso-fareast-font-family:
  等线;mso-font-kerning:0pt'>14.63</span></b><span lang=EN-US style='font-size:
  6.5pt;mso-bidi-font-size:10.0pt;font-family:"Times New Roman",serif;
  mso-fareast-font-family:等线;mso-font-kerning:0pt'><o:p></o:p></span></p>
  </td>
  <td width=79 nowrap colspan=2 style='width:59.1pt;border-top:none;border-left:
  none;border-bottom:solid windowtext 1.0pt;border-right:solid windowtext 1.0pt;
  mso-border-top-alt:solid windowtext .5pt;mso-border-left-alt:solid windowtext .5pt;
  mso-border-alt:solid windowtext .5pt;padding:0cm 5.4pt 0cm 5.4pt;height:1.0pt'>
  <p class=MsoNormal align=center style='text-align:center;mso-pagination:widow-orphan'><span
  lang=EN-US style='font-size:6.5pt;mso-bidi-font-size:10.0pt;font-family:"Times New Roman",serif;
  mso-fareast-font-family:等线;mso-font-kerning:0pt'>MTFAA-LSA<b
  style='mso-bidi-font-weight:normal'><o:p></o:p></b></span></p>
  </td>
  <td width=39 style='width:29.5pt;border-top:none;border-left:none;border-bottom:
  solid windowtext 1.0pt;border-right:solid windowtext 1.0pt;mso-border-top-alt:
  solid windowtext .5pt;mso-border-left-alt:solid windowtext .5pt;mso-border-alt:
  solid windowtext .5pt;padding:0cm 5.4pt 0cm 5.4pt;height:1.0pt'>
  <p class=MsoNormal align=center style='text-align:center;mso-pagination:widow-orphan'><b
  style='mso-bidi-font-weight:normal'><span lang=EN-US style='font-size:6.5pt;
  mso-bidi-font-size:10.0pt;font-family:"Times New Roman",serif;mso-fareast-font-family:
  等线;mso-font-kerning:0pt'>9.840<o:p></o:p></span></b></p>
  </td>
  <td width=39 style='width:29.55pt;border-top:none;border-left:none;
  border-bottom:solid windowtext 1.0pt;border-right:solid windowtext 1.0pt;
  mso-border-top-alt:solid windowtext .5pt;mso-border-left-alt:solid windowtext .5pt;
  mso-border-alt:solid windowtext .5pt;padding:0cm 5.4pt 0cm 5.4pt;height:1.0pt'>
  <p class=MsoNormal align=center style='text-align:center;mso-pagination:widow-orphan'><span
  lang=EN-US style='font-size:6.5pt;mso-bidi-font-size:10.0pt;font-family:"Times New Roman",serif;
  mso-fareast-font-family:等线;mso-font-kerning:0pt'>8.636<o:p></o:p></span></p>
  </td>
  <td width=39 valign=top style='width:29.55pt;border-top:none;border-left:
  none;border-bottom:solid windowtext 1.0pt;border-right:solid windowtext 1.0pt;
  mso-border-top-alt:solid windowtext .5pt;mso-border-left-alt:solid windowtext .5pt;
  mso-border-alt:solid windowtext .5pt;padding:0cm 5.4pt 0cm 5.4pt;height:1.0pt'>
  <p class=MsoNormal align=center style='text-align:center;mso-pagination:widow-orphan'><b
  style='mso-bidi-font-weight:normal'><span lang=EN-US style='font-size:6.5pt;
  mso-bidi-font-size:10.0pt;font-family:"Times New Roman",serif;mso-fareast-font-family:
  等线;mso-font-kerning:0pt'>9.037<o:p></o:p></span></b></p>
  </td>
 </tr>
 <tr style='mso-yfti-irow:5;height:1.0pt'>
  <td width=75 nowrap style='width:56.45pt;border:solid windowtext 1.0pt;
  border-top:none;mso-border-top-alt:solid windowtext .5pt;mso-border-alt:solid windowtext .5pt;
  padding:0cm 5.4pt 0cm 5.4pt;height:1.0pt'>
  <p class=MsoNormal align=center style='text-align:center;mso-pagination:widow-orphan'><span
  lang=EN-US style='font-size:5.5pt;mso-bidi-font-size:10.0pt;font-family:"Times New Roman",serif;
  mso-fareast-font-family:等线;mso-font-kerning:0pt'>MTFAA-MSLSA</span><span
  lang=EN-US style='font-size:6.5pt;mso-bidi-font-size:10.0pt;font-family:"Times New Roman",serif;
  mso-fareast-font-family:等线;mso-font-kerning:0pt'><o:p></o:p></span></p>
  </td>
  <td width=39 nowrap style='width:29.5pt;border-top:none;border-left:none;
  border-bottom:solid windowtext 1.0pt;border-right:solid windowtext 1.0pt;
  mso-border-top-alt:solid windowtext .5pt;mso-border-left-alt:solid windowtext .5pt;
  mso-border-alt:solid windowtext .5pt;padding:0cm 5.4pt 0cm 5.4pt;height:1.0pt'>
  <p class=MsoNormal align=center style='text-align:center;mso-pagination:widow-orphan'><span
  lang=EN-US style='font-size:6.5pt;mso-bidi-font-size:10.0pt;font-family:"Times New Roman",serif;
  mso-fareast-font-family:等线;mso-font-kerning:0pt'>0.809<o:p></o:p></span></p>
  </td>
  <td width=39 nowrap style='width:29.55pt;border-top:none;border-left:none;
  border-bottom:solid windowtext 1.0pt;border-right:solid windowtext 1.0pt;
  mso-border-top-alt:solid windowtext .5pt;mso-border-left-alt:solid windowtext .5pt;
  mso-border-alt:solid windowtext .5pt;padding:0cm 5.4pt 0cm 5.4pt;height:1.0pt'>
  <p class=MsoNormal align=center style='text-align:center;mso-pagination:widow-orphan'><span
  lang=EN-US style='font-size:6.5pt;mso-bidi-font-size:10.0pt;font-family:"Times New Roman",serif;
  mso-fareast-font-family:等线;mso-font-kerning:0pt'>0.880<o:p></o:p></span></p>
  </td>
  <td width=39 nowrap style='width:29.55pt;border-top:none;border-left:none;
  border-bottom:solid windowtext 1.0pt;border-right:solid windowtext 1.0pt;
  mso-border-top-alt:solid windowtext .5pt;mso-border-left-alt:solid windowtext .5pt;
  mso-border-alt:solid windowtext .5pt;padding:0cm 5.4pt 0cm 5.4pt;height:1.0pt'>
  <p class=MsoNormal align=center style='text-align:center;mso-pagination:widow-orphan'><span
  lang=EN-US style='font-size:6.5pt;mso-bidi-font-size:10.0pt;font-family:"Times New Roman",serif;
  mso-fareast-font-family:等线;mso-font-kerning:0pt'>0.859<o:p></o:p></span></p>
  </td>
  <td width=39 nowrap valign=top style='width:29.5pt;border-top:none;
  border-left:none;border-bottom:solid windowtext 1.0pt;border-right:solid windowtext 1.0pt;
  mso-border-top-alt:solid windowtext .5pt;mso-border-left-alt:solid windowtext .5pt;
  mso-border-alt:solid windowtext .5pt;padding:0cm 5.4pt 0cm 5.4pt;height:1.0pt'>
  <p class=MsoNormal align=center style='text-align:center;mso-pagination:widow-orphan'><b
  style='mso-bidi-font-weight:normal'><span lang=EN-US style='font-size:6.5pt;
  mso-bidi-font-size:10.0pt;font-family:"Times New Roman",serif;mso-fareast-font-family:
  等线;mso-font-kerning:0pt'>10.43<o:p></o:p></span></b></p>
  </td>
  <td width=39 nowrap valign=top style='width:29.55pt;border-top:none;
  border-left:none;border-bottom:solid windowtext 1.0pt;border-right:solid windowtext 1.0pt;
  mso-border-top-alt:solid windowtext .5pt;mso-border-left-alt:solid windowtext .5pt;
  mso-border-alt:solid windowtext .5pt;padding:0cm 5.4pt 0cm 5.4pt;height:1.0pt'>
  <p class=MsoNormal align=center style='text-align:center;mso-pagination:widow-orphan'><span
  lang=EN-US style='font-size:6.5pt;mso-bidi-font-size:10.0pt;font-family:"Times New Roman",serif;
  mso-fareast-font-family:等线;mso-font-kerning:0pt'>15.98<o:p></o:p></span></p>
  </td>
  <td width=39 nowrap valign=top style='width:29.55pt;border-top:none;
  border-left:none;border-bottom:solid windowtext 1.0pt;border-right:solid windowtext 1.0pt;
  mso-border-top-alt:solid windowtext .5pt;mso-border-left-alt:solid windowtext .5pt;
  mso-border-alt:solid windowtext .5pt;padding:0cm 5.4pt 0cm 5.4pt;height:1.0pt'>
  <p class=MsoNormal align=center style='text-align:center;mso-pagination:widow-orphan'><span
  lang=EN-US style='font-size:6.5pt;mso-bidi-font-size:10.0pt;font-family:"Times New Roman",serif;
  mso-fareast-font-family:等线;mso-font-kerning:0pt'>14.50<o:p></o:p></span></p>
  </td>
  <td width=39 nowrap style='width:29.5pt;border-top:none;border-left:none;
  border-bottom:solid windowtext 1.0pt;border-right:solid windowtext 1.0pt;
  mso-border-top-alt:solid windowtext .5pt;mso-border-left-alt:solid windowtext .5pt;
  mso-border-alt:solid windowtext .5pt;padding:0cm 5.4pt 0cm 5.4pt;height:1.0pt'>
  <p class=MsoNormal align=center style='text-align:center;mso-pagination:widow-orphan'><b
  style='mso-bidi-font-weight:normal'><span lang=EN-US style='font-size:6.5pt;
  mso-bidi-font-size:10.0pt;font-family:"Times New Roman",serif;mso-fareast-font-family:
  等线;mso-font-kerning:0pt'><o:p>&nbsp;</o:p></span></b></p>
  </td>
  <td width=79 nowrap colspan=2 style='width:59.1pt;border-top:none;border-left:
  none;border-bottom:solid windowtext 1.0pt;border-right:solid windowtext 1.0pt;
  mso-border-top-alt:solid windowtext .5pt;mso-border-left-alt:solid windowtext .5pt;
  mso-border-alt:solid windowtext .5pt;padding:0cm 5.4pt 0cm 5.4pt;height:1.0pt'>
  <p class=MsoNormal align=center style='text-align:center;mso-pagination:widow-orphan'><span
  lang=EN-US style='font-size:5.5pt;mso-bidi-font-size:10.0pt;font-family:"Times New Roman",serif;
  mso-fareast-font-family:等线;mso-font-kerning:0pt'>MTFAA-MSLSA</span><span
  lang=EN-US style='font-size:6.5pt;mso-bidi-font-size:10.0pt;font-family:"Times New Roman",serif;
  mso-fareast-font-family:等线;mso-font-kerning:0pt'><o:p></o:p></span></p>
  </td>
  <td width=39 style='width:29.5pt;border-top:none;border-left:none;border-bottom:
  solid windowtext 1.0pt;border-right:solid windowtext 1.0pt;mso-border-top-alt:
  solid windowtext .5pt;mso-border-left-alt:solid windowtext .5pt;mso-border-alt:
  solid windowtext .5pt;padding:0cm 5.4pt 0cm 5.4pt;height:1.0pt'>
  <p class=MsoNormal align=center style='text-align:center;mso-pagination:widow-orphan'><span
  lang=EN-US style='font-size:6.5pt;mso-bidi-font-size:10.0pt;font-family:"Times New Roman",serif;
  mso-fareast-font-family:等线;mso-font-kerning:0pt'>10.03<o:p></o:p></span></p>
  </td>
  <td width=39 style='width:29.55pt;border-top:none;border-left:none;
  border-bottom:solid windowtext 1.0pt;border-right:solid windowtext 1.0pt;
  mso-border-top-alt:solid windowtext .5pt;mso-border-left-alt:solid windowtext .5pt;
  mso-border-alt:solid windowtext .5pt;padding:0cm 5.4pt 0cm 5.4pt;height:1.0pt'>
  <p class=MsoNormal align=center style='text-align:center;mso-pagination:widow-orphan'><b
  style='mso-bidi-font-weight:normal'><span lang=EN-US style='font-size:6.5pt;
  mso-bidi-font-size:10.0pt;font-family:"Times New Roman",serif;mso-fareast-font-family:
  等线;mso-font-kerning:0pt'>8.623<o:p></o:p></span></b></p>
  </td>
  <td width=39 valign=top style='width:29.55pt;border-top:none;border-left:
  none;border-bottom:solid windowtext 1.0pt;border-right:solid windowtext 1.0pt;
  mso-border-top-alt:solid windowtext .5pt;mso-border-left-alt:solid windowtext .5pt;
  mso-border-alt:solid windowtext .5pt;padding:0cm 5.4pt 0cm 5.4pt;height:1.0pt'>
  <p class=MsoNormal align=center style='text-align:center;mso-pagination:widow-orphan'><span
  lang=EN-US style='font-size:6.5pt;mso-bidi-font-size:10.0pt;font-family:"Times New Roman",serif;
  mso-fareast-font-family:等线;mso-font-kerning:0pt'>9.094<o:p></o:p></span></p>
  </td>
 </tr>
 <tr style='mso-yfti-irow:6;height:15.0pt'>
  <td width=548 nowrap colspan=13 style='width:410.85pt;border:solid windowtext 1.0pt;
  border-top:none;mso-border-top-alt:solid windowtext .5pt;mso-border-alt:solid windowtext .5pt;
  padding:0cm 5.4pt 0cm 5.4pt;height:15.0pt'></td>
 </tr>
 <tr style='mso-yfti-irow:7;height:15.0pt'>
  <td width=75 nowrap style='width:56.45pt;border:solid windowtext 1.0pt;
  border-top:none;mso-border-top-alt:solid windowtext .5pt;mso-border-alt:solid windowtext .5pt;
  padding:0cm 5.4pt 0cm 5.4pt;height:15.0pt'>
  <p class=MsoNormal align=center style='text-align:center;mso-pagination:widow-orphan'><span
  lang=EN-US style='font-size:5.5pt;mso-bidi-font-size:10.0pt;font-family:"Times New Roman",serif;
  mso-fareast-font-family:等线;mso-font-kerning:0pt'>Wideband Metrics</span><span
  lang=EN-US style='font-size:7.5pt;mso-bidi-font-size:10.0pt;font-family:"Times New Roman",serif;
  mso-fareast-font-family:等线;mso-font-kerning:0pt'><o:p></o:p></span></p>
  </td>
  <td width=118 nowrap colspan=3 style='width:88.6pt;border-top:none;
  border-left:none;border-bottom:solid windowtext 1.0pt;border-right:solid windowtext 1.0pt;
  mso-border-top-alt:solid windowtext .5pt;mso-border-left-alt:solid windowtext .5pt;
  mso-border-alt:solid windowtext .5pt;padding:0cm 5.4pt 0cm 5.4pt;height:15.0pt'>
  <p class=MsoNormal align=center style='text-align:center;mso-pagination:widow-orphan'><span
  lang=EN-US style='font-size:7.5pt;mso-bidi-font-size:10.0pt;font-family:"Times New Roman",serif;
  mso-fareast-font-family:等线;mso-font-kerning:0pt'>PESQ<o:p></o:p></span></p>
  </td>
  <td width=118 nowrap colspan=3 style='width:88.6pt;border-top:none;
  border-left:none;border-bottom:solid windowtext 1.0pt;border-right:solid windowtext 1.0pt;
  mso-border-top-alt:solid windowtext .5pt;mso-border-left-alt:solid windowtext .5pt;
  mso-border-alt:solid windowtext .5pt;padding:0cm 5.4pt 0cm 5.4pt;height:15.0pt'>
  <p class=MsoNormal align=center style='text-align:center;mso-pagination:widow-orphan'><span
  lang=EN-US style='font-size:7.5pt;mso-bidi-font-size:10.0pt;font-family:"Times New Roman",serif;
  mso-fareast-font-family:等线;mso-font-kerning:0pt'>CSIG<o:p></o:p></span></p>
  </td>
  <td width=118 nowrap colspan=3 style='width:88.6pt;border-top:none;
  border-left:none;border-bottom:solid windowtext 1.0pt;border-right:solid windowtext 1.0pt;
  mso-border-top-alt:solid windowtext .5pt;mso-border-left-alt:solid windowtext .5pt;
  mso-border-alt:solid windowtext .5pt;padding:0cm 5.4pt 0cm 5.4pt;height:15.0pt'>
  <p class=MsoNormal align=center style='text-align:center;mso-pagination:widow-orphan'><span
  lang=EN-US style='font-size:7.5pt;mso-bidi-font-size:10.0pt;font-family:"Times New Roman",serif;
  mso-fareast-font-family:等线;mso-font-kerning:0pt'>CBAK<o:p></o:p></span></p>
  </td>
  <td width=118 colspan=3 valign=top style='width:88.6pt;border-top:none;
  border-left:none;border-bottom:solid windowtext 1.0pt;border-right:solid windowtext 1.0pt;
  mso-border-top-alt:solid windowtext .5pt;mso-border-left-alt:solid windowtext .5pt;
  mso-border-alt:solid windowtext .5pt;padding:0cm 5.4pt 0cm 5.4pt;height:15.0pt'>
  <p class=MsoNormal align=center style='text-align:center;mso-pagination:widow-orphan'><span
  lang=EN-US style='font-size:7.5pt;mso-bidi-font-size:10.0pt;font-family:"Times New Roman",serif;
  mso-fareast-font-family:等线;mso-font-kerning:0pt'>COVL<o:p></o:p></span></p>
  </td>
 </tr>
 <tr style='mso-yfti-irow:8;height:15.0pt'>
  <td width=75 nowrap style='width:56.45pt;border:solid windowtext 1.0pt;
  border-top:none;mso-border-top-alt:solid windowtext .5pt;mso-border-alt:solid windowtext .5pt;
  padding:0cm 5.4pt 0cm 5.4pt;height:15.0pt'>
  <p class=MsoNormal align=center style='text-align:center;mso-pagination:widow-orphan'><span
  class=GramE><span lang=EN-US style='font-size:6.5pt;mso-bidi-font-size:10.0pt;
  font-family:"Times New Roman",serif;mso-fareast-font-family:等线;mso-font-kerning:
  0pt'>SNR(</span></span><span lang=EN-US style='font-size:6.5pt;mso-bidi-font-size:
  10.0pt;font-family:"Times New Roman",serif;mso-fareast-font-family:等线;
  mso-font-kerning:0pt'>dB)<o:p></o:p></span></p>
  </td>
  <td width=39 nowrap style='width:29.5pt;border-top:none;border-left:none;
  border-bottom:solid windowtext 1.0pt;border-right:solid windowtext 1.0pt;
  mso-border-top-alt:solid windowtext .5pt;mso-border-left-alt:solid windowtext .5pt;
  mso-border-alt:solid windowtext .5pt;padding:0cm 5.4pt 0cm 5.4pt;height:15.0pt'>
  <p class=MsoNormal align=center style='text-align:center;mso-pagination:widow-orphan'><span
  lang=EN-US style='font-size:7.5pt;font-family:"Times New Roman",serif;
  mso-fareast-font-family:等线;mso-font-kerning:0pt'>-5~0<o:p></o:p></span></p>
  </td>
  <td width=39 nowrap style='width:29.55pt;border-top:none;border-left:none;
  border-bottom:solid windowtext 1.0pt;border-right:solid windowtext 1.0pt;
  mso-border-top-alt:solid windowtext .5pt;mso-border-left-alt:solid windowtext .5pt;
  mso-border-alt:solid windowtext .5pt;padding:0cm 5.4pt 0cm 5.4pt;height:15.0pt'>
  <p class=MsoNormal align=center style='text-align:center;mso-pagination:widow-orphan'><span
  lang=EN-US style='font-size:7.5pt;font-family:"Times New Roman",serif;
  mso-fareast-font-family:等线;mso-font-kerning:0pt'>0~15<o:p></o:p></span></p>
  </td>
  <td width=39 nowrap style='width:29.55pt;border-top:none;border-left:none;
  border-bottom:solid windowtext 1.0pt;border-right:solid windowtext 1.0pt;
  mso-border-top-alt:solid windowtext .5pt;mso-border-left-alt:solid windowtext .5pt;
  mso-border-alt:solid windowtext .5pt;padding:0cm 5.4pt 0cm 5.4pt;height:15.0pt'>
  <p class=MsoNormal align=center style='text-align:center;mso-pagination:widow-orphan'><span
  class=SpellE><span lang=EN-US style='font-size:7.5pt;font-family:"Times New Roman",serif;
  mso-fareast-font-family:等线;mso-font-kerning:0pt'>Ovrl</span></span><span
  lang=EN-US style='font-size:7.5pt;font-family:"Times New Roman",serif;
  mso-fareast-font-family:等线;mso-font-kerning:0pt'>.<o:p></o:p></span></p>
  </td>
  <td width=39 nowrap style='width:29.5pt;border-top:none;border-left:none;
  border-bottom:solid windowtext 1.0pt;border-right:solid windowtext 1.0pt;
  mso-border-top-alt:solid windowtext .5pt;mso-border-left-alt:solid windowtext .5pt;
  mso-border-alt:solid windowtext .5pt;padding:0cm 5.4pt 0cm 5.4pt;height:15.0pt'>
  <p class=MsoNormal align=center style='text-align:center;mso-pagination:widow-orphan'><span
  lang=EN-US style='font-size:7.5pt;font-family:"Times New Roman",serif;
  mso-fareast-font-family:等线;mso-font-kerning:0pt'>-5~0<o:p></o:p></span></p>
  </td>
  <td width=39 nowrap style='width:29.55pt;border-top:none;border-left:none;
  border-bottom:solid windowtext 1.0pt;border-right:solid windowtext 1.0pt;
  mso-border-top-alt:solid windowtext .5pt;mso-border-left-alt:solid windowtext .5pt;
  mso-border-alt:solid windowtext .5pt;padding:0cm 5.4pt 0cm 5.4pt;height:15.0pt'>
  <p class=MsoNormal align=center style='text-align:center;mso-pagination:widow-orphan'><span
  lang=EN-US style='font-size:7.5pt;font-family:"Times New Roman",serif;
  mso-fareast-font-family:等线;mso-font-kerning:0pt'>0~15<o:p></o:p></span></p>
  </td>
  <td width=39 nowrap style='width:29.55pt;border-top:none;border-left:none;
  border-bottom:solid windowtext 1.0pt;border-right:solid windowtext 1.0pt;
  mso-border-top-alt:solid windowtext .5pt;mso-border-left-alt:solid windowtext .5pt;
  mso-border-alt:solid windowtext .5pt;padding:0cm 5.4pt 0cm 5.4pt;height:15.0pt'>
  <p class=MsoNormal align=center style='text-align:center;mso-pagination:widow-orphan'><span
  class=SpellE><span lang=EN-US style='font-size:7.5pt;font-family:"Times New Roman",serif;
  mso-fareast-font-family:等线;mso-font-kerning:0pt'>Ovrl</span></span><span
  lang=EN-US style='font-size:7.5pt;font-family:"Times New Roman",serif;
  mso-fareast-font-family:等线;mso-font-kerning:0pt'>.<o:p></o:p></span></p>
  </td>
  <td width=39 nowrap style='width:29.5pt;border-top:none;border-left:none;
  border-bottom:solid windowtext 1.0pt;border-right:solid windowtext 1.0pt;
  mso-border-top-alt:solid windowtext .5pt;mso-border-left-alt:solid windowtext .5pt;
  mso-border-alt:solid windowtext .5pt;padding:0cm 5.4pt 0cm 5.4pt;height:15.0pt'>
  <p class=MsoNormal align=center style='text-align:center;mso-pagination:widow-orphan'><span
  lang=EN-US style='font-size:7.5pt;font-family:"Times New Roman",serif;
  mso-fareast-font-family:等线;mso-font-kerning:0pt'>-5~0<o:p></o:p></span></p>
  </td>
  <td width=39 nowrap style='width:29.55pt;border-top:none;border-left:none;
  border-bottom:solid windowtext 1.0pt;border-right:solid windowtext 1.0pt;
  mso-border-top-alt:solid windowtext .5pt;mso-border-left-alt:solid windowtext .5pt;
  mso-border-alt:solid windowtext .5pt;padding:0cm 5.4pt 0cm 5.4pt;height:15.0pt'>
  <p class=MsoNormal align=center style='text-align:center;mso-pagination:widow-orphan'><span
  lang=EN-US style='font-size:7.5pt;font-family:"Times New Roman",serif;
  mso-fareast-font-family:等线;mso-font-kerning:0pt'>0~15<o:p></o:p></span></p>
  </td>
  <td width=39 nowrap style='width:29.55pt;border-top:none;border-left:none;
  border-bottom:solid windowtext 1.0pt;border-right:solid windowtext 1.0pt;
  mso-border-top-alt:solid windowtext .5pt;mso-border-left-alt:solid windowtext .5pt;
  mso-border-alt:solid windowtext .5pt;padding:0cm 5.4pt 0cm 5.4pt;height:15.0pt'>
  <p class=MsoNormal align=center style='text-align:center;mso-pagination:widow-orphan'><span
  class=SpellE><span lang=EN-US style='font-size:7.5pt;font-family:"Times New Roman",serif;
  mso-fareast-font-family:等线;mso-font-kerning:0pt'>Ovrl</span></span><span
  lang=EN-US style='font-size:7.5pt;font-family:"Times New Roman",serif;
  mso-fareast-font-family:等线;mso-font-kerning:0pt'>.<o:p></o:p></span></p>
  </td>
  <td width=39 style='width:29.5pt;border-top:none;border-left:none;border-bottom:
  solid windowtext 1.0pt;border-right:solid windowtext 1.0pt;mso-border-top-alt:
  solid windowtext .5pt;mso-border-left-alt:solid windowtext .5pt;mso-border-alt:
  solid windowtext .5pt;padding:0cm 5.4pt 0cm 5.4pt;height:15.0pt'>
  <p class=MsoNormal align=center style='text-align:center;mso-pagination:widow-orphan'><span
  lang=EN-US style='font-size:7.5pt;font-family:"Times New Roman",serif;
  mso-fareast-font-family:等线;mso-font-kerning:0pt'>-5~0<o:p></o:p></span></p>
  </td>
  <td width=39 style='width:29.55pt;border-top:none;border-left:none;
  border-bottom:solid windowtext 1.0pt;border-right:solid windowtext 1.0pt;
  mso-border-top-alt:solid windowtext .5pt;mso-border-left-alt:solid windowtext .5pt;
  mso-border-alt:solid windowtext .5pt;padding:0cm 5.4pt 0cm 5.4pt;height:15.0pt'>
  <p class=MsoNormal align=center style='text-align:center;mso-pagination:widow-orphan'><span
  lang=EN-US style='font-size:7.5pt;font-family:"Times New Roman",serif;
  mso-fareast-font-family:等线;mso-font-kerning:0pt'>0~15<o:p></o:p></span></p>
  </td>
  <td width=39 style='width:29.55pt;border-top:none;border-left:none;
  border-bottom:solid windowtext 1.0pt;border-right:solid windowtext 1.0pt;
  mso-border-top-alt:solid windowtext .5pt;mso-border-left-alt:solid windowtext .5pt;
  mso-border-alt:solid windowtext .5pt;padding:0cm 5.4pt 0cm 5.4pt;height:15.0pt'>
  <p class=MsoNormal align=center style='text-align:center;mso-pagination:widow-orphan'><span
  class=SpellE><span lang=EN-US style='font-size:7.5pt;font-family:"Times New Roman",serif;
  mso-fareast-font-family:等线;mso-font-kerning:0pt'>Ovrl</span></span><span
  lang=EN-US style='font-size:7.5pt;font-family:"Times New Roman",serif;
  mso-fareast-font-family:等线;mso-font-kerning:0pt'>.<o:p></o:p></span></p>
  </td>
 </tr>
 <tr style='mso-yfti-irow:9;height:15.6pt'>
  <td width=75 nowrap style='width:56.45pt;border:solid windowtext 1.0pt;
  border-top:none;mso-border-top-alt:solid windowtext .5pt;mso-border-alt:solid windowtext .5pt;
  padding:0cm 5.4pt 0cm 5.4pt;height:15.6pt'>
  <p class=MsoNormal align=center style='text-align:center;mso-pagination:widow-orphan'><span
  lang=EN-US style='font-size:6.5pt;mso-bidi-font-size:10.0pt;font-family:"Times New Roman",serif;
  mso-fareast-font-family:等线;mso-font-kerning:0pt'>Noisy<o:p></o:p></span></p>
  </td>
  <td width=39 nowrap style='width:29.5pt;border-top:none;border-left:none;
  border-bottom:solid windowtext 1.0pt;border-right:solid windowtext 1.0pt;
  mso-border-top-alt:solid windowtext .5pt;mso-border-left-alt:solid windowtext .5pt;
  mso-border-alt:solid windowtext .5pt;padding:0cm 5.4pt 0cm 5.4pt;height:15.6pt'>
  <p class=MsoNormal align=center style='text-align:center;mso-pagination:widow-orphan'><span
  lang=EN-US style='font-size:6.5pt;mso-bidi-font-size:10.0pt;font-family:"Times New Roman",serif;
  mso-fareast-font-family:等线;mso-font-kerning:0pt'>1.160</span><span
  lang=EN-US style='font-size:6.5pt;font-family:"Times New Roman",serif;
  mso-fareast-font-family:等线;mso-font-kerning:0pt'><o:p></o:p></span></p>
  </td>
  <td width=39 nowrap style='width:29.55pt;border-top:none;border-left:none;
  border-bottom:solid windowtext 1.0pt;border-right:solid windowtext 1.0pt;
  mso-border-top-alt:solid windowtext .5pt;mso-border-left-alt:solid windowtext .5pt;
  mso-border-alt:solid windowtext .5pt;padding:0cm 5.4pt 0cm 5.4pt;height:15.6pt'>
  <p class=MsoNormal align=center style='text-align:center;mso-pagination:widow-orphan'><span
  lang=EN-US style='font-size:6.5pt;mso-bidi-font-size:10.0pt;font-family:"Times New Roman",serif;
  mso-fareast-font-family:等线;mso-font-kerning:0pt'>1.446</span><span
  lang=EN-US style='font-size:6.5pt;font-family:"Times New Roman",serif;
  mso-fareast-font-family:等线;mso-font-kerning:0pt'><o:p></o:p></span></p>
  </td>
  <td width=39 nowrap style='width:29.55pt;border-top:none;border-left:none;
  border-bottom:solid windowtext 1.0pt;border-right:solid windowtext 1.0pt;
  mso-border-top-alt:solid windowtext .5pt;mso-border-left-alt:solid windowtext .5pt;
  mso-border-alt:solid windowtext .5pt;padding:0cm 5.4pt 0cm 5.4pt;height:15.6pt'>
  <p class=MsoNormal align=center style='text-align:center;mso-pagination:widow-orphan'><span
  lang=EN-US style='font-size:6.5pt;mso-bidi-font-size:10.0pt;font-family:"Times New Roman",serif;
  mso-fareast-font-family:等线;mso-font-kerning:0pt'>1.364</span><span
  lang=EN-US style='font-size:6.5pt;font-family:"Times New Roman",serif;
  mso-fareast-font-family:等线;mso-font-kerning:0pt'><o:p></o:p></span></p>
  </td>
  <td width=39 nowrap style='width:29.5pt;border-top:none;border-left:none;
  border-bottom:solid windowtext 1.0pt;border-right:solid windowtext 1.0pt;
  mso-border-top-alt:solid windowtext .5pt;mso-border-left-alt:solid windowtext .5pt;
  mso-border-alt:solid windowtext .5pt;padding:0cm 5.4pt 0cm 5.4pt;height:15.6pt'>
  <p class=MsoNormal align=center style='text-align:center;mso-pagination:widow-orphan'><span
  lang=EN-US style='font-size:6.5pt;font-family:"Times New Roman",serif;
  mso-fareast-font-family:等线;mso-font-kerning:0pt'>2.023<o:p></o:p></span></p>
  </td>
  <td width=39 nowrap style='width:29.55pt;border-top:none;border-left:none;
  border-bottom:solid windowtext 1.0pt;border-right:solid windowtext 1.0pt;
  mso-border-top-alt:solid windowtext .5pt;mso-border-left-alt:solid windowtext .5pt;
  mso-border-alt:solid windowtext .5pt;padding:0cm 5.4pt 0cm 5.4pt;height:15.6pt'>
  <p class=MsoNormal align=center style='text-align:center;mso-pagination:widow-orphan'><span
  lang=EN-US style='font-size:6.5pt;font-family:"Times New Roman",serif;
  mso-fareast-font-family:等线;mso-font-kerning:0pt'>2.719<o:p></o:p></span></p>
  </td>
  <td width=39 nowrap style='width:29.55pt;border-top:none;border-left:none;
  border-bottom:solid windowtext 1.0pt;border-right:solid windowtext 1.0pt;
  mso-border-top-alt:solid windowtext .5pt;mso-border-left-alt:solid windowtext .5pt;
  mso-border-alt:solid windowtext .5pt;padding:0cm 5.4pt 0cm 5.4pt;height:15.6pt'>
  <p class=MsoNormal align=center style='text-align:center;mso-pagination:widow-orphan'><span
  lang=EN-US style='font-size:6.5pt;font-family:"Times New Roman",serif;
  mso-fareast-font-family:等线;mso-font-kerning:0pt'>2.517<o:p></o:p></span></p>
  </td>
  <td width=39 nowrap style='width:29.5pt;border-top:none;border-left:none;
  border-bottom:solid windowtext 1.0pt;border-right:solid windowtext 1.0pt;
  mso-border-top-alt:solid windowtext .5pt;mso-border-left-alt:solid windowtext .5pt;
  mso-border-alt:solid windowtext .5pt;padding:0cm 5.4pt 0cm 5.4pt;height:15.6pt'>
  <p class=MsoNormal align=center style='text-align:center;mso-pagination:widow-orphan'><span
  lang=EN-US style='font-size:6.5pt;font-family:"Times New Roman",serif;
  mso-fareast-font-family:等线;mso-font-kerning:0pt'>1.833<o:p></o:p></span></p>
  </td>
  <td width=39 nowrap style='width:29.55pt;border-top:none;border-left:none;
  border-bottom:solid windowtext 1.0pt;border-right:solid windowtext 1.0pt;
  mso-border-top-alt:solid windowtext .5pt;mso-border-left-alt:solid windowtext .5pt;
  mso-border-alt:solid windowtext .5pt;padding:0cm 5.4pt 0cm 5.4pt;height:15.6pt'>
  <p class=MsoNormal align=center style='text-align:center;mso-pagination:widow-orphan'><span
  lang=EN-US style='font-size:6.5pt;font-family:"Times New Roman",serif;
  mso-fareast-font-family:等线;mso-font-kerning:0pt'>2.481<o:p></o:p></span></p>
  </td>
  <td width=39 nowrap style='width:29.55pt;border-top:none;border-left:none;
  border-bottom:solid windowtext 1.0pt;border-right:solid windowtext 1.0pt;
  mso-border-top-alt:solid windowtext .5pt;mso-border-left-alt:solid windowtext .5pt;
  mso-border-alt:solid windowtext .5pt;padding:0cm 5.4pt 0cm 5.4pt;height:15.6pt'>
  <p class=MsoNormal align=center style='text-align:center;mso-pagination:widow-orphan'><span
  lang=EN-US style='font-size:6.5pt;font-family:"Times New Roman",serif;
  mso-fareast-font-family:等线;mso-font-kerning:0pt'>2.293<o:p></o:p></span></p>
  </td>
  <td width=39 style='width:29.5pt;border-top:none;border-left:none;border-bottom:
  solid windowtext 1.0pt;border-right:solid windowtext 1.0pt;mso-border-top-alt:
  solid windowtext .5pt;mso-border-left-alt:solid windowtext .5pt;mso-border-alt:
  solid windowtext .5pt;padding:0cm 5.4pt 0cm 5.4pt;height:15.6pt'>
  <p class=MsoNormal align=center style='text-align:center;mso-pagination:widow-orphan'><span
  lang=EN-US style='font-size:6.5pt;font-family:"Times New Roman",serif;
  mso-fareast-font-family:等线;mso-font-kerning:0pt'>1.571<o:p></o:p></span></p>
  </td>
  <td width=39 style='width:29.55pt;border-top:none;border-left:none;
  border-bottom:solid windowtext 1.0pt;border-right:solid windowtext 1.0pt;
  mso-border-top-alt:solid windowtext .5pt;mso-border-left-alt:solid windowtext .5pt;
  mso-border-alt:solid windowtext .5pt;padding:0cm 5.4pt 0cm 5.4pt;height:15.6pt'>
  <p class=MsoNormal align=center style='text-align:center;mso-pagination:widow-orphan'><span
  lang=EN-US style='font-size:6.5pt;font-family:"Times New Roman",serif;
  mso-fareast-font-family:等线;mso-font-kerning:0pt'>2.095<o:p></o:p></span></p>
  </td>
  <td width=39 style='width:29.55pt;border-top:none;border-left:none;
  border-bottom:solid windowtext 1.0pt;border-right:solid windowtext 1.0pt;
  mso-border-top-alt:solid windowtext .5pt;mso-border-left-alt:solid windowtext .5pt;
  mso-border-alt:solid windowtext .5pt;padding:0cm 5.4pt 0cm 5.4pt;height:15.6pt'>
  <p class=MsoNormal align=center style='text-align:center;mso-pagination:widow-orphan'><span
  lang=EN-US style='font-size:6.5pt;font-family:"Times New Roman",serif;
  mso-fareast-font-family:等线;mso-font-kerning:0pt'>1.943<o:p></o:p></span></p>
  </td>
 </tr>
 <tr style='mso-yfti-irow:10;height:15.6pt'>
  <td width=75 nowrap style='width:56.45pt;border:solid windowtext 1.0pt;
  border-top:none;mso-border-top-alt:solid windowtext .5pt;mso-border-alt:solid windowtext .5pt;
  padding:0cm 5.4pt 0cm 5.4pt;height:15.6pt'>
  <p class=MsoNormal align=center style='text-align:center;mso-pagination:widow-orphan'><span
  lang=EN-US style='font-size:6.5pt;mso-bidi-font-size:10.0pt;font-family:"Times New Roman",serif;
  mso-fareast-font-family:等线;mso-font-kerning:0pt'>MTFAA<o:p></o:p></span></p>
  </td>
  <td width=39 nowrap style='width:29.5pt;border-top:none;border-left:none;
  border-bottom:solid windowtext 1.0pt;border-right:solid windowtext 1.0pt;
  mso-border-top-alt:solid windowtext .5pt;mso-border-left-alt:solid windowtext .5pt;
  mso-border-alt:solid windowtext .5pt;padding:0cm 5.4pt 0cm 5.4pt;height:15.6pt'>
  <p class=MsoNormal align=center style='text-align:center;mso-pagination:widow-orphan'><span
  lang=EN-US style='font-size:6.5pt;mso-bidi-font-size:10.0pt;font-family:"Times New Roman",serif;
  mso-fareast-font-family:等线;mso-font-kerning:0pt'>1.981</span><span
  lang=EN-US style='font-size:6.5pt;font-family:"Times New Roman",serif;
  mso-fareast-font-family:等线;mso-font-kerning:0pt'><o:p></o:p></span></p>
  </td>
  <td width=39 nowrap style='width:29.55pt;border-top:none;border-left:none;
  border-bottom:solid windowtext 1.0pt;border-right:solid windowtext 1.0pt;
  mso-border-top-alt:solid windowtext .5pt;mso-border-left-alt:solid windowtext .5pt;
  mso-border-alt:solid windowtext .5pt;padding:0cm 5.4pt 0cm 5.4pt;height:15.6pt'>
  <p class=MsoNormal align=center style='text-align:center;mso-pagination:widow-orphan'><span
  lang=EN-US style='font-size:6.5pt;mso-bidi-font-size:10.0pt;font-family:"Times New Roman",serif;
  mso-fareast-font-family:等线;mso-font-kerning:0pt'>2.669</span><span
  lang=EN-US style='font-size:6.5pt;font-family:"Times New Roman",serif;
  mso-fareast-font-family:等线;mso-font-kerning:0pt'><o:p></o:p></span></p>
  </td>
  <td width=39 nowrap style='width:29.55pt;border-top:none;border-left:none;
  border-bottom:solid windowtext 1.0pt;border-right:solid windowtext 1.0pt;
  mso-border-top-alt:solid windowtext .5pt;mso-border-left-alt:solid windowtext .5pt;
  mso-border-alt:solid windowtext .5pt;padding:0cm 5.4pt 0cm 5.4pt;height:15.6pt'>
  <p class=MsoNormal align=center style='text-align:center;mso-pagination:widow-orphan'><span
  lang=EN-US style='font-size:6.5pt;mso-bidi-font-size:10.0pt;font-family:"Times New Roman",serif;
  mso-fareast-font-family:等线;mso-font-kerning:0pt'>2.470</span><span
  lang=EN-US style='font-size:6.5pt;font-family:"Times New Roman",serif;
  mso-fareast-font-family:等线;mso-font-kerning:0pt'><o:p></o:p></span></p>
  </td>
  <td width=39 nowrap style='width:29.5pt;border-top:none;border-left:none;
  border-bottom:solid windowtext 1.0pt;border-right:solid windowtext 1.0pt;
  mso-border-top-alt:solid windowtext .5pt;mso-border-left-alt:solid windowtext .5pt;
  mso-border-alt:solid windowtext .5pt;padding:0cm 5.4pt 0cm 5.4pt;height:15.6pt'>
  <p class=MsoNormal align=center style='text-align:center;mso-pagination:widow-orphan'><span
  lang=EN-US style='font-size:6.5pt;font-family:"Times New Roman",serif;
  mso-fareast-font-family:等线;mso-font-kerning:0pt'>3.465<o:p></o:p></span></p>
  </td>
  <td width=39 nowrap style='width:29.55pt;border-top:none;border-left:none;
  border-bottom:solid windowtext 1.0pt;border-right:solid windowtext 1.0pt;
  mso-border-top-alt:solid windowtext .5pt;mso-border-left-alt:solid windowtext .5pt;
  mso-border-alt:solid windowtext .5pt;padding:0cm 5.4pt 0cm 5.4pt;height:15.6pt'>
  <p class=MsoNormal align=center style='text-align:center;mso-pagination:widow-orphan'><span
  lang=EN-US style='font-size:6.5pt;font-family:"Times New Roman",serif;
  mso-fareast-font-family:等线;mso-font-kerning:0pt'>4.113<o:p></o:p></span></p>
  </td>
  <td width=39 nowrap style='width:29.55pt;border-top:none;border-left:none;
  border-bottom:solid windowtext 1.0pt;border-right:solid windowtext 1.0pt;
  mso-border-top-alt:solid windowtext .5pt;mso-border-left-alt:solid windowtext .5pt;
  mso-border-alt:solid windowtext .5pt;padding:0cm 5.4pt 0cm 5.4pt;height:15.6pt'>
  <p class=MsoNormal align=center style='text-align:center;mso-pagination:widow-orphan'><span
  lang=EN-US style='font-size:6.5pt;font-family:"Times New Roman",serif;
  mso-fareast-font-family:等线;mso-font-kerning:0pt'>3.925<o:p></o:p></span></p>
  </td>
  <td width=39 nowrap style='width:29.5pt;border-top:none;border-left:none;
  border-bottom:solid windowtext 1.0pt;border-right:solid windowtext 1.0pt;
  mso-border-top-alt:solid windowtext .5pt;mso-border-left-alt:solid windowtext .5pt;
  mso-border-alt:solid windowtext .5pt;padding:0cm 5.4pt 0cm 5.4pt;height:15.6pt'>
  <p class=MsoNormal align=center style='text-align:center;mso-pagination:widow-orphan'><span
  lang=EN-US style='font-size:6.5pt;font-family:"Times New Roman",serif;
  mso-fareast-font-family:等线;mso-font-kerning:0pt'>2.951<o:p></o:p></span></p>
  </td>
  <td width=39 nowrap style='width:29.55pt;border-top:none;border-left:none;
  border-bottom:solid windowtext 1.0pt;border-right:solid windowtext 1.0pt;
  mso-border-top-alt:solid windowtext .5pt;mso-border-left-alt:solid windowtext .5pt;
  mso-border-alt:solid windowtext .5pt;padding:0cm 5.4pt 0cm 5.4pt;height:15.6pt'>
  <p class=MsoNormal align=center style='text-align:center;mso-pagination:widow-orphan'><span
  lang=EN-US style='font-size:6.5pt;font-family:"Times New Roman",serif;
  mso-fareast-font-family:等线;mso-font-kerning:0pt'>3.523<o:p></o:p></span></p>
  </td>
  <td width=39 nowrap style='width:29.55pt;border-top:none;border-left:none;
  border-bottom:solid windowtext 1.0pt;border-right:solid windowtext 1.0pt;
  mso-border-top-alt:solid windowtext .5pt;mso-border-left-alt:solid windowtext .5pt;
  mso-border-alt:solid windowtext .5pt;padding:0cm 5.4pt 0cm 5.4pt;height:15.6pt'>
  <p class=MsoNormal align=center style='text-align:center;mso-pagination:widow-orphan'><span
  lang=EN-US style='font-size:6.5pt;font-family:"Times New Roman",serif;
  mso-fareast-font-family:等线;mso-font-kerning:0pt'>3.357<o:p></o:p></span></p>
  </td>
  <td width=39 style='width:29.5pt;border-top:none;border-left:none;border-bottom:
  solid windowtext 1.0pt;border-right:solid windowtext 1.0pt;mso-border-top-alt:
  solid windowtext .5pt;mso-border-left-alt:solid windowtext .5pt;mso-border-alt:
  solid windowtext .5pt;padding:0cm 5.4pt 0cm 5.4pt;height:15.6pt'>
  <p class=MsoNormal align=center style='text-align:center;mso-pagination:widow-orphan'><span
  lang=EN-US style='font-size:6.5pt;font-family:"Times New Roman",serif;
  mso-fareast-font-family:等线;mso-font-kerning:0pt'>2.754<o:p></o:p></span></p>
  </td>
  <td width=39 style='width:29.55pt;border-top:none;border-left:none;
  border-bottom:solid windowtext 1.0pt;border-right:solid windowtext 1.0pt;
  mso-border-top-alt:solid windowtext .5pt;mso-border-left-alt:solid windowtext .5pt;
  mso-border-alt:solid windowtext .5pt;padding:0cm 5.4pt 0cm 5.4pt;height:15.6pt'>
  <p class=MsoNormal align=center style='text-align:center;mso-pagination:widow-orphan'><span
  lang=EN-US style='font-size:6.5pt;font-family:"Times New Roman",serif;
  mso-fareast-font-family:等线;mso-font-kerning:0pt'>3.436<o:p></o:p></span></p>
  </td>
  <td width=39 style='width:29.55pt;border-top:none;border-left:none;
  border-bottom:solid windowtext 1.0pt;border-right:solid windowtext 1.0pt;
  mso-border-top-alt:solid windowtext .5pt;mso-border-left-alt:solid windowtext .5pt;
  mso-border-alt:solid windowtext .5pt;padding:0cm 5.4pt 0cm 5.4pt;height:15.6pt'>
  <p class=MsoNormal align=center style='text-align:center;mso-pagination:widow-orphan'><span
  lang=EN-US style='font-size:6.5pt;font-family:"Times New Roman",serif;
  mso-fareast-font-family:等线;mso-font-kerning:0pt'>3.238<o:p></o:p></span></p>
  </td>
 </tr>
 <tr style='mso-yfti-irow:11;height:15.6pt'>
  <td width=75 nowrap style='width:56.45pt;border:solid windowtext 1.0pt;
  border-top:none;mso-border-top-alt:solid windowtext .5pt;mso-border-alt:solid windowtext .5pt;
  padding:0cm 5.4pt 0cm 5.4pt;height:15.6pt'>
  <p class=MsoNormal align=center style='text-align:center;mso-pagination:widow-orphan'><span
  lang=EN-US style='font-size:6.5pt;mso-bidi-font-size:10.0pt;font-family:"Times New Roman",serif;
  mso-fareast-font-family:等线;mso-font-kerning:0pt'>MTFAA-LSA<o:p></o:p></span></p>
  </td>
  <td width=39 nowrap style='width:29.5pt;border-top:none;border-left:none;
  border-bottom:solid windowtext 1.0pt;border-right:solid windowtext 1.0pt;
  mso-border-top-alt:solid windowtext .5pt;mso-border-left-alt:solid windowtext .5pt;
  mso-border-alt:solid windowtext .5pt;padding:0cm 5.4pt 0cm 5.4pt;height:15.6pt'>
  <p class=MsoNormal align=center style='text-align:center;mso-pagination:widow-orphan'><b
  style='mso-bidi-font-weight:normal'><span lang=EN-US style='font-size:6.5pt;
  mso-bidi-font-size:10.0pt;font-family:"Times New Roman",serif;mso-fareast-font-family:
  等线;mso-font-kerning:0pt'>2.084</span></b><b style='mso-bidi-font-weight:normal'><span
  lang=EN-US style='font-size:6.5pt;font-family:"Times New Roman",serif;
  mso-fareast-font-family:等线;mso-font-kerning:0pt'><o:p></o:p></span></b></p>
  </td>
  <td width=39 nowrap style='width:29.55pt;border-top:none;border-left:none;
  border-bottom:solid windowtext 1.0pt;border-right:solid windowtext 1.0pt;
  mso-border-top-alt:solid windowtext .5pt;mso-border-left-alt:solid windowtext .5pt;
  mso-border-alt:solid windowtext .5pt;padding:0cm 5.4pt 0cm 5.4pt;height:15.6pt'>
  <p class=MsoNormal align=center style='text-align:center;mso-pagination:widow-orphan'><b
  style='mso-bidi-font-weight:normal'><span lang=EN-US style='font-size:6.5pt;
  mso-bidi-font-size:10.0pt;font-family:"Times New Roman",serif;mso-fareast-font-family:
  等线;mso-font-kerning:0pt'>2.795</span></b><b style='mso-bidi-font-weight:normal'><span
  lang=EN-US style='font-size:6.5pt;font-family:"Times New Roman",serif;
  mso-fareast-font-family:等线;mso-font-kerning:0pt'><o:p></o:p></span></b></p>
  </td>
  <td width=39 nowrap style='width:29.55pt;border-top:none;border-left:none;
  border-bottom:solid windowtext 1.0pt;border-right:solid windowtext 1.0pt;
  mso-border-top-alt:solid windowtext .5pt;mso-border-left-alt:solid windowtext .5pt;
  mso-border-alt:solid windowtext .5pt;padding:0cm 5.4pt 0cm 5.4pt;height:15.6pt'>
  <p class=MsoNormal align=center style='text-align:center;mso-pagination:widow-orphan'><b
  style='mso-bidi-font-weight:normal'><span lang=EN-US style='font-size:6.5pt;
  mso-bidi-font-size:10.0pt;font-family:"Times New Roman",serif;mso-fareast-font-family:
  等线;mso-font-kerning:0pt'>2.589</span></b><b style='mso-bidi-font-weight:normal'><span
  lang=EN-US style='font-size:6.5pt;font-family:"Times New Roman",serif;
  mso-fareast-font-family:等线;mso-font-kerning:0pt'><o:p></o:p></span></b></p>
  </td>
  <td width=39 nowrap style='width:29.5pt;border-top:none;border-left:none;
  border-bottom:solid windowtext 1.0pt;border-right:solid windowtext 1.0pt;
  mso-border-top-alt:solid windowtext .5pt;mso-border-left-alt:solid windowtext .5pt;
  mso-border-alt:solid windowtext .5pt;padding:0cm 5.4pt 0cm 5.4pt;height:15.6pt'>
  <p class=MsoNormal align=center style='text-align:center;mso-pagination:widow-orphan'><b
  style='mso-bidi-font-weight:normal'><span lang=EN-US style='font-size:6.5pt;
  font-family:"Times New Roman",serif;mso-fareast-font-family:等线;mso-font-kerning:
  0pt'>3.517<o:p></o:p></span></b></p>
  </td>
  <td width=39 nowrap style='width:29.55pt;border-top:none;border-left:none;
  border-bottom:solid windowtext 1.0pt;border-right:solid windowtext 1.0pt;
  mso-border-top-alt:solid windowtext .5pt;mso-border-left-alt:solid windowtext .5pt;
  mso-border-alt:solid windowtext .5pt;padding:0cm 5.4pt 0cm 5.4pt;height:15.6pt'>
  <p class=MsoNormal align=center style='text-align:center;mso-pagination:widow-orphan'><b
  style='mso-bidi-font-weight:normal'><span lang=EN-US style='font-size:6.5pt;
  font-family:"Times New Roman",serif;mso-fareast-font-family:等线;mso-font-kerning:
  0pt'>4.203<o:p></o:p></span></b></p>
  </td>
  <td width=39 nowrap style='width:29.55pt;border-top:none;border-left:none;
  border-bottom:solid windowtext 1.0pt;border-right:solid windowtext 1.0pt;
  mso-border-top-alt:solid windowtext .5pt;mso-border-left-alt:solid windowtext .5pt;
  mso-border-alt:solid windowtext .5pt;padding:0cm 5.4pt 0cm 5.4pt;height:15.6pt'>
  <p class=MsoNormal align=center style='text-align:center;mso-pagination:widow-orphan'><b
  style='mso-bidi-font-weight:normal'><span lang=EN-US style='font-size:6.5pt;
  font-family:"Times New Roman",serif;mso-fareast-font-family:等线;mso-font-kerning:
  0pt'>4.004<o:p></o:p></span></b></p>
  </td>
  <td width=39 nowrap style='width:29.5pt;border-top:none;border-left:none;
  border-bottom:solid windowtext 1.0pt;border-right:solid windowtext 1.0pt;
  mso-border-top-alt:solid windowtext .5pt;mso-border-left-alt:solid windowtext .5pt;
  mso-border-alt:solid windowtext .5pt;padding:0cm 5.4pt 0cm 5.4pt;height:15.6pt'>
  <p class=MsoNormal align=center style='text-align:center;mso-pagination:widow-orphan'><span
  lang=EN-US style='font-size:6.5pt;font-family:"Times New Roman",serif;
  mso-fareast-font-family:等线;mso-font-kerning:0pt'>3.006<o:p></o:p></span></p>
  </td>
  <td width=39 nowrap style='width:29.55pt;border-top:none;border-left:none;
  border-bottom:solid windowtext 1.0pt;border-right:solid windowtext 1.0pt;
  mso-border-top-alt:solid windowtext .5pt;mso-border-left-alt:solid windowtext .5pt;
  mso-border-alt:solid windowtext .5pt;padding:0cm 5.4pt 0cm 5.4pt;height:15.6pt'>
  <p class=MsoNormal align=center style='text-align:center;mso-pagination:widow-orphan'><b
  style='mso-bidi-font-weight:normal'><span lang=EN-US style='font-size:6.5pt;
  font-family:"Times New Roman",serif;mso-fareast-font-family:等线;mso-font-kerning:
  0pt'>3.593<o:p></o:p></span></b></p>
  </td>
  <td width=39 nowrap style='width:29.55pt;border-top:none;border-left:none;
  border-bottom:solid windowtext 1.0pt;border-right:solid windowtext 1.0pt;
  mso-border-top-alt:solid windowtext .5pt;mso-border-left-alt:solid windowtext .5pt;
  mso-border-alt:solid windowtext .5pt;padding:0cm 5.4pt 0cm 5.4pt;height:15.6pt'>
  <p class=MsoNormal align=center style='text-align:center;mso-pagination:widow-orphan'><b
  style='mso-bidi-font-weight:normal'><span lang=EN-US style='font-size:6.5pt;
  font-family:"Times New Roman",serif;mso-fareast-font-family:等线;mso-font-kerning:
  0pt'>3.423<o:p></o:p></span></b></p>
  </td>
  <td width=39 style='width:29.5pt;border-top:none;border-left:none;border-bottom:
  solid windowtext 1.0pt;border-right:solid windowtext 1.0pt;mso-border-top-alt:
  solid windowtext .5pt;mso-border-left-alt:solid windowtext .5pt;mso-border-alt:
  solid windowtext .5pt;padding:0cm 5.4pt 0cm 5.4pt;height:15.6pt'>
  <p class=MsoNormal align=center style='text-align:center;mso-pagination:widow-orphan'><b
  style='mso-bidi-font-weight:normal'><span lang=EN-US style='font-size:6.5pt;
  font-family:"Times New Roman",serif;mso-fareast-font-family:等线;mso-font-kerning:
  0pt'>2.829<o:p></o:p></span></b></p>
  </td>
  <td width=39 style='width:29.55pt;border-top:none;border-left:none;
  border-bottom:solid windowtext 1.0pt;border-right:solid windowtext 1.0pt;
  mso-border-top-alt:solid windowtext .5pt;mso-border-left-alt:solid windowtext .5pt;
  mso-border-alt:solid windowtext .5pt;padding:0cm 5.4pt 0cm 5.4pt;height:15.6pt'>
  <p class=MsoNormal align=center style='text-align:center;mso-pagination:widow-orphan'><b
  style='mso-bidi-font-weight:normal'><span lang=EN-US style='font-size:6.5pt;
  font-family:"Times New Roman",serif;mso-fareast-font-family:等线;mso-font-kerning:
  0pt'>3.547<o:p></o:p></span></b></p>
  </td>
  <td width=39 style='width:29.55pt;border-top:none;border-left:none;
  border-bottom:solid windowtext 1.0pt;border-right:solid windowtext 1.0pt;
  mso-border-top-alt:solid windowtext .5pt;mso-border-left-alt:solid windowtext .5pt;
  mso-border-alt:solid windowtext .5pt;padding:0cm 5.4pt 0cm 5.4pt;height:15.6pt'>
  <p class=MsoNormal align=center style='text-align:center;mso-pagination:widow-orphan'><b
  style='mso-bidi-font-weight:normal'><span lang=EN-US style='font-size:6.5pt;
  font-family:"Times New Roman",serif;mso-fareast-font-family:等线;mso-font-kerning:
  0pt'>3.339<o:p></o:p></span></b></p>
  </td>
 </tr>
 <tr style='mso-yfti-irow:12;height:15.6pt'>
  <td width=75 nowrap style='width:56.45pt;border:solid windowtext 1.0pt;
  border-top:none;mso-border-top-alt:solid windowtext .5pt;mso-border-alt:solid windowtext .5pt;
  padding:0cm 5.4pt 0cm 5.4pt;height:15.6pt'>
  <p class=MsoNormal align=center style='text-align:center;mso-pagination:widow-orphan'><span
  lang=EN-US style='font-size:5.5pt;mso-bidi-font-size:10.0pt;font-family:"Times New Roman",serif;
  mso-fareast-font-family:等线;mso-font-kerning:0pt'>MTFAA-MSLSA</span><span
  lang=EN-US style='font-size:6.5pt;mso-bidi-font-size:10.0pt;font-family:"Times New Roman",serif;
  mso-fareast-font-family:等线;mso-font-kerning:0pt'><o:p></o:p></span></p>
  </td>
  <td width=39 nowrap style='width:29.5pt;border-top:none;border-left:none;
  border-bottom:solid windowtext 1.0pt;border-right:solid windowtext 1.0pt;
  mso-border-top-alt:solid windowtext .5pt;mso-border-left-alt:solid windowtext .5pt;
  mso-border-alt:solid windowtext .5pt;padding:0cm 5.4pt 0cm 5.4pt;height:15.6pt'>
  <p class=MsoNormal align=center style='text-align:center;mso-pagination:widow-orphan'><span
  lang=EN-US style='font-size:6.5pt;mso-bidi-font-size:10.0pt;font-family:"Times New Roman",serif;
  mso-fareast-font-family:等线;mso-font-kerning:0pt'>2.077<o:p></o:p></span></p>
  </td>
  <td width=39 nowrap style='width:29.55pt;border-top:none;border-left:none;
  border-bottom:solid windowtext 1.0pt;border-right:solid windowtext 1.0pt;
  mso-border-top-alt:solid windowtext .5pt;mso-border-left-alt:solid windowtext .5pt;
  mso-border-alt:solid windowtext .5pt;padding:0cm 5.4pt 0cm 5.4pt;height:15.6pt'>
  <p class=MsoNormal align=center style='text-align:center;mso-pagination:widow-orphan'><span
  lang=EN-US style='font-size:6.5pt;mso-bidi-font-size:10.0pt;font-family:"Times New Roman",serif;
  mso-fareast-font-family:等线;mso-font-kerning:0pt'>2.772<o:p></o:p></span></p>
  </td>
  <td width=39 nowrap style='width:29.55pt;border-top:none;border-left:none;
  border-bottom:solid windowtext 1.0pt;border-right:solid windowtext 1.0pt;
  mso-border-top-alt:solid windowtext .5pt;mso-border-left-alt:solid windowtext .5pt;
  mso-border-alt:solid windowtext .5pt;padding:0cm 5.4pt 0cm 5.4pt;height:15.6pt'>
  <p class=MsoNormal align=center style='text-align:center;mso-pagination:widow-orphan'><span
  lang=EN-US style='font-size:6.5pt;mso-bidi-font-size:10.0pt;font-family:"Times New Roman",serif;
  mso-fareast-font-family:等线;mso-font-kerning:0pt'>2.571<o:p></o:p></span></p>
  </td>
  <td width=39 nowrap style='width:29.5pt;border-top:none;border-left:none;
  border-bottom:solid windowtext 1.0pt;border-right:solid windowtext 1.0pt;
  mso-border-top-alt:solid windowtext .5pt;mso-border-left-alt:solid windowtext .5pt;
  mso-border-alt:solid windowtext .5pt;padding:0cm 5.4pt 0cm 5.4pt;height:15.6pt'>
  <p class=MsoNormal align=center style='text-align:center;mso-pagination:widow-orphan'><span
  lang=EN-US style='font-size:6.5pt;font-family:"Times New Roman",serif;
  mso-fareast-font-family:等线;mso-font-kerning:0pt'>3.500<o:p></o:p></span></p>
  </td>
  <td width=39 nowrap style='width:29.55pt;border-top:none;border-left:none;
  border-bottom:solid windowtext 1.0pt;border-right:solid windowtext 1.0pt;
  mso-border-top-alt:solid windowtext .5pt;mso-border-left-alt:solid windowtext .5pt;
  mso-border-alt:solid windowtext .5pt;padding:0cm 5.4pt 0cm 5.4pt;height:15.6pt'>
  <p class=MsoNormal align=center style='text-align:center;mso-pagination:widow-orphan'><span
  lang=EN-US style='font-size:6.5pt;font-family:"Times New Roman",serif;
  mso-fareast-font-family:等线;mso-font-kerning:0pt'>4.167<o:p></o:p></span></p>
  </td>
  <td width=39 nowrap style='width:29.55pt;border-top:none;border-left:none;
  border-bottom:solid windowtext 1.0pt;border-right:solid windowtext 1.0pt;
  mso-border-top-alt:solid windowtext .5pt;mso-border-left-alt:solid windowtext .5pt;
  mso-border-alt:solid windowtext .5pt;padding:0cm 5.4pt 0cm 5.4pt;height:15.6pt'>
  <p class=MsoNormal align=center style='text-align:center;mso-pagination:widow-orphan'><span
  lang=EN-US style='font-size:6.5pt;font-family:"Times New Roman",serif;
  mso-fareast-font-family:等线;mso-font-kerning:0pt'>3.974<o:p></o:p></span></p>
  </td>
  <td width=39 nowrap style='width:29.5pt;border-top:none;border-left:none;
  border-bottom:solid windowtext 1.0pt;border-right:solid windowtext 1.0pt;
  mso-border-top-alt:solid windowtext .5pt;mso-border-left-alt:solid windowtext .5pt;
  mso-border-alt:solid windowtext .5pt;padding:0cm 5.4pt 0cm 5.4pt;height:15.6pt'>
  <p class=MsoNormal align=center style='text-align:center;mso-pagination:widow-orphan'><b
  style='mso-bidi-font-weight:normal'><span lang=EN-US style='font-size:6.5pt;
  font-family:"Times New Roman",serif;mso-fareast-font-family:等线;mso-font-kerning:
  0pt'>3.013<o:p></o:p></span></b></p>
  </td>
  <td width=39 nowrap style='width:29.55pt;border-top:none;border-left:none;
  border-bottom:solid windowtext 1.0pt;border-right:solid windowtext 1.0pt;
  mso-border-top-alt:solid windowtext .5pt;mso-border-left-alt:solid windowtext .5pt;
  mso-border-alt:solid windowtext .5pt;padding:0cm 5.4pt 0cm 5.4pt;height:15.6pt'>
  <p class=MsoNormal align=center style='text-align:center;mso-pagination:widow-orphan'><span
  lang=EN-US style='font-size:6.5pt;font-family:"Times New Roman",serif;
  mso-fareast-font-family:等线;mso-font-kerning:0pt'>3.589<o:p></o:p></span></p>
  </td>
  <td width=39 nowrap style='width:29.55pt;border-top:none;border-left:none;
  border-bottom:solid windowtext 1.0pt;border-right:solid windowtext 1.0pt;
  mso-border-top-alt:solid windowtext .5pt;mso-border-left-alt:solid windowtext .5pt;
  mso-border-alt:solid windowtext .5pt;padding:0cm 5.4pt 0cm 5.4pt;height:15.6pt'>
  <p class=MsoNormal align=center style='text-align:center;mso-pagination:widow-orphan'><span
  lang=EN-US style='font-size:6.5pt;font-family:"Times New Roman",serif;
  mso-fareast-font-family:等线;mso-font-kerning:0pt'>3.422<o:p></o:p></span></p>
  </td>
  <td width=39 style='width:29.5pt;border-top:none;border-left:none;border-bottom:
  solid windowtext 1.0pt;border-right:solid windowtext 1.0pt;mso-border-top-alt:
  solid windowtext .5pt;mso-border-left-alt:solid windowtext .5pt;mso-border-alt:
  solid windowtext .5pt;padding:0cm 5.4pt 0cm 5.4pt;height:15.6pt'>
  <p class=MsoNormal align=center style='text-align:center;mso-pagination:widow-orphan'><span
  lang=EN-US style='font-size:6.5pt;font-family:"Times New Roman",serif;
  mso-fareast-font-family:等线;mso-font-kerning:0pt'>2.820<o:p></o:p></span></p>
  </td>
  <td width=39 style='width:29.55pt;border-top:none;border-left:none;
  border-bottom:solid windowtext 1.0pt;border-right:solid windowtext 1.0pt;
  mso-border-top-alt:solid windowtext .5pt;mso-border-left-alt:solid windowtext .5pt;
  mso-border-alt:solid windowtext .5pt;padding:0cm 5.4pt 0cm 5.4pt;height:15.6pt'>
  <p class=MsoNormal align=center style='text-align:center;mso-pagination:widow-orphan'><span
  lang=EN-US style='font-size:6.5pt;font-family:"Times New Roman",serif;
  mso-fareast-font-family:等线;mso-font-kerning:0pt'>3.517<o:p></o:p></span></p>
  </td>
  <td width=39 style='width:29.55pt;border-top:none;border-left:none;
  border-bottom:solid windowtext 1.0pt;border-right:solid windowtext 1.0pt;
  mso-border-top-alt:solid windowtext .5pt;mso-border-left-alt:solid windowtext .5pt;
  mso-border-alt:solid windowtext .5pt;padding:0cm 5.4pt 0cm 5.4pt;height:15.6pt'>
  <p class=MsoNormal align=center style='text-align:center;mso-pagination:widow-orphan'><span
  lang=EN-US style='font-size:6.5pt;font-family:"Times New Roman",serif;
  mso-fareast-font-family:等线;mso-font-kerning:0pt'>3.314<o:p></o:p></span></p>
  </td>
 </tr>
 <tr style='mso-yfti-irow:13;height:15.0pt;mso-row-margin-right:265.8pt'>
  <td width=75 nowrap style='width:56.45pt;border:solid windowtext 1.0pt;
  border-top:none;mso-border-top-alt:solid windowtext .5pt;mso-border-alt:solid windowtext .5pt;
  padding:0cm 5.4pt 0cm 5.4pt;height:15.0pt'>
  <p class=MsoNormal align=center style='text-align:center;mso-pagination:widow-orphan'><span
  lang=EN-US style='font-size:5.5pt;mso-bidi-font-size:10.0pt;font-family:"Times New Roman",serif;
  mso-fareast-font-family:等线;mso-font-kerning:0pt'>Wideband Metrics</span><span
  lang=EN-US style='font-size:7.5pt;mso-bidi-font-size:10.0pt;font-family:"Times New Roman",serif;
  mso-fareast-font-family:等线;mso-font-kerning:0pt'><o:p></o:p></span></p>
  </td>
  <td width=118 nowrap colspan=3 style='width:88.6pt;border-top:none;
  border-left:none;border-bottom:solid windowtext 1.0pt;border-right:solid windowtext 1.0pt;
  mso-border-top-alt:solid windowtext .5pt;mso-border-left-alt:solid windowtext .5pt;
  mso-border-alt:solid windowtext .5pt;padding:0cm 5.4pt 0cm 5.4pt;height:15.0pt'>
  <p class=MsoNormal align=center style='text-align:center;mso-pagination:widow-orphan'><span
  lang=EN-US style='font-size:7.5pt;mso-bidi-font-size:10.0pt;font-family:"Times New Roman",serif;
  mso-fareast-font-family:等线;mso-font-kerning:0pt'>SSNR (dB)<o:p></o:p></span></p>
  </td>
  <td style='mso-cell-special:placeholder;border:none;padding:0cm 0cm 0cm 0cm'
  width=354 colspan=9><p class='MsoNormal'>&nbsp;</td>
 </tr>
 <tr style='mso-yfti-irow:14;height:15.0pt;mso-row-margin-right:265.8pt'>
  <td width=75 nowrap style='width:56.45pt;border:solid windowtext 1.0pt;
  border-top:none;mso-border-top-alt:solid windowtext .5pt;mso-border-alt:solid windowtext .5pt;
  padding:0cm 5.4pt 0cm 5.4pt;height:15.0pt'>
  <p class=MsoNormal align=center style='text-align:center;mso-pagination:widow-orphan'><span
  class=GramE><span lang=EN-US style='font-size:6.5pt;mso-bidi-font-size:10.0pt;
  font-family:"Times New Roman",serif;mso-fareast-font-family:等线;mso-font-kerning:
  0pt'>SNR(</span></span><span lang=EN-US style='font-size:6.5pt;mso-bidi-font-size:
  10.0pt;font-family:"Times New Roman",serif;mso-fareast-font-family:等线;
  mso-font-kerning:0pt'>dB)<o:p></o:p></span></p>
  </td>
  <td width=39 nowrap style='width:29.5pt;border-top:none;border-left:none;
  border-bottom:solid windowtext 1.0pt;border-right:solid windowtext 1.0pt;
  mso-border-top-alt:solid windowtext .5pt;mso-border-left-alt:solid windowtext .5pt;
  mso-border-alt:solid windowtext .5pt;padding:0cm 5.4pt 0cm 5.4pt;height:15.0pt'>
  <p class=MsoNormal align=center style='text-align:center;mso-pagination:widow-orphan'><span
  lang=EN-US style='font-size:7.5pt;font-family:"Times New Roman",serif;
  mso-fareast-font-family:等线;mso-font-kerning:0pt'>-5~0<o:p></o:p></span></p>
  </td>
  <td width=39 nowrap style='width:29.55pt;border-top:none;border-left:none;
  border-bottom:solid windowtext 1.0pt;border-right:solid windowtext 1.0pt;
  mso-border-top-alt:solid windowtext .5pt;mso-border-left-alt:solid windowtext .5pt;
  mso-border-alt:solid windowtext .5pt;padding:0cm 5.4pt 0cm 5.4pt;height:15.0pt'>
  <p class=MsoNormal align=center style='text-align:center;mso-pagination:widow-orphan'><span
  lang=EN-US style='font-size:7.5pt;font-family:"Times New Roman",serif;
  mso-fareast-font-family:等线;mso-font-kerning:0pt'>0~15<o:p></o:p></span></p>
  </td>
  <td width=39 nowrap style='width:29.55pt;border-top:none;border-left:none;
  border-bottom:solid windowtext 1.0pt;border-right:solid windowtext 1.0pt;
  mso-border-top-alt:solid windowtext .5pt;mso-border-left-alt:solid windowtext .5pt;
  mso-border-alt:solid windowtext .5pt;padding:0cm 5.4pt 0cm 5.4pt;height:15.0pt'>
  <p class=MsoNormal align=center style='text-align:center;mso-pagination:widow-orphan'><span
  class=SpellE><span lang=EN-US style='font-size:7.5pt;font-family:"Times New Roman",serif;
  mso-fareast-font-family:等线;mso-font-kerning:0pt'>Ovrl</span></span><span
  lang=EN-US style='font-size:7.5pt;font-family:"Times New Roman",serif;
  mso-fareast-font-family:等线;mso-font-kerning:0pt'>.<o:p></o:p></span></p>
  </td>
  <td style='mso-cell-special:placeholder;border:none;padding:0cm 0cm 0cm 0cm'
  width=354 colspan=9><p class='MsoNormal'>&nbsp;</td>
 </tr>
 <tr style='mso-yfti-irow:15;height:15.6pt;mso-row-margin-right:265.8pt'>
  <td width=75 nowrap style='width:56.45pt;border:solid windowtext 1.0pt;
  border-top:none;mso-border-top-alt:solid windowtext .5pt;mso-border-alt:solid windowtext .5pt;
  padding:0cm 5.4pt 0cm 5.4pt;height:15.6pt'>
  <p class=MsoNormal align=center style='text-align:center;mso-pagination:widow-orphan'><span
  lang=EN-US style='font-size:6.5pt;mso-bidi-font-size:10.0pt;font-family:"Times New Roman",serif;
  mso-fareast-font-family:等线;mso-font-kerning:0pt'>Noisy<o:p></o:p></span></p>
  </td>
  <td width=39 nowrap style='width:29.5pt;border-top:none;border-left:none;
  border-bottom:solid windowtext 1.0pt;border-right:solid windowtext 1.0pt;
  mso-border-top-alt:solid windowtext .5pt;mso-border-left-alt:solid windowtext .5pt;
  mso-border-alt:solid windowtext .5pt;padding:0cm 5.4pt 0cm 5.4pt;height:15.6pt'>
  <p class=MsoNormal align=center style='text-align:center;mso-pagination:widow-orphan'><span
  lang=EN-US style='font-size:6.5pt;mso-bidi-font-size:10.0pt;font-family:"Times New Roman",serif;
  mso-fareast-font-family:等线;mso-font-kerning:0pt'>-2.291</span><span
  lang=EN-US style='font-size:6.5pt;font-family:"Times New Roman",serif;
  mso-fareast-font-family:等线;mso-font-kerning:0pt'><o:p></o:p></span></p>
  </td>
  <td width=39 nowrap style='width:29.55pt;border-top:none;border-left:none;
  border-bottom:solid windowtext 1.0pt;border-right:solid windowtext 1.0pt;
  mso-border-top-alt:solid windowtext .5pt;mso-border-left-alt:solid windowtext .5pt;
  mso-border-alt:solid windowtext .5pt;padding:0cm 5.4pt 0cm 5.4pt;height:15.6pt'>
  <p class=MsoNormal align=center style='text-align:center;mso-pagination:widow-orphan'><span
  lang=EN-US style='font-size:6.5pt;mso-bidi-font-size:10.0pt;font-family:"Times New Roman",serif;
  mso-fareast-font-family:等线;mso-font-kerning:0pt'>4.19</span><span lang=EN-US
  style='font-size:6.5pt;font-family:"Times New Roman",serif;mso-fareast-font-family:
  等线;mso-font-kerning:0pt'><o:p></o:p></span></p>
  </td>
  <td width=39 nowrap style='width:29.55pt;border-top:none;border-left:none;
  border-bottom:solid windowtext 1.0pt;border-right:solid windowtext 1.0pt;
  mso-border-top-alt:solid windowtext .5pt;mso-border-left-alt:solid windowtext .5pt;
  mso-border-alt:solid windowtext .5pt;padding:0cm 5.4pt 0cm 5.4pt;height:15.6pt'>
  <p class=MsoNormal align=center style='text-align:center;mso-pagination:widow-orphan'><span
  lang=EN-US style='font-size:6.5pt;mso-bidi-font-size:10.0pt;font-family:"Times New Roman",serif;
  mso-fareast-font-family:等线;mso-font-kerning:0pt'>2.307</span><span
  lang=EN-US style='font-size:6.5pt;font-family:"Times New Roman",serif;
  mso-fareast-font-family:等线;mso-font-kerning:0pt'><o:p></o:p></span></p>
  </td>
  <td style='mso-cell-special:placeholder;border:none;padding:0cm 0cm 0cm 0cm'
  width=354 colspan=9><p class='MsoNormal'>&nbsp;</td>
 </tr>
 <tr style='mso-yfti-irow:16;height:15.6pt;mso-row-margin-right:265.8pt'>
  <td width=75 nowrap style='width:56.45pt;border:solid windowtext 1.0pt;
  border-top:none;mso-border-top-alt:solid windowtext .5pt;mso-border-alt:solid windowtext .5pt;
  padding:0cm 5.4pt 0cm 5.4pt;height:15.6pt'>
  <p class=MsoNormal align=center style='text-align:center;mso-pagination:widow-orphan'><span
  lang=EN-US style='font-size:6.5pt;mso-bidi-font-size:10.0pt;font-family:"Times New Roman",serif;
  mso-fareast-font-family:等线;mso-font-kerning:0pt'>MTFAA<o:p></o:p></span></p>
  </td>
  <td width=39 nowrap style='width:29.5pt;border-top:none;border-left:none;
  border-bottom:solid windowtext 1.0pt;border-right:solid windowtext 1.0pt;
  mso-border-top-alt:solid windowtext .5pt;mso-border-left-alt:solid windowtext .5pt;
  mso-border-alt:solid windowtext .5pt;padding:0cm 5.4pt 0cm 5.4pt;height:15.6pt'>
  <p class=MsoNormal align=center style='text-align:center;mso-pagination:widow-orphan'><span
  lang=EN-US style='font-size:6.5pt;mso-bidi-font-size:10.0pt;font-family:"Times New Roman",serif;
  mso-fareast-font-family:等线;mso-font-kerning:0pt'>6.550</span><span
  lang=EN-US style='font-size:6.5pt;font-family:"Times New Roman",serif;
  mso-fareast-font-family:等线;mso-font-kerning:0pt'><o:p></o:p></span></p>
  </td>
  <td width=39 nowrap style='width:29.55pt;border-top:none;border-left:none;
  border-bottom:solid windowtext 1.0pt;border-right:solid windowtext 1.0pt;
  mso-border-top-alt:solid windowtext .5pt;mso-border-left-alt:solid windowtext .5pt;
  mso-border-alt:solid windowtext .5pt;padding:0cm 5.4pt 0cm 5.4pt;height:15.6pt'>
  <p class=MsoNormal align=center style='text-align:center;mso-pagination:widow-orphan'><span
  lang=EN-US style='font-size:6.5pt;mso-bidi-font-size:10.0pt;font-family:"Times New Roman",serif;
  mso-fareast-font-family:等线;mso-font-kerning:0pt'>10.13</span><span
  lang=EN-US style='font-size:6.5pt;font-family:"Times New Roman",serif;
  mso-fareast-font-family:等线;mso-font-kerning:0pt'><o:p></o:p></span></p>
  </td>
  <td width=39 nowrap style='width:29.55pt;border-top:none;border-left:none;
  border-bottom:solid windowtext 1.0pt;border-right:solid windowtext 1.0pt;
  mso-border-top-alt:solid windowtext .5pt;mso-border-left-alt:solid windowtext .5pt;
  mso-border-alt:solid windowtext .5pt;padding:0cm 5.4pt 0cm 5.4pt;height:15.6pt'>
  <p class=MsoNormal align=center style='text-align:center;mso-pagination:widow-orphan'><span
  lang=EN-US style='font-size:6.5pt;mso-bidi-font-size:10.0pt;font-family:"Times New Roman",serif;
  mso-fareast-font-family:等线;mso-font-kerning:0pt'>9.094</span><span
  lang=EN-US style='font-size:6.5pt;font-family:"Times New Roman",serif;
  mso-fareast-font-family:等线;mso-font-kerning:0pt'><o:p></o:p></span></p>
  </td>
  <td style='mso-cell-special:placeholder;border:none;padding:0cm 0cm 0cm 0cm'
  width=354 colspan=9><p class='MsoNormal'>&nbsp;</td>
 </tr>
 <tr style='mso-yfti-irow:17;height:15.6pt;mso-row-margin-right:265.8pt'>
  <td width=75 nowrap style='width:56.45pt;border:solid windowtext 1.0pt;
  border-top:none;mso-border-top-alt:solid windowtext .5pt;mso-border-alt:solid windowtext .5pt;
  padding:0cm 5.4pt 0cm 5.4pt;height:15.6pt'>
  <p class=MsoNormal align=center style='text-align:center;mso-pagination:widow-orphan'><span
  lang=EN-US style='font-size:6.5pt;mso-bidi-font-size:10.0pt;font-family:"Times New Roman",serif;
  mso-fareast-font-family:等线;mso-font-kerning:0pt'>MTFAA-LSA<o:p></o:p></span></p>
  </td>
  <td width=39 nowrap style='width:29.5pt;border-top:none;border-left:none;
  border-bottom:solid windowtext 1.0pt;border-right:solid windowtext 1.0pt;
  mso-border-top-alt:solid windowtext .5pt;mso-border-left-alt:solid windowtext .5pt;
  mso-border-alt:solid windowtext .5pt;padding:0cm 5.4pt 0cm 5.4pt;height:15.6pt'>
  <p class=MsoNormal align=center style='text-align:center;mso-pagination:widow-orphan'><span
  lang=EN-US style='font-size:6.5pt;mso-bidi-font-size:10.0pt;font-family:"Times New Roman",serif;
  mso-fareast-font-family:等线;mso-font-kerning:0pt'>6.609</span><span
  lang=EN-US style='font-size:6.5pt;font-family:"Times New Roman",serif;
  mso-fareast-font-family:等线;mso-font-kerning:0pt'><o:p></o:p></span></p>
  </td>
  <td width=39 nowrap style='width:29.55pt;border-top:none;border-left:none;
  border-bottom:solid windowtext 1.0pt;border-right:solid windowtext 1.0pt;
  mso-border-top-alt:solid windowtext .5pt;mso-border-left-alt:solid windowtext .5pt;
  mso-border-alt:solid windowtext .5pt;padding:0cm 5.4pt 0cm 5.4pt;height:15.6pt'>
  <p class=MsoNormal align=center style='text-align:center;mso-pagination:widow-orphan'><span
  lang=EN-US style='font-size:6.5pt;mso-bidi-font-size:10.0pt;font-family:"Times New Roman",serif;
  mso-fareast-font-family:等线;mso-font-kerning:0pt'>10.26</span><span
  lang=EN-US style='font-size:6.5pt;font-family:"Times New Roman",serif;
  mso-fareast-font-family:等线;mso-font-kerning:0pt'><o:p></o:p></span></p>
  </td>
  <td width=39 nowrap style='width:29.55pt;border-top:none;border-left:none;
  border-bottom:solid windowtext 1.0pt;border-right:solid windowtext 1.0pt;
  mso-border-top-alt:solid windowtext .5pt;mso-border-left-alt:solid windowtext .5pt;
  mso-border-alt:solid windowtext .5pt;padding:0cm 5.4pt 0cm 5.4pt;height:15.6pt'>
  <p class=MsoNormal align=center style='text-align:center;mso-pagination:widow-orphan'><span
  lang=EN-US style='font-size:6.5pt;mso-bidi-font-size:10.0pt;font-family:"Times New Roman",serif;
  mso-fareast-font-family:等线;mso-font-kerning:0pt'>9.200</span><span
  lang=EN-US style='font-size:6.5pt;font-family:"Times New Roman",serif;
  mso-fareast-font-family:等线;mso-font-kerning:0pt'><o:p></o:p></span></p>
  </td>
  <td style='mso-cell-special:placeholder;border:none;padding:0cm 0cm 0cm 0cm'
  width=354 colspan=9><p class='MsoNormal'>&nbsp;</td>
 </tr>
 <tr style='mso-yfti-irow:18;mso-yfti-lastrow:yes;height:15.6pt;mso-row-margin-right:
  265.8pt'>
  <td width=75 nowrap style='width:56.45pt;border:solid windowtext 1.0pt;
  border-top:none;mso-border-top-alt:solid windowtext .5pt;mso-border-alt:solid windowtext .5pt;
  padding:0cm 5.4pt 0cm 5.4pt;height:15.6pt'>
  <p class=MsoNormal align=center style='text-align:center;mso-pagination:widow-orphan'><span
  lang=EN-US style='font-size:5.5pt;mso-bidi-font-size:10.0pt;font-family:"Times New Roman",serif;
  mso-fareast-font-family:等线;mso-font-kerning:0pt'>MTFAA-MSLSA</span><span
  lang=EN-US style='font-size:6.5pt;mso-bidi-font-size:10.0pt;font-family:"Times New Roman",serif;
  mso-fareast-font-family:等线;mso-font-kerning:0pt'><o:p></o:p></span></p>
  </td>
  <td width=39 nowrap style='width:29.5pt;border-top:none;border-left:none;
  border-bottom:solid windowtext 1.0pt;border-right:solid windowtext 1.0pt;
  mso-border-top-alt:solid windowtext .5pt;mso-border-left-alt:solid windowtext .5pt;
  mso-border-alt:solid windowtext .5pt;padding:0cm 5.4pt 0cm 5.4pt;height:15.6pt'>
  <p class=MsoNormal align=center style='text-align:center;mso-pagination:widow-orphan'><b
  style='mso-bidi-font-weight:normal'><span lang=EN-US style='font-size:6.5pt;
  mso-bidi-font-size:10.0pt;font-family:"Times New Roman",serif;mso-fareast-font-family:
  等线;mso-font-kerning:0pt'>6.779<o:p></o:p></span></b></p>
  </td>
  <td width=39 nowrap style='width:29.55pt;border-top:none;border-left:none;
  border-bottom:solid windowtext 1.0pt;border-right:solid windowtext 1.0pt;
  mso-border-top-alt:solid windowtext .5pt;mso-border-left-alt:solid windowtext .5pt;
  mso-border-alt:solid windowtext .5pt;padding:0cm 5.4pt 0cm 5.4pt;height:15.6pt'>
  <p class=MsoNormal align=center style='text-align:center;mso-pagination:widow-orphan'><b
  style='mso-bidi-font-weight:normal'><span lang=EN-US style='font-size:6.5pt;
  mso-bidi-font-size:10.0pt;font-family:"Times New Roman",serif;mso-fareast-font-family:
  等线;mso-font-kerning:0pt'>10.38</span></b><span lang=EN-US style='font-size:
  6.5pt;mso-bidi-font-size:10.0pt;font-family:"Times New Roman",serif;
  mso-fareast-font-family:等线;mso-font-kerning:0pt'><o:p></o:p></span></p>
  </td>
  <td width=39 nowrap style='width:29.55pt;border-top:none;border-left:none;
  border-bottom:solid windowtext 1.0pt;border-right:solid windowtext 1.0pt;
  mso-border-top-alt:solid windowtext .5pt;mso-border-left-alt:solid windowtext .5pt;
  mso-border-alt:solid windowtext .5pt;padding:0cm 5.4pt 0cm 5.4pt;height:15.6pt'>
  <p class=MsoNormal align=center style='text-align:center;mso-pagination:widow-orphan'><b
  style='mso-bidi-font-weight:normal'><span lang=EN-US style='font-size:6.5pt;
  mso-bidi-font-size:10.0pt;font-family:"Times New Roman",serif;mso-fareast-font-family:
  等线;mso-font-kerning:0pt'>9.338<o:p></o:p></span></b></p>
  </td>
  <td style='mso-cell-special:placeholder;border:none;padding:0cm 0cm 0cm 0cm'
  width=354 colspan=9><p class='MsoNormal'>&nbsp;</td>
 </tr>
</table>

</div>

<p class=MsoNormal><span lang=EN-US><o:p>&nbsp;</o:p></span></p>

</div>

</body>

<!-- ![trainloss_mtfaas](https://user-images.githubusercontent.com/103247057/223667340-de14f5f1-5d91-4873-8f49-ea62014a7905.jpeg)
![Validation_mtfaas](https://user-images.githubusercontent.com/103247057/223667446-9e942476-dc0b-4456-b836-58e6eb435006.jpeg) -->

<div align=center><img src="https://user-images.githubusercontent.com/103247057/223667340-de14f5f1-5d91-4873-8f49-ea62014a7905.jpeg" width="320" height="240" /><img src="https://user-images.githubusercontent.com/103247057/223667446-9e942476-dc0b-4456-b836-58e6eb435006.jpeg" width="320" height="240" /></div>

The best performance of MSLSA is slightly worse than that of conventional LSA, while it can be seen from the training process that MSLSA may achieve a more stable result with a higher mean score and lower variance of PESQ in the validation set, which can be seen in the figure below (statistics based on the last 100 epochs)

<!-- ![Validation_mtfaas_box](https://user-images.githubusercontent.com/103247057/223670308-36957894-07d0-4b34-88c5-14b6011c2e7e.jpeg) -->
<div align=center><img src="https://user-images.githubusercontent.com/103247057/223670308-36957894-07d0-4b34-88c5-14b6011c2e7e.jpeg" width="320" height="240" /></div>
