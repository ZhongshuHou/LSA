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

<table class=MsoTableGrid border=1 cellspacing=0 cellpadding=0 width=552
 style='border-collapse:collapse;mso-table-layout-alt:fixed;border:none;
 mso-border-alt:solid windowtext .5pt;mso-yfti-tbllook:1184;mso-padding-alt:
 0cm 5.4pt 0cm 5.4pt'>
 <tr style='mso-yfti-irow:0;mso-yfti-firstrow:yes;height:1.0pt;mso-row-margin-right:
  3.35pt'>
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
  <td width=39 nowrap rowspan=7 style='width:29.5pt;border:none;border-right:
  solid windowtext 1.0pt;mso-border-left-alt:solid windowtext .5pt;mso-border-left-alt:
  solid windowtext .5pt;mso-border-right-alt:solid windowtext .5pt;padding:
  0cm 5.4pt 0cm 5.4pt;height:1.0pt'>
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
  <td style='mso-cell-special:placeholder;border:none;padding:0cm 0cm 0cm 0cm'
  width=4><p class='MsoNormal'>&nbsp;</td>
 </tr>
 <tr style='mso-yfti-irow:1;height:1.0pt;mso-row-margin-right:3.35pt'>
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
  <td width=39 nowrap style='width:29.5pt;border:none;border-bottom:solid windowtext 1.0pt;
  mso-border-top-alt:solid windowtext .5pt;mso-border-left-alt:solid windowtext .5pt;
  mso-border-top-alt:solid windowtext .5pt;mso-border-left-alt:solid windowtext .5pt;
  mso-border-bottom-alt:solid windowtext .5pt;padding:0cm 5.4pt 0cm 5.4pt;
  height:1.0pt'>
  <p class=MsoNormal align=center style='text-align:center;mso-pagination:widow-orphan'><span
  lang=EN-US style='font-size:7.5pt;mso-bidi-font-size:10.0pt;font-family:"Times New Roman",serif;
  mso-fareast-font-family:等线;mso-font-kerning:0pt'>-5~0<o:p></o:p></span></p>
  </td>
  <td width=39 nowrap style='width:29.55pt;border:none;border-bottom:solid windowtext 1.0pt;
  mso-border-top-alt:solid windowtext .5pt;mso-border-top-alt:solid windowtext .5pt;
  mso-border-bottom-alt:solid windowtext .5pt;padding:0cm 5.4pt 0cm 5.4pt;
  height:1.0pt'>
  <p class=MsoNormal align=center style='text-align:center;mso-pagination:widow-orphan'><span
  lang=EN-US style='font-size:7.5pt;mso-bidi-font-size:10.0pt;font-family:"Times New Roman",serif;
  mso-fareast-font-family:等线;mso-font-kerning:0pt'>0~15<o:p></o:p></span></p>
  </td>
  <td width=39 nowrap style='width:29.55pt;border-top:none;border-left:none;
  border-bottom:solid windowtext 1.0pt;border-right:solid windowtext 1.0pt;
  mso-border-top-alt:solid windowtext .5pt;mso-border-top-alt:solid windowtext .5pt;
  mso-border-bottom-alt:solid windowtext .5pt;mso-border-right-alt:solid windowtext .5pt;
  padding:0cm 5.4pt 0cm 5.4pt;height:1.0pt'>
  <p class=MsoNormal align=center style='text-align:center;mso-pagination:widow-orphan'><span
  class=SpellE><span lang=EN-US style='font-size:7.5pt;mso-bidi-font-size:10.0pt;
  font-family:"Times New Roman",serif;mso-fareast-font-family:等线;mso-font-kerning:
  0pt'>Ovrl</span></span><span lang=EN-US style='font-size:7.5pt;mso-bidi-font-size:
  10.0pt;font-family:"Times New Roman",serif;mso-fareast-font-family:等线;
  mso-font-kerning:0pt'>.<o:p></o:p></span></p>
  </td>
  <td width=39 nowrap style='width:29.5pt;border:none;border-bottom:solid windowtext 1.0pt;
  mso-border-top-alt:solid windowtext .5pt;mso-border-left-alt:solid windowtext .5pt;
  mso-border-top-alt:solid windowtext .5pt;mso-border-left-alt:solid windowtext .5pt;
  mso-border-bottom-alt:solid windowtext .5pt;padding:0cm 5.4pt 0cm 5.4pt;
  height:1.0pt'>
  <p class=MsoNormal align=center style='text-align:center;mso-pagination:widow-orphan'><span
  lang=EN-US style='font-size:7.5pt;mso-bidi-font-size:10.0pt;font-family:"Times New Roman",serif;
  mso-fareast-font-family:等线;mso-font-kerning:0pt'>-5~0<o:p></o:p></span></p>
  </td>
  <td width=39 nowrap style='width:29.55pt;border:none;border-bottom:solid windowtext 1.0pt;
  mso-border-top-alt:solid windowtext .5pt;mso-border-top-alt:solid windowtext .5pt;
  mso-border-bottom-alt:solid windowtext .5pt;padding:0cm 5.4pt 0cm 5.4pt;
  height:1.0pt'>
  <p class=MsoNormal align=center style='text-align:center;mso-pagination:widow-orphan'><span
  lang=EN-US style='font-size:7.5pt;mso-bidi-font-size:10.0pt;font-family:"Times New Roman",serif;
  mso-fareast-font-family:等线;mso-font-kerning:0pt'>0~15<o:p></o:p></span></p>
  </td>
  <td width=39 nowrap style='width:29.55pt;border-top:none;border-left:none;
  border-bottom:solid windowtext 1.0pt;border-right:solid windowtext 1.0pt;
  mso-border-top-alt:solid windowtext .5pt;mso-border-top-alt:solid windowtext .5pt;
  mso-border-bottom-alt:solid windowtext .5pt;mso-border-right-alt:solid windowtext .5pt;
  padding:0cm 5.4pt 0cm 5.4pt;height:1.0pt'>
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
  <td width=39 style='width:29.5pt;border:none;border-bottom:solid windowtext 1.0pt;
  mso-border-top-alt:solid windowtext .5pt;mso-border-left-alt:solid windowtext .5pt;
  mso-border-top-alt:solid windowtext .5pt;mso-border-left-alt:solid windowtext .5pt;
  mso-border-bottom-alt:solid windowtext .5pt;padding:0cm 5.4pt 0cm 5.4pt;
  height:1.0pt'>
  <p class=MsoNormal align=center style='text-align:center;mso-pagination:widow-orphan'><span
  lang=EN-US style='font-size:7.5pt;font-family:"Times New Roman",serif;
  mso-fareast-font-family:等线;mso-font-kerning:0pt'>0~8<o:p></o:p></span></p>
  </td>
  <td width=39 style='width:29.55pt;border:none;border-bottom:solid windowtext 1.0pt;
  mso-border-top-alt:solid windowtext .5pt;mso-border-top-alt:solid windowtext .5pt;
  mso-border-bottom-alt:solid windowtext .5pt;padding:0cm 5.4pt 0cm 5.4pt;
  height:1.0pt'>
  <p class=MsoNormal align=center style='text-align:center;mso-pagination:widow-orphan'><span
  lang=EN-US style='font-size:7.5pt;font-family:"Times New Roman",serif;
  mso-fareast-font-family:等线;mso-font-kerning:0pt'>8~24<o:p></o:p></span></p>
  </td>
  <td width=39 style='width:29.55pt;border-top:none;border-left:none;
  border-bottom:solid windowtext 1.0pt;border-right:solid windowtext 1.0pt;
  mso-border-top-alt:solid windowtext .5pt;mso-border-top-alt:solid windowtext .5pt;
  mso-border-bottom-alt:solid windowtext .5pt;mso-border-right-alt:solid windowtext .5pt;
  padding:0cm 5.4pt 0cm 5.4pt;height:1.0pt'>
  <p class=MsoNormal align=center style='text-align:center;mso-pagination:widow-orphan'><span
  lang=EN-US style='font-size:7.5pt;font-family:"Times New Roman",serif;
  mso-fareast-font-family:等线;mso-font-kerning:0pt'>Full.<o:p></o:p></span></p>
  </td>
  <td style='mso-cell-special:placeholder;border:none;padding:0cm 0cm 0cm 0cm'
  width=4><p class='MsoNormal'>&nbsp;</td>
 </tr>
 <tr style='mso-yfti-irow:2;height:1.0pt;mso-row-margin-right:3.35pt'>
  <td width=75 nowrap style='width:56.45pt;border:solid windowtext 1.0pt;
  border-top:none;mso-border-top-alt:solid windowtext .5pt;mso-border-alt:solid windowtext .5pt;
  padding:0cm 5.4pt 0cm 5.4pt;height:1.0pt'>
  <p class=MsoNormal align=center style='text-align:center;mso-pagination:widow-orphan'><span
  lang=EN-US style='font-size:6.5pt;mso-bidi-font-size:10.0pt;font-family:"Times New Roman",serif;
  mso-fareast-font-family:等线;mso-font-kerning:0pt'>Noisy</span><span
  lang=EN-US style='font-size:7.5pt;mso-bidi-font-size:10.0pt;font-family:"Times New Roman",serif;
  mso-fareast-font-family:等线;mso-font-kerning:0pt'><o:p></o:p></span></p>
  </td>
  <td width=39 nowrap style='width:29.5pt;border:none;border-bottom:solid windowtext 1.0pt;
  mso-border-top-alt:solid windowtext .5pt;mso-border-left-alt:solid windowtext .5pt;
  mso-border-top-alt:solid windowtext .5pt;mso-border-left-alt:solid windowtext .5pt;
  mso-border-bottom-alt:solid windowtext .5pt;padding:0cm 5.4pt 0cm 5.4pt;
  height:1.0pt'>
  <p class=MsoNormal align=center style='text-align:center;mso-pagination:widow-orphan'><span
  lang=EN-US style='font-size:6.5pt;mso-bidi-font-size:10.0pt;font-family:"Times New Roman",serif;
  mso-fareast-font-family:等线;mso-font-kerning:0pt'>0.687<o:p></o:p></span></p>
  </td>
  <td width=39 nowrap style='width:29.55pt;border:none;border-bottom:solid windowtext 1.0pt;
  mso-border-top-alt:solid windowtext .5pt;mso-border-top-alt:solid windowtext .5pt;
  mso-border-bottom-alt:solid windowtext .5pt;padding:0cm 5.4pt 0cm 5.4pt;
  height:1.0pt'>
  <p class=MsoNormal align=center style='text-align:center;mso-pagination:widow-orphan'><span
  lang=EN-US style='font-size:6.5pt;mso-bidi-font-size:10.0pt;font-family:"Times New Roman",serif;
  mso-fareast-font-family:等线;mso-font-kerning:0pt'>0.805<o:p></o:p></span></p>
  </td>
  <td width=39 nowrap style='width:29.55pt;border-top:none;border-left:none;
  border-bottom:solid windowtext 1.0pt;border-right:solid windowtext 1.0pt;
  mso-border-top-alt:solid windowtext .5pt;mso-border-top-alt:solid windowtext .5pt;
  mso-border-bottom-alt:solid windowtext .5pt;mso-border-right-alt:solid windowtext .5pt;
  padding:0cm 5.4pt 0cm 5.4pt;height:1.0pt'>
  <p class=MsoNormal align=center style='text-align:center;mso-pagination:widow-orphan'><span
  lang=EN-US style='font-size:6.5pt;mso-bidi-font-size:10.0pt;font-family:"Times New Roman",serif;
  mso-fareast-font-family:等线;mso-font-kerning:0pt'>0.771<o:p></o:p></span></p>
  </td>
  <td width=39 nowrap valign=top style='width:29.5pt;border:none;border-bottom:
  solid windowtext 1.0pt;mso-border-top-alt:solid windowtext .5pt;mso-border-left-alt:
  solid windowtext .5pt;mso-border-top-alt:solid windowtext .5pt;mso-border-left-alt:
  solid windowtext .5pt;mso-border-bottom-alt:solid windowtext .5pt;padding:
  0cm 5.4pt 0cm 5.4pt;height:1.0pt'>
  <p class=MsoNormal align=center style='text-align:center;mso-pagination:widow-orphan'><span
  lang=EN-US style='font-size:6.5pt;mso-bidi-font-size:10.0pt;font-family:"Times New Roman",serif;
  mso-fareast-font-family:等线;mso-font-kerning:0pt'>-2.515<o:p></o:p></span></p>
  </td>
  <td width=39 nowrap valign=top style='width:29.55pt;border:none;border-bottom:
  solid windowtext 1.0pt;mso-border-top-alt:solid windowtext .5pt;mso-border-top-alt:
  solid windowtext .5pt;mso-border-bottom-alt:solid windowtext .5pt;padding:
  0cm 5.4pt 0cm 5.4pt;height:1.0pt'>
  <p class=MsoNormal align=center style='text-align:center;mso-pagination:widow-orphan'><span
  lang=EN-US style='font-size:6.5pt;mso-bidi-font-size:10.0pt;font-family:"Times New Roman",serif;
  mso-fareast-font-family:等线;mso-font-kerning:0pt'>7.971<o:p></o:p></span></p>
  </td>
  <td width=39 nowrap valign=top style='width:29.55pt;border-top:none;
  border-left:none;border-bottom:solid windowtext 1.0pt;border-right:solid windowtext 1.0pt;
  mso-border-top-alt:solid windowtext .5pt;mso-border-top-alt:solid windowtext .5pt;
  mso-border-bottom-alt:solid windowtext .5pt;mso-border-right-alt:solid windowtext .5pt;
  padding:0cm 5.4pt 0cm 5.4pt;height:1.0pt'>
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
  <td width=39 style='width:29.5pt;border:none;border-bottom:solid windowtext 1.0pt;
  mso-border-top-alt:solid windowtext .5pt;mso-border-left-alt:solid windowtext .5pt;
  mso-border-top-alt:solid windowtext .5pt;mso-border-left-alt:solid windowtext .5pt;
  mso-border-bottom-alt:solid windowtext .5pt;padding:0cm 5.4pt 0cm 5.4pt;
  height:1.0pt'>
  <p class=MsoNormal align=center style='text-align:center;mso-pagination:widow-orphan'><span
  lang=EN-US style='font-size:6.5pt;mso-bidi-font-size:10.0pt;font-family:"Times New Roman",serif;
  mso-fareast-font-family:等线;mso-font-kerning:0pt'>18.37<o:p></o:p></span></p>
  </td>
  <td width=39 style='width:29.55pt;border:none;border-bottom:solid windowtext 1.0pt;
  mso-border-top-alt:solid windowtext .5pt;mso-border-top-alt:solid windowtext .5pt;
  mso-border-bottom-alt:solid windowtext .5pt;padding:0cm 5.4pt 0cm 5.4pt;
  height:1.0pt'>
  <p class=MsoNormal align=center style='text-align:center;mso-pagination:widow-orphan'><span
  lang=EN-US style='font-size:6.5pt;mso-bidi-font-size:10.0pt;font-family:"Times New Roman",serif;
  mso-fareast-font-family:等线;mso-font-kerning:0pt'>12.38<o:p></o:p></span></p>
  </td>
  <td width=39 valign=top style='width:29.55pt;border-top:none;border-left:
  none;border-bottom:solid windowtext 1.0pt;border-right:solid windowtext 1.0pt;
  mso-border-top-alt:solid windowtext .5pt;mso-border-top-alt:solid windowtext .5pt;
  mso-border-bottom-alt:solid windowtext .5pt;mso-border-right-alt:solid windowtext .5pt;
  padding:0cm 5.4pt 0cm 5.4pt;height:1.0pt'>
  <p class=MsoNormal align=center style='text-align:center;mso-pagination:widow-orphan'><span
  lang=EN-US style='font-size:6.5pt;mso-bidi-font-size:10.0pt;font-family:"Times New Roman",serif;
  mso-fareast-font-family:等线;mso-font-kerning:0pt'>14.38<o:p></o:p></span></p>
  </td>
  <td style='mso-cell-special:placeholder;border:none;padding:0cm 0cm 0cm 0cm'
  width=4><p class='MsoNormal'>&nbsp;</td>
 </tr>
 <tr style='mso-yfti-irow:3;height:1.0pt;mso-row-margin-right:3.35pt'>
  <td width=75 nowrap style='width:56.45pt;border-top:none;border-left:solid windowtext 1.0pt;
  border-bottom:none;border-right:solid windowtext 1.0pt;mso-border-top-alt:
  solid windowtext .5pt;mso-border-top-alt:solid windowtext .5pt;mso-border-left-alt:
  solid windowtext .5pt;mso-border-right-alt:solid windowtext .5pt;padding:
  0cm 5.4pt 0cm 5.4pt;height:1.0pt'>
  <p class=MsoNormal align=center style='text-align:center;mso-pagination:widow-orphan'><span
  lang=EN-US style='font-size:6.5pt;mso-bidi-font-size:10.0pt;font-family:"Times New Roman",serif;
  mso-fareast-font-family:等线;mso-font-kerning:0pt'>MTFAA<o:p></o:p></span></p>
  </td>
  <td width=39 nowrap style='width:29.5pt;border:none;mso-border-top-alt:solid windowtext .5pt;
  mso-border-left-alt:solid windowtext .5pt;padding:0cm 5.4pt 0cm 5.4pt;
  height:1.0pt'>
  <p class=MsoNormal align=center style='text-align:center;mso-pagination:widow-orphan'><span
  lang=EN-US style='font-size:6.5pt;mso-bidi-font-size:10.0pt;font-family:"Times New Roman",serif;
  mso-fareast-font-family:等线;mso-font-kerning:0pt'>0.805<o:p></o:p></span></p>
  </td>
  <td width=39 nowrap style='width:29.55pt;border:none;mso-border-top-alt:solid windowtext .5pt;
  padding:0cm 5.4pt 0cm 5.4pt;height:1.0pt'>
  <p class=MsoNormal align=center style='text-align:center;mso-pagination:widow-orphan'><span
  lang=EN-US style='font-size:6.5pt;mso-bidi-font-size:10.0pt;font-family:"Times New Roman",serif;
  mso-fareast-font-family:等线;mso-font-kerning:0pt'>0.876<o:p></o:p></span></p>
  </td>
  <td width=39 nowrap style='width:29.55pt;border:none;border-right:solid windowtext 1.0pt;
  mso-border-top-alt:solid windowtext .5pt;mso-border-top-alt:solid windowtext .5pt;
  mso-border-right-alt:solid windowtext .5pt;padding:0cm 5.4pt 0cm 5.4pt;
  height:1.0pt'>
  <p class=MsoNormal align=center style='text-align:center;mso-pagination:widow-orphan'><span
  lang=EN-US style='font-size:6.5pt;mso-bidi-font-size:10.0pt;font-family:"Times New Roman",serif;
  mso-fareast-font-family:等线;mso-font-kerning:0pt'>0.856<o:p></o:p></span></p>
  </td>
  <td width=39 nowrap valign=top style='width:29.5pt;border:none;mso-border-top-alt:
  solid windowtext .5pt;mso-border-left-alt:solid windowtext .5pt;padding:0cm 5.4pt 0cm 5.4pt;
  height:1.0pt'>
  <p class=MsoNormal align=center style='text-align:center;mso-pagination:widow-orphan'><span
  lang=EN-US style='font-size:6.5pt;mso-bidi-font-size:10.0pt;font-family:"Times New Roman",serif;
  mso-fareast-font-family:等线;mso-font-kerning:0pt'>10.10<o:p></o:p></span></p>
  </td>
  <td width=39 nowrap valign=top style='width:29.55pt;border:none;mso-border-top-alt:
  solid windowtext .5pt;padding:0cm 5.4pt 0cm 5.4pt;height:1.0pt'>
  <p class=MsoNormal align=center style='text-align:center;mso-pagination:widow-orphan'><span
  lang=EN-US style='font-size:6.5pt;mso-bidi-font-size:10.0pt;font-family:"Times New Roman",serif;
  mso-fareast-font-family:等线;mso-font-kerning:0pt'>15.74<o:p></o:p></span></p>
  </td>
  <td width=39 nowrap valign=top style='width:29.55pt;border:none;border-right:
  solid windowtext 1.0pt;mso-border-top-alt:solid windowtext .5pt;mso-border-top-alt:
  solid windowtext .5pt;mso-border-right-alt:solid windowtext .5pt;padding:
  0cm 5.4pt 0cm 5.4pt;height:1.0pt'>
  <p class=MsoNormal align=center style='text-align:center;mso-pagination:widow-orphan'><span
  lang=EN-US style='font-size:6.5pt;mso-bidi-font-size:10.0pt;font-family:"Times New Roman",serif;
  mso-fareast-font-family:等线;mso-font-kerning:0pt'>14.23<o:p></o:p></span></p>
  </td>
  <td width=79 nowrap colspan=2 style='width:59.1pt;border:none;border-right:
  solid windowtext 1.0pt;mso-border-top-alt:solid windowtext .5pt;mso-border-left-alt:
  solid windowtext .5pt;mso-border-top-alt:solid windowtext .5pt;mso-border-left-alt:
  solid windowtext .5pt;mso-border-right-alt:solid windowtext .5pt;padding:
  0cm 5.4pt 0cm 5.4pt;height:1.0pt'>
  <p class=MsoNormal align=center style='text-align:center;mso-pagination:widow-orphan'><span
  lang=EN-US style='font-size:6.5pt;mso-bidi-font-size:10.0pt;font-family:"Times New Roman",serif;
  mso-fareast-font-family:等线;mso-font-kerning:0pt'>MTFAA<o:p></o:p></span></p>
  </td>
  <td width=39 style='width:29.5pt;border:none;mso-border-top-alt:solid windowtext .5pt;
  mso-border-left-alt:solid windowtext .5pt;padding:0cm 5.4pt 0cm 5.4pt;
  height:1.0pt'>
  <p class=MsoNormal align=center style='text-align:center;mso-pagination:widow-orphan'><span
  lang=EN-US style='font-size:6.5pt;mso-bidi-font-size:10.0pt;font-family:"Times New Roman",serif;
  mso-fareast-font-family:等线;mso-font-kerning:0pt'>10.33<o:p></o:p></span></p>
  </td>
  <td width=39 style='width:29.55pt;border:none;mso-border-top-alt:solid windowtext .5pt;
  padding:0cm 5.4pt 0cm 5.4pt;height:1.0pt'>
  <p class=MsoNormal align=center style='text-align:center;mso-pagination:widow-orphan'><span
  lang=EN-US style='font-size:6.5pt;mso-bidi-font-size:10.0pt;font-family:"Times New Roman",serif;
  mso-fareast-font-family:等线;mso-font-kerning:0pt'>9.349<o:p></o:p></span></p>
  </td>
  <td width=39 valign=top style='width:29.55pt;border:none;border-right:solid windowtext 1.0pt;
  mso-border-top-alt:solid windowtext .5pt;mso-border-top-alt:solid windowtext .5pt;
  mso-border-right-alt:solid windowtext .5pt;padding:0cm 5.4pt 0cm 5.4pt;
  height:1.0pt'>
  <p class=MsoNormal align=center style='text-align:center;mso-pagination:widow-orphan'><span
  lang=EN-US style='font-size:6.5pt;mso-bidi-font-size:10.0pt;font-family:"Times New Roman",serif;
  mso-fareast-font-family:等线;mso-font-kerning:0pt'>9.678<o:p></o:p></span></p>
  </td>
  <td style='mso-cell-special:placeholder;border:none;padding:0cm 0cm 0cm 0cm'
  width=4><p class='MsoNormal'>&nbsp;</td>
 </tr>
 <tr style='mso-yfti-irow:4;height:1.0pt;mso-row-margin-right:3.35pt'>
  <td width=75 nowrap style='width:56.45pt;border:solid windowtext 1.0pt;
  border-top:none;mso-border-left-alt:solid windowtext .5pt;mso-border-bottom-alt:
  solid windowtext .5pt;mso-border-right-alt:solid windowtext .5pt;padding:
  0cm 5.4pt 0cm 5.4pt;height:1.0pt'>
  <p class=MsoNormal align=center style='text-align:center;mso-pagination:widow-orphan'><span
  lang=EN-US style='font-size:6.5pt;mso-bidi-font-size:10.0pt;font-family:"Times New Roman",serif;
  mso-fareast-font-family:等线;mso-font-kerning:0pt'>MTFAA-LSA</span><span
  lang=EN-US style='font-size:7.5pt;mso-bidi-font-size:10.0pt;font-family:"Times New Roman",serif;
  mso-fareast-font-family:等线;mso-font-kerning:0pt'><o:p></o:p></span></p>
  </td>
  <td width=39 nowrap style='width:29.5pt;border:none;border-bottom:solid windowtext 1.0pt;
  mso-border-left-alt:solid windowtext .5pt;mso-border-left-alt:solid windowtext .5pt;
  mso-border-bottom-alt:solid windowtext .5pt;padding:0cm 5.4pt 0cm 5.4pt;
  height:1.0pt'>
  <p class=MsoNormal align=center style='text-align:center;mso-pagination:widow-orphan'><b
  style='mso-bidi-font-weight:normal'><span lang=EN-US style='font-size:6.5pt;
  mso-bidi-font-size:10.0pt;font-family:"Times New Roman",serif;mso-fareast-font-family:
  等线;mso-font-kerning:0pt'>0.809</span></b><!--[if gte msEquation 12]><m:oMath><b
   style='mso-bidi-font-weight:normal'><span lang=EN-US style='font-size:6.5pt;
   mso-bidi-font-size:10.0pt;font-family:"Cambria Math",serif;mso-fareast-font-family:
   等线;mso-bidi-font-family:"Times New Roman";mso-font-kerning:0pt'><m:r><m:rPr><m:scr
      m:val="roman"/><m:sty m:val="b"/></m:rPr>↑</m:r></span></b></m:oMath><![endif]--><![if !msEquation]><span
  lang=EN-US style='font-size:10.5pt;mso-bidi-font-size:11.0pt;font-family:
  等线;mso-ascii-theme-font:minor-latin;mso-fareast-theme-font:minor-fareast;
  mso-hansi-theme-font:minor-latin;mso-bidi-font-family:"Times New Roman";
  mso-bidi-theme-font:minor-bidi;position:relative;top:5.5pt;mso-text-raise:
  -5.5pt;mso-ansi-language:EN-US;mso-fareast-language:ZH-CN;mso-bidi-language:
  AR-SA'><!--[if gte vml 1]><v:shapetype id="_x0000_t75" coordsize="21600,21600"
   o:spt="75" o:preferrelative="t" path="m@4@5l@4@11@9@11@9@5xe" filled="f"
   stroked="f">
   <v:stroke joinstyle="miter"/>
   <v:formulas>
    <v:f eqn="if lineDrawn pixelLineWidth 0"/>
    <v:f eqn="sum @0 1 0"/>
    <v:f eqn="sum 0 0 @1"/>
    <v:f eqn="prod @2 1 2"/>
    <v:f eqn="prod @3 21600 pixelWidth"/>
    <v:f eqn="prod @3 21600 pixelHeight"/>
    <v:f eqn="sum @0 0 1"/>
    <v:f eqn="prod @6 1 2"/>
    <v:f eqn="prod @7 21600 pixelWidth"/>
    <v:f eqn="sum @8 21600 0"/>
    <v:f eqn="prod @7 21600 pixelHeight"/>
    <v:f eqn="sum @10 21600 0"/>
   </v:formulas>
   <v:path o:extrusionok="f" gradientshapeok="t" o:connecttype="rect"/>
   <o:lock v:ext="edit" aspectratio="t"/>
  </v:shapetype><v:shape id="_x0000_i1025" type="#_x0000_t75" style='width:3pt;
   height:15.75pt'>
   <v:imagedata src="Table_readme.files/image001.png" o:title="" chromakey="white"/>
  </v:shape><![endif]--><![if !vml]><img width=4 height=21
  src="Table_readme.files/image002.png" v:shapes="_x0000_i1025"><![endif]></span><![endif]><b
  style='mso-bidi-font-weight:normal'><span lang=EN-US style='font-size:6.5pt;
  mso-bidi-font-size:10.0pt;font-family:"Times New Roman",serif;mso-fareast-font-family:
  等线;mso-font-kerning:0pt'><o:p></o:p></span></b></p>
  </td>
  <td width=39 nowrap style='width:29.55pt;border:none;border-bottom:solid windowtext 1.0pt;
  mso-border-bottom-alt:solid windowtext .5pt;padding:0cm 5.4pt 0cm 5.4pt;
  height:1.0pt'>
  <p class=MsoNormal align=center style='text-align:center;mso-pagination:widow-orphan'><b
  style='mso-bidi-font-weight:normal'><span lang=EN-US style='font-size:6.5pt;
  mso-bidi-font-size:10.0pt;font-family:"Times New Roman",serif;mso-fareast-font-family:
  等线;mso-font-kerning:0pt'>0.881</span></b><!--[if gte msEquation 12]><m:oMath><b
   style='mso-bidi-font-weight:normal'><span lang=EN-US style='font-size:6.5pt;
   mso-bidi-font-size:10.0pt;font-family:"Cambria Math",serif;mso-fareast-font-family:
   等线;mso-bidi-font-family:"Times New Roman";mso-font-kerning:0pt'><m:r><m:rPr><m:scr
      m:val="roman"/><m:sty m:val="b"/></m:rPr>↑</m:r></span></b></m:oMath><![endif]--><![if !msEquation]><span
  lang=EN-US style='font-size:10.5pt;mso-bidi-font-size:11.0pt;font-family:
  等线;mso-ascii-theme-font:minor-latin;mso-fareast-theme-font:minor-fareast;
  mso-hansi-theme-font:minor-latin;mso-bidi-font-family:"Times New Roman";
  mso-bidi-theme-font:minor-bidi;position:relative;top:5.5pt;mso-text-raise:
  -5.5pt;mso-ansi-language:EN-US;mso-fareast-language:ZH-CN;mso-bidi-language:
  AR-SA'><!--[if gte vml 1]><v:shape id="_x0000_i1025" type="#_x0000_t75"
   style='width:3pt;height:15.75pt'>
   <v:imagedata src="Table_readme.files/image001.png" o:title="" chromakey="white"/>
  </v:shape><![endif]--><![if !vml]><img width=4 height=21
  src="Table_readme.files/image002.png" v:shapes="_x0000_i1025"><![endif]></span><![endif]><b
  style='mso-bidi-font-weight:normal'><span lang=EN-US style='font-size:6.5pt;
  mso-bidi-font-size:10.0pt;font-family:"Times New Roman",serif;mso-fareast-font-family:
  等线;mso-font-kerning:0pt'><o:p></o:p></span></b></p>
  </td>
  <td width=39 nowrap style='width:29.55pt;border-top:none;border-left:none;
  border-bottom:solid windowtext 1.0pt;border-right:solid windowtext 1.0pt;
  mso-border-bottom-alt:solid windowtext .5pt;mso-border-right-alt:solid windowtext .5pt;
  padding:0cm 5.4pt 0cm 5.4pt;height:1.0pt'>
  <p class=MsoNormal align=center style='text-align:center;mso-pagination:widow-orphan'><b
  style='mso-bidi-font-weight:normal'><span lang=EN-US style='font-size:6.5pt;
  mso-bidi-font-size:10.0pt;font-family:"Times New Roman",serif;mso-fareast-font-family:
  等线;mso-font-kerning:0pt'>0.860</span></b><!--[if gte msEquation 12]><m:oMath><b
   style='mso-bidi-font-weight:normal'><span lang=EN-US style='font-size:6.5pt;
   mso-bidi-font-size:10.0pt;font-family:"Cambria Math",serif;mso-fareast-font-family:
   等线;mso-bidi-font-family:"Times New Roman";mso-font-kerning:0pt'><m:r><m:rPr><m:scr
      m:val="roman"/><m:sty m:val="b"/></m:rPr>↑</m:r></span></b></m:oMath><![endif]--><![if !msEquation]><span
  lang=EN-US style='font-size:10.5pt;mso-bidi-font-size:11.0pt;font-family:
  等线;mso-ascii-theme-font:minor-latin;mso-fareast-theme-font:minor-fareast;
  mso-hansi-theme-font:minor-latin;mso-bidi-font-family:"Times New Roman";
  mso-bidi-theme-font:minor-bidi;position:relative;top:5.5pt;mso-text-raise:
  -5.5pt;mso-ansi-language:EN-US;mso-fareast-language:ZH-CN;mso-bidi-language:
  AR-SA'><!--[if gte vml 1]><v:shape id="_x0000_i1025" type="#_x0000_t75"
   style='width:3pt;height:15.75pt'>
   <v:imagedata src="Table_readme.files/image001.png" o:title="" chromakey="white"/>
  </v:shape><![endif]--><![if !vml]><img width=4 height=21
  src="Table_readme.files/image002.png" v:shapes="_x0000_i1025"><![endif]></span><![endif]><b
  style='mso-bidi-font-weight:normal'><span lang=EN-US style='font-size:6.5pt;
  mso-bidi-font-size:10.0pt;font-family:"Times New Roman",serif;mso-fareast-font-family:
  等线;mso-font-kerning:0pt'><o:p></o:p></span></b></p>
  </td>
  <td width=39 nowrap valign=top style='width:29.5pt;border:none;border-bottom:
  solid windowtext 1.0pt;mso-border-left-alt:solid windowtext .5pt;mso-border-left-alt:
  solid windowtext .5pt;mso-border-bottom-alt:solid windowtext .5pt;padding:
  0cm 5.4pt 0cm 5.4pt;height:1.0pt'>
  <p class=MsoNormal align=center style='text-align:center;mso-pagination:widow-orphan'><b
  style='mso-bidi-font-weight:normal'><span lang=EN-US style='font-size:6.5pt;
  mso-bidi-font-size:10.0pt;font-family:"Times New Roman",serif;mso-fareast-font-family:
  等线;mso-font-kerning:0pt'>10.34</span></b><!--[if gte msEquation 12]><m:oMath><b
   style='mso-bidi-font-weight:normal'><span lang=EN-US style='font-size:6.5pt;
   mso-bidi-font-size:10.0pt;font-family:"Cambria Math",serif;mso-fareast-font-family:
   等线;mso-bidi-font-family:"Times New Roman";mso-font-kerning:0pt'><m:r><m:rPr><m:scr
      m:val="roman"/><m:sty m:val="b"/></m:rPr>↑</m:r></span></b></m:oMath><![endif]--><![if !msEquation]><span
  lang=EN-US style='font-size:10.5pt;mso-bidi-font-size:11.0pt;font-family:
  等线;mso-ascii-theme-font:minor-latin;mso-fareast-theme-font:minor-fareast;
  mso-hansi-theme-font:minor-latin;mso-bidi-font-family:"Times New Roman";
  mso-bidi-theme-font:minor-bidi;position:relative;top:5.5pt;mso-text-raise:
  -5.5pt;mso-ansi-language:EN-US;mso-fareast-language:ZH-CN;mso-bidi-language:
  AR-SA'><!--[if gte vml 1]><v:shape id="_x0000_i1025" type="#_x0000_t75"
   style='width:3pt;height:15.75pt'>
   <v:imagedata src="Table_readme.files/image001.png" o:title="" chromakey="white"/>
  </v:shape><![endif]--><![if !vml]><img width=4 height=21
  src="Table_readme.files/image002.png" v:shapes="_x0000_i1025"><![endif]></span><![endif]><b
  style='mso-bidi-font-weight:normal'><span lang=EN-US style='font-size:6.5pt;
  mso-bidi-font-size:10.0pt;font-family:"Times New Roman",serif;mso-fareast-font-family:
  等线;mso-font-kerning:0pt'><o:p></o:p></span></b></p>
  </td>
  <td width=39 nowrap valign=top style='width:29.55pt;border:none;border-bottom:
  solid windowtext 1.0pt;mso-border-bottom-alt:solid windowtext .5pt;
  padding:0cm 5.4pt 0cm 5.4pt;height:1.0pt'>
  <p class=MsoNormal align=center style='text-align:center;mso-pagination:widow-orphan'><b
  style='mso-bidi-font-weight:normal'><span lang=EN-US style='font-size:6.5pt;
  mso-bidi-font-size:10.0pt;font-family:"Times New Roman",serif;mso-fareast-font-family:
  等线;mso-font-kerning:0pt'>16.20</span></b><!--[if gte msEquation 12]><m:oMath><b
   style='mso-bidi-font-weight:normal'><span lang=EN-US style='font-size:6.5pt;
   mso-bidi-font-size:10.0pt;font-family:"Cambria Math",serif;mso-fareast-font-family:
   等线;mso-bidi-font-family:"Times New Roman";mso-font-kerning:0pt'><m:r><m:rPr><m:scr
      m:val="roman"/><m:sty m:val="b"/></m:rPr>↑</m:r></span></b></m:oMath><![endif]--><![if !msEquation]><span
  lang=EN-US style='font-size:10.5pt;mso-bidi-font-size:11.0pt;font-family:
  等线;mso-ascii-theme-font:minor-latin;mso-fareast-theme-font:minor-fareast;
  mso-hansi-theme-font:minor-latin;mso-bidi-font-family:"Times New Roman";
  mso-bidi-theme-font:minor-bidi;position:relative;top:5.5pt;mso-text-raise:
  -5.5pt;mso-ansi-language:EN-US;mso-fareast-language:ZH-CN;mso-bidi-language:
  AR-SA'><!--[if gte vml 1]><v:shape id="_x0000_i1025" type="#_x0000_t75"
   style='width:3pt;height:15.75pt'>
   <v:imagedata src="Table_readme.files/image001.png" o:title="" chromakey="white"/>
  </v:shape><![endif]--><![if !vml]><img width=4 height=21
  src="Table_readme.files/image002.png" v:shapes="_x0000_i1025"><![endif]></span><![endif]><b
  style='mso-bidi-font-weight:normal'><span lang=EN-US style='font-size:6.5pt;
  mso-bidi-font-size:10.0pt;font-family:"Times New Roman",serif;mso-fareast-font-family:
  等线;mso-font-kerning:0pt'><o:p></o:p></span></b></p>
  </td>
  <td width=39 nowrap valign=top style='width:29.55pt;border-top:none;
  border-left:none;border-bottom:solid windowtext 1.0pt;border-right:solid windowtext 1.0pt;
  mso-border-bottom-alt:solid windowtext .5pt;mso-border-right-alt:solid windowtext .5pt;
  padding:0cm 5.4pt 0cm 5.4pt;height:1.0pt'>
  <p class=MsoNormal align=center style='text-align:center;mso-pagination:widow-orphan'><b
  style='mso-bidi-font-weight:normal'><span lang=EN-US style='font-size:6.5pt;
  mso-bidi-font-size:10.0pt;font-family:"Times New Roman",serif;mso-fareast-font-family:
  等线;mso-font-kerning:0pt'>14.63</span></b><!--[if gte msEquation 12]><m:oMath><b
   style='mso-bidi-font-weight:normal'><span lang=EN-US style='font-size:6.5pt;
   mso-bidi-font-size:10.0pt;font-family:"Cambria Math",serif;mso-fareast-font-family:
   等线;mso-bidi-font-family:"Times New Roman";mso-font-kerning:0pt'><m:r><m:rPr><m:scr
      m:val="roman"/><m:sty m:val="b"/></m:rPr>↑</m:r></span></b></m:oMath><![endif]--><![if !msEquation]><span
  lang=EN-US style='font-size:10.5pt;mso-bidi-font-size:11.0pt;font-family:
  等线;mso-ascii-theme-font:minor-latin;mso-fareast-theme-font:minor-fareast;
  mso-hansi-theme-font:minor-latin;mso-bidi-font-family:"Times New Roman";
  mso-bidi-theme-font:minor-bidi;position:relative;top:5.5pt;mso-text-raise:
  -5.5pt;mso-ansi-language:EN-US;mso-fareast-language:ZH-CN;mso-bidi-language:
  AR-SA'><!--[if gte vml 1]><v:shape id="_x0000_i1025" type="#_x0000_t75"
   style='width:3pt;height:15.75pt'>
   <v:imagedata src="Table_readme.files/image001.png" o:title="" chromakey="white"/>
  </v:shape><![endif]--><![if !vml]><img width=4 height=21
  src="Table_readme.files/image002.png" v:shapes="_x0000_i1025"><![endif]></span><![endif]><span
  lang=EN-US style='font-size:6.5pt;mso-bidi-font-size:10.0pt;font-family:"Times New Roman",serif;
  mso-fareast-font-family:等线;mso-font-kerning:0pt'><o:p></o:p></span></p>
  </td>
  <td width=79 nowrap colspan=2 style='width:59.1pt;border-top:none;border-left:
  none;border-bottom:solid windowtext 1.0pt;border-right:solid windowtext 1.0pt;
  mso-border-left-alt:solid windowtext .5pt;mso-border-left-alt:solid windowtext .5pt;
  mso-border-bottom-alt:solid windowtext .5pt;mso-border-right-alt:solid windowtext .5pt;
  padding:0cm 5.4pt 0cm 5.4pt;height:1.0pt'>
  <p class=MsoNormal align=center style='text-align:center;mso-pagination:widow-orphan'><span
  lang=EN-US style='font-size:6.5pt;mso-bidi-font-size:10.0pt;font-family:"Times New Roman",serif;
  mso-fareast-font-family:等线;mso-font-kerning:0pt'>MTFAA-LSA<b
  style='mso-bidi-font-weight:normal'><o:p></o:p></b></span></p>
  </td>
  <td width=39 style='width:29.5pt;border:none;border-bottom:solid windowtext 1.0pt;
  mso-border-left-alt:solid windowtext .5pt;mso-border-left-alt:solid windowtext .5pt;
  mso-border-bottom-alt:solid windowtext .5pt;padding:0cm 5.4pt 0cm 5.4pt;
  height:1.0pt'>
  <p class=MsoNormal align=center style='text-align:center;mso-pagination:widow-orphan'><b
  style='mso-bidi-font-weight:normal'><span lang=EN-US style='font-size:6.5pt;
  mso-bidi-font-size:10.0pt;font-family:"Times New Roman",serif;mso-fareast-font-family:
  等线;mso-font-kerning:0pt'>9.840</span></b><!--[if gte msEquation 12]><m:oMath><b
   style='mso-bidi-font-weight:normal'><span lang=EN-US style='font-size:6.5pt;
   mso-bidi-font-size:10.0pt;font-family:"Cambria Math",serif;mso-fareast-font-family:
   等线;mso-bidi-font-family:"Times New Roman";mso-font-kerning:0pt'><m:r><m:rPr><m:scr
      m:val="roman"/><m:sty m:val="b"/></m:rPr>↓</m:r></span></b></m:oMath><![endif]--><![if !msEquation]><span
  lang=EN-US style='font-size:10.5pt;mso-bidi-font-size:11.0pt;font-family:
  等线;mso-ascii-theme-font:minor-latin;mso-fareast-theme-font:minor-fareast;
  mso-hansi-theme-font:minor-latin;mso-bidi-font-family:"Times New Roman";
  mso-bidi-theme-font:minor-bidi;position:relative;top:5.5pt;mso-text-raise:
  -5.5pt;mso-ansi-language:EN-US;mso-fareast-language:ZH-CN;mso-bidi-language:
  AR-SA'><!--[if gte vml 1]><v:shape id="_x0000_i1025" type="#_x0000_t75"
   style='width:2.85pt;height:15.7pt'>
   <v:imagedata src="Table_readme.files/image003.png" o:title="" chromakey="white"/>
  </v:shape><![endif]--><![if !vml]><img width=4 height=21
  src="Table_readme.files/image004.png" v:shapes="_x0000_i1025"><![endif]></span><![endif]><b
  style='mso-bidi-font-weight:normal'><span lang=EN-US style='font-size:6.5pt;
  mso-bidi-font-size:10.0pt;font-family:"Times New Roman",serif;mso-fareast-font-family:
  等线;mso-font-kerning:0pt'><o:p></o:p></span></b></p>
  </td>
  <td width=39 style='width:29.55pt;border:none;border-bottom:solid windowtext 1.0pt;
  mso-border-bottom-alt:solid windowtext .5pt;padding:0cm 5.4pt 0cm 5.4pt;
  height:1.0pt'>
  <p class=MsoNormal align=center style='text-align:center;mso-pagination:widow-orphan'><b
  style='mso-bidi-font-weight:normal'><span lang=EN-US style='font-size:6.5pt;
  mso-bidi-font-size:10.0pt;font-family:"Times New Roman",serif;mso-fareast-font-family:
  等线;mso-font-kerning:0pt'>8.636</span></b><!--[if gte msEquation 12]><m:oMath><b
   style='mso-bidi-font-weight:normal'><span lang=EN-US style='font-size:6.5pt;
   mso-bidi-font-size:10.0pt;font-family:"Cambria Math",serif;mso-fareast-font-family:
   等线;mso-bidi-font-family:"Times New Roman";mso-font-kerning:0pt'><m:r><m:rPr><m:scr
      m:val="roman"/><m:sty m:val="b"/></m:rPr>↓</m:r></span></b></m:oMath><![endif]--><![if !msEquation]><span
  lang=EN-US style='font-size:10.5pt;mso-bidi-font-size:11.0pt;font-family:
  等线;mso-ascii-theme-font:minor-latin;mso-fareast-theme-font:minor-fareast;
  mso-hansi-theme-font:minor-latin;mso-bidi-font-family:"Times New Roman";
  mso-bidi-theme-font:minor-bidi;position:relative;top:5.5pt;mso-text-raise:
  -5.5pt;mso-ansi-language:EN-US;mso-fareast-language:ZH-CN;mso-bidi-language:
  AR-SA'><!--[if gte vml 1]><v:shape id="_x0000_i1025" type="#_x0000_t75"
   style='width:2.85pt;height:15.7pt'>
   <v:imagedata src="Table_readme.files/image003.png" o:title="" chromakey="white"/>
  </v:shape><![endif]--><![if !vml]><img width=4 height=21
  src="Table_readme.files/image004.png" v:shapes="_x0000_i1025"><![endif]></span><![endif]><b
  style='mso-bidi-font-weight:normal'><span lang=EN-US style='font-size:6.5pt;
  mso-bidi-font-size:10.0pt;font-family:"Times New Roman",serif;mso-fareast-font-family:
  等线;mso-font-kerning:0pt'><o:p></o:p></span></b></p>
  </td>
  <td width=39 valign=top style='width:29.55pt;border-top:none;border-left:
  none;border-bottom:solid windowtext 1.0pt;border-right:solid windowtext 1.0pt;
  mso-border-bottom-alt:solid windowtext .5pt;mso-border-right-alt:solid windowtext .5pt;
  padding:0cm 5.4pt 0cm 5.4pt;height:1.0pt'>
  <p class=MsoNormal align=center style='text-align:center;mso-pagination:widow-orphan'><b
  style='mso-bidi-font-weight:normal'><span lang=EN-US style='font-size:6.5pt;
  mso-bidi-font-size:10.0pt;font-family:"Times New Roman",serif;mso-fareast-font-family:
  等线;mso-font-kerning:0pt'>9.037</span></b><!--[if gte msEquation 12]><m:oMath><b
   style='mso-bidi-font-weight:normal'><span lang=EN-US style='font-size:6.5pt;
   mso-bidi-font-size:10.0pt;font-family:"Cambria Math",serif;mso-fareast-font-family:
   等线;mso-bidi-font-family:"Times New Roman";mso-font-kerning:0pt'><m:r><m:rPr><m:scr
      m:val="roman"/><m:sty m:val="b"/></m:rPr>↓</m:r></span></b></m:oMath><![endif]--><![if !msEquation]><span
  lang=EN-US style='font-size:10.5pt;mso-bidi-font-size:11.0pt;font-family:
  等线;mso-ascii-theme-font:minor-latin;mso-fareast-theme-font:minor-fareast;
  mso-hansi-theme-font:minor-latin;mso-bidi-font-family:"Times New Roman";
  mso-bidi-theme-font:minor-bidi;position:relative;top:5.5pt;mso-text-raise:
  -5.5pt;mso-ansi-language:EN-US;mso-fareast-language:ZH-CN;mso-bidi-language:
  AR-SA'><!--[if gte vml 1]><v:shape id="_x0000_i1025" type="#_x0000_t75"
   style='width:2.85pt;height:15.7pt'>
   <v:imagedata src="Table_readme.files/image003.png" o:title="" chromakey="white"/>
  </v:shape><![endif]--><![if !vml]><img width=4 height=21
  src="Table_readme.files/image004.png" v:shapes="_x0000_i1025"><![endif]></span><![endif]><b
  style='mso-bidi-font-weight:normal'><span lang=EN-US style='font-size:6.5pt;
  mso-bidi-font-size:10.0pt;font-family:"Times New Roman",serif;mso-fareast-font-family:
  等线;mso-font-kerning:0pt'><o:p></o:p></span></b></p>
  </td>
  <td style='mso-cell-special:placeholder;border:none;padding:0cm 0cm 0cm 0cm'
  width=4><p class='MsoNormal'>&nbsp;</td>
 </tr>
 <tr style='mso-yfti-irow:5;height:1.0pt;mso-row-margin-right:3.35pt'>
  <td width=75 nowrap style='width:56.45pt;border-top:none;border-left:solid windowtext 1.0pt;
  border-bottom:none;border-right:solid windowtext 1.0pt;mso-border-top-alt:
  solid windowtext .5pt;mso-border-top-alt:solid windowtext .5pt;mso-border-left-alt:
  solid windowtext .5pt;mso-border-right-alt:solid windowtext .5pt;padding:
  0cm 5.4pt 0cm 5.4pt;height:1.0pt'>
  <p class=MsoNormal align=center style='text-align:center;mso-pagination:widow-orphan'><span
  lang=EN-US style='font-size:6.5pt;mso-bidi-font-size:10.0pt;font-family:"Times New Roman",serif;
  mso-fareast-font-family:等线;mso-font-kerning:0pt'>DPARN<o:p></o:p></span></p>
  </td>
  <td width=39 nowrap style='width:29.5pt;border:none;mso-border-top-alt:solid windowtext .5pt;
  mso-border-left-alt:solid windowtext .5pt;padding:0cm 5.4pt 0cm 5.4pt;
  height:1.0pt'>
  <p class=MsoNormal align=center style='text-align:center;mso-pagination:widow-orphan'><span
  lang=EN-US style='font-size:6.5pt;mso-bidi-font-size:10.0pt;font-family:"Times New Roman",serif;
  mso-fareast-font-family:等线;mso-font-kerning:0pt'>0.752<o:p></o:p></span></p>
  </td>
  <td width=39 nowrap style='width:29.55pt;border:none;mso-border-top-alt:solid windowtext .5pt;
  padding:0cm 5.4pt 0cm 5.4pt;height:1.0pt'>
  <p class=MsoNormal align=center style='text-align:center;mso-pagination:widow-orphan'><span
  lang=EN-US style='font-size:6.5pt;mso-bidi-font-size:10.0pt;font-family:"Times New Roman",serif;
  mso-fareast-font-family:等线;mso-font-kerning:0pt'>0.858<o:p></o:p></span></p>
  </td>
  <td width=39 nowrap style='width:29.55pt;border:none;border-right:solid windowtext 1.0pt;
  mso-border-top-alt:solid windowtext .5pt;mso-border-top-alt:solid windowtext .5pt;
  mso-border-right-alt:solid windowtext .5pt;padding:0cm 5.4pt 0cm 5.4pt;
  height:1.0pt'>
  <p class=MsoNormal align=center style='text-align:center;mso-pagination:widow-orphan'><span
  lang=EN-US style='font-size:6.5pt;mso-bidi-font-size:10.0pt;font-family:"Times New Roman",serif;
  mso-fareast-font-family:等线;mso-font-kerning:0pt'>0.828<o:p></o:p></span></p>
  </td>
  <td width=39 nowrap valign=top style='width:29.5pt;border:none;mso-border-top-alt:
  solid windowtext .5pt;mso-border-left-alt:solid windowtext .5pt;padding:0cm 5.4pt 0cm 5.4pt;
  height:1.0pt'>
  <p class=MsoNormal align=center style='text-align:center;mso-pagination:widow-orphan'><span
  lang=EN-US style='font-size:6.5pt;mso-bidi-font-size:10.0pt;font-family:"Times New Roman",serif;
  mso-fareast-font-family:等线;mso-font-kerning:0pt'>8.461<o:p></o:p></span></p>
  </td>
  <td width=39 nowrap valign=top style='width:29.55pt;border:none;mso-border-top-alt:
  solid windowtext .5pt;padding:0cm 5.4pt 0cm 5.4pt;height:1.0pt'>
  <p class=MsoNormal align=center style='text-align:center;mso-pagination:widow-orphan'><span
  lang=EN-US style='font-size:6.5pt;mso-bidi-font-size:10.0pt;font-family:"Times New Roman",serif;
  mso-fareast-font-family:等线;mso-font-kerning:0pt'>13.71<o:p></o:p></span></p>
  </td>
  <td width=39 nowrap valign=top style='width:29.55pt;border:none;border-right:
  solid windowtext 1.0pt;mso-border-top-alt:solid windowtext .5pt;mso-border-top-alt:
  solid windowtext .5pt;mso-border-right-alt:solid windowtext .5pt;padding:
  0cm 5.4pt 0cm 5.4pt;height:1.0pt'>
  <p class=MsoNormal align=center style='text-align:center;mso-pagination:widow-orphan'><span
  lang=EN-US style='font-size:6.5pt;mso-bidi-font-size:10.0pt;font-family:"Times New Roman",serif;
  mso-fareast-font-family:等线;mso-font-kerning:0pt'>12.31<o:p></o:p></span></p>
  </td>
  <td width=79 nowrap colspan=2 style='width:59.1pt;border:none;border-right:
  solid windowtext 1.0pt;mso-border-top-alt:solid windowtext .5pt;mso-border-left-alt:
  solid windowtext .5pt;mso-border-top-alt:solid windowtext .5pt;mso-border-left-alt:
  solid windowtext .5pt;mso-border-right-alt:solid windowtext .5pt;padding:
  0cm 5.4pt 0cm 5.4pt;height:1.0pt'>
  <p class=MsoNormal align=center style='text-align:center;mso-pagination:widow-orphan'><span
  lang=EN-US style='font-size:6.5pt;mso-bidi-font-size:10.0pt;font-family:"Times New Roman",serif;
  mso-fareast-font-family:等线;mso-font-kerning:0pt'>DPARN<o:p></o:p></span></p>
  </td>
  <td width=39 style='width:29.5pt;border:none;mso-border-top-alt:solid windowtext .5pt;
  mso-border-left-alt:solid windowtext .5pt;padding:0cm 5.4pt 0cm 5.4pt;
  height:1.0pt'>
  <p class=MsoNormal align=center style='text-align:center;mso-pagination:widow-orphan'><span
  lang=EN-US style='font-size:6.5pt;mso-bidi-font-size:10.0pt;font-family:"Times New Roman",serif;
  mso-fareast-font-family:等线;mso-font-kerning:0pt'>10.92<o:p></o:p></span></p>
  </td>
  <td width=39 style='width:29.55pt;border:none;mso-border-top-alt:solid windowtext .5pt;
  padding:0cm 5.4pt 0cm 5.4pt;height:1.0pt'>
  <p class=MsoNormal align=center style='text-align:center;mso-pagination:widow-orphan'><span
  lang=EN-US style='font-size:6.5pt;mso-bidi-font-size:10.0pt;font-family:"Times New Roman",serif;
  mso-fareast-font-family:等线;mso-font-kerning:0pt'>13.11<o:p></o:p></span></p>
  </td>
  <td width=39 valign=top style='width:29.55pt;border:none;border-right:solid windowtext 1.0pt;
  mso-border-top-alt:solid windowtext .5pt;mso-border-top-alt:solid windowtext .5pt;
  mso-border-right-alt:solid windowtext .5pt;padding:0cm 5.4pt 0cm 5.4pt;
  height:1.0pt'>
  <p class=MsoNormal align=center style='text-align:center;mso-pagination:widow-orphan'><span
  lang=EN-US style='font-size:6.5pt;mso-bidi-font-size:10.0pt;font-family:"Times New Roman",serif;
  mso-fareast-font-family:等线;mso-font-kerning:0pt'>12.38<o:p></o:p></span></p>
  </td>
  <td style='mso-cell-special:placeholder;border:none;padding:0cm 0cm 0cm 0cm'
  width=4><p class='MsoNormal'>&nbsp;</td>
 </tr>
 <tr style='mso-yfti-irow:6;height:1.0pt;mso-row-margin-right:3.35pt'>
  <td width=75 nowrap style='width:56.45pt;border:solid windowtext 1.0pt;
  border-top:none;mso-border-left-alt:solid windowtext .5pt;mso-border-bottom-alt:
  solid windowtext .5pt;mso-border-right-alt:solid windowtext .5pt;padding:
  0cm 5.4pt 0cm 5.4pt;height:1.0pt'>
  <p class=MsoNormal align=center style='text-align:center;mso-pagination:widow-orphan'><span
  lang=EN-US style='font-size:6.5pt;mso-bidi-font-size:10.0pt;font-family:"Times New Roman",serif;
  mso-fareast-font-family:等线;mso-font-kerning:0pt'>DPARN-LSA<o:p></o:p></span></p>
  </td>
  <td width=39 nowrap style='width:29.5pt;border:none;border-bottom:solid windowtext 1.0pt;
  mso-border-left-alt:solid windowtext .5pt;mso-border-left-alt:solid windowtext .5pt;
  mso-border-bottom-alt:solid windowtext .5pt;padding:0cm 5.4pt 0cm 5.4pt;
  height:1.0pt'>
  <p class=MsoNormal align=center style='text-align:center;mso-pagination:widow-orphan'><b
  style='mso-bidi-font-weight:normal'><span lang=EN-US style='font-size:6.5pt;
  mso-bidi-font-size:10.0pt;font-family:"Times New Roman",serif;mso-fareast-font-family:
  等线;mso-font-kerning:0pt'>0.757</span></b><!--[if gte msEquation 12]><m:oMath><b
   style='mso-bidi-font-weight:normal'><span lang=EN-US style='font-size:6.5pt;
   mso-bidi-font-size:10.0pt;font-family:"Cambria Math",serif;mso-fareast-font-family:
   等线;mso-bidi-font-family:"Times New Roman";mso-font-kerning:0pt'><m:r><m:rPr><m:scr
      m:val="roman"/><m:sty m:val="b"/></m:rPr>↑</m:r></span></b></m:oMath><![endif]--><![if !msEquation]><span
  lang=EN-US style='font-size:10.5pt;mso-bidi-font-size:11.0pt;font-family:
  等线;mso-ascii-theme-font:minor-latin;mso-fareast-theme-font:minor-fareast;
  mso-hansi-theme-font:minor-latin;mso-bidi-font-family:"Times New Roman";
  mso-bidi-theme-font:minor-bidi;position:relative;top:5.5pt;mso-text-raise:
  -5.5pt;mso-ansi-language:EN-US;mso-fareast-language:ZH-CN;mso-bidi-language:
  AR-SA'><!--[if gte vml 1]><v:shape id="_x0000_i1025" type="#_x0000_t75"
   style='width:2.85pt;height:15.7pt'>
   <v:imagedata src="Table_readme.files/image001.png" o:title="" chromakey="white"/>
  </v:shape><![endif]--><![if !vml]><img width=4 height=21
  src="Table_readme.files/image002.png" v:shapes="_x0000_i1025"><![endif]></span><![endif]><b
  style='mso-bidi-font-weight:normal'><span lang=EN-US style='font-size:6.5pt;
  mso-bidi-font-size:10.0pt;font-family:"Times New Roman",serif;mso-fareast-font-family:
  等线;mso-font-kerning:0pt'><o:p></o:p></span></b></p>
  </td>
  <td width=39 nowrap style='width:29.55pt;border:none;border-bottom:solid windowtext 1.0pt;
  mso-border-bottom-alt:solid windowtext .5pt;padding:0cm 5.4pt 0cm 5.4pt;
  height:1.0pt'>
  <p class=MsoNormal align=center style='text-align:center;mso-pagination:widow-orphan'><b
  style='mso-bidi-font-weight:normal'><span lang=EN-US style='font-size:6.5pt;
  mso-bidi-font-size:10.0pt;font-family:"Times New Roman",serif;mso-fareast-font-family:
  等线;mso-font-kerning:0pt'>0.861</span></b><!--[if gte msEquation 12]><m:oMath><b
   style='mso-bidi-font-weight:normal'><span lang=EN-US style='font-size:6.5pt;
   mso-bidi-font-size:10.0pt;font-family:"Cambria Math",serif;mso-fareast-font-family:
   等线;mso-bidi-font-family:"Times New Roman";mso-font-kerning:0pt'><m:r><m:rPr><m:scr
      m:val="roman"/><m:sty m:val="b"/></m:rPr>↑</m:r></span></b></m:oMath><![endif]--><![if !msEquation]><span
  lang=EN-US style='font-size:10.5pt;mso-bidi-font-size:11.0pt;font-family:
  等线;mso-ascii-theme-font:minor-latin;mso-fareast-theme-font:minor-fareast;
  mso-hansi-theme-font:minor-latin;mso-bidi-font-family:"Times New Roman";
  mso-bidi-theme-font:minor-bidi;position:relative;top:5.5pt;mso-text-raise:
  -5.5pt;mso-ansi-language:EN-US;mso-fareast-language:ZH-CN;mso-bidi-language:
  AR-SA'><!--[if gte vml 1]><v:shape id="_x0000_i1025" type="#_x0000_t75"
   style='width:2.85pt;height:15.7pt'>
   <v:imagedata src="Table_readme.files/image001.png" o:title="" chromakey="white"/>
  </v:shape><![endif]--><![if !vml]><img width=4 height=21
  src="Table_readme.files/image002.png" v:shapes="_x0000_i1025"><![endif]></span><![endif]><b
  style='mso-bidi-font-weight:normal'><span lang=EN-US style='font-size:6.5pt;
  mso-bidi-font-size:10.0pt;font-family:"Times New Roman",serif;mso-fareast-font-family:
  等线;mso-font-kerning:0pt'><o:p></o:p></span></b></p>
  </td>
  <td width=39 nowrap style='width:29.55pt;border-top:none;border-left:none;
  border-bottom:solid windowtext 1.0pt;border-right:solid windowtext 1.0pt;
  mso-border-bottom-alt:solid windowtext .5pt;mso-border-right-alt:solid windowtext .5pt;
  padding:0cm 5.4pt 0cm 5.4pt;height:1.0pt'>
  <p class=MsoNormal align=center style='text-align:center;mso-pagination:widow-orphan'><b
  style='mso-bidi-font-weight:normal'><span lang=EN-US style='font-size:6.5pt;
  mso-bidi-font-size:10.0pt;font-family:"Times New Roman",serif;mso-fareast-font-family:
  等线;mso-font-kerning:0pt'>0.831</span></b><!--[if gte msEquation 12]><m:oMath><b
   style='mso-bidi-font-weight:normal'><span lang=EN-US style='font-size:6.5pt;
   mso-bidi-font-size:10.0pt;font-family:"Cambria Math",serif;mso-fareast-font-family:
   等线;mso-bidi-font-family:"Times New Roman";mso-font-kerning:0pt'><m:r><m:rPr><m:scr
      m:val="roman"/><m:sty m:val="b"/></m:rPr>↑</m:r></span></b></m:oMath><![endif]--><![if !msEquation]><span
  lang=EN-US style='font-size:10.5pt;mso-bidi-font-size:11.0pt;font-family:
  等线;mso-ascii-theme-font:minor-latin;mso-fareast-theme-font:minor-fareast;
  mso-hansi-theme-font:minor-latin;mso-bidi-font-family:"Times New Roman";
  mso-bidi-theme-font:minor-bidi;position:relative;top:5.5pt;mso-text-raise:
  -5.5pt;mso-ansi-language:EN-US;mso-fareast-language:ZH-CN;mso-bidi-language:
  AR-SA'><!--[if gte vml 1]><v:shape id="_x0000_i1025" type="#_x0000_t75"
   style='width:2.85pt;height:15.7pt'>
   <v:imagedata src="Table_readme.files/image001.png" o:title="" chromakey="white"/>
  </v:shape><![endif]--><![if !vml]><img width=4 height=21
  src="Table_readme.files/image002.png" v:shapes="_x0000_i1025"><![endif]></span><![endif]><b
  style='mso-bidi-font-weight:normal'><span lang=EN-US style='font-size:6.5pt;
  mso-bidi-font-size:10.0pt;font-family:"Times New Roman",serif;mso-fareast-font-family:
  等线;mso-font-kerning:0pt'><o:p></o:p></span></b></p>
  </td>
  <td width=39 nowrap valign=top style='width:29.5pt;border:none;border-bottom:
  solid windowtext 1.0pt;mso-border-left-alt:solid windowtext .5pt;mso-border-left-alt:
  solid windowtext .5pt;mso-border-bottom-alt:solid windowtext .5pt;padding:
  0cm 5.4pt 0cm 5.4pt;height:1.0pt'>
  <p class=MsoNormal align=center style='text-align:center;mso-pagination:widow-orphan'><b
  style='mso-bidi-font-weight:normal'><span lang=EN-US style='font-size:6.5pt;
  mso-bidi-font-size:10.0pt;font-family:"Times New Roman",serif;mso-fareast-font-family:
  等线;mso-font-kerning:0pt'>8.617</span></b><!--[if gte msEquation 12]><m:oMath><b
   style='mso-bidi-font-weight:normal'><span lang=EN-US style='font-size:6.5pt;
   mso-bidi-font-size:10.0pt;font-family:"Cambria Math",serif;mso-fareast-font-family:
   等线;mso-bidi-font-family:"Times New Roman";mso-font-kerning:0pt'><m:r><m:rPr><m:scr
      m:val="roman"/><m:sty m:val="b"/></m:rPr>↑</m:r></span></b></m:oMath><![endif]--><![if !msEquation]><span
  lang=EN-US style='font-size:10.5pt;mso-bidi-font-size:11.0pt;font-family:
  等线;mso-ascii-theme-font:minor-latin;mso-fareast-theme-font:minor-fareast;
  mso-hansi-theme-font:minor-latin;mso-bidi-font-family:"Times New Roman";
  mso-bidi-theme-font:minor-bidi;position:relative;top:5.5pt;mso-text-raise:
  -5.5pt;mso-ansi-language:EN-US;mso-fareast-language:ZH-CN;mso-bidi-language:
  AR-SA'><!--[if gte vml 1]><v:shape id="_x0000_i1025" type="#_x0000_t75"
   style='width:2.85pt;height:15.7pt'>
   <v:imagedata src="Table_readme.files/image001.png" o:title="" chromakey="white"/>
  </v:shape><![endif]--><![if !vml]><img width=4 height=21
  src="Table_readme.files/image002.png" v:shapes="_x0000_i1025"><![endif]></span><![endif]><b
  style='mso-bidi-font-weight:normal'><span lang=EN-US style='font-size:6.5pt;
  mso-bidi-font-size:10.0pt;font-family:"Times New Roman",serif;mso-fareast-font-family:
  等线;mso-font-kerning:0pt'><o:p></o:p></span></b></p>
  </td>
  <td width=39 nowrap valign=top style='width:29.55pt;border:none;border-bottom:
  solid windowtext 1.0pt;mso-border-bottom-alt:solid windowtext .5pt;
  padding:0cm 5.4pt 0cm 5.4pt;height:1.0pt'>
  <p class=MsoNormal align=center style='text-align:center;mso-pagination:widow-orphan'><b
  style='mso-bidi-font-weight:normal'><span lang=EN-US style='font-size:6.5pt;
  mso-bidi-font-size:10.0pt;font-family:"Times New Roman",serif;mso-fareast-font-family:
  等线;mso-font-kerning:0pt'>13.84</span></b><!--[if gte msEquation 12]><m:oMath><b
   style='mso-bidi-font-weight:normal'><span lang=EN-US style='font-size:6.5pt;
   mso-bidi-font-size:10.0pt;font-family:"Cambria Math",serif;mso-fareast-font-family:
   等线;mso-bidi-font-family:"Times New Roman";mso-font-kerning:0pt'><m:r><m:rPr><m:scr
      m:val="roman"/><m:sty m:val="b"/></m:rPr>↑</m:r></span></b></m:oMath><![endif]--><![if !msEquation]><span
  lang=EN-US style='font-size:10.5pt;mso-bidi-font-size:11.0pt;font-family:
  等线;mso-ascii-theme-font:minor-latin;mso-fareast-theme-font:minor-fareast;
  mso-hansi-theme-font:minor-latin;mso-bidi-font-family:"Times New Roman";
  mso-bidi-theme-font:minor-bidi;position:relative;top:5.5pt;mso-text-raise:
  -5.5pt;mso-ansi-language:EN-US;mso-fareast-language:ZH-CN;mso-bidi-language:
  AR-SA'><!--[if gte vml 1]><v:shape id="_x0000_i1025" type="#_x0000_t75"
   style='width:2.85pt;height:15.7pt'>
   <v:imagedata src="Table_readme.files/image001.png" o:title="" chromakey="white"/>
  </v:shape><![endif]--><![if !vml]><img width=4 height=21
  src="Table_readme.files/image002.png" v:shapes="_x0000_i1025"><![endif]></span><![endif]><b
  style='mso-bidi-font-weight:normal'><span lang=EN-US style='font-size:6.5pt;
  mso-bidi-font-size:10.0pt;font-family:"Times New Roman",serif;mso-fareast-font-family:
  等线;mso-font-kerning:0pt'><o:p></o:p></span></b></p>
  </td>
  <td width=39 nowrap valign=top style='width:29.55pt;border-top:none;
  border-left:none;border-bottom:solid windowtext 1.0pt;border-right:solid windowtext 1.0pt;
  mso-border-bottom-alt:solid windowtext .5pt;mso-border-right-alt:solid windowtext .5pt;
  padding:0cm 5.4pt 0cm 5.4pt;height:1.0pt'>
  <p class=MsoNormal align=center style='text-align:center;mso-pagination:widow-orphan'><b
  style='mso-bidi-font-weight:normal'><span lang=EN-US style='font-size:6.5pt;
  mso-bidi-font-size:10.0pt;font-family:"Times New Roman",serif;mso-fareast-font-family:
  等线;mso-font-kerning:0pt'>12.47</span></b><!--[if gte msEquation 12]><m:oMath><b
   style='mso-bidi-font-weight:normal'><span lang=EN-US style='font-size:6.5pt;
   mso-bidi-font-size:10.0pt;font-family:"Cambria Math",serif;mso-fareast-font-family:
   等线;mso-bidi-font-family:"Times New Roman";mso-font-kerning:0pt'><m:r><m:rPr><m:scr
      m:val="roman"/><m:sty m:val="b"/></m:rPr>↑</m:r></span></b></m:oMath><![endif]--><![if !msEquation]><span
  lang=EN-US style='font-size:10.5pt;mso-bidi-font-size:11.0pt;font-family:
  等线;mso-ascii-theme-font:minor-latin;mso-fareast-theme-font:minor-fareast;
  mso-hansi-theme-font:minor-latin;mso-bidi-font-family:"Times New Roman";
  mso-bidi-theme-font:minor-bidi;position:relative;top:5.5pt;mso-text-raise:
  -5.5pt;mso-ansi-language:EN-US;mso-fareast-language:ZH-CN;mso-bidi-language:
  AR-SA'><!--[if gte vml 1]><v:shape id="_x0000_i1025" type="#_x0000_t75"
   style='width:2.85pt;height:15.7pt'>
   <v:imagedata src="Table_readme.files/image001.png" o:title="" chromakey="white"/>
  </v:shape><![endif]--><![if !vml]><img width=4 height=21
  src="Table_readme.files/image002.png" v:shapes="_x0000_i1025"><![endif]></span><![endif]><b
  style='mso-bidi-font-weight:normal'><span lang=EN-US style='font-size:6.5pt;
  mso-bidi-font-size:10.0pt;font-family:"Times New Roman",serif;mso-fareast-font-family:
  等线;mso-font-kerning:0pt'><o:p></o:p></span></b></p>
  </td>
  <td width=79 nowrap colspan=2 style='width:59.1pt;border-top:none;border-left:
  none;border-bottom:solid windowtext 1.0pt;border-right:solid windowtext 1.0pt;
  mso-border-left-alt:solid windowtext .5pt;mso-border-left-alt:solid windowtext .5pt;
  mso-border-bottom-alt:solid windowtext .5pt;mso-border-right-alt:solid windowtext .5pt;
  padding:0cm 5.4pt 0cm 5.4pt;height:1.0pt'>
  <p class=MsoNormal align=center style='text-align:center;mso-pagination:widow-orphan'><span
  lang=EN-US style='font-size:6.5pt;mso-bidi-font-size:10.0pt;font-family:"Times New Roman",serif;
  mso-fareast-font-family:等线;mso-font-kerning:0pt'>DPARN-LSA<o:p></o:p></span></p>
  </td>
  <td width=39 style='width:29.5pt;border:none;border-bottom:solid windowtext 1.0pt;
  mso-border-left-alt:solid windowtext .5pt;mso-border-left-alt:solid windowtext .5pt;
  mso-border-bottom-alt:solid windowtext .5pt;padding:0cm 5.4pt 0cm 5.4pt;
  height:1.0pt'>
  <p class=MsoNormal align=center style='text-align:center;mso-pagination:widow-orphan'><b
  style='mso-bidi-font-weight:normal'><span lang=EN-US style='font-size:6.5pt;
  mso-bidi-font-size:10.0pt;font-family:"Times New Roman",serif;mso-fareast-font-family:
  等线;mso-font-kerning:0pt'>10.76</span></b><!--[if gte msEquation 12]><m:oMath><b
   style='mso-bidi-font-weight:normal'><span lang=EN-US style='font-size:6.5pt;
   mso-bidi-font-size:10.0pt;font-family:"Cambria Math",serif;mso-fareast-font-family:
   等线;mso-bidi-font-family:"Times New Roman";mso-font-kerning:0pt'><m:r><m:rPr><m:scr
      m:val="roman"/><m:sty m:val="b"/></m:rPr>↓</m:r></span></b></m:oMath><![endif]--><![if !msEquation]><span
  lang=EN-US style='font-size:10.5pt;mso-bidi-font-size:11.0pt;font-family:
  等线;mso-ascii-theme-font:minor-latin;mso-fareast-theme-font:minor-fareast;
  mso-hansi-theme-font:minor-latin;mso-bidi-font-family:"Times New Roman";
  mso-bidi-theme-font:minor-bidi;position:relative;top:5.5pt;mso-text-raise:
  -5.5pt;mso-ansi-language:EN-US;mso-fareast-language:ZH-CN;mso-bidi-language:
  AR-SA'><!--[if gte vml 1]><v:shape id="_x0000_i1025" type="#_x0000_t75"
   style='width:2.85pt;height:15.7pt'>
   <v:imagedata src="Table_readme.files/image003.png" o:title="" chromakey="white"/>
  </v:shape><![endif]--><![if !vml]><img width=4 height=21
  src="Table_readme.files/image004.png" v:shapes="_x0000_i1025"><![endif]></span><![endif]><b
  style='mso-bidi-font-weight:normal'><span lang=EN-US style='font-size:6.5pt;
  mso-bidi-font-size:10.0pt;font-family:"Times New Roman",serif;mso-fareast-font-family:
  等线;mso-font-kerning:0pt'><o:p></o:p></span></b></p>
  </td>
  <td width=39 style='width:29.55pt;border:none;border-bottom:solid windowtext 1.0pt;
  mso-border-bottom-alt:solid windowtext .5pt;padding:0cm 5.4pt 0cm 5.4pt;
  height:1.0pt'>
  <p class=MsoNormal align=center style='text-align:center;mso-pagination:widow-orphan'><b
  style='mso-bidi-font-weight:normal'><span lang=EN-US style='font-size:6.5pt;
  mso-bidi-font-size:10.0pt;font-family:"Times New Roman",serif;mso-fareast-font-family:
  等线;mso-font-kerning:0pt'>12.99</span></b><!--[if gte msEquation 12]><m:oMath><b
   style='mso-bidi-font-weight:normal'><span lang=EN-US style='font-size:6.5pt;
   mso-bidi-font-size:10.0pt;font-family:"Cambria Math",serif;mso-fareast-font-family:
   等线;mso-bidi-font-family:"Times New Roman";mso-font-kerning:0pt'><m:r><m:rPr><m:scr
      m:val="roman"/><m:sty m:val="b"/></m:rPr>↓</m:r></span></b></m:oMath><![endif]--><![if !msEquation]><span
  lang=EN-US style='font-size:10.5pt;mso-bidi-font-size:11.0pt;font-family:
  等线;mso-ascii-theme-font:minor-latin;mso-fareast-theme-font:minor-fareast;
  mso-hansi-theme-font:minor-latin;mso-bidi-font-family:"Times New Roman";
  mso-bidi-theme-font:minor-bidi;position:relative;top:5.5pt;mso-text-raise:
  -5.5pt;mso-ansi-language:EN-US;mso-fareast-language:ZH-CN;mso-bidi-language:
  AR-SA'><!--[if gte vml 1]><v:shape id="_x0000_i1025" type="#_x0000_t75"
   style='width:2.85pt;height:15.7pt'>
   <v:imagedata src="Table_readme.files/image003.png" o:title="" chromakey="white"/>
  </v:shape><![endif]--><![if !vml]><img width=4 height=21
  src="Table_readme.files/image004.png" v:shapes="_x0000_i1025"><![endif]></span><![endif]><b
  style='mso-bidi-font-weight:normal'><span lang=EN-US style='font-size:6.5pt;
  mso-bidi-font-size:10.0pt;font-family:"Times New Roman",serif;mso-fareast-font-family:
  等线;mso-font-kerning:0pt'><o:p></o:p></span></b></p>
  </td>
  <td width=39 valign=top style='width:29.55pt;border-top:none;border-left:
  none;border-bottom:solid windowtext 1.0pt;border-right:solid windowtext 1.0pt;
  mso-border-bottom-alt:solid windowtext .5pt;mso-border-right-alt:solid windowtext .5pt;
  padding:0cm 5.4pt 0cm 5.4pt;height:1.0pt'>
  <p class=MsoNormal align=center style='text-align:center;mso-pagination:widow-orphan'><b
  style='mso-bidi-font-weight:normal'><span lang=EN-US style='font-size:6.5pt;
  mso-bidi-font-size:10.0pt;font-family:"Times New Roman",serif;mso-fareast-font-family:
  等线;mso-font-kerning:0pt'>12.25</span></b><!--[if gte msEquation 12]><m:oMath><b
   style='mso-bidi-font-weight:normal'><span lang=EN-US style='font-size:6.5pt;
   mso-bidi-font-size:10.0pt;font-family:"Cambria Math",serif;mso-fareast-font-family:
   等线;mso-bidi-font-family:"Times New Roman";mso-font-kerning:0pt'><m:r><m:rPr><m:scr
      m:val="roman"/><m:sty m:val="b"/></m:rPr>↓</m:r></span></b></m:oMath><![endif]--><![if !msEquation]><span
  lang=EN-US style='font-size:10.5pt;mso-bidi-font-size:11.0pt;font-family:
  等线;mso-ascii-theme-font:minor-latin;mso-fareast-theme-font:minor-fareast;
  mso-hansi-theme-font:minor-latin;mso-bidi-font-family:"Times New Roman";
  mso-bidi-theme-font:minor-bidi;position:relative;top:5.5pt;mso-text-raise:
  -5.5pt;mso-ansi-language:EN-US;mso-fareast-language:ZH-CN;mso-bidi-language:
  AR-SA'><!--[if gte vml 1]><v:shape id="_x0000_i1025" type="#_x0000_t75"
   style='width:2.85pt;height:15.7pt'>
   <v:imagedata src="Table_readme.files/image003.png" o:title="" chromakey="white"/>
  </v:shape><![endif]--><![if !vml]><img width=4 height=21
  src="Table_readme.files/image004.png" v:shapes="_x0000_i1025"><![endif]></span><![endif]><b
  style='mso-bidi-font-weight:normal'><span lang=EN-US style='font-size:6.5pt;
  mso-bidi-font-size:10.0pt;font-family:"Times New Roman",serif;mso-fareast-font-family:
  等线;mso-font-kerning:0pt'><o:p></o:p></span></b></p>
  </td>
  <td style='mso-cell-special:placeholder;border:none;padding:0cm 0cm 0cm 0cm'
  width=4><p class='MsoNormal'>&nbsp;</td>
 </tr>
 <tr style='mso-yfti-irow:7;height:15.0pt'>
  <td width=552 nowrap colspan=14 style='width:414.2pt;border:none;padding:
  0cm 5.4pt 0cm 5.4pt;height:15.0pt'></td>
 </tr>
 <tr style='mso-yfti-irow:8;height:15.0pt;mso-row-margin-right:3.35pt'>
  <td width=75 nowrap style='width:56.45pt;border:solid windowtext 1.0pt;
  mso-border-alt:solid windowtext .5pt;padding:0cm 5.4pt 0cm 5.4pt;height:15.0pt'>
  <p class=MsoNormal align=center style='text-align:center;mso-pagination:widow-orphan'><span
  lang=EN-US style='font-size:5.5pt;mso-bidi-font-size:10.0pt;font-family:"Times New Roman",serif;
  mso-fareast-font-family:等线;mso-font-kerning:0pt'>Wideband Metrics</span><span
  lang=EN-US style='font-size:7.5pt;mso-bidi-font-size:10.0pt;font-family:"Times New Roman",serif;
  mso-fareast-font-family:等线;mso-font-kerning:0pt'><o:p></o:p></span></p>
  </td>
  <td width=118 nowrap colspan=3 style='width:88.6pt;border:solid windowtext 1.0pt;
  border-left:none;mso-border-left-alt:solid windowtext .5pt;mso-border-alt:
  solid windowtext .5pt;padding:0cm 5.4pt 0cm 5.4pt;height:15.0pt'>
  <p class=MsoNormal align=center style='text-align:center;mso-pagination:widow-orphan'><span
  lang=EN-US style='font-size:7.5pt;mso-bidi-font-size:10.0pt;font-family:"Times New Roman",serif;
  mso-fareast-font-family:等线;mso-font-kerning:0pt'>PESQ<o:p></o:p></span></p>
  </td>
  <td width=118 nowrap colspan=3 style='width:88.6pt;border:solid windowtext 1.0pt;
  border-left:none;mso-border-left-alt:solid windowtext .5pt;mso-border-alt:
  solid windowtext .5pt;padding:0cm 5.4pt 0cm 5.4pt;height:15.0pt'>
  <p class=MsoNormal align=center style='text-align:center;mso-pagination:widow-orphan'><span
  lang=EN-US style='font-size:7.5pt;mso-bidi-font-size:10.0pt;font-family:"Times New Roman",serif;
  mso-fareast-font-family:等线;mso-font-kerning:0pt'>CSIG<o:p></o:p></span></p>
  </td>
  <td width=118 nowrap colspan=3 style='width:88.6pt;border:solid windowtext 1.0pt;
  border-left:none;mso-border-left-alt:solid windowtext .5pt;mso-border-alt:
  solid windowtext .5pt;padding:0cm 5.4pt 0cm 5.4pt;height:15.0pt'>
  <p class=MsoNormal align=center style='text-align:center;mso-pagination:widow-orphan'><span
  lang=EN-US style='font-size:7.5pt;mso-bidi-font-size:10.0pt;font-family:"Times New Roman",serif;
  mso-fareast-font-family:等线;mso-font-kerning:0pt'>CBAK<o:p></o:p></span></p>
  </td>
  <td width=118 colspan=3 valign=top style='width:88.6pt;border:solid windowtext 1.0pt;
  border-left:none;mso-border-left-alt:solid windowtext .5pt;mso-border-alt:
  solid windowtext .5pt;padding:0cm 5.4pt 0cm 5.4pt;height:15.0pt'>
  <p class=MsoNormal align=center style='text-align:center;mso-pagination:widow-orphan'><span
  lang=EN-US style='font-size:7.5pt;mso-bidi-font-size:10.0pt;font-family:"Times New Roman",serif;
  mso-fareast-font-family:等线;mso-font-kerning:0pt'>COVL<o:p></o:p></span></p>
  </td>
  <td style='mso-cell-special:placeholder;border:none;padding:0cm 0cm 0cm 0cm'
  width=4><p class='MsoNormal'>&nbsp;</td>
 </tr>
 <tr style='mso-yfti-irow:9;height:15.0pt;mso-row-margin-right:3.35pt'>
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
  <td width=39 nowrap style='width:29.5pt;border:none;border-bottom:solid windowtext 1.0pt;
  mso-border-top-alt:solid windowtext .5pt;mso-border-left-alt:solid windowtext .5pt;
  mso-border-top-alt:solid windowtext .5pt;mso-border-left-alt:solid windowtext .5pt;
  mso-border-bottom-alt:solid windowtext .5pt;padding:0cm 5.4pt 0cm 5.4pt;
  height:15.0pt'>
  <p class=MsoNormal align=center style='text-align:center;mso-pagination:widow-orphan'><span
  lang=EN-US style='font-size:7.5pt;font-family:"Times New Roman",serif;
  mso-fareast-font-family:等线;mso-font-kerning:0pt'>-5~0<o:p></o:p></span></p>
  </td>
  <td width=39 nowrap style='width:29.55pt;border:none;border-bottom:solid windowtext 1.0pt;
  mso-border-top-alt:solid windowtext .5pt;mso-border-top-alt:solid windowtext .5pt;
  mso-border-bottom-alt:solid windowtext .5pt;padding:0cm 5.4pt 0cm 5.4pt;
  height:15.0pt'>
  <p class=MsoNormal align=center style='text-align:center;mso-pagination:widow-orphan'><span
  lang=EN-US style='font-size:7.5pt;font-family:"Times New Roman",serif;
  mso-fareast-font-family:等线;mso-font-kerning:0pt'>0~15<o:p></o:p></span></p>
  </td>
  <td width=39 nowrap style='width:29.55pt;border-top:none;border-left:none;
  border-bottom:solid windowtext 1.0pt;border-right:solid windowtext 1.0pt;
  mso-border-top-alt:solid windowtext .5pt;mso-border-top-alt:solid windowtext .5pt;
  mso-border-bottom-alt:solid windowtext .5pt;mso-border-right-alt:solid windowtext .5pt;
  padding:0cm 5.4pt 0cm 5.4pt;height:15.0pt'>
  <p class=MsoNormal align=center style='text-align:center;mso-pagination:widow-orphan'><span
  class=SpellE><span lang=EN-US style='font-size:7.5pt;font-family:"Times New Roman",serif;
  mso-fareast-font-family:等线;mso-font-kerning:0pt'>Ovrl</span></span><span
  lang=EN-US style='font-size:7.5pt;font-family:"Times New Roman",serif;
  mso-fareast-font-family:等线;mso-font-kerning:0pt'>.<o:p></o:p></span></p>
  </td>
  <td width=39 nowrap style='width:29.5pt;border:none;border-bottom:solid windowtext 1.0pt;
  mso-border-top-alt:solid windowtext .5pt;mso-border-left-alt:solid windowtext .5pt;
  mso-border-top-alt:solid windowtext .5pt;mso-border-left-alt:solid windowtext .5pt;
  mso-border-bottom-alt:solid windowtext .5pt;padding:0cm 5.4pt 0cm 5.4pt;
  height:15.0pt'>
  <p class=MsoNormal align=center style='text-align:center;mso-pagination:widow-orphan'><span
  lang=EN-US style='font-size:7.5pt;font-family:"Times New Roman",serif;
  mso-fareast-font-family:等线;mso-font-kerning:0pt'>-5~0<o:p></o:p></span></p>
  </td>
  <td width=39 nowrap style='width:29.55pt;border:none;border-bottom:solid windowtext 1.0pt;
  mso-border-top-alt:solid windowtext .5pt;mso-border-top-alt:solid windowtext .5pt;
  mso-border-bottom-alt:solid windowtext .5pt;padding:0cm 5.4pt 0cm 5.4pt;
  height:15.0pt'>
  <p class=MsoNormal align=center style='text-align:center;mso-pagination:widow-orphan'><span
  lang=EN-US style='font-size:7.5pt;font-family:"Times New Roman",serif;
  mso-fareast-font-family:等线;mso-font-kerning:0pt'>0~15<o:p></o:p></span></p>
  </td>
  <td width=39 nowrap style='width:29.55pt;border-top:none;border-left:none;
  border-bottom:solid windowtext 1.0pt;border-right:solid windowtext 1.0pt;
  mso-border-top-alt:solid windowtext .5pt;mso-border-top-alt:solid windowtext .5pt;
  mso-border-bottom-alt:solid windowtext .5pt;mso-border-right-alt:solid windowtext .5pt;
  padding:0cm 5.4pt 0cm 5.4pt;height:15.0pt'>
  <p class=MsoNormal align=center style='text-align:center;mso-pagination:widow-orphan'><span
  class=SpellE><span lang=EN-US style='font-size:7.5pt;font-family:"Times New Roman",serif;
  mso-fareast-font-family:等线;mso-font-kerning:0pt'>Ovrl</span></span><span
  lang=EN-US style='font-size:7.5pt;font-family:"Times New Roman",serif;
  mso-fareast-font-family:等线;mso-font-kerning:0pt'>.<o:p></o:p></span></p>
  </td>
  <td width=39 nowrap style='width:29.5pt;border:none;border-bottom:solid windowtext 1.0pt;
  mso-border-top-alt:solid windowtext .5pt;mso-border-left-alt:solid windowtext .5pt;
  mso-border-top-alt:solid windowtext .5pt;mso-border-left-alt:solid windowtext .5pt;
  mso-border-bottom-alt:solid windowtext .5pt;padding:0cm 5.4pt 0cm 5.4pt;
  height:15.0pt'>
  <p class=MsoNormal align=center style='text-align:center;mso-pagination:widow-orphan'><span
  lang=EN-US style='font-size:7.5pt;font-family:"Times New Roman",serif;
  mso-fareast-font-family:等线;mso-font-kerning:0pt'>-5~0<o:p></o:p></span></p>
  </td>
  <td width=39 nowrap style='width:29.55pt;border:none;border-bottom:solid windowtext 1.0pt;
  mso-border-top-alt:solid windowtext .5pt;mso-border-top-alt:solid windowtext .5pt;
  mso-border-bottom-alt:solid windowtext .5pt;padding:0cm 5.4pt 0cm 5.4pt;
  height:15.0pt'>
  <p class=MsoNormal align=center style='text-align:center;mso-pagination:widow-orphan'><span
  lang=EN-US style='font-size:7.5pt;font-family:"Times New Roman",serif;
  mso-fareast-font-family:等线;mso-font-kerning:0pt'>0~15<o:p></o:p></span></p>
  </td>
  <td width=39 nowrap style='width:29.55pt;border-top:none;border-left:none;
  border-bottom:solid windowtext 1.0pt;border-right:solid windowtext 1.0pt;
  mso-border-top-alt:solid windowtext .5pt;mso-border-top-alt:solid windowtext .5pt;
  mso-border-bottom-alt:solid windowtext .5pt;mso-border-right-alt:solid windowtext .5pt;
  padding:0cm 5.4pt 0cm 5.4pt;height:15.0pt'>
  <p class=MsoNormal align=center style='text-align:center;mso-pagination:widow-orphan'><span
  class=SpellE><span lang=EN-US style='font-size:7.5pt;font-family:"Times New Roman",serif;
  mso-fareast-font-family:等线;mso-font-kerning:0pt'>Ovrl</span></span><span
  lang=EN-US style='font-size:7.5pt;font-family:"Times New Roman",serif;
  mso-fareast-font-family:等线;mso-font-kerning:0pt'>.<o:p></o:p></span></p>
  </td>
  <td width=39 style='width:29.5pt;border:none;border-bottom:solid windowtext 1.0pt;
  mso-border-top-alt:solid windowtext .5pt;mso-border-left-alt:solid windowtext .5pt;
  mso-border-top-alt:solid windowtext .5pt;mso-border-left-alt:solid windowtext .5pt;
  mso-border-bottom-alt:solid windowtext .5pt;padding:0cm 5.4pt 0cm 5.4pt;
  height:15.0pt'>
  <p class=MsoNormal align=center style='text-align:center;mso-pagination:widow-orphan'><span
  lang=EN-US style='font-size:7.5pt;font-family:"Times New Roman",serif;
  mso-fareast-font-family:等线;mso-font-kerning:0pt'>-5~0<o:p></o:p></span></p>
  </td>
  <td width=39 style='width:29.55pt;border:none;border-bottom:solid windowtext 1.0pt;
  mso-border-top-alt:solid windowtext .5pt;mso-border-top-alt:solid windowtext .5pt;
  mso-border-bottom-alt:solid windowtext .5pt;padding:0cm 5.4pt 0cm 5.4pt;
  height:15.0pt'>
  <p class=MsoNormal align=center style='text-align:center;mso-pagination:widow-orphan'><span
  lang=EN-US style='font-size:7.5pt;font-family:"Times New Roman",serif;
  mso-fareast-font-family:等线;mso-font-kerning:0pt'>0~15<o:p></o:p></span></p>
  </td>
  <td width=39 style='width:29.55pt;border-top:none;border-left:none;
  border-bottom:solid windowtext 1.0pt;border-right:solid windowtext 1.0pt;
  mso-border-top-alt:solid windowtext .5pt;mso-border-top-alt:solid windowtext .5pt;
  mso-border-bottom-alt:solid windowtext .5pt;mso-border-right-alt:solid windowtext .5pt;
  padding:0cm 5.4pt 0cm 5.4pt;height:15.0pt'>
  <p class=MsoNormal align=center style='text-align:center;mso-pagination:widow-orphan'><span
  class=SpellE><span lang=EN-US style='font-size:7.5pt;font-family:"Times New Roman",serif;
  mso-fareast-font-family:等线;mso-font-kerning:0pt'>Ovrl</span></span><span
  lang=EN-US style='font-size:7.5pt;font-family:"Times New Roman",serif;
  mso-fareast-font-family:等线;mso-font-kerning:0pt'>.<o:p></o:p></span></p>
  </td>
  <td style='mso-cell-special:placeholder;border:none;padding:0cm 0cm 0cm 0cm'
  width=4><p class='MsoNormal'>&nbsp;</td>
 </tr>
 <tr style='mso-yfti-irow:10;height:15.6pt;mso-row-margin-right:3.35pt'>
  <td width=75 nowrap style='width:56.45pt;border:solid windowtext 1.0pt;
  border-top:none;mso-border-top-alt:solid windowtext .5pt;mso-border-alt:solid windowtext .5pt;
  padding:0cm 5.4pt 0cm 5.4pt;height:15.6pt'>
  <p class=MsoNormal align=center style='text-align:center;mso-pagination:widow-orphan'><span
  lang=EN-US style='font-size:6.5pt;mso-bidi-font-size:10.0pt;font-family:"Times New Roman",serif;
  mso-fareast-font-family:等线;mso-font-kerning:0pt'>Noisy<o:p></o:p></span></p>
  </td>
  <td width=39 nowrap style='width:29.5pt;border:none;border-bottom:solid windowtext 1.0pt;
  mso-border-top-alt:solid windowtext .5pt;mso-border-left-alt:solid windowtext .5pt;
  mso-border-top-alt:solid windowtext .5pt;mso-border-left-alt:solid windowtext .5pt;
  mso-border-bottom-alt:solid windowtext .5pt;padding:0cm 5.4pt 0cm 5.4pt;
  height:15.6pt'>
  <p class=MsoNormal align=center style='text-align:center;mso-pagination:widow-orphan'><span
  lang=EN-US style='font-size:6.5pt;mso-bidi-font-size:10.0pt;font-family:"Times New Roman",serif;
  mso-fareast-font-family:等线;mso-font-kerning:0pt'>1.160</span><span
  lang=EN-US style='font-size:6.5pt;font-family:"Times New Roman",serif;
  mso-fareast-font-family:等线;mso-font-kerning:0pt'><o:p></o:p></span></p>
  </td>
  <td width=39 nowrap style='width:29.55pt;border:none;border-bottom:solid windowtext 1.0pt;
  mso-border-top-alt:solid windowtext .5pt;mso-border-top-alt:solid windowtext .5pt;
  mso-border-bottom-alt:solid windowtext .5pt;padding:0cm 5.4pt 0cm 5.4pt;
  height:15.6pt'>
  <p class=MsoNormal align=center style='text-align:center;mso-pagination:widow-orphan'><span
  lang=EN-US style='font-size:6.5pt;mso-bidi-font-size:10.0pt;font-family:"Times New Roman",serif;
  mso-fareast-font-family:等线;mso-font-kerning:0pt'>1.446</span><span
  lang=EN-US style='font-size:6.5pt;font-family:"Times New Roman",serif;
  mso-fareast-font-family:等线;mso-font-kerning:0pt'><o:p></o:p></span></p>
  </td>
  <td width=39 nowrap style='width:29.55pt;border-top:none;border-left:none;
  border-bottom:solid windowtext 1.0pt;border-right:solid windowtext 1.0pt;
  mso-border-top-alt:solid windowtext .5pt;mso-border-top-alt:solid windowtext .5pt;
  mso-border-bottom-alt:solid windowtext .5pt;mso-border-right-alt:solid windowtext .5pt;
  padding:0cm 5.4pt 0cm 5.4pt;height:15.6pt'>
  <p class=MsoNormal align=center style='text-align:center;mso-pagination:widow-orphan'><span
  lang=EN-US style='font-size:6.5pt;mso-bidi-font-size:10.0pt;font-family:"Times New Roman",serif;
  mso-fareast-font-family:等线;mso-font-kerning:0pt'>1.364</span><span
  lang=EN-US style='font-size:6.5pt;font-family:"Times New Roman",serif;
  mso-fareast-font-family:等线;mso-font-kerning:0pt'><o:p></o:p></span></p>
  </td>
  <td width=39 nowrap style='width:29.5pt;border:none;border-bottom:solid windowtext 1.0pt;
  mso-border-top-alt:solid windowtext .5pt;mso-border-left-alt:solid windowtext .5pt;
  mso-border-top-alt:solid windowtext .5pt;mso-border-left-alt:solid windowtext .5pt;
  mso-border-bottom-alt:solid windowtext .5pt;padding:0cm 5.4pt 0cm 5.4pt;
  height:15.6pt'>
  <p class=MsoNormal align=center style='text-align:center;mso-pagination:widow-orphan'><span
  lang=EN-US style='font-size:6.5pt;font-family:"Times New Roman",serif;
  mso-fareast-font-family:等线;mso-font-kerning:0pt'>2.023<o:p></o:p></span></p>
  </td>
  <td width=39 nowrap style='width:29.55pt;border:none;border-bottom:solid windowtext 1.0pt;
  mso-border-top-alt:solid windowtext .5pt;mso-border-top-alt:solid windowtext .5pt;
  mso-border-bottom-alt:solid windowtext .5pt;padding:0cm 5.4pt 0cm 5.4pt;
  height:15.6pt'>
  <p class=MsoNormal align=center style='text-align:center;mso-pagination:widow-orphan'><span
  lang=EN-US style='font-size:6.5pt;font-family:"Times New Roman",serif;
  mso-fareast-font-family:等线;mso-font-kerning:0pt'>2.719<o:p></o:p></span></p>
  </td>
  <td width=39 nowrap style='width:29.55pt;border-top:none;border-left:none;
  border-bottom:solid windowtext 1.0pt;border-right:solid windowtext 1.0pt;
  mso-border-top-alt:solid windowtext .5pt;mso-border-top-alt:solid windowtext .5pt;
  mso-border-bottom-alt:solid windowtext .5pt;mso-border-right-alt:solid windowtext .5pt;
  padding:0cm 5.4pt 0cm 5.4pt;height:15.6pt'>
  <p class=MsoNormal align=center style='text-align:center;mso-pagination:widow-orphan'><span
  lang=EN-US style='font-size:6.5pt;font-family:"Times New Roman",serif;
  mso-fareast-font-family:等线;mso-font-kerning:0pt'>2.517<o:p></o:p></span></p>
  </td>
  <td width=39 nowrap style='width:29.5pt;border:none;border-bottom:solid windowtext 1.0pt;
  mso-border-top-alt:solid windowtext .5pt;mso-border-left-alt:solid windowtext .5pt;
  mso-border-top-alt:solid windowtext .5pt;mso-border-left-alt:solid windowtext .5pt;
  mso-border-bottom-alt:solid windowtext .5pt;padding:0cm 5.4pt 0cm 5.4pt;
  height:15.6pt'>
  <p class=MsoNormal align=center style='text-align:center;mso-pagination:widow-orphan'><span
  lang=EN-US style='font-size:6.5pt;font-family:"Times New Roman",serif;
  mso-fareast-font-family:等线;mso-font-kerning:0pt'>1.833<o:p></o:p></span></p>
  </td>
  <td width=39 nowrap style='width:29.55pt;border:none;border-bottom:solid windowtext 1.0pt;
  mso-border-top-alt:solid windowtext .5pt;mso-border-top-alt:solid windowtext .5pt;
  mso-border-bottom-alt:solid windowtext .5pt;padding:0cm 5.4pt 0cm 5.4pt;
  height:15.6pt'>
  <p class=MsoNormal align=center style='text-align:center;mso-pagination:widow-orphan'><span
  lang=EN-US style='font-size:6.5pt;font-family:"Times New Roman",serif;
  mso-fareast-font-family:等线;mso-font-kerning:0pt'>2.481<o:p></o:p></span></p>
  </td>
  <td width=39 nowrap style='width:29.55pt;border-top:none;border-left:none;
  border-bottom:solid windowtext 1.0pt;border-right:solid windowtext 1.0pt;
  mso-border-top-alt:solid windowtext .5pt;mso-border-top-alt:solid windowtext .5pt;
  mso-border-bottom-alt:solid windowtext .5pt;mso-border-right-alt:solid windowtext .5pt;
  padding:0cm 5.4pt 0cm 5.4pt;height:15.6pt'>
  <p class=MsoNormal align=center style='text-align:center;mso-pagination:widow-orphan'><span
  lang=EN-US style='font-size:6.5pt;font-family:"Times New Roman",serif;
  mso-fareast-font-family:等线;mso-font-kerning:0pt'>2.293<o:p></o:p></span></p>
  </td>
  <td width=39 style='width:29.5pt;border:none;border-bottom:solid windowtext 1.0pt;
  mso-border-top-alt:solid windowtext .5pt;mso-border-left-alt:solid windowtext .5pt;
  mso-border-top-alt:solid windowtext .5pt;mso-border-left-alt:solid windowtext .5pt;
  mso-border-bottom-alt:solid windowtext .5pt;padding:0cm 5.4pt 0cm 5.4pt;
  height:15.6pt'>
  <p class=MsoNormal align=center style='text-align:center;mso-pagination:widow-orphan'><span
  lang=EN-US style='font-size:6.5pt;font-family:"Times New Roman",serif;
  mso-fareast-font-family:等线;mso-font-kerning:0pt'>1.571<o:p></o:p></span></p>
  </td>
  <td width=39 style='width:29.55pt;border:none;border-bottom:solid windowtext 1.0pt;
  mso-border-top-alt:solid windowtext .5pt;mso-border-top-alt:solid windowtext .5pt;
  mso-border-bottom-alt:solid windowtext .5pt;padding:0cm 5.4pt 0cm 5.4pt;
  height:15.6pt'>
  <p class=MsoNormal align=center style='text-align:center;mso-pagination:widow-orphan'><span
  lang=EN-US style='font-size:6.5pt;font-family:"Times New Roman",serif;
  mso-fareast-font-family:等线;mso-font-kerning:0pt'>2.095<o:p></o:p></span></p>
  </td>
  <td width=39 style='width:29.55pt;border-top:none;border-left:none;
  border-bottom:solid windowtext 1.0pt;border-right:solid windowtext 1.0pt;
  mso-border-top-alt:solid windowtext .5pt;mso-border-top-alt:solid windowtext .5pt;
  mso-border-bottom-alt:solid windowtext .5pt;mso-border-right-alt:solid windowtext .5pt;
  padding:0cm 5.4pt 0cm 5.4pt;height:15.6pt'>
  <p class=MsoNormal align=center style='text-align:center;mso-pagination:widow-orphan'><span
  lang=EN-US style='font-size:6.5pt;font-family:"Times New Roman",serif;
  mso-fareast-font-family:等线;mso-font-kerning:0pt'>1.943<o:p></o:p></span></p>
  </td>
  <td style='mso-cell-special:placeholder;border:none;padding:0cm 0cm 0cm 0cm'
  width=4><p class='MsoNormal'>&nbsp;</td>
 </tr>
 <tr style='mso-yfti-irow:11;height:15.6pt;mso-row-margin-right:3.35pt'>
  <td width=75 nowrap style='width:56.45pt;border-top:none;border-left:solid windowtext 1.0pt;
  border-bottom:none;border-right:solid windowtext 1.0pt;mso-border-top-alt:
  solid windowtext .5pt;mso-border-top-alt:solid windowtext .5pt;mso-border-left-alt:
  solid windowtext .5pt;mso-border-right-alt:solid windowtext .5pt;padding:
  0cm 5.4pt 0cm 5.4pt;height:15.6pt'>
  <p class=MsoNormal align=center style='text-align:center;mso-pagination:widow-orphan'><span
  lang=EN-US style='font-size:6.5pt;mso-bidi-font-size:10.0pt;font-family:"Times New Roman",serif;
  mso-fareast-font-family:等线;mso-font-kerning:0pt'>MTFAA<o:p></o:p></span></p>
  </td>
  <td width=39 nowrap style='width:29.5pt;border:none;mso-border-top-alt:solid windowtext .5pt;
  mso-border-left-alt:solid windowtext .5pt;padding:0cm 5.4pt 0cm 5.4pt;
  height:15.6pt'>
  <p class=MsoNormal align=center style='text-align:center;mso-pagination:widow-orphan'><span
  lang=EN-US style='font-size:6.5pt;mso-bidi-font-size:10.0pt;font-family:"Times New Roman",serif;
  mso-fareast-font-family:等线;mso-font-kerning:0pt'>1.981</span><span
  lang=EN-US style='font-size:6.5pt;font-family:"Times New Roman",serif;
  mso-fareast-font-family:等线;mso-font-kerning:0pt'><o:p></o:p></span></p>
  </td>
  <td width=39 nowrap style='width:29.55pt;border:none;mso-border-top-alt:solid windowtext .5pt;
  padding:0cm 5.4pt 0cm 5.4pt;height:15.6pt'>
  <p class=MsoNormal align=center style='text-align:center;mso-pagination:widow-orphan'><span
  lang=EN-US style='font-size:6.5pt;mso-bidi-font-size:10.0pt;font-family:"Times New Roman",serif;
  mso-fareast-font-family:等线;mso-font-kerning:0pt'>2.669</span><span
  lang=EN-US style='font-size:6.5pt;font-family:"Times New Roman",serif;
  mso-fareast-font-family:等线;mso-font-kerning:0pt'><o:p></o:p></span></p>
  </td>
  <td width=39 nowrap style='width:29.55pt;border:none;border-right:solid windowtext 1.0pt;
  mso-border-top-alt:solid windowtext .5pt;mso-border-top-alt:solid windowtext .5pt;
  mso-border-right-alt:solid windowtext .5pt;padding:0cm 5.4pt 0cm 5.4pt;
  height:15.6pt'>
  <p class=MsoNormal align=center style='text-align:center;mso-pagination:widow-orphan'><span
  lang=EN-US style='font-size:6.5pt;mso-bidi-font-size:10.0pt;font-family:"Times New Roman",serif;
  mso-fareast-font-family:等线;mso-font-kerning:0pt'>2.470</span><span
  lang=EN-US style='font-size:6.5pt;font-family:"Times New Roman",serif;
  mso-fareast-font-family:等线;mso-font-kerning:0pt'><o:p></o:p></span></p>
  </td>
  <td width=39 nowrap style='width:29.5pt;border:none;mso-border-top-alt:solid windowtext .5pt;
  mso-border-left-alt:solid windowtext .5pt;padding:0cm 5.4pt 0cm 5.4pt;
  height:15.6pt'>
  <p class=MsoNormal align=center style='text-align:center;mso-pagination:widow-orphan'><span
  lang=EN-US style='font-size:6.5pt;font-family:"Times New Roman",serif;
  mso-fareast-font-family:等线;mso-font-kerning:0pt'>3.465<o:p></o:p></span></p>
  </td>
  <td width=39 nowrap style='width:29.55pt;border:none;mso-border-top-alt:solid windowtext .5pt;
  padding:0cm 5.4pt 0cm 5.4pt;height:15.6pt'>
  <p class=MsoNormal align=center style='text-align:center;mso-pagination:widow-orphan'><span
  lang=EN-US style='font-size:6.5pt;font-family:"Times New Roman",serif;
  mso-fareast-font-family:等线;mso-font-kerning:0pt'>4.113<o:p></o:p></span></p>
  </td>
  <td width=39 nowrap style='width:29.55pt;border:none;border-right:solid windowtext 1.0pt;
  mso-border-top-alt:solid windowtext .5pt;mso-border-top-alt:solid windowtext .5pt;
  mso-border-right-alt:solid windowtext .5pt;padding:0cm 5.4pt 0cm 5.4pt;
  height:15.6pt'>
  <p class=MsoNormal align=center style='text-align:center;mso-pagination:widow-orphan'><span
  lang=EN-US style='font-size:6.5pt;font-family:"Times New Roman",serif;
  mso-fareast-font-family:等线;mso-font-kerning:0pt'>3.925<o:p></o:p></span></p>
  </td>
  <td width=39 nowrap style='width:29.5pt;border:none;mso-border-top-alt:solid windowtext .5pt;
  mso-border-left-alt:solid windowtext .5pt;padding:0cm 5.4pt 0cm 5.4pt;
  height:15.6pt'>
  <p class=MsoNormal align=center style='text-align:center;mso-pagination:widow-orphan'><span
  lang=EN-US style='font-size:6.5pt;font-family:"Times New Roman",serif;
  mso-fareast-font-family:等线;mso-font-kerning:0pt'>2.951<o:p></o:p></span></p>
  </td>
  <td width=39 nowrap style='width:29.55pt;border:none;mso-border-top-alt:solid windowtext .5pt;
  padding:0cm 5.4pt 0cm 5.4pt;height:15.6pt'>
  <p class=MsoNormal align=center style='text-align:center;mso-pagination:widow-orphan'><span
  lang=EN-US style='font-size:6.5pt;font-family:"Times New Roman",serif;
  mso-fareast-font-family:等线;mso-font-kerning:0pt'>3.523<o:p></o:p></span></p>
  </td>
  <td width=39 nowrap style='width:29.55pt;border:none;border-right:solid windowtext 1.0pt;
  mso-border-top-alt:solid windowtext .5pt;mso-border-top-alt:solid windowtext .5pt;
  mso-border-right-alt:solid windowtext .5pt;padding:0cm 5.4pt 0cm 5.4pt;
  height:15.6pt'>
  <p class=MsoNormal align=center style='text-align:center;mso-pagination:widow-orphan'><span
  lang=EN-US style='font-size:6.5pt;font-family:"Times New Roman",serif;
  mso-fareast-font-family:等线;mso-font-kerning:0pt'>3.357<o:p></o:p></span></p>
  </td>
  <td width=39 style='width:29.5pt;border:none;mso-border-top-alt:solid windowtext .5pt;
  mso-border-left-alt:solid windowtext .5pt;padding:0cm 5.4pt 0cm 5.4pt;
  height:15.6pt'>
  <p class=MsoNormal align=center style='text-align:center;mso-pagination:widow-orphan'><span
  lang=EN-US style='font-size:6.5pt;font-family:"Times New Roman",serif;
  mso-fareast-font-family:等线;mso-font-kerning:0pt'>2.754<o:p></o:p></span></p>
  </td>
  <td width=39 style='width:29.55pt;border:none;mso-border-top-alt:solid windowtext .5pt;
  padding:0cm 5.4pt 0cm 5.4pt;height:15.6pt'>
  <p class=MsoNormal align=center style='text-align:center;mso-pagination:widow-orphan'><span
  lang=EN-US style='font-size:6.5pt;font-family:"Times New Roman",serif;
  mso-fareast-font-family:等线;mso-font-kerning:0pt'>3.436<o:p></o:p></span></p>
  </td>
  <td width=39 style='width:29.55pt;border:none;border-right:solid windowtext 1.0pt;
  mso-border-top-alt:solid windowtext .5pt;mso-border-top-alt:solid windowtext .5pt;
  mso-border-right-alt:solid windowtext .5pt;padding:0cm 5.4pt 0cm 5.4pt;
  height:15.6pt'>
  <p class=MsoNormal align=center style='text-align:center;mso-pagination:widow-orphan'><span
  lang=EN-US style='font-size:6.5pt;font-family:"Times New Roman",serif;
  mso-fareast-font-family:等线;mso-font-kerning:0pt'>3.238<o:p></o:p></span></p>
  </td>
  <td style='mso-cell-special:placeholder;border:none;padding:0cm 0cm 0cm 0cm'
  width=4><p class='MsoNormal'>&nbsp;</td>
 </tr>
 <tr style='mso-yfti-irow:12;height:15.6pt;mso-row-margin-right:3.35pt'>
  <td width=75 nowrap style='width:56.45pt;border:solid windowtext 1.0pt;
  border-top:none;mso-border-left-alt:solid windowtext .5pt;mso-border-bottom-alt:
  solid windowtext .5pt;mso-border-right-alt:solid windowtext .5pt;padding:
  0cm 5.4pt 0cm 5.4pt;height:15.6pt'>
  <p class=MsoNormal align=center style='text-align:center;mso-pagination:widow-orphan'><span
  lang=EN-US style='font-size:6.5pt;mso-bidi-font-size:10.0pt;font-family:"Times New Roman",serif;
  mso-fareast-font-family:等线;mso-font-kerning:0pt'>MTFAA-LSA<o:p></o:p></span></p>
  </td>
  <td width=39 nowrap style='width:29.5pt;border:none;border-bottom:solid windowtext 1.0pt;
  mso-border-left-alt:solid windowtext .5pt;mso-border-left-alt:solid windowtext .5pt;
  mso-border-bottom-alt:solid windowtext .5pt;padding:0cm 5.4pt 0cm 5.4pt;
  height:15.6pt'>
  <p class=MsoNormal align=center style='text-align:center;mso-pagination:widow-orphan'><b
  style='mso-bidi-font-weight:normal'><span lang=EN-US style='font-size:6.5pt;
  mso-bidi-font-size:10.0pt;font-family:"Times New Roman",serif;mso-fareast-font-family:
  等线;mso-font-kerning:0pt'>2.084</span></b><!--[if gte msEquation 12]><m:oMath><b
   style='mso-bidi-font-weight:normal'><span lang=EN-US style='font-size:6.5pt;
   mso-bidi-font-size:10.0pt;font-family:"Cambria Math",serif;mso-fareast-font-family:
   等线;mso-bidi-font-family:"Times New Roman";mso-font-kerning:0pt'><m:r><m:rPr><m:scr
      m:val="roman"/><m:sty m:val="b"/></m:rPr>↑</m:r></span></b></m:oMath><![endif]--><![if !msEquation]><span
  lang=EN-US style='font-size:10.5pt;mso-bidi-font-size:11.0pt;font-family:
  等线;mso-ascii-theme-font:minor-latin;mso-fareast-theme-font:minor-fareast;
  mso-hansi-theme-font:minor-latin;mso-bidi-font-family:"Times New Roman";
  mso-bidi-theme-font:minor-bidi;position:relative;top:5.5pt;mso-text-raise:
  -5.5pt;mso-ansi-language:EN-US;mso-fareast-language:ZH-CN;mso-bidi-language:
  AR-SA'><!--[if gte vml 1]><v:shape id="_x0000_i1025" type="#_x0000_t75"
   style='width:2.85pt;height:15.7pt'>
   <v:imagedata src="Table_readme.files/image001.png" o:title="" chromakey="white"/>
  </v:shape><![endif]--><![if !vml]><img width=4 height=21
  src="Table_readme.files/image002.png" v:shapes="_x0000_i1025"><![endif]></span><![endif]><b
  style='mso-bidi-font-weight:normal'><span lang=EN-US style='font-size:6.5pt;
  font-family:"Times New Roman",serif;mso-fareast-font-family:等线;mso-font-kerning:
  0pt'><o:p></o:p></span></b></p>
  </td>
  <td width=39 nowrap style='width:29.55pt;border:none;border-bottom:solid windowtext 1.0pt;
  mso-border-bottom-alt:solid windowtext .5pt;padding:0cm 5.4pt 0cm 5.4pt;
  height:15.6pt'>
  <p class=MsoNormal align=center style='text-align:center;mso-pagination:widow-orphan'><b
  style='mso-bidi-font-weight:normal'><span lang=EN-US style='font-size:6.5pt;
  mso-bidi-font-size:10.0pt;font-family:"Times New Roman",serif;mso-fareast-font-family:
  等线;mso-font-kerning:0pt'>2.795</span></b><!--[if gte msEquation 12]><m:oMath><b
   style='mso-bidi-font-weight:normal'><span lang=EN-US style='font-size:6.5pt;
   mso-bidi-font-size:10.0pt;font-family:"Cambria Math",serif;mso-fareast-font-family:
   等线;mso-bidi-font-family:"Times New Roman";mso-font-kerning:0pt'><m:r><m:rPr><m:scr
      m:val="roman"/><m:sty m:val="b"/></m:rPr>↑</m:r></span></b></m:oMath><![endif]--><![if !msEquation]><span
  lang=EN-US style='font-size:10.5pt;mso-bidi-font-size:11.0pt;font-family:
  等线;mso-ascii-theme-font:minor-latin;mso-fareast-theme-font:minor-fareast;
  mso-hansi-theme-font:minor-latin;mso-bidi-font-family:"Times New Roman";
  mso-bidi-theme-font:minor-bidi;position:relative;top:5.5pt;mso-text-raise:
  -5.5pt;mso-ansi-language:EN-US;mso-fareast-language:ZH-CN;mso-bidi-language:
  AR-SA'><!--[if gte vml 1]><v:shape id="_x0000_i1025" type="#_x0000_t75"
   style='width:2.85pt;height:15.7pt'>
   <v:imagedata src="Table_readme.files/image001.png" o:title="" chromakey="white"/>
  </v:shape><![endif]--><![if !vml]><img width=4 height=21
  src="Table_readme.files/image002.png" v:shapes="_x0000_i1025"><![endif]></span><![endif]><b
  style='mso-bidi-font-weight:normal'><span lang=EN-US style='font-size:6.5pt;
  font-family:"Times New Roman",serif;mso-fareast-font-family:等线;mso-font-kerning:
  0pt'><o:p></o:p></span></b></p>
  </td>
  <td width=39 nowrap style='width:29.55pt;border-top:none;border-left:none;
  border-bottom:solid windowtext 1.0pt;border-right:solid windowtext 1.0pt;
  mso-border-bottom-alt:solid windowtext .5pt;mso-border-right-alt:solid windowtext .5pt;
  padding:0cm 5.4pt 0cm 5.4pt;height:15.6pt'>
  <p class=MsoNormal align=center style='text-align:center;mso-pagination:widow-orphan'><b
  style='mso-bidi-font-weight:normal'><span lang=EN-US style='font-size:6.5pt;
  mso-bidi-font-size:10.0pt;font-family:"Times New Roman",serif;mso-fareast-font-family:
  等线;mso-font-kerning:0pt'>2.589</span></b><!--[if gte msEquation 12]><m:oMath><b
   style='mso-bidi-font-weight:normal'><span lang=EN-US style='font-size:6.5pt;
   mso-bidi-font-size:10.0pt;font-family:"Cambria Math",serif;mso-fareast-font-family:
   等线;mso-bidi-font-family:"Times New Roman";mso-font-kerning:0pt'><m:r><m:rPr><m:scr
      m:val="roman"/><m:sty m:val="b"/></m:rPr>↑</m:r></span></b></m:oMath><![endif]--><![if !msEquation]><span
  lang=EN-US style='font-size:10.5pt;mso-bidi-font-size:11.0pt;font-family:
  等线;mso-ascii-theme-font:minor-latin;mso-fareast-theme-font:minor-fareast;
  mso-hansi-theme-font:minor-latin;mso-bidi-font-family:"Times New Roman";
  mso-bidi-theme-font:minor-bidi;position:relative;top:5.5pt;mso-text-raise:
  -5.5pt;mso-ansi-language:EN-US;mso-fareast-language:ZH-CN;mso-bidi-language:
  AR-SA'><!--[if gte vml 1]><v:shape id="_x0000_i1025" type="#_x0000_t75"
   style='width:2.85pt;height:15.7pt'>
   <v:imagedata src="Table_readme.files/image001.png" o:title="" chromakey="white"/>
  </v:shape><![endif]--><![if !vml]><img width=4 height=21
  src="Table_readme.files/image002.png" v:shapes="_x0000_i1025"><![endif]></span><![endif]><b
  style='mso-bidi-font-weight:normal'><span lang=EN-US style='font-size:6.5pt;
  font-family:"Times New Roman",serif;mso-fareast-font-family:等线;mso-font-kerning:
  0pt'><o:p></o:p></span></b></p>
  </td>
  <td width=39 nowrap style='width:29.5pt;border:none;border-bottom:solid windowtext 1.0pt;
  mso-border-left-alt:solid windowtext .5pt;mso-border-left-alt:solid windowtext .5pt;
  mso-border-bottom-alt:solid windowtext .5pt;padding:0cm 5.4pt 0cm 5.4pt;
  height:15.6pt'>
  <p class=MsoNormal align=center style='text-align:center;mso-pagination:widow-orphan'><b
  style='mso-bidi-font-weight:normal'><span lang=EN-US style='font-size:6.5pt;
  font-family:"Times New Roman",serif;mso-fareast-font-family:等线;mso-font-kerning:
  0pt'>3.517</span></b><!--[if gte msEquation 12]><m:oMath><b style='mso-bidi-font-weight:
   normal'><span lang=EN-US style='font-size:6.5pt;mso-bidi-font-size:10.0pt;
   font-family:"Cambria Math",serif;mso-fareast-font-family:等线;mso-bidi-font-family:
   "Times New Roman";mso-font-kerning:0pt'><m:r><m:rPr><m:scr m:val="roman"/><m:sty
      m:val="b"/></m:rPr>↑</m:r></span></b></m:oMath><![endif]--><![if !msEquation]><span
  lang=EN-US style='font-size:10.5pt;mso-bidi-font-size:11.0pt;font-family:
  等线;mso-ascii-theme-font:minor-latin;mso-fareast-theme-font:minor-fareast;
  mso-hansi-theme-font:minor-latin;mso-bidi-font-family:"Times New Roman";
  mso-bidi-theme-font:minor-bidi;position:relative;top:5.5pt;mso-text-raise:
  -5.5pt;mso-ansi-language:EN-US;mso-fareast-language:ZH-CN;mso-bidi-language:
  AR-SA'><!--[if gte vml 1]><v:shape id="_x0000_i1025" type="#_x0000_t75"
   style='width:2.85pt;height:15.7pt'>
   <v:imagedata src="Table_readme.files/image001.png" o:title="" chromakey="white"/>
  </v:shape><![endif]--><![if !vml]><img width=4 height=21
  src="Table_readme.files/image002.png" v:shapes="_x0000_i1025"><![endif]></span><![endif]><b
  style='mso-bidi-font-weight:normal'><span lang=EN-US style='font-size:6.5pt;
  font-family:"Times New Roman",serif;mso-fareast-font-family:等线;mso-font-kerning:
  0pt'><o:p></o:p></span></b></p>
  </td>
  <td width=39 nowrap style='width:29.55pt;border:none;border-bottom:solid windowtext 1.0pt;
  mso-border-bottom-alt:solid windowtext .5pt;padding:0cm 5.4pt 0cm 5.4pt;
  height:15.6pt'>
  <p class=MsoNormal align=center style='text-align:center;mso-pagination:widow-orphan'><b
  style='mso-bidi-font-weight:normal'><span lang=EN-US style='font-size:6.5pt;
  font-family:"Times New Roman",serif;mso-fareast-font-family:等线;mso-font-kerning:
  0pt'>4.203</span></b><!--[if gte msEquation 12]><m:oMath><b style='mso-bidi-font-weight:
   normal'><span lang=EN-US style='font-size:6.5pt;mso-bidi-font-size:10.0pt;
   font-family:"Cambria Math",serif;mso-fareast-font-family:等线;mso-bidi-font-family:
   "Times New Roman";mso-font-kerning:0pt'><m:r><m:rPr><m:scr m:val="roman"/><m:sty
      m:val="b"/></m:rPr>↑</m:r></span></b></m:oMath><![endif]--><![if !msEquation]><span
  lang=EN-US style='font-size:10.5pt;mso-bidi-font-size:11.0pt;font-family:
  等线;mso-ascii-theme-font:minor-latin;mso-fareast-theme-font:minor-fareast;
  mso-hansi-theme-font:minor-latin;mso-bidi-font-family:"Times New Roman";
  mso-bidi-theme-font:minor-bidi;position:relative;top:5.5pt;mso-text-raise:
  -5.5pt;mso-ansi-language:EN-US;mso-fareast-language:ZH-CN;mso-bidi-language:
  AR-SA'><!--[if gte vml 1]><v:shape id="_x0000_i1025" type="#_x0000_t75"
   style='width:2.85pt;height:15.7pt'>
   <v:imagedata src="Table_readme.files/image001.png" o:title="" chromakey="white"/>
  </v:shape><![endif]--><![if !vml]><img width=4 height=21
  src="Table_readme.files/image002.png" v:shapes="_x0000_i1025"><![endif]></span><![endif]><b
  style='mso-bidi-font-weight:normal'><span lang=EN-US style='font-size:6.5pt;
  font-family:"Times New Roman",serif;mso-fareast-font-family:等线;mso-font-kerning:
  0pt'><o:p></o:p></span></b></p>
  </td>
  <td width=39 nowrap style='width:29.55pt;border-top:none;border-left:none;
  border-bottom:solid windowtext 1.0pt;border-right:solid windowtext 1.0pt;
  mso-border-bottom-alt:solid windowtext .5pt;mso-border-right-alt:solid windowtext .5pt;
  padding:0cm 5.4pt 0cm 5.4pt;height:15.6pt'>
  <p class=MsoNormal align=center style='text-align:center;mso-pagination:widow-orphan'><b
  style='mso-bidi-font-weight:normal'><span lang=EN-US style='font-size:6.5pt;
  font-family:"Times New Roman",serif;mso-fareast-font-family:等线;mso-font-kerning:
  0pt'>4.004</span></b><!--[if gte msEquation 12]><m:oMath><b style='mso-bidi-font-weight:
   normal'><span lang=EN-US style='font-size:6.5pt;mso-bidi-font-size:10.0pt;
   font-family:"Cambria Math",serif;mso-fareast-font-family:等线;mso-bidi-font-family:
   "Times New Roman";mso-font-kerning:0pt'><m:r><m:rPr><m:scr m:val="roman"/><m:sty
      m:val="b"/></m:rPr>↑</m:r></span></b></m:oMath><![endif]--><![if !msEquation]><span
  lang=EN-US style='font-size:10.5pt;mso-bidi-font-size:11.0pt;font-family:
  等线;mso-ascii-theme-font:minor-latin;mso-fareast-theme-font:minor-fareast;
  mso-hansi-theme-font:minor-latin;mso-bidi-font-family:"Times New Roman";
  mso-bidi-theme-font:minor-bidi;position:relative;top:5.5pt;mso-text-raise:
  -5.5pt;mso-ansi-language:EN-US;mso-fareast-language:ZH-CN;mso-bidi-language:
  AR-SA'><!--[if gte vml 1]><v:shape id="_x0000_i1025" type="#_x0000_t75"
   style='width:2.85pt;height:15.7pt'>
   <v:imagedata src="Table_readme.files/image001.png" o:title="" chromakey="white"/>
  </v:shape><![endif]--><![if !vml]><img width=4 height=21
  src="Table_readme.files/image002.png" v:shapes="_x0000_i1025"><![endif]></span><![endif]><b
  style='mso-bidi-font-weight:normal'><span lang=EN-US style='font-size:6.5pt;
  font-family:"Times New Roman",serif;mso-fareast-font-family:等线;mso-font-kerning:
  0pt'><o:p></o:p></span></b></p>
  </td>
  <td width=39 nowrap style='width:29.5pt;border:none;border-bottom:solid windowtext 1.0pt;
  mso-border-left-alt:solid windowtext .5pt;mso-border-left-alt:solid windowtext .5pt;
  mso-border-bottom-alt:solid windowtext .5pt;padding:0cm 5.4pt 0cm 5.4pt;
  height:15.6pt'>
  <p class=MsoNormal align=center style='text-align:center;mso-pagination:widow-orphan'><b
  style='mso-bidi-font-weight:normal'><span lang=EN-US style='font-size:6.5pt;
  font-family:"Times New Roman",serif;mso-fareast-font-family:等线;mso-font-kerning:
  0pt'>3.006</span></b><!--[if gte msEquation 12]><m:oMath><b style='mso-bidi-font-weight:
   normal'><span lang=EN-US style='font-size:6.5pt;mso-bidi-font-size:10.0pt;
   font-family:"Cambria Math",serif;mso-fareast-font-family:等线;mso-bidi-font-family:
   "Times New Roman";mso-font-kerning:0pt'><m:r><m:rPr><m:scr m:val="roman"/><m:sty
      m:val="b"/></m:rPr>↑</m:r></span></b></m:oMath><![endif]--><![if !msEquation]><span
  lang=EN-US style='font-size:10.5pt;mso-bidi-font-size:11.0pt;font-family:
  等线;mso-ascii-theme-font:minor-latin;mso-fareast-theme-font:minor-fareast;
  mso-hansi-theme-font:minor-latin;mso-bidi-font-family:"Times New Roman";
  mso-bidi-theme-font:minor-bidi;position:relative;top:5.5pt;mso-text-raise:
  -5.5pt;mso-ansi-language:EN-US;mso-fareast-language:ZH-CN;mso-bidi-language:
  AR-SA'><!--[if gte vml 1]><v:shape id="_x0000_i1025" type="#_x0000_t75"
   style='width:2.85pt;height:15.7pt'>
   <v:imagedata src="Table_readme.files/image001.png" o:title="" chromakey="white"/>
  </v:shape><![endif]--><![if !vml]><img width=4 height=21
  src="Table_readme.files/image002.png" v:shapes="_x0000_i1025"><![endif]></span><![endif]><b
  style='mso-bidi-font-weight:normal'><span lang=EN-US style='font-size:6.5pt;
  font-family:"Times New Roman",serif;mso-fareast-font-family:等线;mso-font-kerning:
  0pt'><o:p></o:p></span></b></p>
  </td>
  <td width=39 nowrap style='width:29.55pt;border:none;border-bottom:solid windowtext 1.0pt;
  mso-border-bottom-alt:solid windowtext .5pt;padding:0cm 5.4pt 0cm 5.4pt;
  height:15.6pt'>
  <p class=MsoNormal align=center style='text-align:center;mso-pagination:widow-orphan'><b
  style='mso-bidi-font-weight:normal'><span lang=EN-US style='font-size:6.5pt;
  font-family:"Times New Roman",serif;mso-fareast-font-family:等线;mso-font-kerning:
  0pt'>3.593</span></b><!--[if gte msEquation 12]><m:oMath><b style='mso-bidi-font-weight:
   normal'><span lang=EN-US style='font-size:6.5pt;mso-bidi-font-size:10.0pt;
   font-family:"Cambria Math",serif;mso-fareast-font-family:等线;mso-bidi-font-family:
   "Times New Roman";mso-font-kerning:0pt'><m:r><m:rPr><m:scr m:val="roman"/><m:sty
      m:val="b"/></m:rPr>↑</m:r></span></b></m:oMath><![endif]--><![if !msEquation]><span
  lang=EN-US style='font-size:10.5pt;mso-bidi-font-size:11.0pt;font-family:
  等线;mso-ascii-theme-font:minor-latin;mso-fareast-theme-font:minor-fareast;
  mso-hansi-theme-font:minor-latin;mso-bidi-font-family:"Times New Roman";
  mso-bidi-theme-font:minor-bidi;position:relative;top:5.5pt;mso-text-raise:
  -5.5pt;mso-ansi-language:EN-US;mso-fareast-language:ZH-CN;mso-bidi-language:
  AR-SA'><!--[if gte vml 1]><v:shape id="_x0000_i1025" type="#_x0000_t75"
   style='width:2.85pt;height:15.7pt'>
   <v:imagedata src="Table_readme.files/image001.png" o:title="" chromakey="white"/>
  </v:shape><![endif]--><![if !vml]><img width=4 height=21
  src="Table_readme.files/image002.png" v:shapes="_x0000_i1025"><![endif]></span><![endif]><b
  style='mso-bidi-font-weight:normal'><span lang=EN-US style='font-size:6.5pt;
  font-family:"Times New Roman",serif;mso-fareast-font-family:等线;mso-font-kerning:
  0pt'><o:p></o:p></span></b></p>
  </td>
  <td width=39 nowrap style='width:29.55pt;border-top:none;border-left:none;
  border-bottom:solid windowtext 1.0pt;border-right:solid windowtext 1.0pt;
  mso-border-bottom-alt:solid windowtext .5pt;mso-border-right-alt:solid windowtext .5pt;
  padding:0cm 5.4pt 0cm 5.4pt;height:15.6pt'>
  <p class=MsoNormal align=center style='text-align:center;mso-pagination:widow-orphan'><b
  style='mso-bidi-font-weight:normal'><span lang=EN-US style='font-size:6.5pt;
  font-family:"Times New Roman",serif;mso-fareast-font-family:等线;mso-font-kerning:
  0pt'>3.423</span></b><!--[if gte msEquation 12]><m:oMath><b style='mso-bidi-font-weight:
   normal'><span lang=EN-US style='font-size:6.5pt;mso-bidi-font-size:10.0pt;
   font-family:"Cambria Math",serif;mso-fareast-font-family:等线;mso-bidi-font-family:
   "Times New Roman";mso-font-kerning:0pt'><m:r><m:rPr><m:scr m:val="roman"/><m:sty
      m:val="b"/></m:rPr>↑</m:r></span></b></m:oMath><![endif]--><![if !msEquation]><span
  lang=EN-US style='font-size:10.5pt;mso-bidi-font-size:11.0pt;font-family:
  等线;mso-ascii-theme-font:minor-latin;mso-fareast-theme-font:minor-fareast;
  mso-hansi-theme-font:minor-latin;mso-bidi-font-family:"Times New Roman";
  mso-bidi-theme-font:minor-bidi;position:relative;top:5.5pt;mso-text-raise:
  -5.5pt;mso-ansi-language:EN-US;mso-fareast-language:ZH-CN;mso-bidi-language:
  AR-SA'><!--[if gte vml 1]><v:shape id="_x0000_i1025" type="#_x0000_t75"
   style='width:2.85pt;height:15.7pt'>
   <v:imagedata src="Table_readme.files/image001.png" o:title="" chromakey="white"/>
  </v:shape><![endif]--><![if !vml]><img width=4 height=21
  src="Table_readme.files/image002.png" v:shapes="_x0000_i1025"><![endif]></span><![endif]><b
  style='mso-bidi-font-weight:normal'><span lang=EN-US style='font-size:6.5pt;
  font-family:"Times New Roman",serif;mso-fareast-font-family:等线;mso-font-kerning:
  0pt'><o:p></o:p></span></b></p>
  </td>
  <td width=39 style='width:29.5pt;border:none;border-bottom:solid windowtext 1.0pt;
  mso-border-left-alt:solid windowtext .5pt;mso-border-left-alt:solid windowtext .5pt;
  mso-border-bottom-alt:solid windowtext .5pt;padding:0cm 5.4pt 0cm 5.4pt;
  height:15.6pt'>
  <p class=MsoNormal align=center style='text-align:center;mso-pagination:widow-orphan'><b
  style='mso-bidi-font-weight:normal'><span lang=EN-US style='font-size:6.5pt;
  font-family:"Times New Roman",serif;mso-fareast-font-family:等线;mso-font-kerning:
  0pt'>2.829</span></b><!--[if gte msEquation 12]><m:oMath><b style='mso-bidi-font-weight:
   normal'><span lang=EN-US style='font-size:6.5pt;mso-bidi-font-size:10.0pt;
   font-family:"Cambria Math",serif;mso-fareast-font-family:等线;mso-bidi-font-family:
   "Times New Roman";mso-font-kerning:0pt'><m:r><m:rPr><m:scr m:val="roman"/><m:sty
      m:val="b"/></m:rPr>↑</m:r></span></b></m:oMath><![endif]--><![if !msEquation]><span
  lang=EN-US style='font-size:10.5pt;mso-bidi-font-size:11.0pt;font-family:
  等线;mso-ascii-theme-font:minor-latin;mso-fareast-theme-font:minor-fareast;
  mso-hansi-theme-font:minor-latin;mso-bidi-font-family:"Times New Roman";
  mso-bidi-theme-font:minor-bidi;position:relative;top:5.5pt;mso-text-raise:
  -5.5pt;mso-ansi-language:EN-US;mso-fareast-language:ZH-CN;mso-bidi-language:
  AR-SA'><!--[if gte vml 1]><v:shape id="_x0000_i1025" type="#_x0000_t75"
   style='width:2.85pt;height:15.7pt'>
   <v:imagedata src="Table_readme.files/image001.png" o:title="" chromakey="white"/>
  </v:shape><![endif]--><![if !vml]><img width=4 height=21
  src="Table_readme.files/image002.png" v:shapes="_x0000_i1025"><![endif]></span><![endif]><b
  style='mso-bidi-font-weight:normal'><span lang=EN-US style='font-size:6.5pt;
  font-family:"Times New Roman",serif;mso-fareast-font-family:等线;mso-font-kerning:
  0pt'><o:p></o:p></span></b></p>
  </td>
  <td width=39 style='width:29.55pt;border:none;border-bottom:solid windowtext 1.0pt;
  mso-border-bottom-alt:solid windowtext .5pt;padding:0cm 5.4pt 0cm 5.4pt;
  height:15.6pt'>
  <p class=MsoNormal align=center style='text-align:center;mso-pagination:widow-orphan'><b
  style='mso-bidi-font-weight:normal'><span lang=EN-US style='font-size:6.5pt;
  font-family:"Times New Roman",serif;mso-fareast-font-family:等线;mso-font-kerning:
  0pt'>3.547</span></b><!--[if gte msEquation 12]><m:oMath><b style='mso-bidi-font-weight:
   normal'><span lang=EN-US style='font-size:6.5pt;mso-bidi-font-size:10.0pt;
   font-family:"Cambria Math",serif;mso-fareast-font-family:等线;mso-bidi-font-family:
   "Times New Roman";mso-font-kerning:0pt'><m:r><m:rPr><m:scr m:val="roman"/><m:sty
      m:val="b"/></m:rPr>↑</m:r></span></b></m:oMath><![endif]--><![if !msEquation]><span
  lang=EN-US style='font-size:10.5pt;mso-bidi-font-size:11.0pt;font-family:
  等线;mso-ascii-theme-font:minor-latin;mso-fareast-theme-font:minor-fareast;
  mso-hansi-theme-font:minor-latin;mso-bidi-font-family:"Times New Roman";
  mso-bidi-theme-font:minor-bidi;position:relative;top:5.5pt;mso-text-raise:
  -5.5pt;mso-ansi-language:EN-US;mso-fareast-language:ZH-CN;mso-bidi-language:
  AR-SA'><!--[if gte vml 1]><v:shape id="_x0000_i1025" type="#_x0000_t75"
   style='width:2.85pt;height:15.7pt'>
   <v:imagedata src="Table_readme.files/image001.png" o:title="" chromakey="white"/>
  </v:shape><![endif]--><![if !vml]><img width=4 height=21
  src="Table_readme.files/image002.png" v:shapes="_x0000_i1025"><![endif]></span><![endif]><b
  style='mso-bidi-font-weight:normal'><span lang=EN-US style='font-size:6.5pt;
  font-family:"Times New Roman",serif;mso-fareast-font-family:等线;mso-font-kerning:
  0pt'><o:p></o:p></span></b></p>
  </td>
  <td width=39 style='width:29.55pt;border-top:none;border-left:none;
  border-bottom:solid windowtext 1.0pt;border-right:solid windowtext 1.0pt;
  mso-border-bottom-alt:solid windowtext .5pt;mso-border-right-alt:solid windowtext .5pt;
  padding:0cm 5.4pt 0cm 5.4pt;height:15.6pt'>
  <p class=MsoNormal align=center style='text-align:center;mso-pagination:widow-orphan'><b
  style='mso-bidi-font-weight:normal'><span lang=EN-US style='font-size:6.5pt;
  font-family:"Times New Roman",serif;mso-fareast-font-family:等线;mso-font-kerning:
  0pt'>3.339</span></b><!--[if gte msEquation 12]><m:oMath><b style='mso-bidi-font-weight:
   normal'><span lang=EN-US style='font-size:6.5pt;mso-bidi-font-size:10.0pt;
   font-family:"Cambria Math",serif;mso-fareast-font-family:等线;mso-bidi-font-family:
   "Times New Roman";mso-font-kerning:0pt'><m:r><m:rPr><m:scr m:val="roman"/><m:sty
      m:val="b"/></m:rPr>↑</m:r></span></b></m:oMath><![endif]--><![if !msEquation]><span
  lang=EN-US style='font-size:10.5pt;mso-bidi-font-size:11.0pt;font-family:
  等线;mso-ascii-theme-font:minor-latin;mso-fareast-theme-font:minor-fareast;
  mso-hansi-theme-font:minor-latin;mso-bidi-font-family:"Times New Roman";
  mso-bidi-theme-font:minor-bidi;position:relative;top:5.5pt;mso-text-raise:
  -5.5pt;mso-ansi-language:EN-US;mso-fareast-language:ZH-CN;mso-bidi-language:
  AR-SA'><!--[if gte vml 1]><v:shape id="_x0000_i1025" type="#_x0000_t75"
   style='width:2.85pt;height:15.7pt'>
   <v:imagedata src="Table_readme.files/image001.png" o:title="" chromakey="white"/>
  </v:shape><![endif]--><![if !vml]><img width=4 height=21
  src="Table_readme.files/image002.png" v:shapes="_x0000_i1025"><![endif]></span><![endif]><b
  style='mso-bidi-font-weight:normal'><span lang=EN-US style='font-size:6.5pt;
  font-family:"Times New Roman",serif;mso-fareast-font-family:等线;mso-font-kerning:
  0pt'><o:p></o:p></span></b></p>
  </td>
  <td style='mso-cell-special:placeholder;border:none;padding:0cm 0cm 0cm 0cm'
  width=4><p class='MsoNormal'>&nbsp;</td>
 </tr>
 <tr style='mso-yfti-irow:13;height:15.6pt;mso-row-margin-right:3.35pt'>
  <td width=75 nowrap style='width:56.45pt;border-top:none;border-left:solid windowtext 1.0pt;
  border-bottom:none;border-right:solid windowtext 1.0pt;mso-border-top-alt:
  solid windowtext .5pt;mso-border-top-alt:solid windowtext .5pt;mso-border-left-alt:
  solid windowtext .5pt;mso-border-right-alt:solid windowtext .5pt;padding:
  0cm 5.4pt 0cm 5.4pt;height:15.6pt'>
  <p class=MsoNormal align=center style='text-align:center;mso-pagination:widow-orphan'><span
  lang=EN-US style='font-size:6.5pt;mso-bidi-font-size:10.0pt;font-family:"Times New Roman",serif;
  mso-fareast-font-family:等线;mso-font-kerning:0pt'>DPARN<o:p></o:p></span></p>
  </td>
  <td width=39 nowrap style='width:29.5pt;border:none;mso-border-top-alt:solid windowtext .5pt;
  mso-border-left-alt:solid windowtext .5pt;padding:0cm 5.4pt 0cm 5.4pt;
  height:15.6pt'>
  <p class=MsoNormal align=center style='text-align:center;mso-pagination:widow-orphan'><span
  lang=EN-US style='font-size:6.5pt;mso-bidi-font-size:10.0pt;font-family:"Times New Roman",serif;
  mso-fareast-font-family:等线;mso-font-kerning:0pt'>1.702<o:p></o:p></span></p>
  </td>
  <td width=39 nowrap style='width:29.55pt;border:none;mso-border-top-alt:solid windowtext .5pt;
  padding:0cm 5.4pt 0cm 5.4pt;height:15.6pt'>
  <p class=MsoNormal align=center style='text-align:center;mso-pagination:widow-orphan'><span
  lang=EN-US style='font-size:6.5pt;mso-bidi-font-size:10.0pt;font-family:"Times New Roman",serif;
  mso-fareast-font-family:等线;mso-font-kerning:0pt'>2.309<o:p></o:p></span></p>
  </td>
  <td width=39 nowrap style='width:29.55pt;border:none;border-right:solid windowtext 1.0pt;
  mso-border-top-alt:solid windowtext .5pt;mso-border-top-alt:solid windowtext .5pt;
  mso-border-right-alt:solid windowtext .5pt;padding:0cm 5.4pt 0cm 5.4pt;
  height:15.6pt'>
  <p class=MsoNormal align=center style='text-align:center;mso-pagination:widow-orphan'><span
  lang=EN-US style='font-size:6.5pt;mso-bidi-font-size:10.0pt;font-family:"Times New Roman",serif;
  mso-fareast-font-family:等线;mso-font-kerning:0pt'>2.134<o:p></o:p></span></p>
  </td>
  <td width=39 nowrap style='width:29.5pt;border:none;mso-border-top-alt:solid windowtext .5pt;
  mso-border-left-alt:solid windowtext .5pt;padding:0cm 5.4pt 0cm 5.4pt;
  height:15.6pt'>
  <p class=MsoNormal align=center style='text-align:center;mso-pagination:widow-orphan'><span
  lang=EN-US style='font-size:6.5pt;font-family:"Times New Roman",serif;
  mso-fareast-font-family:等线;mso-font-kerning:0pt'>3.136<o:p></o:p></span></p>
  </td>
  <td width=39 nowrap style='width:29.55pt;border:none;mso-border-top-alt:solid windowtext .5pt;
  padding:0cm 5.4pt 0cm 5.4pt;height:15.6pt'>
  <p class=MsoNormal align=center style='text-align:center;mso-pagination:widow-orphan'><span
  lang=EN-US style='font-size:6.5pt;font-family:"Times New Roman",serif;
  mso-fareast-font-family:等线;mso-font-kerning:0pt'>3.759<o:p></o:p></span></p>
  </td>
  <td width=39 nowrap style='width:29.55pt;border:none;border-right:solid windowtext 1.0pt;
  mso-border-top-alt:solid windowtext .5pt;mso-border-top-alt:solid windowtext .5pt;
  mso-border-right-alt:solid windowtext .5pt;padding:0cm 5.4pt 0cm 5.4pt;
  height:15.6pt'>
  <p class=MsoNormal align=center style='text-align:center;mso-pagination:widow-orphan'><span
  lang=EN-US style='font-size:6.5pt;font-family:"Times New Roman",serif;
  mso-fareast-font-family:等线;mso-font-kerning:0pt'>3.580<o:p></o:p></span></p>
  </td>
  <td width=39 nowrap style='width:29.5pt;border:none;mso-border-top-alt:solid windowtext .5pt;
  mso-border-left-alt:solid windowtext .5pt;padding:0cm 5.4pt 0cm 5.4pt;
  height:15.6pt'>
  <p class=MsoNormal align=center style='text-align:center;mso-pagination:widow-orphan'><span
  lang=EN-US style='font-size:6.5pt;font-family:"Times New Roman",serif;
  mso-fareast-font-family:等线;mso-font-kerning:0pt'>2.505<o:p></o:p></span></p>
  </td>
  <td width=39 nowrap style='width:29.55pt;border:none;mso-border-top-alt:solid windowtext .5pt;
  padding:0cm 5.4pt 0cm 5.4pt;height:15.6pt'>
  <p class=MsoNormal align=center style='text-align:center;mso-pagination:widow-orphan'><span
  lang=EN-US style='font-size:6.5pt;font-family:"Times New Roman",serif;
  mso-fareast-font-family:等线;mso-font-kerning:0pt'>2.859<o:p></o:p></span></p>
  </td>
  <td width=39 nowrap style='width:29.55pt;border:none;border-right:solid windowtext 1.0pt;
  mso-border-top-alt:solid windowtext .5pt;mso-border-top-alt:solid windowtext .5pt;
  mso-border-right-alt:solid windowtext .5pt;padding:0cm 5.4pt 0cm 5.4pt;
  height:15.6pt'>
  <p class=MsoNormal align=center style='text-align:center;mso-pagination:widow-orphan'><span
  lang=EN-US style='font-size:6.5pt;font-family:"Times New Roman",serif;
  mso-fareast-font-family:等线;mso-font-kerning:0pt'>2.757<o:p></o:p></span></p>
  </td>
  <td width=39 style='width:29.5pt;border:none;mso-border-top-alt:solid windowtext .5pt;
  mso-border-left-alt:solid windowtext .5pt;padding:0cm 5.4pt 0cm 5.4pt;
  height:15.6pt'>
  <p class=MsoNormal align=center style='text-align:center;mso-pagination:widow-orphan'><span
  lang=EN-US style='font-size:6.5pt;font-family:"Times New Roman",serif;
  mso-fareast-font-family:等线;mso-font-kerning:0pt'>2.447<o:p></o:p></span></p>
  </td>
  <td width=39 style='width:29.55pt;border:none;mso-border-top-alt:solid windowtext .5pt;
  padding:0cm 5.4pt 0cm 5.4pt;height:15.6pt'>
  <p class=MsoNormal align=center style='text-align:center;mso-pagination:widow-orphan'><span
  lang=EN-US style='font-size:6.5pt;font-family:"Times New Roman",serif;
  mso-fareast-font-family:等线;mso-font-kerning:0pt'>3.069<o:p></o:p></span></p>
  </td>
  <td width=39 style='width:29.55pt;border:none;border-right:solid windowtext 1.0pt;
  mso-border-top-alt:solid windowtext .5pt;mso-border-top-alt:solid windowtext .5pt;
  mso-border-right-alt:solid windowtext .5pt;padding:0cm 5.4pt 0cm 5.4pt;
  height:15.6pt'>
  <p class=MsoNormal align=center style='text-align:center;mso-pagination:widow-orphan'><span
  lang=EN-US style='font-size:6.5pt;font-family:"Times New Roman",serif;
  mso-fareast-font-family:等线;mso-font-kerning:0pt'>2.890<o:p></o:p></span></p>
  </td>
  <td style='mso-cell-special:placeholder;border:none;padding:0cm 0cm 0cm 0cm'
  width=4><p class='MsoNormal'>&nbsp;</td>
 </tr>
 <tr style='mso-yfti-irow:14;mso-yfti-lastrow:yes;height:15.6pt;mso-row-margin-right:
  3.35pt'>
  <td width=75 nowrap style='width:56.45pt;border:solid windowtext 1.0pt;
  border-top:none;mso-border-left-alt:solid windowtext .5pt;mso-border-bottom-alt:
  solid windowtext .5pt;mso-border-right-alt:solid windowtext .5pt;padding:
  0cm 5.4pt 0cm 5.4pt;height:15.6pt'>
  <p class=MsoNormal align=center style='text-align:center;mso-pagination:widow-orphan'><span
  lang=EN-US style='font-size:6.5pt;mso-bidi-font-size:10.0pt;font-family:"Times New Roman",serif;
  mso-fareast-font-family:等线;mso-font-kerning:0pt'>DPARN-LSA<o:p></o:p></span></p>
  </td>
  <td width=39 nowrap style='width:29.5pt;border:none;border-bottom:solid windowtext 1.0pt;
  mso-border-left-alt:solid windowtext .5pt;mso-border-left-alt:solid windowtext .5pt;
  mso-border-bottom-alt:solid windowtext .5pt;padding:0cm 5.4pt 0cm 5.4pt;
  height:15.6pt'>
  <p class=MsoNormal align=center style='text-align:center;mso-pagination:widow-orphan'><b
  style='mso-bidi-font-weight:normal'><span lang=EN-US style='font-size:6.5pt;
  mso-bidi-font-size:10.0pt;font-family:"Times New Roman",serif;mso-fareast-font-family:
  等线;mso-font-kerning:0pt'>1.776</span></b><!--[if gte msEquation 12]><m:oMath><b
   style='mso-bidi-font-weight:normal'><span lang=EN-US style='font-size:6.5pt;
   mso-bidi-font-size:10.0pt;font-family:"Cambria Math",serif;mso-fareast-font-family:
   等线;mso-bidi-font-family:"Times New Roman";mso-font-kerning:0pt'><m:r><m:rPr><m:scr
      m:val="roman"/><m:sty m:val="b"/></m:rPr>↑</m:r></span></b></m:oMath><![endif]--><![if !msEquation]><span
  lang=EN-US style='font-size:10.5pt;mso-bidi-font-size:11.0pt;font-family:
  等线;mso-ascii-theme-font:minor-latin;mso-fareast-theme-font:minor-fareast;
  mso-hansi-theme-font:minor-latin;mso-bidi-font-family:"Times New Roman";
  mso-bidi-theme-font:minor-bidi;position:relative;top:5.5pt;mso-text-raise:
  -5.5pt;mso-ansi-language:EN-US;mso-fareast-language:ZH-CN;mso-bidi-language:
  AR-SA'><!--[if gte vml 1]><v:shape id="_x0000_i1025" type="#_x0000_t75"
   style='width:2.85pt;height:15.7pt'>
   <v:imagedata src="Table_readme.files/image001.png" o:title="" chromakey="white"/>
  </v:shape><![endif]--><![if !vml]><img width=4 height=21
  src="Table_readme.files/image002.png" v:shapes="_x0000_i1025"><![endif]></span><![endif]><b
  style='mso-bidi-font-weight:normal'><span lang=EN-US style='font-size:6.5pt;
  mso-bidi-font-size:10.0pt;font-family:"Times New Roman",serif;mso-fareast-font-family:
  等线;mso-font-kerning:0pt'><o:p></o:p></span></b></p>
  </td>
  <td width=39 nowrap style='width:29.55pt;border:none;border-bottom:solid windowtext 1.0pt;
  mso-border-bottom-alt:solid windowtext .5pt;padding:0cm 5.4pt 0cm 5.4pt;
  height:15.6pt'>
  <p class=MsoNormal align=center style='text-align:center;mso-pagination:widow-orphan'><b
  style='mso-bidi-font-weight:normal'><span lang=EN-US style='font-size:6.5pt;
  mso-bidi-font-size:10.0pt;font-family:"Times New Roman",serif;mso-fareast-font-family:
  等线;mso-font-kerning:0pt'>2.423</span></b><!--[if gte msEquation 12]><m:oMath><b
   style='mso-bidi-font-weight:normal'><span lang=EN-US style='font-size:6.5pt;
   mso-bidi-font-size:10.0pt;font-family:"Cambria Math",serif;mso-fareast-font-family:
   等线;mso-bidi-font-family:"Times New Roman";mso-font-kerning:0pt'><m:r><m:rPr><m:scr
      m:val="roman"/><m:sty m:val="b"/></m:rPr>↑</m:r></span></b></m:oMath><![endif]--><![if !msEquation]><span
  lang=EN-US style='font-size:10.5pt;mso-bidi-font-size:11.0pt;font-family:
  等线;mso-ascii-theme-font:minor-latin;mso-fareast-theme-font:minor-fareast;
  mso-hansi-theme-font:minor-latin;mso-bidi-font-family:"Times New Roman";
  mso-bidi-theme-font:minor-bidi;position:relative;top:5.5pt;mso-text-raise:
  -5.5pt;mso-ansi-language:EN-US;mso-fareast-language:ZH-CN;mso-bidi-language:
  AR-SA'><!--[if gte vml 1]><v:shape id="_x0000_i1025" type="#_x0000_t75"
   style='width:2.85pt;height:15.7pt'>
   <v:imagedata src="Table_readme.files/image001.png" o:title="" chromakey="white"/>
  </v:shape><![endif]--><![if !vml]><img width=4 height=21
  src="Table_readme.files/image002.png" v:shapes="_x0000_i1025"><![endif]></span><![endif]><b
  style='mso-bidi-font-weight:normal'><span lang=EN-US style='font-size:6.5pt;
  mso-bidi-font-size:10.0pt;font-family:"Times New Roman",serif;mso-fareast-font-family:
  等线;mso-font-kerning:0pt'><o:p></o:p></span></b></p>
  </td>
  <td width=39 nowrap style='width:29.55pt;border-top:none;border-left:none;
  border-bottom:solid windowtext 1.0pt;border-right:solid windowtext 1.0pt;
  mso-border-bottom-alt:solid windowtext .5pt;mso-border-right-alt:solid windowtext .5pt;
  padding:0cm 5.4pt 0cm 5.4pt;height:15.6pt'>
  <p class=MsoNormal align=center style='text-align:center;mso-pagination:widow-orphan'><b
  style='mso-bidi-font-weight:normal'><span lang=EN-US style='font-size:6.5pt;
  mso-bidi-font-size:10.0pt;font-family:"Times New Roman",serif;mso-fareast-font-family:
  等线;mso-font-kerning:0pt'>2.237</span></b><!--[if gte msEquation 12]><m:oMath><b
   style='mso-bidi-font-weight:normal'><span lang=EN-US style='font-size:6.5pt;
   mso-bidi-font-size:10.0pt;font-family:"Cambria Math",serif;mso-fareast-font-family:
   等线;mso-bidi-font-family:"Times New Roman";mso-font-kerning:0pt'><m:r><m:rPr><m:scr
      m:val="roman"/><m:sty m:val="b"/></m:rPr>↑</m:r></span></b></m:oMath><![endif]--><![if !msEquation]><span
  lang=EN-US style='font-size:10.5pt;mso-bidi-font-size:11.0pt;font-family:
  等线;mso-ascii-theme-font:minor-latin;mso-fareast-theme-font:minor-fareast;
  mso-hansi-theme-font:minor-latin;mso-bidi-font-family:"Times New Roman";
  mso-bidi-theme-font:minor-bidi;position:relative;top:5.5pt;mso-text-raise:
  -5.5pt;mso-ansi-language:EN-US;mso-fareast-language:ZH-CN;mso-bidi-language:
  AR-SA'><!--[if gte vml 1]><v:shape id="_x0000_i1025" type="#_x0000_t75"
   style='width:2.85pt;height:15.7pt'>
   <v:imagedata src="Table_readme.files/image001.png" o:title="" chromakey="white"/>
  </v:shape><![endif]--><![if !vml]><img width=4 height=21
  src="Table_readme.files/image002.png" v:shapes="_x0000_i1025"><![endif]></span><![endif]><b
  style='mso-bidi-font-weight:normal'><span lang=EN-US style='font-size:6.5pt;
  mso-bidi-font-size:10.0pt;font-family:"Times New Roman",serif;mso-fareast-font-family:
  等线;mso-font-kerning:0pt'><o:p></o:p></span></b></p>
  </td>
  <td width=39 nowrap style='width:29.5pt;border:none;border-bottom:solid windowtext 1.0pt;
  mso-border-left-alt:solid windowtext .5pt;mso-border-left-alt:solid windowtext .5pt;
  mso-border-bottom-alt:solid windowtext .5pt;padding:0cm 5.4pt 0cm 5.4pt;
  height:15.6pt'>
  <p class=MsoNormal align=center style='text-align:center;mso-pagination:widow-orphan'><b
  style='mso-bidi-font-weight:normal'><span lang=EN-US style='font-size:6.5pt;
  font-family:"Times New Roman",serif;mso-fareast-font-family:等线;mso-font-kerning:
  0pt'>3.179</span></b><!--[if gte msEquation 12]><m:oMath><b style='mso-bidi-font-weight:
   normal'><span lang=EN-US style='font-size:6.5pt;mso-bidi-font-size:10.0pt;
   font-family:"Cambria Math",serif;mso-fareast-font-family:等线;mso-bidi-font-family:
   "Times New Roman";mso-font-kerning:0pt'><m:r><m:rPr><m:scr m:val="roman"/><m:sty
      m:val="b"/></m:rPr>↑</m:r></span></b></m:oMath><![endif]--><![if !msEquation]><span
  lang=EN-US style='font-size:10.5pt;mso-bidi-font-size:11.0pt;font-family:
  等线;mso-ascii-theme-font:minor-latin;mso-fareast-theme-font:minor-fareast;
  mso-hansi-theme-font:minor-latin;mso-bidi-font-family:"Times New Roman";
  mso-bidi-theme-font:minor-bidi;position:relative;top:5.5pt;mso-text-raise:
  -5.5pt;mso-ansi-language:EN-US;mso-fareast-language:ZH-CN;mso-bidi-language:
  AR-SA'><!--[if gte vml 1]><v:shape id="_x0000_i1025" type="#_x0000_t75"
   style='width:2.85pt;height:15.7pt'>
   <v:imagedata src="Table_readme.files/image001.png" o:title="" chromakey="white"/>
  </v:shape><![endif]--><![if !vml]><img width=4 height=21
  src="Table_readme.files/image002.png" v:shapes="_x0000_i1025"><![endif]></span><![endif]><b
  style='mso-bidi-font-weight:normal'><span lang=EN-US style='font-size:6.5pt;
  font-family:"Times New Roman",serif;mso-fareast-font-family:等线;mso-font-kerning:
  0pt'><o:p></o:p></span></b></p>
  </td>
  <td width=39 nowrap style='width:29.55pt;border:none;border-bottom:solid windowtext 1.0pt;
  mso-border-bottom-alt:solid windowtext .5pt;padding:0cm 5.4pt 0cm 5.4pt;
  height:15.6pt'>
  <p class=MsoNormal align=center style='text-align:center;mso-pagination:widow-orphan'><b
  style='mso-bidi-font-weight:normal'><span lang=EN-US style='font-size:6.5pt;
  font-family:"Times New Roman",serif;mso-fareast-font-family:等线;mso-font-kerning:
  0pt'>3.829</span></b><!--[if gte msEquation 12]><m:oMath><b style='mso-bidi-font-weight:
   normal'><span lang=EN-US style='font-size:6.5pt;mso-bidi-font-size:10.0pt;
   font-family:"Cambria Math",serif;mso-fareast-font-family:等线;mso-bidi-font-family:
   "Times New Roman";mso-font-kerning:0pt'><m:r><m:rPr><m:scr m:val="roman"/><m:sty
      m:val="b"/></m:rPr>↑</m:r></span></b></m:oMath><![endif]--><![if !msEquation]><span
  lang=EN-US style='font-size:10.5pt;mso-bidi-font-size:11.0pt;font-family:
  等线;mso-ascii-theme-font:minor-latin;mso-fareast-theme-font:minor-fareast;
  mso-hansi-theme-font:minor-latin;mso-bidi-font-family:"Times New Roman";
  mso-bidi-theme-font:minor-bidi;position:relative;top:5.5pt;mso-text-raise:
  -5.5pt;mso-ansi-language:EN-US;mso-fareast-language:ZH-CN;mso-bidi-language:
  AR-SA'><!--[if gte vml 1]><v:shape id="_x0000_i1025" type="#_x0000_t75"
   style='width:2.85pt;height:15.7pt'>
   <v:imagedata src="Table_readme.files/image001.png" o:title="" chromakey="white"/>
  </v:shape><![endif]--><![if !vml]><img width=4 height=21
  src="Table_readme.files/image002.png" v:shapes="_x0000_i1025"><![endif]></span><![endif]><b
  style='mso-bidi-font-weight:normal'><span lang=EN-US style='font-size:6.5pt;
  font-family:"Times New Roman",serif;mso-fareast-font-family:等线;mso-font-kerning:
  0pt'><o:p></o:p></span></b></p>
  </td>
  <td width=39 nowrap style='width:29.55pt;border-top:none;border-left:none;
  border-bottom:solid windowtext 1.0pt;border-right:solid windowtext 1.0pt;
  mso-border-bottom-alt:solid windowtext .5pt;mso-border-right-alt:solid windowtext .5pt;
  padding:0cm 5.4pt 0cm 5.4pt;height:15.6pt'>
  <p class=MsoNormal align=center style='text-align:center;mso-pagination:widow-orphan'><b
  style='mso-bidi-font-weight:normal'><span lang=EN-US style='font-size:6.5pt;
  font-family:"Times New Roman",serif;mso-fareast-font-family:等线;mso-font-kerning:
  0pt'>3.642</span></b><!--[if gte msEquation 12]><m:oMath><b style='mso-bidi-font-weight:
   normal'><span lang=EN-US style='font-size:6.5pt;mso-bidi-font-size:10.0pt;
   font-family:"Cambria Math",serif;mso-fareast-font-family:等线;mso-bidi-font-family:
   "Times New Roman";mso-font-kerning:0pt'><m:r><m:rPr><m:scr m:val="roman"/><m:sty
      m:val="b"/></m:rPr>↑</m:r></span></b></m:oMath><![endif]--><![if !msEquation]><span
  lang=EN-US style='font-size:10.5pt;mso-bidi-font-size:11.0pt;font-family:
  等线;mso-ascii-theme-font:minor-latin;mso-fareast-theme-font:minor-fareast;
  mso-hansi-theme-font:minor-latin;mso-bidi-font-family:"Times New Roman";
  mso-bidi-theme-font:minor-bidi;position:relative;top:5.5pt;mso-text-raise:
  -5.5pt;mso-ansi-language:EN-US;mso-fareast-language:ZH-CN;mso-bidi-language:
  AR-SA'><!--[if gte vml 1]><v:shape id="_x0000_i1025" type="#_x0000_t75"
   style='width:2.85pt;height:15.7pt'>
   <v:imagedata src="Table_readme.files/image001.png" o:title="" chromakey="white"/>
  </v:shape><![endif]--><![if !vml]><img width=4 height=21
  src="Table_readme.files/image002.png" v:shapes="_x0000_i1025"><![endif]></span><![endif]><b
  style='mso-bidi-font-weight:normal'><span lang=EN-US style='font-size:6.5pt;
  font-family:"Times New Roman",serif;mso-fareast-font-family:等线;mso-font-kerning:
  0pt'><o:p></o:p></span></b></p>
  </td>
  <td width=39 nowrap style='width:29.5pt;border:none;border-bottom:solid windowtext 1.0pt;
  mso-border-left-alt:solid windowtext .5pt;mso-border-left-alt:solid windowtext .5pt;
  mso-border-bottom-alt:solid windowtext .5pt;padding:0cm 5.4pt 0cm 5.4pt;
  height:15.6pt'>
  <p class=MsoNormal align=center style='text-align:center;mso-pagination:widow-orphan'><b
  style='mso-bidi-font-weight:normal'><span lang=EN-US style='font-size:6.5pt;
  font-family:"Times New Roman",serif;mso-fareast-font-family:等线;mso-font-kerning:
  0pt'>2.619</span></b><!--[if gte msEquation 12]><m:oMath><b style='mso-bidi-font-weight:
   normal'><span lang=EN-US style='font-size:6.5pt;mso-bidi-font-size:10.0pt;
   font-family:"Cambria Math",serif;mso-fareast-font-family:等线;mso-bidi-font-family:
   "Times New Roman";mso-font-kerning:0pt'><m:r><m:rPr><m:scr m:val="roman"/><m:sty
      m:val="b"/></m:rPr>↑</m:r></span></b></m:oMath><![endif]--><![if !msEquation]><span
  lang=EN-US style='font-size:10.5pt;mso-bidi-font-size:11.0pt;font-family:
  等线;mso-ascii-theme-font:minor-latin;mso-fareast-theme-font:minor-fareast;
  mso-hansi-theme-font:minor-latin;mso-bidi-font-family:"Times New Roman";
  mso-bidi-theme-font:minor-bidi;position:relative;top:5.5pt;mso-text-raise:
  -5.5pt;mso-ansi-language:EN-US;mso-fareast-language:ZH-CN;mso-bidi-language:
  AR-SA'><!--[if gte vml 1]><v:shape id="_x0000_i1025" type="#_x0000_t75"
   style='width:2.85pt;height:15.7pt'>
   <v:imagedata src="Table_readme.files/image001.png" o:title="" chromakey="white"/>
  </v:shape><![endif]--><![if !vml]><img width=4 height=21
  src="Table_readme.files/image002.png" v:shapes="_x0000_i1025"><![endif]></span><![endif]><b
  style='mso-bidi-font-weight:normal'><span lang=EN-US style='font-size:6.5pt;
  font-family:"Times New Roman",serif;mso-fareast-font-family:等线;mso-font-kerning:
  0pt'><o:p></o:p></span></b></p>
  </td>
  <td width=39 nowrap style='width:29.55pt;border:none;border-bottom:solid windowtext 1.0pt;
  mso-border-bottom-alt:solid windowtext .5pt;padding:0cm 5.4pt 0cm 5.4pt;
  height:15.6pt'>
  <p class=MsoNormal align=center style='text-align:center;mso-pagination:widow-orphan'><b
  style='mso-bidi-font-weight:normal'><span lang=EN-US style='font-size:6.5pt;
  font-family:"Times New Roman",serif;mso-fareast-font-family:等线;mso-font-kerning:
  0pt'>3.030</span></b><!--[if gte msEquation 12]><m:oMath><b style='mso-bidi-font-weight:
   normal'><span lang=EN-US style='font-size:6.5pt;mso-bidi-font-size:10.0pt;
   font-family:"Cambria Math",serif;mso-fareast-font-family:等线;mso-bidi-font-family:
   "Times New Roman";mso-font-kerning:0pt'><m:r><m:rPr><m:scr m:val="roman"/><m:sty
      m:val="b"/></m:rPr>↑</m:r></span></b></m:oMath><![endif]--><![if !msEquation]><span
  lang=EN-US style='font-size:10.5pt;mso-bidi-font-size:11.0pt;font-family:
  等线;mso-ascii-theme-font:minor-latin;mso-fareast-theme-font:minor-fareast;
  mso-hansi-theme-font:minor-latin;mso-bidi-font-family:"Times New Roman";
  mso-bidi-theme-font:minor-bidi;position:relative;top:5.5pt;mso-text-raise:
  -5.5pt;mso-ansi-language:EN-US;mso-fareast-language:ZH-CN;mso-bidi-language:
  AR-SA'><!--[if gte vml 1]><v:shape id="_x0000_i1025" type="#_x0000_t75"
   style='width:2.85pt;height:15.7pt'>
   <v:imagedata src="Table_readme.files/image001.png" o:title="" chromakey="white"/>
  </v:shape><![endif]--><![if !vml]><img width=4 height=21
  src="Table_readme.files/image002.png" v:shapes="_x0000_i1025"><![endif]></span><![endif]><b
  style='mso-bidi-font-weight:normal'><span lang=EN-US style='font-size:6.5pt;
  font-family:"Times New Roman",serif;mso-fareast-font-family:等线;mso-font-kerning:
  0pt'><o:p></o:p></span></b></p>
  </td>
  <td width=39 nowrap style='width:29.55pt;border-top:none;border-left:none;
  border-bottom:solid windowtext 1.0pt;border-right:solid windowtext 1.0pt;
  mso-border-bottom-alt:solid windowtext .5pt;mso-border-right-alt:solid windowtext .5pt;
  padding:0cm 5.4pt 0cm 5.4pt;height:15.6pt'>
  <p class=MsoNormal align=center style='text-align:center;mso-pagination:widow-orphan'><b
  style='mso-bidi-font-weight:normal'><span lang=EN-US style='font-size:6.5pt;
  font-family:"Times New Roman",serif;mso-fareast-font-family:等线;mso-font-kerning:
  0pt'>2.912</span></b><!--[if gte msEquation 12]><m:oMath><b style='mso-bidi-font-weight:
   normal'><span lang=EN-US style='font-size:6.5pt;mso-bidi-font-size:10.0pt;
   font-family:"Cambria Math",serif;mso-fareast-font-family:等线;mso-bidi-font-family:
   "Times New Roman";mso-font-kerning:0pt'><m:r><m:rPr><m:scr m:val="roman"/><m:sty
      m:val="b"/></m:rPr>↑</m:r></span></b></m:oMath><![endif]--><![if !msEquation]><span
  lang=EN-US style='font-size:10.5pt;mso-bidi-font-size:11.0pt;font-family:
  等线;mso-ascii-theme-font:minor-latin;mso-fareast-theme-font:minor-fareast;
  mso-hansi-theme-font:minor-latin;mso-bidi-font-family:"Times New Roman";
  mso-bidi-theme-font:minor-bidi;position:relative;top:5.5pt;mso-text-raise:
  -5.5pt;mso-ansi-language:EN-US;mso-fareast-language:ZH-CN;mso-bidi-language:
  AR-SA'><!--[if gte vml 1]><v:shape id="_x0000_i1025" type="#_x0000_t75"
   style='width:2.85pt;height:15.7pt'>
   <v:imagedata src="Table_readme.files/image001.png" o:title="" chromakey="white"/>
  </v:shape><![endif]--><![if !vml]><img width=4 height=21
  src="Table_readme.files/image002.png" v:shapes="_x0000_i1025"><![endif]></span><![endif]><b
  style='mso-bidi-font-weight:normal'><span lang=EN-US style='font-size:6.5pt;
  font-family:"Times New Roman",serif;mso-fareast-font-family:等线;mso-font-kerning:
  0pt'><o:p></o:p></span></b></p>
  </td>
  <td width=39 style='width:29.5pt;border:none;border-bottom:solid windowtext 1.0pt;
  mso-border-left-alt:solid windowtext .5pt;mso-border-left-alt:solid windowtext .5pt;
  mso-border-bottom-alt:solid windowtext .5pt;padding:0cm 5.4pt 0cm 5.4pt;
  height:15.6pt'>
  <p class=MsoNormal align=center style='text-align:center;mso-pagination:widow-orphan'><b
  style='mso-bidi-font-weight:normal'><span lang=EN-US style='font-size:6.5pt;
  font-family:"Times New Roman",serif;mso-fareast-font-family:等线;mso-font-kerning:
  0pt'>2.507</span></b><!--[if gte msEquation 12]><m:oMath><b style='mso-bidi-font-weight:
   normal'><span lang=EN-US style='font-size:6.5pt;mso-bidi-font-size:10.0pt;
   font-family:"Cambria Math",serif;mso-fareast-font-family:等线;mso-bidi-font-family:
   "Times New Roman";mso-font-kerning:0pt'><m:r><m:rPr><m:scr m:val="roman"/><m:sty
      m:val="b"/></m:rPr>↑</m:r></span></b></m:oMath><![endif]--><![if !msEquation]><span
  lang=EN-US style='font-size:10.5pt;mso-bidi-font-size:11.0pt;font-family:
  等线;mso-ascii-theme-font:minor-latin;mso-fareast-theme-font:minor-fareast;
  mso-hansi-theme-font:minor-latin;mso-bidi-font-family:"Times New Roman";
  mso-bidi-theme-font:minor-bidi;position:relative;top:5.5pt;mso-text-raise:
  -5.5pt;mso-ansi-language:EN-US;mso-fareast-language:ZH-CN;mso-bidi-language:
  AR-SA'><!--[if gte vml 1]><v:shape id="_x0000_i1025" type="#_x0000_t75"
   style='width:2.85pt;height:15.7pt'>
   <v:imagedata src="Table_readme.files/image001.png" o:title="" chromakey="white"/>
  </v:shape><![endif]--><![if !vml]><img width=4 height=21
  src="Table_readme.files/image002.png" v:shapes="_x0000_i1025"><![endif]></span><![endif]><b
  style='mso-bidi-font-weight:normal'><span lang=EN-US style='font-size:6.5pt;
  font-family:"Times New Roman",serif;mso-fareast-font-family:等线;mso-font-kerning:
  0pt'><o:p></o:p></span></b></p>
  </td>
  <td width=39 style='width:29.55pt;border:none;border-bottom:solid windowtext 1.0pt;
  mso-border-bottom-alt:solid windowtext .5pt;padding:0cm 5.4pt 0cm 5.4pt;
  height:15.6pt'>
  <p class=MsoNormal align=center style='text-align:center;mso-pagination:widow-orphan'><b
  style='mso-bidi-font-weight:normal'><span lang=EN-US style='font-size:6.5pt;
  font-family:"Times New Roman",serif;mso-fareast-font-family:等线;mso-font-kerning:
  0pt'>3.166</span></b><!--[if gte msEquation 12]><m:oMath><b style='mso-bidi-font-weight:
   normal'><span lang=EN-US style='font-size:6.5pt;mso-bidi-font-size:10.0pt;
   font-family:"Cambria Math",serif;mso-fareast-font-family:等线;mso-bidi-font-family:
   "Times New Roman";mso-font-kerning:0pt'><m:r><m:rPr><m:scr m:val="roman"/><m:sty
      m:val="b"/></m:rPr>↑</m:r></span></b></m:oMath><![endif]--><![if !msEquation]><span
  lang=EN-US style='font-size:10.5pt;mso-bidi-font-size:11.0pt;font-family:
  等线;mso-ascii-theme-font:minor-latin;mso-fareast-theme-font:minor-fareast;
  mso-hansi-theme-font:minor-latin;mso-bidi-font-family:"Times New Roman";
  mso-bidi-theme-font:minor-bidi;position:relative;top:5.5pt;mso-text-raise:
  -5.5pt;mso-ansi-language:EN-US;mso-fareast-language:ZH-CN;mso-bidi-language:
  AR-SA'><!--[if gte vml 1]><v:shape id="_x0000_i1025" type="#_x0000_t75"
   style='width:2.85pt;height:15.7pt'>
   <v:imagedata src="Table_readme.files/image001.png" o:title="" chromakey="white"/>
  </v:shape><![endif]--><![if !vml]><img width=4 height=21
  src="Table_readme.files/image002.png" v:shapes="_x0000_i1025"><![endif]></span><![endif]><b
  style='mso-bidi-font-weight:normal'><span lang=EN-US style='font-size:6.5pt;
  font-family:"Times New Roman",serif;mso-fareast-font-family:等线;mso-font-kerning:
  0pt'><o:p></o:p></span></b></p>
  </td>
  <td width=39 style='width:29.55pt;border-top:none;border-left:none;
  border-bottom:solid windowtext 1.0pt;border-right:solid windowtext 1.0pt;
  mso-border-bottom-alt:solid windowtext .5pt;mso-border-right-alt:solid windowtext .5pt;
  padding:0cm 5.4pt 0cm 5.4pt;height:15.6pt'>
  <p class=MsoNormal align=center style='text-align:center;mso-pagination:widow-orphan'><b
  style='mso-bidi-font-weight:normal'><span lang=EN-US style='font-size:6.5pt;
  font-family:"Times New Roman",serif;mso-fareast-font-family:等线;mso-font-kerning:
  0pt'>2.977</span></b><!--[if gte msEquation 12]><m:oMath><b style='mso-bidi-font-weight:
   normal'><span lang=EN-US style='font-size:6.5pt;mso-bidi-font-size:10.0pt;
   font-family:"Cambria Math",serif;mso-fareast-font-family:等线;mso-bidi-font-family:
   "Times New Roman";mso-font-kerning:0pt'><m:r><m:rPr><m:scr m:val="roman"/><m:sty
      m:val="b"/></m:rPr>↑</m:r></span></b></m:oMath><![endif]--><![if !msEquation]><span
  lang=EN-US style='font-size:10.5pt;mso-bidi-font-size:11.0pt;font-family:
  等线;mso-ascii-theme-font:minor-latin;mso-fareast-theme-font:minor-fareast;
  mso-hansi-theme-font:minor-latin;mso-bidi-font-family:"Times New Roman";
  mso-bidi-theme-font:minor-bidi;position:relative;top:5.5pt;mso-text-raise:
  -5.5pt;mso-ansi-language:EN-US;mso-fareast-language:ZH-CN;mso-bidi-language:
  AR-SA'><!--[if gte vml 1]><v:shape id="_x0000_i1025" type="#_x0000_t75"
   style='width:2.85pt;height:15.7pt'>
   <v:imagedata src="Table_readme.files/image001.png" o:title="" chromakey="white"/>
  </v:shape><![endif]--><![if !vml]><img width=4 height=21
  src="Table_readme.files/image002.png" v:shapes="_x0000_i1025"><![endif]></span><![endif]><b
  style='mso-bidi-font-weight:normal'><span lang=EN-US style='font-size:6.5pt;
  font-family:"Times New Roman",serif;mso-fareast-font-family:等线;mso-font-kerning:
  0pt'><o:p></o:p></span></b></p>
  </td>
  <td style='mso-cell-special:placeholder;border:none;padding:0cm 0cm 0cm 0cm'
  width=4><p class='MsoNormal'>&nbsp;</td>
 </tr>
</table>

</div>

<p class=MsoNormal><span lang=EN-US><o:p>&nbsp;</o:p></span></p>

</div>

</body>

