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
The training process can be seen in figures below, where both LSA-based models achieve better convergence compared with the original models. 
<!-- ![trainloss_dparn](https://user-images.githubusercontent.com/103247057/222072535-ab00598c-448d-47ff-9cf3-77e7da8d302c.jpeg) -->
<!-- ![Validation](https://user-images.githubusercontent.com/103247057/222072910-c3c7730d-b7be-45ca-9591-5706c395f1ad.jpeg) -->
<div align=center><img src="https://user-images.githubusercontent.com/103247057/222072535-ab00598c-448d-47ff-9cf3-77e7da8d302c.jpeg" width="320" height="240" /><img src="https://user-images.githubusercontent.com/103247057/222072910-c3c7730d-b7be-45ca-9591-5706c395f1ad.jpeg" width="320" height="240" /></div>
<!-- ![trainloss_mtfaa](https://user-images.githubusercontent.com/103247057/222074817-58680383-6d61-42c6-9f94-eb4c58ecefab.jpeg)
![Validation_mtfaa](https://user-images.githubusercontent.com/103247057/222074828-44bc05e4-c05e-40bc-b4b5-060cdb7030a7.jpeg) -->
<div align=center><img src="https://user-images.githubusercontent.com/103247057/222074817-58680383-6d61-42c6-9f94-eb4c58ecefab.jpeg" width="320" height="240" /><img src="https://user-images.githubusercontent.com/103247057/222074828-44bc05e4-c05e-40bc-b4b5-060cdb7030a7.jpeg" width="320" height="240" /></div>

<table class=MsoNormalTable border=0 cellspacing=0 cellpadding=0 width=246
 style='width:184.3pt;border-collapse:collapse;mso-yfti-tbllook:1184;
 mso-padding-alt:0cm 1.4pt 0cm 1.4pt'>
 <tr style='mso-yfti-irow:0;mso-yfti-firstrow:yes;height:11.35pt;mso-row-margin-right:
  7.1pt'>
  <td width=57 colspan=2 valign=bottom style='width:42.55pt;border:solid windowtext 1.0pt;
  border-left:none;mso-border-top-alt:solid windowtext .5pt;mso-border-bottom-alt:
  solid windowtext .5pt;mso-border-right-alt:solid windowtext .5pt;padding:
  0cm 1.4pt 0cm 1.4pt;height:11.35pt'>
  <p class=MsoNormal align=center style='text-align:center;mso-pagination:widow-orphan'><span
  lang=EN-US style='font-size:6.5pt;mso-bidi-font-size:7.5pt;font-family:"Times New Roman",serif;
  mso-fareast-font-family:等线;color:black;mso-bidi-font-weight:bold'>Config.<o:p></o:p></span></p>
  </td>
  <td width=113 nowrap colspan=4 valign=bottom style='width:3.0cm;border:solid windowtext 1.0pt;
  border-left:none;mso-border-left-alt:solid windowtext .5pt;mso-border-alt:
  solid windowtext .5pt;padding:0cm 1.4pt 0cm 1.4pt;height:11.35pt'>
  <p class=MsoNormal align=center style='text-align:center;mso-pagination:widow-orphan'><span
  lang=EN-US style='font-size:6.5pt;mso-bidi-font-size:7.5pt;font-family:"Times New Roman",serif;
  mso-fareast-font-family:等线;color:black'>Wideband Metrics<o:p></o:p></span></p>
  </td>
  <td width=66 nowrap colspan=2 valign=bottom style='width:49.6pt;border-top:
  solid windowtext 1.0pt;border-left:none;border-bottom:solid windowtext 1.0pt;
  border-right:none;mso-border-left-alt:solid windowtext .5pt;mso-border-top-alt:
  solid windowtext .5pt;mso-border-left-alt:solid windowtext .5pt;mso-border-bottom-alt:
  solid windowtext .5pt;padding:0cm 1.4pt 0cm 1.4pt;height:11.35pt'>
  <p class=MsoNormal align=center style='text-align:center;mso-pagination:widow-orphan'><span
  lang=EN-US style='font-size:6.5pt;mso-bidi-font-size:7.5pt;font-family:"Times New Roman",serif;
  mso-fareast-font-family:等线;color:black'>Full-band Metrics<o:p></o:p></span></p>
  </td>
  <td style='mso-cell-special:placeholder;border:none;border-bottom:solid windowtext 1.0pt'
  width=9><p class='MsoNormal'>&nbsp;</td>
 </tr>
 <tr style='mso-yfti-irow:1;height:11.35pt'>
  <td width=28 valign=bottom style='width:21.3pt;border-top:none;border-left:
  none;border-bottom:solid windowtext 1.0pt;border-right:solid windowtext 1.0pt;
  mso-border-top-alt:solid windowtext .5pt;mso-border-top-alt:solid windowtext .5pt;
  mso-border-bottom-alt:solid windowtext .5pt;mso-border-right-alt:solid windowtext .5pt;
  padding:0cm 1.4pt 0cm 1.4pt;height:11.35pt'>
  <p class=MsoNormal align=center style='text-align:center;mso-pagination:widow-orphan'><span
  lang=EN-US style='font-size:5.5pt;mso-bidi-font-size:6.5pt;font-family:"Times New Roman",serif;
  mso-fareast-font-family:等线;color:black;mso-bidi-font-weight:bold'>Model<o:p></o:p></span></p>
  </td>
  <td width=28 valign=bottom style='width:21.25pt;border-top:none;border-left:
  none;border-bottom:solid windowtext 1.0pt;border-right:solid windowtext 1.0pt;
  mso-border-top-alt:solid windowtext .5pt;mso-border-left-alt:solid windowtext .5pt;
  mso-border-alt:solid windowtext .5pt;padding:0cm 1.4pt 0cm 1.4pt;height:11.35pt'>
  <p class=MsoNormal align=center style='text-align:center;mso-pagination:widow-orphan'><!--[if gte msEquation 12]><m:oMathPara><m:oMath><m:sSub><m:sSubPr><span
      style='font-size:5.5pt;mso-ansi-font-size:5.5pt;mso-bidi-font-size:6.5pt;
      font-family:"Cambria Math",serif;mso-ascii-font-family:"Cambria Math";
      mso-hansi-font-family:"Cambria Math";mso-bidi-font-family:"Times New Roman";
      font-style:italic;mso-bidi-font-style:normal'><m:ctrlPr></m:ctrlPr></span></m:sSubPr><m:e><i
      style='mso-bidi-font-style:normal'><span lang=EN-US style='font-size:
      5.5pt;mso-bidi-font-size:6.5pt;font-family:"Cambria Math",serif;
      mso-bidi-font-family:"Times New Roman"'><m:r>N</m:r></span></i></m:e><m:sub><i
      style='mso-bidi-font-style:normal'><span lang=EN-US style='font-size:
      5.5pt;mso-bidi-font-size:6.5pt;font-family:"Cambria Math",serif;
      mso-bidi-font-family:"Times New Roman"'><m:r>l</m:r></span></i></m:sub></m:sSub></m:oMath></m:oMathPara><![endif]--><![if !msEquation]><span
  lang=EN-US style='font-size:10.5pt;mso-bidi-font-size:11.0pt;font-family:
  等线;mso-ascii-theme-font:minor-latin;mso-fareast-theme-font:minor-fareast;
  mso-hansi-theme-font:minor-latin;mso-bidi-font-family:"Times New Roman";
  mso-bidi-theme-font:minor-bidi;mso-ansi-language:EN-US;mso-fareast-language:
  ZH-CN;mso-bidi-language:AR-SA'><!--[if gte vml 1]><v:shape id="_x0000_i1025"
   type="#_x0000_t75" style='width:5.2pt;height:15.55pt'>
   <v:imagedata src="Table_readme.files/image003.png" o:title="" chromakey="white"/>
  </v:shape><![endif]--><![if !vml]><img width=7 height=21
  src="Table_readme.files/image004.png" v:shapes="_x0000_i1025"><![endif]></span><![endif]><span
  lang=EN-US style='font-size:5.5pt;mso-bidi-font-size:6.5pt;font-family:"Times New Roman",serif;
  mso-fareast-font-family:等线;color:black;mso-bidi-font-weight:bold'><o:p></o:p></span></p>
  </td>
  <td width=28 nowrap valign=bottom style='width:21.25pt;border:none;
  border-bottom:solid windowtext 1.0pt;mso-border-top-alt:solid windowtext .5pt;
  mso-border-left-alt:solid windowtext .5pt;mso-border-top-alt:solid windowtext .5pt;
  mso-border-left-alt:solid windowtext .5pt;mso-border-bottom-alt:solid windowtext .5pt;
  padding:0cm 1.4pt 0cm 1.4pt;height:11.35pt'>
  <p class=MsoNormal align=center style='text-align:center;mso-pagination:widow-orphan'><span
  lang=EN-US style='font-size:5.5pt;mso-bidi-font-size:6.5pt;font-family:"Times New Roman",serif;
  mso-fareast-font-family:等线;color:black'>PESQ<o:p></o:p></span></p>
  </td>
  <td width=28 valign=bottom style='width:21.25pt;border:none;border-bottom:
  solid windowtext 1.0pt;mso-border-top-alt:solid windowtext .5pt;mso-border-top-alt:
  solid windowtext .5pt;mso-border-bottom-alt:solid windowtext .5pt;padding:
  0cm 1.4pt 0cm 1.4pt;height:11.35pt'>
  <p class=MsoNormal align=center style='text-align:center;mso-pagination:widow-orphan'><span
  lang=EN-US style='font-size:5.5pt;mso-bidi-font-size:6.5pt;font-family:"Times New Roman",serif;
  mso-fareast-font-family:等线;color:black'>CSIG<o:p></o:p></span></p>
  </td>
  <td width=28 valign=bottom style='width:21.25pt;border:none;border-bottom:
  solid windowtext 1.0pt;mso-border-top-alt:solid windowtext .5pt;mso-border-top-alt:
  solid windowtext .5pt;mso-border-bottom-alt:solid windowtext .5pt;padding:
  0cm 1.4pt 0cm 1.4pt;height:11.35pt'>
  <p class=MsoNormal align=center style='text-align:center;mso-pagination:widow-orphan'><span
  lang=EN-US style='font-size:5.5pt;mso-bidi-font-size:6.5pt;font-family:"Times New Roman",serif;
  mso-fareast-font-family:等线;color:black'>CBAK<o:p></o:p></span></p>
  </td>
  <td width=28 valign=bottom style='width:21.3pt;border-top:none;border-left:
  none;border-bottom:solid windowtext 1.0pt;border-right:solid windowtext 1.0pt;
  mso-border-top-alt:solid windowtext .5pt;mso-border-top-alt:solid windowtext .5pt;
  mso-border-bottom-alt:solid windowtext .5pt;mso-border-right-alt:solid windowtext .5pt;
  padding:0cm 1.4pt 0cm 1.4pt;height:11.35pt'>
  <p class=MsoNormal align=center style='text-align:center;mso-pagination:widow-orphan'><span
  lang=EN-US style='font-size:5.5pt;mso-bidi-font-size:6.5pt;font-family:"Times New Roman",serif;
  mso-fareast-font-family:等线;color:black'>COVL<o:p></o:p></span></p>
  </td>
  <td width=38 nowrap valign=bottom style='width:1.0cm;border:none;border-bottom:
  solid windowtext 1.0pt;mso-border-top-alt:solid windowtext .5pt;mso-border-left-alt:
  solid windowtext .5pt;mso-border-top-alt:solid windowtext .5pt;mso-border-left-alt:
  solid windowtext .5pt;mso-border-bottom-alt:solid windowtext .5pt;padding:
  0cm 1.4pt 0cm 1.4pt;height:11.35pt'>
  <p class=MsoNormal align=center style='text-align:center;mso-pagination:widow-orphan'><span
  class=GramE><span lang=EN-US style='font-size:5.5pt;mso-bidi-font-size:6.5pt;
  font-family:"Times New Roman",serif;mso-fareast-font-family:等线;color:black'>STOI(</span></span><span
  lang=EN-US style='font-size:5.5pt;mso-bidi-font-size:6.5pt;font-family:"Times New Roman",serif;
  mso-fareast-font-family:等线;color:black'>%)<o:p></o:p></span></p>
  </td>
  <td width=38 nowrap colspan=2 valign=bottom style='width:1.0cm;border:none;
  border-bottom:solid windowtext 1.0pt;mso-border-top-alt:solid windowtext .5pt;
  mso-border-top-alt:solid windowtext .5pt;mso-border-bottom-alt:solid windowtext .5pt;
  padding:0cm 1.4pt 0cm 1.4pt;height:11.35pt'>
  <p class=MsoNormal align=center style='text-align:center;mso-pagination:widow-orphan'><span
  lang=EN-US style='font-size:5.5pt;mso-bidi-font-size:6.5pt;font-family:"Times New Roman",serif;
  mso-fareast-font-family:等线;color:black'>SiSDR(dB)<o:p></o:p></span></p>
  </td>
 </tr>
 <tr style='mso-yfti-irow:2;height:11.35pt'>
  <td width=28 rowspan=3 style='width:21.3pt;border:solid windowtext 1.0pt;
  border-top:none;mso-border-top-alt:solid windowtext .5pt;mso-border-alt:solid windowtext .5pt;
  padding:0cm 1.4pt 0cm 1.4pt;height:11.35pt'>
  <p class=MsoNormal align=center style='text-align:center;mso-pagination:widow-orphan'><span
  lang=EN-US style='font-size:5.0pt;mso-bidi-font-size:5.5pt;font-family:"Times New Roman",serif;
  mso-fareast-font-family:等线;color:black'>MTFAA<o:p></o:p></span></p>
  </td>
  <td width=28 style='width:21.25pt;border:none;border-right:solid windowtext 1.0pt;
  mso-border-top-alt:solid windowtext .5pt;mso-border-left-alt:solid windowtext .5pt;
  mso-border-top-alt:solid windowtext .5pt;mso-border-left-alt:solid windowtext .5pt;
  mso-border-right-alt:solid windowtext .5pt;padding:0cm 1.4pt 0cm 1.4pt;
  height:11.35pt'>
  <p class=MsoNormal align=center style='text-align:center;mso-pagination:widow-orphan'><!--[if gte msEquation 12]><m:oMathPara><m:oMath><m:sSup><m:sSupPr><span
      style='font-size:6.5pt;mso-ansi-font-size:6.5pt;mso-bidi-font-size:7.5pt;
      font-family:"Cambria Math",serif;mso-ascii-font-family:"Cambria Math";
      mso-hansi-font-family:"Cambria Math";mso-bidi-font-family:"Times New Roman";
      font-style:italic;mso-bidi-font-style:normal'><m:ctrlPr></m:ctrlPr></span></m:sSupPr><m:e><i
      style='mso-bidi-font-style:normal'><span lang=EN-US style='font-size:
      6.5pt;mso-bidi-font-size:7.5pt;font-family:"Cambria Math",serif;
      mso-bidi-font-family:"Times New Roman"'><m:r>F</m:r></span></i></m:e><m:sup><i
      style='mso-bidi-font-style:normal'><span lang=EN-US style='font-size:
      6.5pt;mso-bidi-font-size:7.5pt;font-family:"Cambria Math",serif;
      mso-bidi-font-family:"Times New Roman"'><m:r>'</m:r></span></i></m:sup></m:sSup><i
    style='mso-bidi-font-style:normal'><span lang=EN-US style='font-size:6.5pt;
    mso-bidi-font-size:7.5pt;font-family:"Cambria Math",serif;mso-bidi-font-family:
    "Times New Roman"'><m:r>/2</m:r></span></i></m:oMath></m:oMathPara><![endif]--><![if !msEquation]><span
  lang=EN-US style='font-size:10.5pt;mso-bidi-font-size:11.0pt;font-family:
  等线;mso-ascii-theme-font:minor-latin;mso-fareast-theme-font:minor-fareast;
  mso-hansi-theme-font:minor-latin;mso-bidi-font-family:"Times New Roman";
  mso-bidi-theme-font:minor-bidi;mso-ansi-language:EN-US;mso-fareast-language:
  ZH-CN;mso-bidi-language:AR-SA'><!--[if gte vml 1]><v:shape id="_x0000_i1025"
   type="#_x0000_t75" style='width:12.65pt;height:15.55pt'>
   <v:imagedata src="Table_readme.files/image005.png" o:title="" chromakey="white"/>
  </v:shape><![endif]--><![if !vml]><img width=17 height=21
  src="Table_readme.files/image006.png" v:shapes="_x0000_i1025"><![endif]></span><![endif]><span
  lang=EN-US style='font-size:6.5pt;mso-bidi-font-size:7.5pt;font-family:"Times New Roman",serif;
  mso-fareast-font-family:等线;color:black'><o:p></o:p></span></p>
  </td>
  <td width=28 nowrap style='width:21.25pt;border:none;mso-border-top-alt:solid windowtext .5pt;
  mso-border-left-alt:solid windowtext .5pt;padding:0cm 1.4pt 0cm 1.4pt;
  height:11.35pt'>
  <p class=MsoNormal align=center style='text-align:center;mso-pagination:widow-orphan'><span
  lang=EN-US style='font-size:7.5pt;mso-bidi-font-size:8.0pt;font-family:"Times New Roman",serif;
  mso-fareast-font-family:等线;color:black'>3.16<o:p></o:p></span></p>
  </td>
  <td width=28 style='width:21.25pt;border:none;mso-border-top-alt:solid windowtext .5pt;
  padding:0cm 1.4pt 0cm 1.4pt;height:11.35pt'>
  <p class=MsoNormal align=center style='text-align:center;mso-pagination:widow-orphan'><span
  lang=EN-US style='font-size:7.5pt;mso-bidi-font-size:8.0pt;font-family:"Times New Roman",serif;
  mso-fareast-font-family:等线;color:black'>4.34<o:p></o:p></span></p>
  </td>
  <td width=28 style='width:21.25pt;border:none;mso-border-top-alt:solid windowtext .5pt;
  padding:0cm 1.4pt 0cm 1.4pt;height:11.35pt'>
  <p class=MsoNormal align=center style='text-align:center;mso-pagination:widow-orphan'><b><span
  lang=EN-US style='font-size:7.5pt;mso-bidi-font-size:8.0pt;font-family:"Times New Roman",serif;
  mso-fareast-font-family:等线;color:black'>3.63<o:p></o:p></span></b></p>
  </td>
  <td width=28 style='width:21.3pt;border:none;border-right:solid windowtext 1.0pt;
  mso-border-top-alt:solid windowtext .5pt;mso-border-top-alt:solid windowtext .5pt;
  mso-border-right-alt:solid windowtext .5pt;padding:0cm 1.4pt 0cm 1.4pt;
  height:11.35pt'>
  <p class=MsoNormal align=center style='text-align:center;mso-pagination:widow-orphan'><span
  lang=EN-US style='font-size:7.5pt;mso-bidi-font-size:8.0pt;font-family:"Times New Roman",serif;
  mso-fareast-font-family:等线;color:black'>3.77<o:p></o:p></span></p>
  </td>
  <td width=38 nowrap style='width:1.0cm;border:none;mso-border-top-alt:solid windowtext .5pt;
  mso-border-left-alt:solid windowtext .5pt;padding:0cm 1.4pt 0cm 1.4pt;
  height:11.35pt'>
  <p class=MsoNormal align=center style='text-align:center;mso-pagination:widow-orphan'><span
  lang=EN-US style='font-size:7.5pt;mso-bidi-font-size:8.0pt;font-family:"Times New Roman",serif;
  mso-fareast-font-family:等线;color:black'>94.7<o:p></o:p></span></p>
  </td>
  <td width=38 nowrap colspan=2 style='width:1.0cm;border:none;mso-border-top-alt:
  solid windowtext .5pt;padding:0cm 1.4pt 0cm 1.4pt;height:11.35pt'>
  <p class=MsoNormal align=center style='text-align:center;mso-pagination:widow-orphan'><span
  lang=EN-US style='font-size:7.5pt;mso-bidi-font-size:8.0pt;font-family:"Times New Roman",serif;
  mso-fareast-font-family:等线;color:black'>18.5<o:p></o:p></span></p>
  </td>
 </tr>
 <tr style='mso-yfti-irow:3;height:11.35pt'>
  <td width=28 style='width:21.25pt;border:none;border-right:solid windowtext 1.0pt;
  mso-border-left-alt:solid windowtext .5pt;mso-border-left-alt:solid windowtext .5pt;
  mso-border-right-alt:solid windowtext .5pt;padding:0cm 1.4pt 0cm 1.4pt;
  height:11.35pt'>
  <p class=MsoNormal align=center style='text-align:center;mso-pagination:widow-orphan'><!--[if gte msEquation 12]><m:oMathPara><m:oMath><m:sSup><m:sSupPr><span
      style='font-size:6.5pt;mso-ansi-font-size:6.5pt;mso-bidi-font-size:7.5pt;
      font-family:"Cambria Math",serif;mso-ascii-font-family:"Cambria Math";
      mso-hansi-font-family:"Cambria Math";mso-bidi-font-family:"Times New Roman";
      font-style:italic;mso-bidi-font-style:normal'><m:ctrlPr></m:ctrlPr></span></m:sSupPr><m:e><i
      style='mso-bidi-font-style:normal'><span lang=EN-US style='font-size:
      6.5pt;mso-bidi-font-size:7.5pt;font-family:"Cambria Math",serif;
      mso-bidi-font-family:"Times New Roman"'><m:r>F</m:r></span></i></m:e><m:sup><i
      style='mso-bidi-font-style:normal'><span lang=EN-US style='font-size:
      6.5pt;mso-bidi-font-size:7.5pt;font-family:"Cambria Math",serif;
      mso-bidi-font-family:"Times New Roman"'><m:r>'</m:r></span></i></m:sup></m:sSup><i
    style='mso-bidi-font-style:normal'><span lang=EN-US style='font-size:6.5pt;
    mso-bidi-font-size:7.5pt;font-family:"Cambria Math",serif;mso-bidi-font-family:
    "Times New Roman"'><m:r>/4</m:r></span></i></m:oMath></m:oMathPara><![endif]--><![if !msEquation]><span
  lang=EN-US style='font-size:10.5pt;mso-bidi-font-size:11.0pt;font-family:
  等线;mso-ascii-theme-font:minor-latin;mso-fareast-theme-font:minor-fareast;
  mso-hansi-theme-font:minor-latin;mso-bidi-font-family:"Times New Roman";
  mso-bidi-theme-font:minor-bidi;mso-ansi-language:EN-US;mso-fareast-language:
  ZH-CN;mso-bidi-language:AR-SA'><!--[if gte vml 1]><v:shape id="_x0000_i1025"
   type="#_x0000_t75" style='width:12.65pt;height:15.55pt'>
   <v:imagedata src="Table_readme.files/image007.png" o:title="" chromakey="white"/>
  </v:shape><![endif]--><![if !vml]><img width=17 height=21
  src="Table_readme.files/image008.png" v:shapes="_x0000_i1025"><![endif]></span><![endif]><span
  lang=EN-US style='font-size:6.5pt;mso-bidi-font-size:7.5pt;font-family:"Times New Roman",serif;
  mso-fareast-font-family:等线'><o:p></o:p></span></p>
  </td>
  <td width=28 nowrap style='width:21.25pt;border:none;mso-border-left-alt:
  solid windowtext .5pt;padding:0cm 1.4pt 0cm 1.4pt;height:11.35pt'>
  <p class=MsoNormal align=center style='text-align:center;mso-pagination:widow-orphan'><span
  lang=EN-US style='font-size:7.5pt;mso-bidi-font-size:8.0pt;font-family:"Times New Roman",serif;
  mso-fareast-font-family:等线;color:black'>3.15<o:p></o:p></span></p>
  </td>
  <td width=28 style='width:21.25pt;padding:0cm 1.4pt 0cm 1.4pt;height:11.35pt'>
  <p class=MsoNormal align=center style='text-align:center;mso-pagination:widow-orphan'><span
  lang=EN-US style='font-size:7.5pt;mso-bidi-font-size:8.0pt;font-family:"Times New Roman",serif;
  mso-fareast-font-family:等线;color:black'>4.32<o:p></o:p></span></p>
  </td>
  <td width=28 style='width:21.25pt;padding:0cm 1.4pt 0cm 1.4pt;height:11.35pt'>
  <p class=MsoNormal align=center style='text-align:center;mso-pagination:widow-orphan'><span
  lang=EN-US style='font-size:7.5pt;mso-bidi-font-size:8.0pt;font-family:"Times New Roman",serif;
  mso-fareast-font-family:等线;color:black'>3.58<o:p></o:p></span></p>
  </td>
  <td width=28 style='width:21.3pt;border:none;border-right:solid windowtext 1.0pt;
  mso-border-right-alt:solid windowtext .5pt;padding:0cm 1.4pt 0cm 1.4pt;
  height:11.35pt'>
  <p class=MsoNormal align=center style='text-align:center;mso-pagination:widow-orphan'><span
  lang=EN-US style='font-size:7.5pt;mso-bidi-font-size:8.0pt;font-family:"Times New Roman",serif;
  mso-fareast-font-family:等线;color:black'>3.76<o:p></o:p></span></p>
  </td>
  <td width=38 nowrap style='width:1.0cm;border:none;mso-border-left-alt:solid windowtext .5pt;
  padding:0cm 1.4pt 0cm 1.4pt;height:11.35pt'>
  <p class=MsoNormal align=center style='text-align:center;mso-pagination:widow-orphan'><span
  lang=EN-US style='font-size:7.5pt;mso-bidi-font-size:8.0pt;font-family:"Times New Roman",serif;
  mso-fareast-font-family:等线;color:black'>94.6<o:p></o:p></span></p>
  </td>
  <td width=38 nowrap colspan=2 style='width:1.0cm;padding:0cm 1.4pt 0cm 1.4pt;
  height:11.35pt'>
  <p class=MsoNormal align=center style='text-align:center;mso-pagination:widow-orphan'><span
  lang=EN-US style='font-size:7.5pt;mso-bidi-font-size:8.0pt;font-family:"Times New Roman",serif;
  mso-fareast-font-family:等线;color:black'>18.1<o:p></o:p></span></p>
  </td>
 </tr>
 <tr style='mso-yfti-irow:4;height:11.35pt'>
  <td width=28 style='width:21.25pt;border-top:none;border-left:none;
  border-bottom:solid windowtext 1.0pt;border-right:solid windowtext 1.0pt;
  mso-border-left-alt:solid windowtext .5pt;mso-border-left-alt:solid windowtext .5pt;
  mso-border-bottom-alt:solid windowtext .5pt;mso-border-right-alt:solid windowtext .5pt;
  padding:0cm 1.4pt 0cm 1.4pt;height:11.35pt'>
  <p class=MsoNormal align=center style='text-align:center;mso-pagination:widow-orphan'><!--[if gte msEquation 12]><m:oMathPara><m:oMath><m:rad><m:radPr><m:degHide
       m:val="on"/><span style='font-size:6.5pt;mso-ansi-font-size:6.5pt;
      mso-bidi-font-size:7.5pt;font-family:"Cambria Math",serif;mso-ascii-font-family:
      "Cambria Math";mso-hansi-font-family:"Cambria Math";mso-bidi-font-family:
      "Times New Roman";color:black;font-style:italic;mso-bidi-font-style:normal'><m:ctrlPr></m:ctrlPr></span></m:radPr><m:deg></m:deg><m:e><i
      style='mso-bidi-font-style:normal'><span lang=EN-US style='font-size:
      6.5pt;mso-bidi-font-size:7.5pt;font-family:"Cambria Math",serif;
      mso-bidi-font-family:"Times New Roman"'><m:r>F</m:r><m:r>'</m:r></span></i></m:e></m:rad></m:oMath></m:oMathPara><![endif]--><![if !msEquation]><span
  lang=EN-US style='font-size:10.5pt;mso-bidi-font-size:11.0pt;font-family:
  等线;mso-ascii-theme-font:minor-latin;mso-fareast-theme-font:minor-fareast;
  mso-hansi-theme-font:minor-latin;mso-bidi-font-family:"Times New Roman";
  mso-bidi-theme-font:minor-bidi;mso-ansi-language:EN-US;mso-fareast-language:
  ZH-CN;mso-bidi-language:AR-SA'><!--[if gte vml 1]><v:shape id="_x0000_i1025"
   type="#_x0000_t75" style='width:10.35pt;height:15.55pt'>
   <v:imagedata src="Table_readme.files/image009.png" o:title="" chromakey="white"/>
  </v:shape><![endif]--><![if !vml]><img width=14 height=21
  src="Table_readme.files/image010.png" v:shapes="_x0000_i1025"><![endif]></span><![endif]><span
  lang=EN-US style='font-size:6.5pt;mso-bidi-font-size:7.5pt;font-family:"Times New Roman",serif;
  color:black'><o:p></o:p></span></p>
  </td>
  <td width=28 nowrap style='width:21.25pt;border:none;border-bottom:solid windowtext 1.0pt;
  mso-border-left-alt:solid windowtext .5pt;mso-border-left-alt:solid windowtext .5pt;
  mso-border-bottom-alt:solid windowtext .5pt;padding:0cm 1.4pt 0cm 1.4pt;
  height:11.35pt'>
  <p class=MsoNormal align=center style='text-align:center;mso-pagination:widow-orphan'><b
  style='mso-bidi-font-weight:normal'><span lang=EN-US style='font-size:7.5pt;
  mso-bidi-font-size:8.0pt;font-family:"Times New Roman",serif;mso-fareast-font-family:
  等线;color:black'>3.16<o:p></o:p></span></b></p>
  </td>
  <td width=28 style='width:21.25pt;border:none;border-bottom:solid windowtext 1.0pt;
  mso-border-bottom-alt:solid windowtext .5pt;padding:0cm 1.4pt 0cm 1.4pt;
  height:11.35pt'>
  <p class=MsoNormal align=center style='text-align:center;mso-pagination:widow-orphan'><b
  style='mso-bidi-font-weight:normal'><span lang=EN-US style='font-size:7.5pt;
  mso-bidi-font-size:8.0pt;font-family:"Times New Roman",serif;mso-fareast-font-family:
  等线;color:black'>4.35<o:p></o:p></span></b></p>
  </td>
  <td width=28 style='width:21.25pt;border:none;border-bottom:solid windowtext 1.0pt;
  mso-border-bottom-alt:solid windowtext .5pt;padding:0cm 1.4pt 0cm 1.4pt;
  height:11.35pt'>
  <p class=MsoNormal align=center style='text-align:center;mso-pagination:widow-orphan'><span
  lang=EN-US style='font-size:7.5pt;mso-bidi-font-size:8.0pt;font-family:"Times New Roman",serif;
  mso-fareast-font-family:等线;color:black;mso-bidi-font-weight:bold'>3.61<o:p></o:p></span></p>
  </td>
  <td width=28 style='width:21.3pt;border-top:none;border-left:none;border-bottom:
  solid windowtext 1.0pt;border-right:solid windowtext 1.0pt;mso-border-bottom-alt:
  solid windowtext .5pt;mso-border-right-alt:solid windowtext .5pt;padding:
  0cm 1.4pt 0cm 1.4pt;height:11.35pt'>
  <p class=MsoNormal align=center style='text-align:center;mso-pagination:widow-orphan'><b
  style='mso-bidi-font-weight:normal'><span lang=EN-US style='font-size:7.5pt;
  mso-bidi-font-size:8.0pt;font-family:"Times New Roman",serif;mso-fareast-font-family:
  等线;color:black'>3.78<o:p></o:p></span></b></p>
  </td>
  <td width=38 nowrap style='width:1.0cm;border:none;border-bottom:solid windowtext 1.0pt;
  mso-border-left-alt:solid windowtext .5pt;mso-border-left-alt:solid windowtext .5pt;
  mso-border-bottom-alt:solid windowtext .5pt;padding:0cm 1.4pt 0cm 1.4pt;
  height:11.35pt'>
  <p class=MsoNormal align=center style='text-align:center;mso-pagination:widow-orphan'><b
  style='mso-bidi-font-weight:normal'><span lang=EN-US style='font-size:7.5pt;
  mso-bidi-font-size:8.0pt;font-family:"Times New Roman",serif;mso-fareast-font-family:
  等线;color:black'>94.7<o:p></o:p></span></b></p>
  </td>
  <td width=38 nowrap colspan=2 style='width:1.0cm;border:none;border-bottom:
  solid windowtext 1.0pt;mso-border-bottom-alt:solid windowtext .5pt;
  padding:0cm 1.4pt 0cm 1.4pt;height:11.35pt'>
  <p class=MsoNormal align=center style='text-align:center;mso-pagination:widow-orphan'><b
  style='mso-bidi-font-weight:normal'><span lang=EN-US style='font-size:7.5pt;
  mso-bidi-font-size:8.0pt;font-family:"Times New Roman",serif;mso-fareast-font-family:
  等线;color:black'>18.8<o:p></o:p></span></b></p>
  </td>
 </tr>
 <tr style='mso-yfti-irow:5;height:11.35pt'>
  <td width=28 rowspan=3 style='width:21.3pt;border-top:none;border-left:none;
  border-bottom:solid windowtext 1.0pt;border-right:solid windowtext 1.0pt;
  mso-border-top-alt:solid windowtext .5pt;mso-border-top-alt:solid windowtext .5pt;
  mso-border-bottom-alt:solid windowtext .5pt;mso-border-right-alt:solid windowtext .5pt;
  padding:0cm 1.4pt 0cm 1.4pt;height:11.35pt'>
  <p class=MsoNormal align=center style='text-align:center'><span lang=EN-US
  style='font-size:5.0pt;mso-bidi-font-size:5.5pt;font-family:"Times New Roman",serif;
  color:black'>DPARN<o:p></o:p></span></p>
  </td>
  <td width=28 style='width:21.25pt;border:none;border-right:solid windowtext 1.0pt;
  mso-border-top-alt:solid windowtext .5pt;mso-border-left-alt:solid windowtext .5pt;
  mso-border-top-alt:solid windowtext .5pt;mso-border-left-alt:solid windowtext .5pt;
  mso-border-right-alt:solid windowtext .5pt;padding:0cm 1.4pt 0cm 1.4pt;
  height:11.35pt'>
  <p class=MsoNormal align=center style='text-align:center'><!--[if gte msEquation 12]><m:oMathPara><m:oMath><m:sSup><m:sSupPr><span
      style='font-size:6.5pt;mso-ansi-font-size:6.5pt;mso-bidi-font-size:7.5pt;
      font-family:"Cambria Math",serif;mso-ascii-font-family:"Cambria Math";
      mso-hansi-font-family:"Cambria Math";mso-bidi-font-family:"Times New Roman";
      font-style:italic;mso-bidi-font-style:normal'><m:ctrlPr></m:ctrlPr></span></m:sSupPr><m:e><i
      style='mso-bidi-font-style:normal'><span lang=EN-US style='font-size:
      6.5pt;mso-bidi-font-size:7.5pt;font-family:"Cambria Math",serif;
      mso-bidi-font-family:"Times New Roman"'><m:r>F</m:r></span></i></m:e><m:sup><i
      style='mso-bidi-font-style:normal'><span lang=EN-US style='font-size:
      6.5pt;mso-bidi-font-size:7.5pt;font-family:"Cambria Math",serif;
      mso-bidi-font-family:"Times New Roman"'><m:r>'</m:r></span></i></m:sup></m:sSup><i
    style='mso-bidi-font-style:normal'><span lang=EN-US style='font-size:6.5pt;
    mso-bidi-font-size:7.5pt;font-family:"Cambria Math",serif;mso-bidi-font-family:
    "Times New Roman"'><m:r>/2</m:r></span></i></m:oMath></m:oMathPara><![endif]--><![if !msEquation]><span
  lang=EN-US style='font-size:10.5pt;mso-bidi-font-size:11.0pt;font-family:
  等线;mso-ascii-theme-font:minor-latin;mso-fareast-theme-font:minor-fareast;
  mso-hansi-theme-font:minor-latin;mso-bidi-font-family:"Times New Roman";
  mso-bidi-theme-font:minor-bidi;mso-ansi-language:EN-US;mso-fareast-language:
  ZH-CN;mso-bidi-language:AR-SA'><!--[if gte vml 1]><v:shape id="_x0000_i1025"
   type="#_x0000_t75" style='width:12.65pt;height:15.55pt'>
   <v:imagedata src="Table_readme.files/image005.png" o:title="" chromakey="white"/>
  </v:shape><![endif]--><![if !vml]><img width=17 height=21
  src="Table_readme.files/image006.png" v:shapes="_x0000_i1025"><![endif]></span><![endif]><span
  lang=EN-US style='font-size:6.5pt;mso-bidi-font-size:7.5pt;mso-fareast-font-family:
  等线;color:black'><o:p></o:p></span></p>
  </td>
  <td width=28 nowrap style='width:21.25pt;border:none;mso-border-top-alt:solid windowtext .5pt;
  mso-border-left-alt:solid windowtext .5pt;padding:0cm 1.4pt 0cm 1.4pt;
  height:11.35pt'>
  <p class=MsoNormal align=center style='text-align:center'><b><span
  lang=EN-US style='font-size:7.5pt;mso-bidi-font-size:8.0pt;font-family:"Times New Roman",serif;
  color:black'>2.96</span></b><span lang=EN-US style='font-size:7.5pt;
  mso-bidi-font-size:8.0pt;font-family:"Times New Roman",serif;color:black'><o:p></o:p></span></p>
  </td>
  <td width=28 style='width:21.25pt;border:none;mso-border-top-alt:solid windowtext .5pt;
  padding:0cm 1.4pt 0cm 1.4pt;height:11.35pt'>
  <p class=MsoNormal align=center style='text-align:center'><b><span
  lang=EN-US style='font-size:7.5pt;mso-bidi-font-size:8.0pt;font-family:"Times New Roman",serif;
  color:black'>4.29</span></b><span lang=EN-US style='font-size:7.5pt;
  mso-bidi-font-size:8.0pt;font-family:"Times New Roman",serif;color:black;
  mso-fareast-language:EN-US'><o:p></o:p></span></p>
  </td>
  <td width=28 style='width:21.25pt;border:none;mso-border-top-alt:solid windowtext .5pt;
  padding:0cm 1.4pt 0cm 1.4pt;height:11.35pt'>
  <p class=MsoNormal align=center style='text-align:center'><span lang=EN-US
  style='font-size:7.5pt;mso-bidi-font-size:8.0pt;font-family:"Times New Roman",serif;
  color:black'>3.63<o:p></o:p></span></p>
  </td>
  <td width=28 style='width:21.3pt;border:none;border-right:solid windowtext 1.0pt;
  mso-border-top-alt:solid windowtext .5pt;mso-border-top-alt:solid windowtext .5pt;
  mso-border-right-alt:solid windowtext .5pt;padding:0cm 1.4pt 0cm 1.4pt;
  height:11.35pt'>
  <p class=MsoNormal align=center style='text-align:center'><b><span
  lang=EN-US style='font-size:7.5pt;mso-bidi-font-size:8.0pt;font-family:"Times New Roman",serif;
  color:black'>3.68</span></b><span lang=EN-US style='font-size:7.5pt;
  mso-bidi-font-size:8.0pt;font-family:"Times New Roman",serif;color:black'><o:p></o:p></span></p>
  </td>
  <td width=38 nowrap style='width:1.0cm;border:none;mso-border-top-alt:solid windowtext .5pt;
  mso-border-left-alt:solid windowtext .5pt;padding:0cm 1.4pt 0cm 1.4pt;
  height:11.35pt'>
  <p class=MsoNormal align=center style='text-align:center'><b><span
  lang=EN-US style='font-size:7.5pt;mso-bidi-font-size:8.0pt;font-family:"Times New Roman",serif;
  color:black'>94.2</span></b><span lang=EN-US style='font-size:7.5pt;
  mso-bidi-font-size:8.0pt;font-family:"Times New Roman",serif;color:black'><o:p></o:p></span></p>
  </td>
  <td width=38 nowrap colspan=2 style='width:1.0cm;border:none;mso-border-top-alt:
  solid windowtext .5pt;padding:0cm 1.4pt 0cm 1.4pt;height:11.35pt'>
  <p class=MsoNormal align=center style='text-align:center'><span lang=EN-US
  style='font-size:7.5pt;mso-bidi-font-size:8.0pt;font-family:"Times New Roman",serif;
  color:black'>18.7<o:p></o:p></span></p>
  </td>
 </tr>
 <tr style='mso-yfti-irow:6;height:11.35pt'>
  <td width=28 style='width:21.25pt;border:none;border-right:solid windowtext 1.0pt;
  mso-border-left-alt:solid windowtext .5pt;mso-border-left-alt:solid windowtext .5pt;
  mso-border-right-alt:solid windowtext .5pt;padding:0cm 1.4pt 0cm 1.4pt;
  height:11.35pt'>
  <p class=MsoNormal align=center style='text-align:center'><!--[if gte msEquation 12]><m:oMathPara><m:oMath><m:sSup><m:sSupPr><span
      style='font-size:6.5pt;mso-ansi-font-size:6.5pt;mso-bidi-font-size:7.5pt;
      font-family:"Cambria Math",serif;mso-ascii-font-family:"Cambria Math";
      mso-hansi-font-family:"Cambria Math";mso-bidi-font-family:"Times New Roman";
      font-style:italic;mso-bidi-font-style:normal'><m:ctrlPr></m:ctrlPr></span></m:sSupPr><m:e><i
      style='mso-bidi-font-style:normal'><span lang=EN-US style='font-size:
      6.5pt;mso-bidi-font-size:7.5pt;font-family:"Cambria Math",serif;
      mso-bidi-font-family:"Times New Roman"'><m:r>F</m:r></span></i></m:e><m:sup><i
      style='mso-bidi-font-style:normal'><span lang=EN-US style='font-size:
      6.5pt;mso-bidi-font-size:7.5pt;font-family:"Cambria Math",serif;
      mso-bidi-font-family:"Times New Roman"'><m:r>'</m:r></span></i></m:sup></m:sSup><i
    style='mso-bidi-font-style:normal'><span lang=EN-US style='font-size:6.5pt;
    mso-bidi-font-size:7.5pt;font-family:"Cambria Math",serif;mso-bidi-font-family:
    "Times New Roman"'><m:r>/4</m:r></span></i></m:oMath></m:oMathPara><![endif]--><![if !msEquation]><span
  lang=EN-US style='font-size:10.5pt;mso-bidi-font-size:11.0pt;font-family:
  等线;mso-ascii-theme-font:minor-latin;mso-fareast-theme-font:minor-fareast;
  mso-hansi-theme-font:minor-latin;mso-bidi-font-family:"Times New Roman";
  mso-bidi-theme-font:minor-bidi;mso-ansi-language:EN-US;mso-fareast-language:
  ZH-CN;mso-bidi-language:AR-SA'><!--[if gte vml 1]><v:shape id="_x0000_i1025"
   type="#_x0000_t75" style='width:12.65pt;height:15.55pt'>
   <v:imagedata src="Table_readme.files/image007.png" o:title="" chromakey="white"/>
  </v:shape><![endif]--><![if !vml]><img width=17 height=21
  src="Table_readme.files/image008.png" v:shapes="_x0000_i1025"><![endif]></span><![endif]><span
  lang=EN-US style='font-size:6.5pt;mso-bidi-font-size:7.5pt;color:black'><o:p></o:p></span></p>
  </td>
  <td width=28 nowrap style='width:21.25pt;border:none;mso-border-left-alt:
  solid windowtext .5pt;padding:0cm 1.4pt 0cm 1.4pt;height:11.35pt'>
  <p class=MsoNormal align=center style='text-align:center'><span lang=EN-US
  style='font-size:7.5pt;mso-bidi-font-size:8.0pt;font-family:"Times New Roman",serif;
  color:black'>2.95<o:p></o:p></span></p>
  </td>
  <td width=28 style='width:21.25pt;padding:0cm 1.4pt 0cm 1.4pt;height:11.35pt'>
  <p class=MsoNormal align=center style='text-align:center'><span lang=EN-US
  style='font-size:7.5pt;mso-bidi-font-size:8.0pt;font-family:"Times New Roman",serif;
  color:black'>4.27</span><span lang=EN-US style='font-size:7.5pt;mso-bidi-font-size:
  8.0pt;font-family:"Times New Roman",serif;color:black;mso-fareast-language:
  EN-US'><o:p></o:p></span></p>
  </td>
  <td width=28 style='width:21.25pt;padding:0cm 1.4pt 0cm 1.4pt;height:11.35pt'>
  <p class=MsoNormal align=center style='text-align:center'><b><span
  lang=EN-US style='font-size:7.5pt;mso-bidi-font-size:8.0pt;font-family:"Times New Roman",serif;
  color:black'>3.65<o:p></o:p></span></b></p>
  </td>
  <td width=28 style='width:21.3pt;border:none;border-right:solid windowtext 1.0pt;
  mso-border-right-alt:solid windowtext .5pt;padding:0cm 1.4pt 0cm 1.4pt;
  height:11.35pt'>
  <p class=MsoNormal align=center style='text-align:center'><span lang=EN-US
  style='font-size:7.5pt;mso-bidi-font-size:8.0pt;font-family:"Times New Roman",serif;
  color:black'>3.68<o:p></o:p></span></p>
  </td>
  <td width=38 nowrap style='width:1.0cm;border:none;mso-border-left-alt:solid windowtext .5pt;
  padding:0cm 1.4pt 0cm 1.4pt;height:11.35pt'>
  <p class=MsoNormal align=center style='text-align:center'><span lang=EN-US
  style='font-size:7.5pt;mso-bidi-font-size:8.0pt;font-family:"Times New Roman",serif;
  color:black'>94.2<o:p></o:p></span></p>
  </td>
  <td width=38 nowrap colspan=2 style='width:1.0cm;padding:0cm 1.4pt 0cm 1.4pt;
  height:11.35pt'>
  <p class=MsoNormal align=center style='text-align:center'><b><span
  lang=EN-US style='font-size:7.5pt;mso-bidi-font-size:8.0pt;font-family:"Times New Roman",serif;
  color:black'>18.8<o:p></o:p></span></b></p>
  </td>
 </tr>
 <tr style='mso-yfti-irow:7;mso-yfti-lastrow:yes;height:11.35pt'>
  <td width=28 style='width:21.25pt;border-top:none;border-left:none;
  border-bottom:solid windowtext 1.0pt;border-right:solid windowtext 1.0pt;
  mso-border-left-alt:solid windowtext .5pt;mso-border-left-alt:solid windowtext .5pt;
  mso-border-bottom-alt:solid windowtext .5pt;mso-border-right-alt:solid windowtext .5pt;
  padding:0cm 1.4pt 0cm 1.4pt;height:11.35pt'>
  <p class=MsoNormal align=center style='text-align:center'><!--[if gte msEquation 12]><m:oMathPara><m:oMath><m:rad><m:radPr><m:degHide
       m:val="on"/><span style='font-size:6.5pt;mso-ansi-font-size:6.5pt;
      mso-bidi-font-size:7.5pt;font-family:"Cambria Math",serif;mso-ascii-font-family:
      "Cambria Math";mso-hansi-font-family:"Cambria Math";mso-bidi-font-family:
      "Times New Roman";color:black;font-style:italic;mso-bidi-font-style:normal'><m:ctrlPr></m:ctrlPr></span></m:radPr><m:deg></m:deg><m:e><i
      style='mso-bidi-font-style:normal'><span lang=EN-US style='font-size:
      6.5pt;mso-bidi-font-size:7.5pt;font-family:"Cambria Math",serif;
      mso-bidi-font-family:"Times New Roman"'><m:r>F</m:r><m:r>'</m:r></span></i></m:e></m:rad></m:oMath></m:oMathPara><![endif]--><![if !msEquation]><span
  lang=EN-US style='font-size:10.5pt;mso-bidi-font-size:11.0pt;font-family:
  等线;mso-ascii-theme-font:minor-latin;mso-fareast-theme-font:minor-fareast;
  mso-hansi-theme-font:minor-latin;mso-bidi-font-family:"Times New Roman";
  mso-bidi-theme-font:minor-bidi;mso-ansi-language:EN-US;mso-fareast-language:
  ZH-CN;mso-bidi-language:AR-SA'><!--[if gte vml 1]><v:shape id="_x0000_i1025"
   type="#_x0000_t75" style='width:10.35pt;height:15.55pt'>
   <v:imagedata src="Table_readme.files/image009.png" o:title="" chromakey="white"/>
  </v:shape><![endif]--><![if !vml]><img width=14 height=21
  src="Table_readme.files/image010.png" v:shapes="_x0000_i1025"><![endif]></span><![endif]><span
  lang=EN-US style='font-size:6.5pt;mso-bidi-font-size:7.5pt;font-family:"Times New Roman",serif;
  mso-fareast-font-family:等线'><o:p></o:p></span></p>
  </td>
  <td width=28 nowrap style='width:21.25pt;border:none;border-bottom:solid windowtext 1.0pt;
  mso-border-left-alt:solid windowtext .5pt;mso-border-left-alt:solid windowtext .5pt;
  mso-border-bottom-alt:solid windowtext .5pt;padding:0cm 1.4pt 0cm 1.4pt;
  height:11.35pt'>
  <p class=MsoNormal align=center style='text-align:center'><span lang=EN-US
  style='font-size:7.5pt;mso-bidi-font-size:8.0pt;font-family:"Times New Roman",serif;
  color:black'>2.94<b><o:p></o:p></b></span></p>
  </td>
  <td width=28 style='width:21.25pt;border:none;border-bottom:solid windowtext 1.0pt;
  mso-border-bottom-alt:solid windowtext .5pt;padding:0cm 1.4pt 0cm 1.4pt;
  height:11.35pt'>
  <p class=MsoNormal align=center style='text-align:center'><span lang=EN-US
  style='font-size:7.5pt;mso-bidi-font-size:8.0pt;font-family:"Times New Roman",serif;
  color:black'>4.27<b><o:p></o:p></b></span></p>
  </td>
  <td width=28 style='width:21.25pt;border:none;border-bottom:solid windowtext 1.0pt;
  mso-border-bottom-alt:solid windowtext .5pt;padding:0cm 1.4pt 0cm 1.4pt;
  height:11.35pt'>
  <p class=MsoNormal align=center style='text-align:center'><span lang=EN-US
  style='font-size:7.5pt;mso-bidi-font-size:8.0pt;font-family:"Times New Roman",serif;
  color:black'>3.62<b><o:p></o:p></b></span></p>
  </td>
  <td width=28 style='width:21.3pt;border-top:none;border-left:none;border-bottom:
  solid windowtext 1.0pt;border-right:solid windowtext 1.0pt;mso-border-bottom-alt:
  solid windowtext .5pt;mso-border-right-alt:solid windowtext .5pt;padding:
  0cm 1.4pt 0cm 1.4pt;height:11.35pt'>
  <p class=MsoNormal align=center style='text-align:center'><span lang=EN-US
  style='font-size:7.5pt;mso-bidi-font-size:8.0pt;font-family:"Times New Roman",serif;
  color:black'>3.67<b><o:p></o:p></b></span></p>
  </td>
  <td width=38 nowrap style='width:1.0cm;border:none;border-bottom:solid windowtext 1.0pt;
  mso-border-left-alt:solid windowtext .5pt;mso-border-left-alt:solid windowtext .5pt;
  mso-border-bottom-alt:solid windowtext .5pt;padding:0cm 1.4pt 0cm 1.4pt;
  height:11.35pt'>
  <p class=MsoNormal align=center style='text-align:center'><span lang=EN-US
  style='font-size:7.5pt;mso-bidi-font-size:8.0pt;font-family:"Times New Roman",serif;
  color:black'>94.1<b><o:p></o:p></b></span></p>
  </td>
  <td width=38 nowrap colspan=2 style='width:1.0cm;border:none;border-bottom:
  solid windowtext 1.0pt;mso-border-bottom-alt:solid windowtext .5pt;
  padding:0cm 1.4pt 0cm 1.4pt;height:11.35pt'>
  <p class=MsoNormal align=center style='text-align:center'><span lang=EN-US
  style='font-size:7.5pt;mso-bidi-font-size:8.0pt;font-family:"Times New Roman",serif;
  color:black'>18.5<b><o:p></o:p></b></span></p>
  </td>
 </tr>
 <![if !supportMisalignedColumns]>
 <tr height=0>
  <td width=37 style='border:none'></td>
  <td width=37 style='border:none'></td>
  <td width=37 style='border:none'></td>
  <td width=37 style='border:none'></td>
  <td width=37 style='border:none'></td>
  <td width=37 style='border:none'></td>
  <td width=49 style='border:none'></td>
  <td width=37 style='border:none'></td>
  <td width=12 style='border:none'></td>
 </tr>
 <![endif]>
</table>


