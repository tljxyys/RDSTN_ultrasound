# Residual Dense Swin Transformer for Continuous Depth-Independent Ultrasound Imaging🚀 (ICASSP2024)
[![](https://img.shields.io/badge/Project-Page-green.svg)](https://github.com/tljxyys/RDSTN_ultrasound) [![](https://img.shields.io/badge/Paper-ArXiv-red.svg)](https://arxiv.org/abs/2403.16384) [![](https://img.shields.io/badge/Dataset-🔰BUSI-blue.svg)](https://www.kaggle.com/datasets/aryashah2k/breast-ultrasound-images-dataset) [![](https://img.shields.io/badge/Dataset-🔰USenhance-blue.svg)](https://ultrasoundenhance2023.grand-challenge.org/) 

***
>**Abstract**: _Ultrasound imaging is crucial for evaluating organ morphology and function, yet depth adjustment can degrade image quality and field-of-view, presenting a depth-dependent dilemma. 
Traditional interpolation-based zoom-in techniques often sacrifice detail and introduce artifacts. Motivated by the potential of arbitrary-scale super-resolution to naturally address these 
inherent challenges, we present the Residual Dense Swin Transformer Network (RDSTN), designed to capture the non-local characteristics and long-range dependencies intrinsic to ultrasound images.
It comprises a linear embedding module for feature enhancement, an encoder with shifted-window attention for modeling non-locality, and an MLP decoder for continuous detail reconstruction. 
This strategy streamlines balancing image quality and field-of-view, which offers superior textures over traditional methods. Experimentally, RDSTN outperforms existing approaches while requiring 
fewer parameters. In conclusion, RDSTN shows promising potential for ultrasound image enhancement by overcoming the limitations of conventional interpolation-based methods and achieving depth-independent imaging._
>

![image](https://github.com/tljxyys/RDSTN_ultrasound/blob/main/fig/Figure%202.png)
***

## 1. Background
Ultrasound imaging serves as a pivotal tool in medical diagnostics for its non-invasive nature and real-time imaging capabilities, allowing visualization of superficial and deep structures. However, adjusting the imaging depth presents challenges that impact image quality and field-of-view. Modifying the imaging depth in ultrasound requires altering the echo reception time. Longer reception times, necessary for deeper imaging, tend to lower the frame rate, subsequently reducing temporal resolution. A shallow imaging depth, however, may lead to interference from adjacent echo signals, compromising image quality. Therefore, selecting the appropriate depth threshold is crucial. Traditionally, zoom-in operations utilizing interpolation have been employed to counterbalance unsatisfactory image quality during depth adjustments. This often results in the loss of intricate details and the emergence of aliasing artifacts. Addressing this challenge, our study presents the arbitrary-scale super-resolution (ASSR) as a cutting-edge approach that offers an effective solution within the desired depth threshold.

![image](https://github.com/tljxyys/RDSTN_ultrasound/blob/main/fig/Figure%201.png)

## 2. Dependencies and Installation

- Clone this repo and create a conda virtual environment:
```
https://github.com/tljxyys/RDSTN_ultrasound.git
cd RDSTN_ultrasound
conda create -n rdstn python=3.7 -y
conda activate rdstn
```
- Install necessary packages:
```
conda install pytorch==1.8.0 torchvision==0.9.0 cudatoolkit=10.2 -c pytorch
pip install timm==0.4.12
pip install opencv-python==4.4.0.46 termcolor==1.1.0 yacs==0.1.8 pyyaml scipy
pip install TensorboardX yaml numpy tqdm imageio
```

## 3. Data Preparation
- The BUSI dataset we used are provided by Al-Dhabyani W, Gomaa M, Khaled H, Fahmy A, which can be downloaded from [![](https://img.shields.io/badge/Dataset-🔰BUSI-blue.svg)](https://www.kaggle.com/datasets/aryashah2k/breast-ultrasound-images-dataset). If you would like to use the preprocessed data, please use it for research purposes and do not redistribute it. The USenhance dataset can be obtained from [![](https://img.shields.io/badge/Dataset-🔰USenhance-blue.svg)](https://ultrasoundenhance2023.grand-challenge.org/). Please follow the instructions and regulations set by the official releaser of these two datasets. 

## 4. Training/Testing
- Training. Run the train script on BUSI dataset. The batch size and epoch we used is 16 and 1000, respectively.
```
python train.py --config train/BUSI/train_rdn-liif.yaml
```
- Testing. Achieving 1.6x to 10x super-resolution on BUSI dataset or USenhance data.
```
python test.py --config test/BUSI/test-x1.6.yaml --model save/_train_rdn-liif/epoch-last.pth
```
```
python test.py --config test/MICCAI_ultrasound/breast/test-x1.6.yaml --model save/_train_rdn-liif/epoch-last.pth
```

## 5. Results

- **Quantitative comparison in terms of PSNR(dB).** The evaluation is performed on the BUSI testing set. The models are trained with continuous scale sampled from U(1, 4). Best result of each scale is in **bold**.

| Methods | Num. of Parameters | ×1.6 | ×1.7 | ×1.8 | ×1.9 | ×2 | ×3 | ×4 | ×6 | ×8 | ×10 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Bicubic | – | 40.21 | 39.36 | 38.88 | 38.21 | 38.68 | 33.17 | 30.40 | 26.88 | 24.86 | 23.64 |
| EDSR-LIIF | 496.4K | 43.92 | 43.06 | 42.26 | 41.50 | 40.80 | 35.80 | 32.87 | 29.42 | 27.34 | 26.04 |
| RDN-LIIF | 5.8M | 44.71 | 43.81 | 43.03 | 42.28 | 41.57 | **36.36** | **33.22** | 29.58 | 27.46 | 26.12 | 
| Unet | 31.4M | 42.39 | 41.71 | 41.05 | 40.42 | 39.83 | 35.24 | 32.55 | 29.20 | 27.16 | 25.91 |
| Resnet50 | 4.1M | 42.86 | 42.07 | 41.35 | 40.62 | 39.95 | 35.17 | 32.46 | 29.12 | 27.11 | 25.87 |
| **RDSTN (ours)** | 3.2M | **44.78** | **43.89** | **43.10** | **42.35** | **41.62** | 36.34 | 33.20 | **29.64** | **27.54** | **26.18** |

- **Ablation study of RDSTN on Local Feature Fusion (LFF) and Global Feature Fusion (GFF).** The evaluation is performed on the BUSI testing set, with a focus on measuring the peak signal-to-noise ratio (PSNR(dB)) to assess the performance of these strategies. The best result of each scale is in **bold**.

| Methods | LFF | GFF | ×1.2 | ×1.4 | ×1.6 | ×1.8 | ×2 | ×3 | ×4 | ×6 | ×8 | ×10 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| S1 | ✖️ | ✖️ | 48.65 | 46.17 | 44.36 | 42.69 | 41.21 | 36.04 | 33.01 | 29.51 | 27.42 | 26.07 |
| S2 | ✖️ | ✔️ | 48.71 | 46.23 | 44.42 | 42.73 | 41.27 | 36.07 | 33.03 | 29.54 | 27.46 | 26.11 |
| S3 | ✔️ | ✖️ | 48.89 | 46.40 | 44.61 | 42.94 | 41.46 | 36.23 | 33.13 | 29.59 | 27.50 | 26.14 |
| S4 | ✔️ | ✔️ | **49.27** | **46.62** | **44.78** | **43.10** | **41.62** | **36.34** | **33.20** | **29.64** | **27.54** | **26.18** | 

```HTML
<video width="320" height="240" controls>
    <source src="https://github.com/tljxyys/RDSTN_ultrasound/blob/main/fig/1220.mp4" type="video/mp4">
</video>
```

<img src="https://github.com/tljxyys/RDSTN_ultrasound/blob/main/fig/1215.gif" onload="this.onload=null;this.play();" /> <img src="https://github.com/tljxyys/RDSTN_ultrasound/blob/main/fig/1220.gif" onload="this.onload=null;this.play();" /> <img src="https://github.com/tljxyys/RDSTN_ultrasound/blob/main/fig/1222.gif" onload="this.onload=null;this.play();" />

The figures above are all gif file and will only play once. if you want to see the gif effect, please refresh the page.

## Bibtex
```
@INPROCEEDINGS{10447712,
  author={Hu, Jintong and Che, Hui and Li, Zishuo and Yang, Wenming},
  booktitle={ICASSP 2024 - 2024 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP)}, 
  title={Residual Dense Swin Transformer for Continuous Depth-Independent Ultrasound Imaging}, 
  year={2024},
  volume={},
  number={},
  pages={2280-2284},
  keywords={Image quality;Visualization;Ultrasonic imaging;Superresolution;Imaging;Streaming media;Transformers;Ultrasound imaging;Arbitrary-scale image super-resolution;Depth-independent imaging;Non-local implicit representation},
  doi={10.1109/ICASSP48485.2024.10447712}}
```
