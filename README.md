# Residual Dense Swin Transformer for Continuous Depth-Independent Ultrasound Imaging (ICASSP2024)
[paper](https://ieeexplore.ieee.org/document/10447712) | [code](https://github.com/tljxyys/RDSTN_ultrasound)
***
>Abstract: _Ultrasound imaging is crucial for evaluating organ morphology and function, yet depth adjustment can degrade image quality and field-of-view, presenting a depth-dependent dilemma. 
Traditional interpolation-based zoom-in techniques often sacrifice detail and introduce artifacts. Motivated by the potential of arbitrary-scale super-resolution to naturally address these 
inherent challenges, we present the Residual Dense Swin Transformer Network (RDSTN), designed to capture the non-local characteristics and long-range dependencies intrinsic to ultrasound images.
It comprises a linear embedding module for feature enhancement, an encoder with shifted-window attention for modeling non-locality, and an MLP decoder for continuous detail reconstruction. 
This strategy streamlines balancing image quality and field-of-view, which offers superior textures over traditional methods. Experimentally, RDSTN outperforms existing approaches while requiring 
fewer parameters. In conclusion, RDSTN shows promising potential for ultrasound image enhancement by overcoming the limitations of conventional interpolation-based methods and achieving depth-independent imaging._
>
![image](https://github.com/tljxyys/RDSTN_ultrasound/blob/main/fig/Figure%201.png)
***
![image](https://github.com/tljxyys/RDSTN_ultrasound/blob/main/fig/Figure%202.png)
***
## Dependencies and Installation
* Clone this repo:
```

```

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
