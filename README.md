# Convolutional demosaicing network for joint chromatic and polarimetric imagery

![Python 3.5](https://img.shields.io/badge/python-3.5-DodgerBlue.svg?style=plastic)
![TensorFlow 1.14](https://img.shields.io/badge/TensorFlow-1.14.0-DodgerBlue.svg?style=plastic)
![CUDA 10.0](https://img.shields.io/badge/cuda-10.0-DodgerBlue.svg?style=plastic)
![License CC BY-NC](https://img.shields.io/badge/license-CC_BY--NC-DodgerBlue.svg?style=plastic)

 Our paper is accepted by **Optics Letters 2019**. 

<div align=center>  <img src="figures/network1.jpg" alt="Teaser" width="700" align="bottom" /> </div>

**Picture:**  *Architecture of the proposed CPDNet.*



<div align=center>  <img src="./figures/Comparesion5.jpg" alt="MPI Results" width="1000" align="center" /> </div>

**Picture:**  *Visual results.*



This repository contains the official Tensorflow implementation of the following paper:

> **Convolutional Demosaicing Network for Joint Chromatic and Polarimetric Imagery**<br>
>  Sijia Wen, Yinqiang Zheng, Feng Lu, Qinping Zhao <br> https://www.osapublishing.org/ol/abstract.cfm?uri=ol-44-22-5646
> 
>**Abstract:**  Due to the latest progress in image sensor manufacturing technology, the emergence of a sensor equipped with an RGGB Bayer filter and a directional polarizing filter has brought significant advantages to computer vision tasks where RGB and polarization information is required. In this regard, joint chromatic and polarimetric image demosaicing is indispensable. However, as a new type of array pattern, there is no dedicated method for this challenging task. In this Letter, we collect, to the best of our knowledge, the first chromatic-polarization dataset and propose a chromatic-polarization demosaicing network (CPDNet) to address this joint chromatic and polarimetric image demosaicing issue. The proposed CPDNet is composed of the residual block and the multi-task structure with the costumed loss function. The experimental results show that our proposed methods are capable of faithfully recovering full 12-channel chromatic and polarimetric information for each pixel from a single mosaic image in terms of quantitative measures and visual quality. 

## Resources

Material related to our paper is available via the following links:

- Paper:   https://www.osapublishing.org/ol/abstract.cfm?uri=ol-44-22-5646
- Code: https://github.com/wsj890411/CPDNet

## System requirements

* 64-bit Python 3.5 installation. 
* TensowFlow 1.14.0 or newer with GPU support.
* One or more high-end NVIDIA GPUs with at least 8GB of DRAM.

## Dataset

Download (Dropbox):   https://www.dropbox.com/s/4t3hw6i9tb0nvh2/OL.zip?dl=0

Download (BaiduPan):  https://pan.baidu.com/s/1R6UttPRd7t_-aq337-RS9w Code：9x25

The downloaded and unzipped polarized color images should be organized as follows:

```
data_root
    |-- Train
        |-- train_dataset
            |-- image-ID
            	|-- gt_0
            		|-- image-ID.png
                |-- gt_45
            		|-- image-ID.png
            	|-- gt_90
            		|-- image-ID.png
            	|-- gt_135
            		|-- image-ID.png
            	|-- net_input
            		|-- image-ID.png
            ...
    |-- Test
        |-- test_dataset
            |-- image-ID
            	|-- gt_0
            		|-- image-ID.png
                |-- gt_45
            		|-- image-ID.png
            	|-- gt_90
            		|-- image-ID.png
            	|-- gt_135
            		|-- image-ID.png
            	|-- net_input
            		|-- image-ID.png
            ...
         |-- polarized_dataset
         |-- real_captured
```

## Playing with pre-trained networks and training

### Test

1. The pretrained model is provided in `./OL_CPDNET_mse_ssim_gra_stoke/checkpoint/OL_final/`
2. One can modify the configuration in `./OL_CPDNET_mse_ssim_gra_stoke/main.py`

Then then results will be generated in the `./OL_CPDNET_mse_ssim_gra_stoke/results` dir.

### Train
1. Download the dataset to your `<your_dataset_root>`.
2. Split the dataset and organize them in `OL_CPDNET_DATA`
3. One can modify the configuration in `./OL_CPDNET_mse_ssim_gra_stoke/main.py`

## Citation

If you find this work or code is helpful in your research, please cite:

```latex
@article{wen2019convolutional,
  title={Convolutional demosaicing network for joint chromatic and polarimetric imagery},
  author={Wen, Sijia and Zheng, Yinqiang and Lu, Feng and Zhao, Qinping},
  journal={Optics letters},
  volume={44},
  number={22},
  pages={5646--5649},
  year={2019},
  publisher={Optical Society of America}
}
```


