# AlignGAN

This is the official implementation for AlignGAN(ICCV2019). Please refer our paper for more details:

**RGB-Infrared Cross-Modality Person Re-Identification via Joint Pixel and Feature Alignment** [[Paper](https://arxiv.org/abs/1910.05839)]

Guan'an Wang, Tianzhu Zhang, Jian Cheng, Si Liu, Yang Yang and Zengguang Hou


## Bibtex

If you find the code useful, please consider citing our paper:
```
@InProceedings{Wang_2019_AlignGAN,
author = {Wang, Guan'an and Zhang, Tianzhu and Cheng, Jian and Liu, Si and Yang, Yang and Hou, Zengguang},
title = {RGB-Infrared Cross-Modality Person Re-Identification via Joint Pixel and Feature Alignment},
booktitle = {The IEEE International Conference on Computer Vision (ICCV)},
year = {2019}
}
```


## Dependencies
* [Anaconda (Python 3.7)](https://www.anaconda.com/download/)
* [PyTorch 1.1.0](http://pytorch.org/)
* GPU Memory >= 40G
* Memory >= 20G


## Dataset Preparation
* SYSU-MM01 Dataset [[link](https://github.com/wuancong/SYSU-MM01)]
* Download and extract it anywhere


## Run
```
# train
python main.py --dataset_path sysu-mm01-path --mode train
```


## Contacts
If you have any question about the project, please feel free to contact with me.

E-mail: guan.wang0706@gmail.com
