# AlignGAN

![](https://github.com/wangguanan/AlignGAN/blob/master/images/framework.jpg)

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


## Train
```
# train, please replace sysu-mm01-path with your own path
python main.py --dataset_path sysu-mm01-path --mode train
```

## Test with Pre-trained Model
* Download pretrained model from [Google Drive](https://drive.google.com/drive/folders/1FGKrs02Z7Omw3z5wOqClpuzYNFo-LrWw?usp=sharing) 
* test 
```
# test with pretrained model, please sysu-mm01-path and pretrained-model-path with your own paths
python main.py --dataset_path sysu-mm01-path --mode test --pretrained_model_path pretrained-model-path --pretrained_model_index 250
```




## Contacts
If you have any question about the project, please feel free to contact with me.

E-mail: guan.wang0706@gmail.com
