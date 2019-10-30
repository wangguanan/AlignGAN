# AlignGAN



This is the official implementation for AlignGAN(ICCV2019). Please refer our paper for more details:

**[[Paper](http://openaccess.thecvf.com/content_ICCV_2019/papers/Wang_RGB-Infrared_Cross-Modality_Person_Re-Identification_via_Joint_Pixel_and_Feature_Alignment_ICCV_2019_paper.pdf)] RGB-Infrared Cross-Modality Person Re-Identification via Joint Pixel and Feature Alignment** 

Guan'an Wang, Tianzhu Zhang, Jian Cheng, Si Liu, Yang Yang and Zengguang Hou

![](https://github.com/wangguanan/AlignGAN/blob/master/images/framework.jpg)


## Bibtex

If you find the code useful, please consider citing our paper:
```
@InProceedings{Wang_2019_ICCV,
author = {Wang, Guan'an and Zhang, Tianzhu and Cheng, Jian and Liu, Si and Yang, Yang and Hou, Zengguang},
title = {RGB-Infrared Cross-Modality Person Re-Identification via Joint Pixel and Feature Alignment},
booktitle = {The IEEE International Conference on Computer Vision (ICCV)},
month = {October},
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


## Train
```
# train, please replace sysu-mm01-path with your own path
python main.py --dataset_path sysu-mm01-path --mode train
```

## Test with Pre-trained Model
* pretrained model ([Google Drive](https://drive.google.com/drive/folders/1FGKrs02Z7Omw3z5wOqClpuzYNFo-LrWw?usp=sharing)), please download all the 8 files into a folder. 
* test with the pre-trained model
```
# test with pretrained model, please sysu-mm01-path and pretrained-model-path with your own paths
python main.py --dataset_path sysu-mm01-path --mode test --pretrained_model_path pretrained-model-path --pretrained_model_index 250
```

## Experimental Results

* settings
  * we train our model with 4 GTX1080ti GPUs.

* Comparison with SOTA
![](https://github.com/wangguanan/AlignGAN/blob/master/images/results.png)

* Pixel Alignment Module
![](https://github.com/wangguanan/AlignGAN/blob/master/images/images.png)

* Feature ALignment Module
![](https://github.com/wangguanan/AlignGAN/blob/master/images/feature-visualization.png)



## Contacts
If you have any question about the project, please feel free to contact with me.

E-mail: guan.wang0706@gmail.com
