## ARFlow &mdash; Official PyTorch Implementation

![Python 3.6](https://img.shields.io/badge/python-3.6-green.svg?style=plastic) ![PyTorch 1.1.0](https://img.shields.io/badge/pytorch-1.1.0-green.svg?style=plastic) ![License MIT](https://img.shields.io/github/license/lliuz/ARFlow)

This repository contains the official PyTorch implementation of the paper "[Learning by Analogy: Reliable Supervision from Transformations for Unsupervised Optical Flow Estimation](https://arxiv.org/abs/2003.13045)".

For any inquiries, please contact Liang Liu at [leonliuz@zju.edu.cn](mailto:leonliuz@zju.edu.cn)

## Using the Code

### Requirements

This code has been developed under Python3, PyTorch 1.1.0 and CUDA 9.0 on Ubuntu 16.04. 

We strongly recommend that using docker to ensure you can get the same results as us. The [Dockerfile](./Dockerfile) is available. Also, you can build the environment by following:

```shell
# Install python packages
pip3 install -r requirements.txt

# Compile the coorelation package with gcc and g++ >= 4.9
cd ./models/correlation_package
python3 setup.py install

# Additional dependencies for training
sudo apt-get update && apt-get install -y libsm6 libxext6 libxrender-dev
pip3 install 'opencv-python>=3.0,<4.0' path.py tensorboardX fast_slic
```

If you have any trouble with the correlation package, we also provide an alternative implementation. You can modify the import lines in [models/pwclite.py](./models/pwclite.py#L7) to use it.

### Inference

The [checkpoints](./checkpoints) folder contains the pre-trained models of ARFlow and ARFlow-mv for various datasets.

A minimal example of using a pre-trained is given in [inference.py](./inference.py). For two-view models, just run with:

```shell
python3 inference.py -m checkpoints/KITTI15/pwclite_ar.tar -s 384 640 \
  -i examples/img1.png examples/img2.png
```

For multi-view model, input with 3 frames:

```shell
python3 inference.py -m checkpoints/KITTI15/pwclite_ar_mv.tar -s 384 640 \
  -i examples/img0.png examples/img1.png examples/img2.png
```

We recommend input with 384x640 for KITTI and Cityscapes models, 448x1024 for Sintel models.

### Training

Here we provide the complete training pipeline for ARFlow on Sintel and KITTI datasets:

#### Sintel dataset

1. Pre-train on the Sintel raw movie. Also, you can skip this step with [our pretrained model](./checkpoints/Sintel/pwclite_raw.tar). 

   ```shell
   python3 train.py -c sintel_raw.json
   ```

2. Fine-tune on the Sintel training set. We provide both settings for training with or without AR for your convenience.

   ```shell
   # without AR
   python3 train.py -c sintel_ft.json
   # with AR
   python3 train.py -c sintel_ft_ar.json
   ```

> The default configuration uses the whole training set for training and validation. We strongly recommend you use the re-split sets as in our ablation studies. You can modify the config file by setting `train_subsplit` to 'train' and setting  `val_subsplit` to 'val'.

#### KITTI dataset

The pipeline is similar to Sintel, refer to [configs](./configs) for more details. 

> You can pre-train on KITTI raw data, and then fine-tuning on the multi-view extension, or directly train on the multi-view extension. The final results should be similar.   

### Evaluation

Also, a complete evaluation for a model can be simply run with the option `-e`, for example:

```shell
python3 train.py -c configs/sintel_ft.json -m checkpoints/Sintel/pwclite_ar.tar -e 
```

## Datasets in the paper

Due to copyright issues, please download the dataset from the official websites.

- [MPI Sintel Dataset](http://sintel.is.tue.mpg.de/downloads) and [Sintel Raw Movie](https://www.youtube.com/watch?v=eRsGyueVLvQ)
- [KITTI Raw](http://www.cvlibs.net/datasets/kitti/raw_data.php), [KITTI 2015](http://www.cvlibs.net/download.php?file=data_scene_flow_multiview.zip), [KITTI 2012](http://www.cvlibs.net/download.php?file=data_stereo_flow_multiview.zip)
- [FlyingChairs](https://lmb.informatik.uni-freiburg.de/resources/datasets/FlyingChairs.en.html#flyingchairs), [CityScapes](https://www.cityscapes-dataset.com/downloads/) 

We have upload the Sintel Raw dataset to [Google Drive](https://drive.google.com/file/d/1sDujszN5S0BZ2Eiwzh9vXQiOoen7gyYd/view?usp=sharing) and [Baidu Cloud (Key: mxe9)](https://pan.baidu.com/s/10P0UsaFw5z0ey1rdhN97LA). This dataset is created by manually dividing all frames into folders by shot. It should follow the license in the Sintel benchmark.

### Citation

If you think this work is useful for your research, please consider citing:

```
@inproceedings{liu2020learning,
   title = {Learning by Analogy: Reliable Supervision from Transformations for Unsupervised Optical Flow Estimation},
   author = {Liu, Liang and Zhang, Jiangning and He, Ruifei and Liu, Yong and Wang, Yabiao and Tai, Ying and Luo, Donghao and Wang, Chengjie and Li, Jilin and Huang, Feiyue},
   booktitle = {IEEE Conference on Computer Vision and Pattern Recognition(CVPR)},
   year = {2020}
}
```

### Acknowledgements

We thank [Pengpeng Liu](https://github.com/ppliuboy) for in-depth discussions and helpful comments. Also, we thank for portions of the source code from some great works such as [Fast-SLIC](https://github.com/Algy/fast-slic), [IRR](https://github.com/visinf/irr).