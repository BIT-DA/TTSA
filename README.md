# Target-oriented Transferable Semantic Augmentation

Pytorch Implementation for TPAMI manuscript "Adapting Across Domains via Target-oriented Transferable Semantic Augmentation under Prototype Constraint"

## Abstract

We present a Target-oriented
Transferable Semantic Augmentation (<img src="https://latex.codecogs.com/svg.image?T^{2}SA" title="T^{2}SA" />) method, which enhances the generalization ability of the classifier by training it with a
target-like augmented domain, constructed by semantically augmenting source data towards target at the feature level. Moreover, we
achieve the augmentation implicitly by minimizing the upper bound of the expected Angular-softmax loss over the augmented domain,
which is of high efficiency. Additionally, to further ensure that the augmented domain can imitate target domain nicely and
discriminatively, the prototype constraint is enforced on augmented features class-wisely, which minimizes the expected distance
between augmented features and corresponding target prototype (i.e., average representation) in Euclidean space.

<div align=center><img src="./Figures/TTSA.png" width="100%"></div>

## Prerequisites
```
CUDA 11.2
Python 3.7
torch 1.7.0+cu110
torchvision 0.8.1+cu110
pillow 7.2.0
numpy
argparse
```

## Datasets

### Office-31
Office-31 dataset can be found [here](https://people.eecs.berkeley.edu/~jhoffman/domainadapt/).

### Office-Home
Office-Home dataset can be found [here](http://hemanthdv.org/OfficeHome-Dataset/).

### VisDA 2017

VisDA 2017 dataset can be found [here](https://github.com/VisionLearningGroup/taskcv-2017-public).

### DomainNet

DomainNet dataset can be found [here](http://ai.bu.edu/M3SDA/).

### PACS

PACS dataset can be found [here](https://dali-dl.github.io/project_iccv2017.html).

## Running the code

In the following, we provide the training scripts for different settings.

For unsupervised domain adaptation (UDA),
```
sh TTSA.sh
```

For multi-source domain adaptation (MSDA),
```
sh TTSA_for_MSDA.sh
```

For domain generalization (DG),
```
sh TTSA_for_DG.sh
```

## Evaluate

Several pre-trained models of TTSA can be downloaded [here](https://github.com/BIT-DA/TTSA/releases) and put in <root_dir>/Checkpoint

evaluate on Office-31 for UDA tasks
```
python3 evaluate_TTSA.py --gpu_id 2 --arch resnet50 --dset office --t_test_path ./data/DA_list/office/webcam_31.txt --weight_path ./Checkpoint/amazon-webcam.pth.zip
```

## Acknowledgements
Some codes are adapted from [ISDA](https://github.com/blackfeather-wang/ISDA-for-Deep-Networks), [FACT](https://github.com/MediaBrain-SJTU/FACT) and 
[Transfer-Learning-Library](https://github.com/thuml/Transfer-Learning-Library). We thank them for their excellent projects.
```
@inproceedings{NIPS2019_9426,
    title = {Implicit Semantic Data Augmentation for Deep Networks},
    author = {Wang, Yulin and Pan, Xuran and Song, Shiji and Zhang, Hong and Huang, Gao and Wu, Cheng},
    booktitle = {Advances in Neural Information Processing Systems (NeurIPS)},
    pages = {12635--12644},
    year = {2019},
}

@InProceedings{Xu_2021_CVPR,
    author = {Xu, Qinwei and Zhang, Ruipeng and Zhang, Ya and Wang, Yanfeng and Tian, Qi},
    title = {A Fourier-Based Framework for Domain Generalization},
    booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
    month = {June},
    year = {2021},
    pages = {14383-14392}
}

@misc{dalib,
    author = {Junguang Jiang, Baixu Chen, Bo Fu, Mingsheng Long},
    title = {Transfer-Learning-library},
    year = {2020},
    publisher = {GitHub},
    journal = {GitHub repository},
    howpublished = {\url{https://github.com/thuml/Transfer-Learning-Library}},
}
```

## Contact
If you have any problem about our code, feel free to contact
- shuangli@bit.edu.cn
- michellexie102@gmail.com
- kxgong@bit.edu.cn

or describe your problem in Issues.
