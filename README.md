**3DmFV** : Three-Dimensional Point Cloud Classification in Real-Time using Convolutional Neural Networks
---
Created by [Yizhak Ben-Shabat (Itzik)](http://www.itzikbs.com), [Michael Lindenbaum](http://www.cs.technion.ac.il/people/mic/index.html), and [Anath Fischer](https://meeng.technion.ac.il/members/anath-fischer/) from Technion, I.I.T.

![3DmFV Architecture](https://github.com/sitzikbs/3DmFV-Net/blob/master/doc/3dmfvnet_architecture.PNG)


### Introduction
This is the code for training a point cloud classification network using 3D modified Fisher Vectors.

This work will be presented in IROS 2018  in Madrid, Spain and will also be published in 
 Robotics and Automation Letters. 
 
 
Modern robotic systems are often equipped with a direct 3D data acquisition device, e.g. LiDAR, which provides a rich 
3D point cloud representation of the surroundings. This representation is commonly used for obstacle avoidance and 
mapping. Here, we propose a new approach for using point clouds for another critical robotic capability, semantic 
understanding of the environment (i.e. object classification).
Convolutional neural networks (CNN), that perform extremely well for object classification in 2D images, are not easily 
extendible to 3D point clouds analysis. It is not straightforward due to point clouds' irregular format and a varying 
number of points. The common solution of transforming the point cloud data into a 3D voxel grid needs to address severe
 accuracy vs memory size tradeoffs.  In this paper we propose a novel, intuitively interpretable, 3D point cloud
  representation called 3D Modified Fisher Vectors (3DmFV). Our representation is hybrid as it combines a 
  coarse discrete grid structure with continuous generalized Fisher vectors. Using the grid enables us to design a new 
  CNN architecture for real-time point cloud classification. In a series of performance analysis experiments, we 
  demonstrate competitive results or even better than state-of-the-art on challenging benchmark datasets while 
  maintaining robustness to various data corruptions. 
  
### Citation
If you find our work useful in your research, please cite our work:

    @article{ben20173d,
      title={3D Point Cloud Classification and Segmentation using 3D Modified Fisher Vector Representation for Convolutional Neural Networks},
      author={Ben-Shabat, Yizhak and Lindenbaum, Michael and Fischer, Anath},
      journal={arXiv preprint arXiv:1711.08241},
      year={2017}
    }

*** citation will be updated once the paper is published (RA-L) ***

### Installation
Install [Tensorflow](https://www.tensorflow.org). You will also need to install h5py, [scikit-learn](http://scikit-learn.org/stable/).
 
The code was tested with Python 2.7, TensorFlow 1.2.1, CUDA 8.0.61 and cuDNN 5105 on Ubuntu 16.04.


This code uses the infrastructure of [PointNet](https://github.com/charlesq34/pointnet) as a template,
 however, many substantial changes have been made to the CNN model and point cloud representation.


#### Classification
Train the point cloud classification model using the default settings on ModelNet40, using: 

    python train_cls.py

Alternatively, you can tweak the different GMM parameters (e.g. number of gaussians ) or learning parameters (e.g. learning rate) using

    python train_cls.py  --gpu=0 --log_dir='log' --batch_size=64 --num_point=1024 --num_gaussians=8 --gmm_variance=0.0156 
    --gmm_type='grid' --learning_rate=0.001  --model='voxnet_pfv' --max_epoch=200 --momentum=0.9 --optimizer='adam'
     --decay_step=200000  --weight_decay=0.0 --decay_rate=0.7
 
The model will be saved to `log` directory.
Consecutive runs with the same directory name will be saved in numbered subdirectories in order to prevent accidental 
overwrite of trained models.  

### License
Our code is released under MIT License (see LICENSE file for details).

### Disclaimer
I am a mechanical engine, not a software engineer. git is relatively new to me. Therefore, if you find any place I have 
made an error or have an improvement recommendation, I would appreciate your advice.
