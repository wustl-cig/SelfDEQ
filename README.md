## [Self-Supervised Deep Equilibrium Models for Inverse Problems with Theoretical Guarantees and Applications to MRI Reconstruction (IEEE TCI 2023)](https://ieeexplore.ieee.org/abstract/document/10214618)

[[IEEE](https://ieeexplore.ieee.org/abstract/document/10214618)] [[arXiv](https://arxiv.org/abs/2210.03837)]

Deep equilibrium models (DEQ) have emerged as a powerful alternative to deep unfolding (DU) for image reconstruction. DEQ models—implicit neural networks with effectively infinite number of layers—were shown to achieve state-of-the-art image reconstruction without the memory complexity associated with DU. While the performance of DEQ has been widely investigated, the existing work has primarily focused on the settings where groundtruth data is available for training. We present self-supervised deep equilibrium model (SelfDEQ) as the first self-supervised reconstruction framework for training model-based implicit networks from undersampled and noisy MRI measurements. Our theoretical results show that SelfDEQ can compensate for unbalanced sampling across multiple acquisitions and match the performance of fully supervised DEQ. Our numerical results on in-vivo MRI data show that SelfDEQ leads to state-of-the-art performance using only undersampled and noisy training data.

![](/fig/Fig_method_wDU.png)

## Preparation

### (1) Download the dataset from [this link](https://wustl.box.com/s/vld318wlm8b05hfi4jkfjk0erl26au9o) and put it into the folder 

```
dataset/mri_factor8_sigma0.01_acquisition_times2_is_completeTrue_ACS_percentage0.175_is_fixed_ACSFalse/
```

This dataset is based on dataset provided in this [repo](https://github.com/hkaggarwal/modl).

### (2) Setup the environment

We use Conda to set up a new environment by running 

```
conda env create --file "requirement.yml" -n selfdeq
```

and then activate this new environment

```
conda activate selfdeq
```

## Run the SelfDEQ testing

We provide the checkpoint of the weighted self-supervised DEQ (SelfDEQ). Please run

```
python main.py --setting.gpu_index GPU_INDEX  --setting.mode test
```

The results can be found at the folder `./experiment`.

## Run the SelfDEQ training

We also provide training script for SelfDEQ under full-rank sampling.

```
python main.py --setting.gpu_index GPU_INDEX --setting.mode train
```

The results can be found at the folder `./experiment`. Full training logs and testing results can be found in this folder. 

## Citation
```
@article{gan2023self,
  title={Self-supervised deep equilibrium models with theoretical guarantees and applications to mri reconstruction},
  author={Gan, Weijie and Ying, Chunwei and Boroojeni, Parna Eshraghi and Wang, Tongyao and Eldeniz, Cihat and Hu, Yuyang and Liu, Jiaming and Chen, Yasheng and An, Hongyu and Kamilov, Ulugbek S},
  journal={IEEE Transactions on Computational Imaging},
  year={2023},
  publisher={IEEE}
}
```