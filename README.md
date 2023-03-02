# Guiding Energy-based Models via Contrastive Latent Variables

PyTorch implementation for "[Guiding Energy-based Models via Contrastive Latent Variables](https://openreview.net/forum?id=CZmHHj9MgkP)" (accepted as a Spotlight presentation in ICLR 2023)

<img width="928" alt="스크린샷 2023-03-02 20 59 24" src="https://user-images.githubusercontent.com/4075389/222422785-e3996a2c-499f-42e3-b53e-ac6b885a2c2f.png">

**TL;DR:** A simple yet effective framework for improving energy-based models (EBMs) via contrastive representation learning.

## Install

```bash
conda create -n ebm python=3.9
conda activate ebm
conda install pytorch torchvision cudatoolkit=11.3 -c pytorch
conda install torchmetrics -c conda-forge
conda install ignite -c pytorch-nightly
pip install omegaconf
pip install torch-fidelity
pip install kornia==0.6.3
pip install tensorboard
pip install sklearn
```

## Training

```bash
export CUDA_VISIBLE_DEVICES=0
python train.py configs/cifar10.yaml
```

You can modify options using YAML config files or `key=value` command-line arguments. See `utils.parse_config()` and [OmegaConf](https://omegaconf.readthedocs.io) for details.

## Generation

```bash
python test_fid.py logs/cifar10/resnet_resnet18/ours/config.yaml use_ema=true
```

This command saves 50k generated samples into `samples.pth` in the log directory. You can use this file for [official pytorch FID evaluation](https://github.com/mseitzer/pytorch-fid). Note that the FID value obtained from our code is similar to that from the official evaluation.

## Out-of-distribution Detection

```bash
python test_ood.py logs/cifar10/resnet_resnet18/ours/config.yaml use_ema=true \
    ood_data.name=svhn ood_data.root=/data \
    model.beta=0.1 model.ebm_augmentation=none
```

