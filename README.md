![Python](https://img.shields.io/badge/python-3.12-blue)
![License](https://img.shields.io/badge/license-MIT-green)

# 3DeepFlow: learning-based optical flow for 3D biological images.

## Table of Contents
- [Overview](#overview)
- [3DeeFlow](#3deepflow)
- [Synthetic Elastic Cell Generation](#synthetic-elastic-cell-generation)

---

<p align="center">
  <img src="docs/figure_method_V1.pdf" width="600" alt="Description"/>
</p>
## Overview

The repository is divided into three main parts:

**1. 3DeepFlow** — Learning-based optical flow method for 3D anisotropic biological images. It is trained on synthetic fractal images. 

**2. Synthetic elastic cell generation** - Generates synthetic 3D images of deformed elastic cells using FEniCSx (finite element simulation).

---

## 3DeepFlow

Create a conda environment

```bash
micromamba env create -f environment.yml
```

### Fractal Generation

Fractal images are created using 

```bash
python -m data_generation.fractals.fractals_generation --config=configs/fractal_params.yaml
```

This will create and store images of synthetic fractal images of shape $(32 \times 128 \times 128)$ randomly deformed. 
The random deformation is either generated using Perlin noise, generated using a Gaussian random field, generated with rigid transformation (rotation + translation) or generated with rigid transformation + a Gaussian random field. The last configuration is the one used for in the training dataset for 3DeepFlow.

`configs/fractal_params.yaml` contains all the parameters related to 3D fractal images creation.

### Model

3DeepFlow is a U-Net developed for learning displacement fields on anisotropic 3D images.
Use `sripts/train` with a train config file to train 3DeepFlow. Train config files are in `configs/train_configs`. They contain parameters like the training dataset, the learning rate $l_r$, the random seed, the weights of the different loss parts, etc.

### Optical Flow methods

Optical flow methods to compare 3DeepFlow's results to are available in `optical_flow/algorithms`. THey include Horn and Shcunck's seminal method, a Hessian-regularized Horn&Schunck method, a TV-L1 regularized and Iterative Lucas Kanade algorithms.
The parameters for each method can be modified in `configs/optical_flow.yaml`. 
To compute and store the results of different optical flow methods on images contained in a dataset, run 

```bash
python -m optical_flow.compute_save_of --config=configs/save_params.yaml
```

with `configs/save_params.yaml` containing the images folder, and the folder where to save the results. 

To test different parameters for different OF methods, run

```bash
python -m optical_flow.fine_tuning --config=configs/of_fine_tuning.yaml
```

where `configs/fine_tuning.yaml` contains the parameters to test.

### Inference, mechanobiology

We measure the quality of every method on synthetic images with avaiable Ground-Truth displacement, strain and traction. We use NRMSE to quantify the distance to GT for each method. 

#### Simple synthetic experiment

To infer a displacement field and mechanical values using 3DeepFlow and compare them to those obtained by different OF algorithms, use

```bash
python -m experiment.exp_synthetic
```

Change the important arguments direclty in the python file. 

#### Ablation experiments 

To test different configurations for the model, we use 

```bash
python -m experiment.exp_ablation
```

And to test different training dataset sizes, we use

```bash
python -m experiment.exp_ablation_training
```

Here as well, change the important arguments direclty in the python file. 
> [!NOTE]
> These experiments require different trained models to run.

#### Noise experiments

To test robustness to noise, we use 

```bash
python -m experiment.exp_ablation_training
```

#### Real microscopy image experiment 

To observe the results on a real microscopy image, we use 

```bash
python -m experiment.exp_micro
```
> [!NOTE]
> Preprocessing steps (channel extraction, reshaping) are handled direclt inside the `exp_micro.py` python file. They may need to be adapted to your image format.

---

## Synthetic Elastic Cell Generation

Create a separate conda environment: 

```bash
micromamba env create -f 3denvironment.yml
```

### Generate 3D elastic cell images 

Elastic cell images are created using 

```bash
python -m data_generation.cells.generate_3d_elastic_dataset --config=configs/three_dims_params.yaml
```

Parameters are defined in `configs/three_dims_params.yaml`

> [!NOTE]
> This experiment relies on synthetic cell images from the cell tracking challenge (dataset Fluo-C3DH-A549-SIM).
> You need to download the dataset and input the correct path.