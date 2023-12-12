# DeMAT: Diffusion-enhanced Mask Aware Transformer for Deep Generative In-painting

## Overview
DeMAT (Diffusion-enhanced Mask Aware Transformer) is a novel approach to deep generative in-painting. This project addresses the challenge of filling in missing regions in images, which is crucial for image editing, re-targeting, restoration, and object removal.

## Background
Our approach leverages the latest advancements in transformer architecture and deep learning to model non-local relationships, effectively overcoming the limitations of traditional CNNs in complex in-painting tasks. By integrating techniques from the Mask Aware Transformer (MAT) and Variational Autoencoder, DeMAT excels in reconstructing high-resolution images where masked areas are seamlessly reconstructed or transformed.

## Setup and Installation

### Prerequisites
- Anaconda or Miniconda
- Python 3.7

### Set up the environment
Before proceeding with the setup, ensure you have a working Conda environment with Python 3.7. To create a new environment, run:

```bash
conda create -n demat-env python=3.7
conda activate demat-env
```

After this, you need to install the requirements from the root directory and the requirements under `MAT` directory. To do that, run:

```bash
cd MAT
pip install -r requirements.txt
cd ..
pip install -r requirements.txt
```

Next, you have to install pytorch from this [website](https://pytorch.org/get-started/locally/)

### Usage

#### MAT
To run MAT training and inference, follow the instruction from `README.md` from `MAT` directory. 

#### VAE
The code for running VAE encoder and decoder is available under `transformation.py` script. You need to download the pre-trained VAE from [huggingface](https://huggingface.co/stabilityai/stable-diffusion-2/tree/main). Download the `vae` directory and put it inside `MAT` directory.

### Note
The `MAT` directory contains the cloned repository of the implementation of the original *MAT: Mask-Aware Transformer for Large Hole Image Inpainting* paper. The `train_test_split.py` and `encode_vae.py` are not from the original repository. Those are a helper scripts to do training using our dataset.
