
# SegAug-SimCLR: Segmentation-Based Augmentations for Contrastive Learning

Bachelor's Thesis · University of Bamberg · 2025

[Read the full thesis](https://www.uni-bamberg.de/fileadmin/xai/studies/theses/2025/Bachelor_Thesis_Sascha_Wolf.pdf)

## Overview

This project reproduces SimCLR and investigates whether segmentation-based augmentations can improve self-supervised visual representation learning.

The proposed extension, **SegAug-SimCLR**, uses semantic segmentation to apply different augmentations to foreground objects and image backgrounds.

The goal is to preserve relevant object information while increasing the diversity of augmented views.

## Method

SegAug-SimCLR extends the standard SimCLR augmentation pipeline:

1. Use a pretrained DeepLabV3 model to estimate foreground and background regions.
2. Apply region-specific image transformations.
3. Generate two augmented views of the same image.
4. Train a shared encoder and projection head using the NT-Xent contrastive loss.
5. Evaluate the learned representations using linear classification and downstream transfer learning.

<img width="675" height="242" alt="image" src="https://github.com/user-attachments/assets/1fbe7f34-147e-4ca8-a45d-1018819d6a28" />

## Results

SegAug-SimCLR was evaluated against standard SimCLR using ImageNet and downstream transfer-learning benchmarks.

- **ImageNet:** +0.6 percentage points in Top-1 accuracy using 10% of the ImageNet training data for pretraining.
- **Oxford Flowers:** +1.0 percentage points in transfer-learning accuracy.
- **Other datasets:** Results varied across tasks, showing that segmentation-based augmentations do not consistently improve representation quality.

The complete experimental results and discussion are available in the thesis.

## Repository Structure

- `train_simclr.py` — Contrastive pretraining
- `augmentations.py` — Standard SimCLR augmentations
- `segmentation.py` — DeepLabV3 model initialization
- `segmention_main.py` — Segmentation-based augmentations
- `linear_eval.py` — Linear classification
- `transfer_learning.py` — Downstream transfer-learning evaluation
- `fine_tune_eval.py` — Fine-tuning experiments
- `models/` — Encoder, projection head and contrastive loss

## Reproducibility

This repository contains the experimental implementation developed for the Bachelor's thesis.

Training requires the corresponding datasets, model checkpoints and a compatible PyTorch environment. Dataset and checkpoint paths must be configured before running the experiments.


## Thesis

Sascha Alexander Wolf (2025)

*Exploring Self-Supervise Learning Through SimCLR: Reproducing and Evaluating Natural Image Classification and Transfer Learning Accuracy.*

University of Bamberg, xAILab.
