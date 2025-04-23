# Numpy-MNIST-From-Scratch

A neural network implementation for MNIST handwritten digit recognition using only NumPy - no deep learning frameworks.

## Project Overview

This project implements a neural network from scratch using only NumPy to solve the MNIST handwritten digit classification problem. Instead of relying on deep learning frameworks like PyTorch or TensorFlow, all components (forward/backward propagation, optimizers, loss functions) are implemented manually, providing deeper insights into how neural networks actually work.

## Features

- **Core Neural Network Components**:
  - Linear layers with Xavier initialization
  - 2D Convolution layers (without padding)
  - MaxPooling layers
  - BatchNormalization layers
  - Activation functions (ReLU, Softmax)
  - Multi-class Cross Entropy Loss with label smoothing

- **Optimization Methods**:
  - Stochastic Gradient Descent (SGD)
  - Momentum Gradient Descent
  - Adam Optimizer

- **Additional Features**:
  - Learning rate schedulers
  - Weight decay (L2 regularization)
  - Data augmentation (rotation, flipping, translation)
  - Weight visualization for convolutional layers

## Model Architecture

The best performing model is a CNN with the following architecture:
- Conv1 (1→16 channels, 5×5 kernel) → BatchNorm → ReLU → MaxPool
- Conv2 (16→32 channels, 5×5 kernel) → BatchNorm → ReLU → MaxPool
- Flatten → Linear (512→10) → Softmax

The model achieved **98.85%** accuracy on the test set.

## Experimental Results

Various experiments were conducted to analyze the impact of different factors:

| Factor | Impact | Best Setting |
|--------|--------|--------------|
| Network Structure | Significant positive | CNN with BatchNorm |
| Weight Initialization | Significant positive | Xavier initialization |
| Optimizer | Significant positive | Adam (β₁=0.9, β₂=0.999) |
| Learning Rate | Significant | 0.01-0.02 |
| Weight Decay | Significant negative | No weight decay (0) |
| Data Augmentation | Significant negative | No augmentation for MNIST |
| Label Smoothing | Minimal | 0.3 |
| Batch Normalization | Significant positive | Applied after convolutions |

## Usage

### Setup

First, explore the dataset with `dataset_explore.ipynb` to get familiar with the data.

### Training

To train the model:
1. Open `test_train.py`
2. Modify parameters as needed
3. Run the script

You can modify the training data by changing the values of `train_images_path` and `train_labels_path`.

### Testing

To test the model:
1. Open `test_model.py`
2. Specify the saved model's path and test dataset's path
3. Run the script to get accuracy on the test dataset

## Repository Structure

- `op.py`: Core operations (Linear, Conv2D, Loss functions, etc.)
- `models.py`: Model architecture definitions
- `optimizer.py`: Implementation of optimizers (SGD, MomentGD, Adam)
- `mynn/lr_scheduler.py`: Learning rate schedulers
- `runner.py`: Training loop and utilities
- `weight_visualization.py`: Tools for visualizing CNN weights

## Resources

- Dataset and saved models: [Google Drive](https://drive.google.com/drive/folders/1OfznFNnOanHMrzV77jk9Ch662AvIyXBU?usp=sharing)
- GitHub Repository: [Numpy-MNIST-From-Scratch](https://github.com/jitehabosmys/Numpy-MNIST-From-Scratch) 