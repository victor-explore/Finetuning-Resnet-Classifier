# Fine-tuning ResNet Classifier

This project implements a classifier on a given dataset by fine-tuning a ResNet-50 pre-trained on ImageNet. The classifier is designed to categorize images into 90 different animal classes.

## Project Overview

The main goal of this project is to leverage transfer learning by fine-tuning a pre-trained ResNet-50 model on a custom dataset of animal images. This approach allows us to benefit from the robust feature extraction capabilities of ResNet while adapting it to our specific classification task.

## Key Features

1. **Model Architecture**: Utilizes ResNet-50 pre-trained on ImageNet as the base model. 
2. **Fine-tuning**: Modifies the final fully connected layer to output 90 classes instead of the original 1000 ImageNet classes. 
3. **Partial Freezing**: Freezes initial layers while allowing the last convolution layer and the new fully connected layer to be trainable. 
4. **Data Augmentation**: Implements random horizontal flips and rotations to enhance model generalization. 
5. **Learning Rate Scheduling**: Uses StepLR scheduler to adjust the learning rate during training. 
6. **Performance Metrics**: Reports loss, accuracy, and F1 score for comprehensive model evaluation. 
## Dataset

The dataset consists of animal images divided into 90 classes. It is split into training (60%), validation (20%), and test (20%) sets.

## Implementation Details

- **Framework**: PyTorch
- **Base Model**: ResNet-50 (torchvision.models)
- **Optimizer**: SGD with momentum
- **Loss Function**: Cross-Entropy Loss
- **Learning Rate**: Starting at 0.1, reduced by a factor of 0.1 every 25 epochs
- **Batch Size**: 128
- **Number of Epochs**: 50

## Usage

The main implementation can be found in the `Finetuning Resnet Classifier.ipynb` notebook. This notebook contains:

- Data loading and preprocessing
- Model architecture modification
- Training loop with validation
- Evaluation on the test set
- Model saving functionality

## Results

The notebook reports the following metrics for each epoch:
- Training and validation loss
- Training and validation accuracy
- Training and validation F1 score

Final test set performance is also reported, including test loss, accuracy, and F1 score.

## Requirements

- PyTorch
- torchvision
- numpy
- matplotlib
- tqdm

## Future Work

Potential areas for improvement and exploration include:
1. Experimenting with different ResNet architectures (e.g., ResNet-32, ResNet-101)
2. Implementing more advanced data augmentation techniques
3. Exploring different learning rate scheduling strategies
4. Applying the model to different datasets or expanding the current dataset
