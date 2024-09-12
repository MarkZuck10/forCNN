# LeNet-5 Implementation on MNIST Dataset

This project demonstrates an implementation of the **LeNet-5** Convolutional Neural Network (CNN) architecture, a pioneering model in deep learning designed for handwritten digit classification, specifically tested on the **MNIST** dataset.

## Structure

This part focuses on the LeNet architecture. The model has been built from scratch using Keras, trained on the MNIST dataset, and evaluated with testing data to achieve high classification accuracy.

## Architecture Overview

LeNet-5 consists of 7 layers (not counting the input layer), including:
- **Convolutional layers** for feature extraction
- **Pooling layers** for dimensionality reduction
- **Fully Connected layers** for classification

### Layer-wise Breakdown

1. **Input Layer**: 
   - Input size: 32 x 32 (MNIST images are padded from 28 x 28)
   
2. **C1 - First Convolutional Layer**:
   - Filters: 6 filters of size 5 x 5
   - Output: 28 x 28 x 6

3. **S2 - First Pooling Layer (Average Pooling)**:
   - Filter: 2 x 2
   - Output: 14 x 14 x 6

4. **C3 - Second Convolutional Layer**:
   - Filters: 16 filters of size 5 x 5
   - Output: 10 x 10 x 16

5. **S4 - Second Pooling Layer (Average Pooling)**:
   - Filter: 2 x 2
   - Output: 5 x 5 x 16

6. **C5 - Third Convolutional Layer**:
   - Filters: 120 filters of size 5 x 5
   - Output: 1 x 1 x 120

7. **F6 - Fully Connected Layer**:
   - Input: 120 units from the previous layer
   - Output: 84 units

8. **Output Layer**:
   - Input: 84 units
   - Output: 10 units (for digits 0-9)

### Mathematical Operations

#### Convolutional Layer 1 (C1):
- Input: 32 x 32, Filters: 6, Kernel: 5 x 5
- Operation: 
  \[
  ((N + 2P - F) / S) + 1 = ((32 + 0 - 5) / 1) + 1 = 28 * 28 * 6
  \]
  
#### Pooling Layer 1 (S2):
- Input: 28 x 28 x 6, Filter: 2 x 2, Stride: 2
- Operation: 
  \[
  ((N + 2P - F) / S) + 1 = ((28 + 0 - 2) / 2) + 1 = 14 * 14 * 6
  \]

#### Convolutional Layer 2 (C3):
- Input: 14 x 14 x 6, Filters: 16, Kernel: 5 x 5
- Operation:
  \[
  ((N + 2P - F) / S) + 1 = ((14 + 0 - 5) / 1) + 1 = 10 * 10 * 16
  \]

#### Pooling Layer 2 (S4):
- Input: 10 x 10 x 16, Filter: 2 x 2, Stride: 2
- Operation:
  \[
  ((N + 2P - F) / S) + 1 = ((10 + 0 - 2) / 2) + 1 = 5 * 5 * 16
  \]

#### Fully Connected Layer (F6):
- Input: Flattened 5 x 5 x 16 = 400 units
- Output: 120 units

## Training and Testing

- **Dataset**: MNIST Handwritten Digits Dataset
- **Training Accuracy**: Achieved **99.4%** training accuracy after 10 epochs
- **Testing Accuracy**: Achieved **98.8%** on the test dataset

## Results

- The LeNet-5 architecture successfully classifies handwritten digits in the MNIST dataset with high accuracy.
- The model has been visualized at various stages, showing the layers, filters, and intermediate outputs.

## Dataset

- **MNIST**: The dataset consists of 70,000 grayscale images of handwritten digits. 60,000 images are used for training, and 10,000 for testing.

## Model Working

For implementation, check this workbook
