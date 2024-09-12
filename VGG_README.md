# VGG-16 Implementation on CIFAR-10 Dataset

This part showcases the application of the **VGG-16** Convolutional Neural Network (CNN) architecture, which is renowned for its depth and simplicity. The model is tested on the **CIFAR-10** dataset, a common benchmark for image classification tasks.

## Structure

This section provides an overview of the VGG-16 architecture and its adaptation to the CIFAR-10 dataset. The focus is on resizing the dataset to match the VGG-16 input requirements.

## Architecture Overview

The VGG-16 model consists of 16 layers (13 convolutional layers and 3 fully connected layers). Here’s a breakdown of the architecture:

### Layer-wise Breakdown

1. **Input Layer:**
   - **Input Size:** 224 x 224 x 3 (Resized CIFAR-10 images)

2. **Convolutional Blocks:**
   - **First Block:**
     - **Conv2D (64 filters, 3x3):** Output size = 224 x 224 x 64
     - **Conv2D (64 filters, 3x3):** Output size = 224 x 224 x 64
     - **MaxPooling2D (2x2):** Output size = 112 x 112 x 64

   - **Second Block:**
     - **Conv2D (128 filters, 3x3):** Output size = 112 x 112 x 128
     - **Conv2D (128 filters, 3x3):** Output size = 112 x 112 x 128
     - **MaxPooling2D (2x2):** Output size = 56 x 56 x 128

   - **Third Block:**
     - **Conv2D (256 filters, 3x3):** Output size = 56 x 56 x 256
     - **Conv2D (256 filters, 3x3):** Output size = 56 x 56 x 256
     - **Conv2D (256 filters, 3x3):** Output size = 56 x 56 x 256
     - **MaxPooling2D (2x2):** Output size = 28 x 28 x 256

   - **Fourth Block:**
     - **Conv2D (512 filters, 3x3):** Output size = 28 x 28 x 512
     - **Conv2D (512 filters, 3x3):** Output size = 28 x 28 x 512
     - **Conv2D (512 filters, 3x3):** Output size = 28 x 28 x 512
     - **MaxPooling2D (2x2):** Output size = 14 x 14 x 512

   - **Fifth Block:**
     - **Conv2D (512 filters, 3x3):** Output size = 14x14x512
     - **Conv2D (512 filters, 3x3):** Output size = 14x14x512
     - **Conv2D (512 filters, 3x3):** Output size = 14x14x512
     - **MaxPooling2D (2x2):** Output size = 7x7x512

3. **Fully Connected Layers:**
   - **Flatten:** Converts 7 x 7 x 512 feature maps to a 1D vector of size 25,088.
   - **Dense (256 units, ReLU):** Output size = 256
   - **Dropout (0.5):** Applied to reduce overfitting.
   - **Dense (10 units, Softmax):** Output size = 10 (classification into CIFAR-10 classes)

### Mathematical Operations

- **Convolutional Layers:** Use small 3x3 kernels with padding='same' to maintain spatial dimensions until pooling layers.
- **Pooling Layers:** Apply 2x2 max pooling to reduce spatial dimensions by half at each step.
- **Fully Connected Layers:** Final dense layers for classification, with dropout applied to mitigate overfitting.

### Implementation

```python
model = models.Sequential()

# First Conv Block
model.add(layers.Conv2D(64, (3, 3), padding='same', activation='relu', input_shape=(32, 32, 3)))
model.add(layers.Conv2D(64, (3, 3), padding='same', activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))

# Second Conv Block
model.add(layers.Conv2D(128, (3, 3), padding='same', activation='relu'))
model.add(layers.Conv2D(128, (3, 3), padding='same', activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))

# Third Conv Block
model.add(layers.Conv2D(256, (3, 3), padding='same', activation='relu'))
model.add(layers.Conv2D(256, (3, 3), padding='same', activation='relu'))
model.add(layers.Conv2D(256, (3, 3), padding='same', activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))

# Fourth Conv Block
model.add(layers.Conv2D(512, (3, 3), padding='same', activation='relu'))
model.add(layers.Conv2D(512, (3, 3), padding='same', activation='relu'))
model.add(layers.Conv2D(512, (3, 3), padding='same', activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))

# Fifth Conv Block
model.add(layers.Conv2D(512, (3, 3), padding='same', activation='relu'))
model.add(layers.Conv2D(512, (3, 3), padding='same', activation='relu'))
model.add(layers.Conv2D(512, (3, 3), padding='same', activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))

# Fully connected layers
model.add(layers.Flatten())
model.add(layers.Dense(4096, activation='relu'))
model.add(layers.Dense(4096, activation='relu'))
model.add(layers.Dense(10, activation='softmax'))
```

## Results

- The VGG-16 architecture effectively processes CIFAR-10 images, with resizing and normalization aligning with VGG-16 input requirements.
- The model is prepared for feature extraction using pre-trained weights, though training is not performed due to resource constraints.

## Dataset

- **CIFAR-10**: A dataset consisting of 60,000 32x32 color images in 10 classes, with 6,000 images per class. For this project, a subset of 1,000 images for training and 200 images for testing is utilized.

## Visualization

Architecture Diagram

## Demonstration

Note: Training is not performed in this demonstration due to limitations on local resources, we focus on using pre-trained weights and feature extraction.
**Link**: Notebook Link
