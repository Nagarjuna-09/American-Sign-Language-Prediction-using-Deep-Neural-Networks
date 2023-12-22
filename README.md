# American-Sign-Language-Prediction-using-Deep-Neural-Networks in Real-time

![Sign Language Prediction](path/to/your/image.png)

## Overview

This project focuses on predicting sign language gestures in real-time using deep neural networks. The final tuned and trained model achieved an accuracy of 94.85%.

## About the Dataset

### Supervised Learning Dataset

The Sign Language dataset consists of 28x28 images of hands depicting the 26 letters of the English alphabet. The data is pre-processed and fed into a convolutional neural network to correctly classify each image as the letter it represents.

### Dataset Details

- **Training Dataset:** [sign_mnist_train.csv](./sign_mnist_train.csv)
- **Test Dataset:** [sign_mnist_test.csv](./sign_mnist_test.csv)

## Getting Started

### Prerequisites

Make sure you have the following libraries installed:

```bash
pip install numpy tensorflow matplotlib
import csv
import string
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.image import ImageDataGenerator, array_to_img
