# American-Sign-Language-Prediction-using-Deep-Neural-Networks in Real-time

![Sign Language Prediction](https://miro.medium.com/v2/resize:fit:665/1*MLudTwKUYiCYQE0cV7p6aQ.png)

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
import pandas as pd
```
## Parsing the dataset (Splitting data into Training & Test datasets)
This function reads a file passed as input and return 2 numpy arrays, one containing the labels and one containing the 28x28 representation of each image within the file.

The first row contains the column headers, so you should ignore it.

Each successive row contains 785 comma-separated values between 0 and 255

The first value is the label

The rest are the pixel values for that picture
```bash
def parse_data_from_input(filename):
  with open(filename) as file:
    # Initialize empty lists to store labels and pixel data
    labels = []
    pixel_data = []

    # Using csv.reader and passing in the appropriate delimiter
    csv_reader = csv.reader(file, delimiter=',')

    # Skipping the header row (header row contains 'label', 'pixel1', 'pixel2', 'pixel3', 'pixel4')
    header = next(csv_reader)

    # Iterate over each row in the CSV file
    for row in csv_reader:
        # Extract the label and pixel data into label and pixel arrays
        label = int(row[0])  # 'label' is in the first column of the dataset
        pixels = [int(x) for x in row[1:]]  # pixel data starts from the second column to the end

        # Append the label and pixel data to their respective lists
        labels.append(label)
        pixel_data.append(pixels)

    # Converting the list of pixel data to a NumPy array with the desired shape and dtype
    images = np.array(pixel_data, dtype=float)  # Convert to float64
    images = images.reshape(-1, 28, 28)  # Reshaping 1D array of data into (27455, 28, 28)
    labels = np.array(labels, dtype=float)

    return images, labels
```
