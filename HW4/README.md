# Homework 4 - Image Classification

## Packages:
- numpy
- pandas
- pytorch
- torchvision

## Usage:
1. 利用WiderResNet進行Fashion MNIST圖片辨識，另外有利用Random Erasing等Data augmentation增加圖片多樣性
    ```bash
    $ python 0750730.py
    ```
    - Input File:
        - Place training data and testing data in
        ```bash
        ./data/
        ```
    - Data Format:
        - 28x28 image
        - Pixel: from 0(white) to 255(Black)
    - Output File Format:
        - submission.csv
        - ID: the index 
        - label: 10 classes
        ```bash
        ID,ans
        0,3
        1,0
        2,1
        ...
        ```