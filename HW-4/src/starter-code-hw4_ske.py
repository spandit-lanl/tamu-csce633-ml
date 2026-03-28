import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset
from PIL import Image
import argparse
import numpy as np
import glob
from rich.progress import Progress, TextColumn, BarColumn, TaskProgressColumn
from time import time

import torchvision

class SUN397Dataset(Dataset):
    """
    A custom dataset class for loading the SUN397 dataset.
    """

    def __init__(self, data_dir, transform=None):
        """
        Initializes the dataset with images and labels.
        Args:
            data_dir (str): Path to the data directory.
            transform (callable, optional): Optional transform to be applied on an image.
        """
        pass

    def __len__(self):
        """
        Returns the number of images in the dataset.
        """
        pass

    def __getitem__(self, idx):
        """
        Retrieves an image and its label at the specified index.

        Args:
            idx (int): Index of the image to retrieve.

        Returns:
            tuple: (image, label)
        """
        pass


class CNN(nn.Module):
    """
    Define your CNN Model here
    """
    def __init__(self, num_classes=10):
        """
        Initializes the layers of the CNN model.

        Args:
            num_classes (int): Number of output classes.
        """
        pass

    def forward(self, x):
        """
        Defines the forward pass of the model.

        Args:
            x (Tensor): Input tensor.

        Returns:
            Tensor: Output of the model.
        """
        pass


def calculate_mean_std(**kwargs):
    """
    Fill in the per channel mean and standard deviation of the dataset.
    Just fill in the values, no need to compute them.
    """
    # return [mean_r, mean_g, mean_b], [std_r, std_g, std_b]
    pass

'''
All of the following functions are optional. They are provided to help you get started.
'''

def train(model, train_loader, **kwargs):
    pass

def test(model, test_loader, **kwargs):
    pass

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_dir', type=str,
                        default='welcome/to/CNN/homework',
                        help='Path to training data directory')

    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed for reproducibility')

    return parser.parse_args()

def main():
    args = parse_args()
    torch.manual_seed(args.seed)


if __name__ == "__main__":
    main()
