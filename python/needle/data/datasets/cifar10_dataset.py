import os
import pickle
from typing import Iterator, Optional, List, Sized, Union, Iterable, Any
import numpy as np
from ..data_basic import Dataset

class CIFAR10Dataset(Dataset):
    def __init__(
        self,
        base_folder: str,
        train: bool,
        p: Optional[int] = 0.5,
        transforms: Optional[List] = None
    ):
        """
        Parameters:
        base_folder - cifar-10-batches-py folder filepath
        train - bool, if True load training dataset, else load test dataset
        Divide pixel values by 255. so that images are in 0-1 range.
        Attributes:
        X - numpy array of images
        y - numpy array of labels
        """
        ### BEGIN YOUR SOLUTION
        self.p = p
        self.transforms = transforms
        if train:
            files = [f'{base_folder}/data_batch_{i+1}' for i in range(5)]
        else:
            files = [f'{base_folder}/test_batch']
        n = 10000
        self.N, self.H, self.W, self.C = len(files)*n, 32, 32, 3
        X = np.zeros((self.N, self.C, self.H, self.W)) # N * H * W * C
        y = np.zeros(self.N) # N
        for i in range(len(files)):
            with open(files[i], 'rb') as fo:
                dict = pickle.load(fo, encoding='bytes')
                data = dict[b'data']
                labels = dict[b'labels']
                print(f'data.shape={data.shape}')
                assert data.shape == (n, self.H*self.W*self.C)
                assert len(labels) == n
                X[i*n:(i+1)*n] = data.reshape(n, self.C, self.H, self.W) / 255
                y[i*n:(i+1)*n] = labels
        self.dataset = [X, y]
        ### END YOUR SOLUTION

    def __getitem__(self, index) -> object:
        """
        Returns the image, label at given index
        Image should be of shape (3, 32, 32)
        """
        ### BEGIN YOUR SOLUTION
        img = self.apply_transforms(self.dataset[0][index])
        label = self.dataset[1][index]
        return (img, label)
        ### END YOUR SOLUTION

    def __len__(self) -> int:
        """
        Returns the total number of examples in the dataset
        """
        ### BEGIN YOUR SOLUTION
        return self.N
        ### END YOUR SOLUTION
