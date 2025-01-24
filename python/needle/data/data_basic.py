import numpy as np
from ..autograd import Tensor

from typing import Iterator, Optional, List, Sized, Union, Iterable, Any



class Dataset:
    r"""An abstract class representing a `Dataset`.

    在一个数据集中可能存在多个部分的数据 比如MNIST_DATASET中存在imgs和labels两部分数据
    它们共同组成了一个数据集，所以 dataset = [data1, data2, ...]

    All subclasses should overwrite :meth:`__getitem__`, supporting fetching a
    data sample for a given key. Subclasses must also overwrite
    :meth:`__len__`, which is expected to return the size of the dataset.
    """

    def __init__(self, transforms: Optional[List] = None):
        self.transforms = transforms

    def __getitem__(self, index) -> object:
        r"""
            函数所返回的就是每部分数据中索引为index的部分 -> (data1_ele, data2_ele, ...)
        """
        raise NotImplementedError

    def __len__(self) -> int:
        raise NotImplementedError
    
    def apply_transforms(self, x):
        if self.transforms is not None:
            # apply the transforms
            for tform in self.transforms:
                x = tform(x)
        return x


class DataLoader:
    r"""
    Data loader. Combines a dataset and a sampler, and provides an iterable over
    the given dataset.
    Args:
        dataset (Dataset): dataset from which to load the data.
        batch_size (int, optional): how many samples per batch to load
            (default: ``1``).
        shuffle (bool, optional): set to ``True`` to have the data reshuffled
            at every epoch (default: ``False``).
     """
    dataset: Dataset
    batch_size: Optional[int]

    def __init__(
        self,
        dataset: Dataset,
        batch_size: Optional[int] = 1,
        shuffle: bool = False,
    ):

        self.dataset = dataset
        self.shuffle = shuffle
        self.batch_size = batch_size
        if not self.shuffle:
            self.ordering = np.array_split(np.arange(len(dataset)), 
                                           range(batch_size, len(dataset), batch_size))

    def __iter__(self):
        ### BEGIN YOUR SOLUTION
        if self.shuffle:
            ######################## How to shuffle ########################
            # 1. shuffle based on the pervious epoch's dataset order. ×    #
            #    may introduce the dependence.                             #
            # 2. shuffle based on the original dataset order. ✔           #
            ################################################################
            indices = np.arange(len(self.dataset))
            np.random.shuffle(indices)
            self.ordering = np.array_split(indices, 
                                           range(self.batch_size, len(self.dataset), self.batch_size))
        self.itr = 0
        ### END YOUR SOLUTION
        return self

    def __next__(self):
        ### BEGIN YOUR SOLUTION
        if self.itr == len(self.ordering):
            raise StopIteration
        sample = [self.dataset[i] for i in self.ordering[self.itr]]
        # 这里需要sample中相同部分的数据聚合起来
        # sample = [(data1_ele1, data2_ele1, ...), (data1_ele2, data2_ele2, ...), ...]
        # 需要返回的是 ([data1_ele1, data1_ele2, ...], [data2_ele1, data2_ele2, ...], ...)
        batch = ()
        for i in range(len(sample[0])):
            item = Tensor([eles[i] for eles in sample])
            batch += (item,)
        self.itr += 1
        return batch
        ### END YOUR SOLUTION

