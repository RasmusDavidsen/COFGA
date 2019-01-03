# creates a customized pytorch dataset object
# this allows for data augmentation on the fly
# as weel as allowing use of dataloader pytorch class

# inspired by: https://pytorch.org/tutorials/beginner/data_loading_tutorial.html


import os
import torch
import pandas as pd
from skimage import io, transform
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image


class CofgaDataset(Dataset):
    """Cofga dataset."""

    def __init__(self, csv_file, root_dir, transform=None):
        """
        Arguments:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the resized training images.
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        self.labels = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        img_name = os.path.join(self.root_dir,
                                self.labels.iloc[idx, 0])
        image = io.imread(img_name+'.png')




        if self.transform:
            # changeing image to PIL image, makes it possible to use augmentation
            image = Image.fromarray(image)


            # transforming image
            image = self.transform(image)

            # back to array
            image = np.asarray(image)

        # reshaping the image
        image = np.asarray(image)
        image = image.reshape(3,224,224)

        image = torch.from_numpy(image)

        labels = self.labels.iloc[idx, 1:].values.astype('uint8')
        labels = np.array(labels)

        labels = torch.from_numpy(labels)

        sample = {'image': image, 'labels': labels}

        return sample
