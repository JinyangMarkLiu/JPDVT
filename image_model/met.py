import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import glob
from PIL import Image
import torchvision.transforms as T
from einops import rearrange
from einops.layers.torch import Rearrange

from torchvision import transforms
from sklearn.model_selection import train_test_split
import cv2

class MET(Dataset):
    def __init__(self, image_dir,split):
        seed = 42
        torch.manual_seed(seed)
        self.split = split

        all_files = os.listdir(image_dir)
        self.image_files = [os.path.join(image_dir,all_files[0])+'/' + k for k in os.listdir(os.path.join(image_dir,all_files[0]))]
        self.image_files += [os.path.join(image_dir,all_files[1])+'/' + k for k in os.listdir(os.path.join(image_dir,all_files[1]))]
        self.image_files += [os.path.join(image_dir,all_files[2])+'/' + k for k in os.listdir(os.path.join(image_dir,all_files[2]))]
        # +os.listdir(os.path.join(image_dir,all_files[1]))+os.listdir(os.path.join(image_dir,all_files[2]))
        for image in self.image_files:
            if '.jpg' not in image:
                self.image_files.remove(image)
        dataset_indices = list(range(len( self.image_files)))

        train_indices, test_indices = train_test_split(dataset_indices, test_size=2000, random_state=seed)
        train_indices, val_indices = train_test_split(train_indices, test_size=1000, random_state=seed) 
        self.train_indices = train_indices
        self.test_indices = test_indices
        self.val_indices = val_indices

        # Define the color jitter parameters
        brightness = 0.4  # Randomly adjust brightness with a maximum factor of 0.4
        contrast = 0.4    # Randomly adjust contrast with a maximum factor of 0.4
        saturation = 0.4  # Randomly adjust saturation with a maximum factor of 0.4
        hue = 0.1         # Randomly adjust hue with a maximum factor of 0.1

        flip_probability = 0.5
        self.transform1 = transforms.Compose([
            transforms.Resize(398), 
            transforms.RandomCrop((398,398)),
            transforms.RandomHorizontalFlip(p=flip_probability),  # Horizontal flipping with 0.5 probability
            # transforms.RandomVerticalFlip(p=flip_probability), 
            transforms.ColorJitter(brightness=brightness, contrast=contrast, saturation=saturation, hue=hue),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], inplace=True)
        ])

        self.transform2 = transforms.Compose([
            transforms.Resize(398), 
            transforms.CenterCrop((398,398)), 
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], inplace=True)
        ])

    def __len__(self):
        if self.split == 'train':
            return len(self.train_indices)
        elif self.split == 'val':
            return len(self.val_indices)
        elif self.split == 'test':
            return len(self.test_indices)

    def erode(self,image,n_patches):
        output = torch.zeros(3,96*3,96*3)
        crop = transforms.RandomCrop((96,96))
        gap = 48
        patch_size = 100
        for i in range(n_patches):
            for j in range(n_patches):
                left = i * (patch_size + gap)
                upper = j * (patch_size + gap)
                right = left + patch_size
                lower = upper + patch_size

                patch = crop(image[:,left:right, upper:lower])
                output[:,i*96:i*96+96,j*96:j*96+96] = patch

        return output

    def erode1(self,image,n_patches):
        output = torch.zeros(3,96*3,96*3)
        crop = transforms.RandomCrop((96,96))
        gap = 48
        patch_size = 100
        for i in range(n_patches):
            for j in range(n_patches):
                left = i * (patch_size + gap)
                upper = j * (patch_size + gap)
                right = left + patch_size
                lower = upper + patch_size

                patch = crop(image[:,left:right, upper:lower])
                output[:,i*96:i*96+96,j*96:j*96+96] = patch

        return output
    

    def __getitem__(self, idx):
        if self.split == 'train':
            index = self.train_indices[idx]
            image = self.transform1(Image.open(self.image_files[index]))
            image = self.erode(image,3)
        elif self.split == 'val':
            index = self.val_indices[idx]
            image = self.transform2(Image.open(self.image_files[index]))
            image = self.erode1(image,3)
        elif self.split == 'test':
            index = self.test_indices[idx]
            image = self.transform2(Image.open(self.image_files[index]))
            image = self.erode1(image,3)

        return image






