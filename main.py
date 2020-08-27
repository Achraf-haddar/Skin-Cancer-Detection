import os
import torch

import albumentations
import pretrainedmodels
from dataset import ClassificationDataset

import numpy as np
import pandas as pd

import torch.nn as nn
import torch.nn.Functional as F

class SEResNex50_32x4d(nn.Module):
    def __init__(self, pretrained="imagenet"):
        super(SEResNex50_32x4d, self).__init__()
        self.model = pretrainedmodels.__dict__['se_resnext50_32_x4d'](pretrained=pretrained)
        # To check the number of output features
        # Run this line and check the in_features 
        # pretrained.__dict__["se_resnext50_32_x4d"]()
        self.out = nn.Lineazr(2048, 1)
    
    def forward(self, image):
        bs, _, _, _ = image.shape
        x = self.model.features(image)
        x = F.adaptive_avg_pool2d(x, 1)
        x = x.reshape(bs, -1)
        out = self.out(x)
        return out    
    def train(fold):
        training_data_path = "/home/achraf/Desktop/workspace/SkinCancerDetection/Dataset/256x256/train"
        df = pd.read_csv("/home/achraf/Desktop/workspace/SkinCancerDetection/Dataset/256x256/train_folds.csv")
        device = "cuda"
        epochs = 50
        train_bs = 32
        valid_bs = 16
        mean = (0.485, 0.456, 0.406)
        std = (0.229, 0.224, 0.225)

        df_train = df[df["kfold"] != fold].reset_index(drop=True)
        df_valid = df[df["kfold"] == fold].reset_index(drop=True)

        # Normalization inside augmentation
        train_aug = albumentations.Compose(
            [
                albumentations.Normalize(mean, std, max_pixel_value=255.0, always_apply=True),
            ]
        )
        valid_aug = albumentations.Compose(
            [
                albumentations.Normalize(mean, std, max_pixel_value=255.0, always_apply=True),
            ]
        )

        train_images = df_train.image_name.values.tolist()
        train_images = [os.path.join(training_data_path, i + ".jpg") for i in train_images]
        train_targets = df_train.target.values

        valid_images = df_valid.image_name.values.tolist()
        valid_images = [os.path.join(training_data_path, i + ".jpg") for i in valid_images]
        valid_targets = df_valid.target.values

        train_dataset = ClassificationDataset(
            image_path=train_images, 
            targets=train_targets,
            resize=None,
            augmentations=train_aug
        )
        valid_dataset = ClassificationDataset(
            image_path=valid_images,
            targets=valid_targets,
            resize=None,
            augmentations=valid_aug
        )
        