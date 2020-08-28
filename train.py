import pandas as pd
import albumentations
import os
import torch
import torch.nn as nn
from model import SEResNex50_32x4d
from wtfml.utils import EarlyStopping
import engine
from sklearn import metrics
from dataset import ClassificationDataset
# apex
#from apex import amp

def train(fold):
    training_data_path = "/home/achraf/Desktop/workspace/SkinCancerDetection/Dataset/256x256/train"
    model_path = "/home/achraf/Desktop/workspace/SkinCancerDetection/models"
    df = pd.read_csv("/home/achraf/Desktop/workspace/SkinCancerDetection/Dataset/256x256/train_folds.csv")
    device = "cuda"
    epochs = 50
    train_bs = 16
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
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=train_bs,
        shuffle=True,
        num_workers=4
    )

    valid_dataset = ClassificationDataset(
        image_path=valid_images,
        targets=valid_targets,
        resize=None,
        augmentations=valid_aug
    )
    valid_loader = torch.utils.data.DataLoader(
        valid_dataset,
        batch_size=valid_bs,
        shuffle=False,
        num_workers=4
    )

    model = SEResNex50_32x4d(pretrained="imagenet")
    model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        patience=3,
        mode="max"  # use it when the metric is AUC
    )
    """
    # Using Apex for training a little bit faster without occupying a lot of memory 
    model, optimizer = amp.initialize(
        model,
        optimizer,
        opt_level="01",
        verbosity=0
    )
    """
    es = EarlyStopping(patience=5, mode="max")
    for epoch in range(epochs):
        engine.train(train_loader, model, optimizer, device=device)
        predictions, valid_targets = engine.evaluate(
            valid_loader, model, device=device
        )
        auc = metrics.roc_auc_score(valid_targets, predictions)
        scheduler.step(auc)
        print(f"epoch={epoch}, auc={auc}")
        es(auc, model, model_path)
        if es.early_stop:
            print("early stopping")
            break
