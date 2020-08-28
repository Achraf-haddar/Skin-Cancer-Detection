import pandas as pd
import os
import albumentations
from dataset import ClassificationDataset
from model import SEResNex50_32x4d
import torch
import engine

def predict(fold):
    test_data_path = "/home/achraf/Desktop/workspace/SkinCancerDetection/Dataset/256x256/test"
    df_test = pd.read_csv("/home/achraf/Desktop/workspace/SkinCancerDetection/Dataset/256x256/test.csv")
    df_test.loc[:, 'target'] = 0

    device = "cuda"
    epochs = 5
    test_bs = 16
    mean = (0.458, 0.456, 0.406)  # mean for this model
    std = (0.229, 0.224, 0.225)  # std for this model

    test_aug = albumentations.Compose(
        [
            albumentations.Normalize(mean, std, max_pixel_value=255.0, always_apply=True)
        ]
    )
    test_images = df_test.image_name.values.tolist()
    test_images = [os.path.join(test_data_path, i + ".jpg") for i in test_images]
    test_targets = df_test.target.values

    # Test data loader
    test_dataset = ClassificationDataset(
        image_path=test_data_path,
        targets=test_targets,
        resize=None,
        augmentations=test_aug
    )
    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=test_bs,
        shuffle=False,
        num_workers=4
    )
    model = SEResNex50_32x4d(pretrained="imagenet")
    model_path = os.path.join("/home/achraf/Desktop/workspace/SkinCancerDetection/models", "model_fold_" + str(fold) + ".bin")
    print(model_path)
    model.load_state_dict(torch.load(f"{model_path}"))
    model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    predictions = engine.evaluate(
        data_loader=test_loader,
        model=model,
        device=device
    )
    return np.vstack((predictions)).ravel()