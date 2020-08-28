import torch.nn as nn
import torch.nn.functional as F
import pretrainedmodels


class SEResNex50_32x4d(nn.Module):
    def __init__(self, pretrained="imagenet"):
        super(SEResNex50_32x4d, self).__init__()
        self.model = pretrainedmodels.__dict__['se_resnext50_32x4d'](pretrained=pretrained)
        # To check the number of output features
        # Run this line and check the in_features 
        # pretrained.__dict__["se_resnext50_32_x4d"]()
        self.out = nn.Linear(2048, 1)
    
    def forward(self, image):
        bs, _, _, _ = image.shape
        x = self.model.features(image)
        x = F.adaptive_avg_pool2d(x, 1)
        x = x.reshape(bs, -1)
        out = self.out(x)
        return out 