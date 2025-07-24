import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models

class ContentLoss(nn.Module):
    def __init__(self):
        super(ContentLoss, self).__init__()
    def forward(self, sr, hr):
        return F.mse_loss(sr, hr)

class PerceptualLoss(nn.Module):
    def __init__(self, vgg_layer='features.35', device='cpu'):
        super(PerceptualLoss, self).__init__()
        vgg19 = models.vgg19(weights=models.VGG19_Weights.DEFAULT).features.to(device).eval()
        for param in vgg19.parameters():
            param.requires_grad = False
        self.vgg_features = nn.Sequential(*list(vgg19.children())[:int(vgg_layer.split('.')[-1]) + 1])
    def forward(self, sr, hr):
        sr_features = self.vgg_features(sr)
        hr_features = self.vgg_features(hr).detach()
        return F.mse_loss(sr_features, hr_features)

class DiscriminatorLoss(nn.Module):
    def __init__(self):
        super(DiscriminatorLoss, self).__init__()
        self.bce_loss = nn.BCEWithLogitsLoss()
    def forward(self, real_output, fake_output):
        real_loss = self.bce_loss(real_output, torch.ones_like(real_output))
        fake_loss = self.bce_loss(fake_output, torch.zeros_like(fake_output))
        return real_loss + fake_loss

class GeneratorAdversarialLoss(nn.Module):
    def __init__(self):
        super(GeneratorAdversarialLoss, self).__init__()
        self.bce_loss = nn.BCEWithLogitsLoss()
    def forward(self, fake_output):
        return self.bce_loss(fake_output, torch.ones_like(fake_output))
