import torch
import torch.nn as nn
from torchvision import models


class AgeResNet(nn.Module):
    def __init__(self):
        super(AgeResNet, self).__init__()
        # ImageNet ağırlıklı ResNet18 (Transfer Learning)
        # Weights parametresi internet varsa indirir, yoksa hata verebilir.
        self.net = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)

        # Son katmanı yaş tahmini (1 çıktı) için değiştiriyoruz
        n_features = self.net.fc.in_features
        self.net.fc = nn.Sequential(
            nn.Linear(n_features, 256), nn.ReLU(), nn.Dropout(0.4), nn.Linear(256, 1)
        )

    def forward(self, x):
        return self.net(x)
