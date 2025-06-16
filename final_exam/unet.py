import torch
import torch.nn as nn
import torch.nn.functional as F

class SimpleUNet3ch(nn.Module):
    def __init__(self):
        super().__init__()
        self.enc1 = nn.Sequential(nn.Conv2d(3, 16, 3, padding=1), nn.ReLU())
        self.enc2 = nn.Sequential(nn.Conv2d(16, 32, 3, padding=1), nn.ReLU())
        self.enc3 = nn.Sequential(nn.Conv2d(32, 64, 3, padding=1), nn.ReLU())
        self.pool = nn.MaxPool2d(2)
        self.dec3 = nn.Conv2d(64, 32, 3, padding=1)
        self.dec2 = nn.Conv2d(32, 16, 3, padding=1)
        self.final = nn.Conv2d(16, 3, 1)  # 3채널 출력 (왼쪽, 중앙, 오른쪽 차선)

    def forward(self, x):
        e1 = self.enc1(x)                      # [B, 16, 360, 640]
        e2 = self.enc2(self.pool(e1))          # [B, 32, 180, 320]
        e3 = self.enc3(self.pool(e2))          # [B, 64, 90, 160]

        d3 = self.dec3(F.interpolate(e3, scale_factor=2))  # [B, 32, 180, 320]
        d2 = self.dec2(F.interpolate(d3, scale_factor=2))  # [B, 16, 360, 640]
        out = self.final(d2)                   # [B, 3, 360, 640]
        return torch.sigmoid(out)
