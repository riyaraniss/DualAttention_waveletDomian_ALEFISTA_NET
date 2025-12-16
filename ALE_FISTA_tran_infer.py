"""
Wavelet-Domain FISTA-Net with Dual Attention and ALE Preprocessing
Author: (for reproducibility appendix)
Python 3.10 | PyTorch 2.0 | CUDA 11.2
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pywt

# ============================================================
# 1. ALE PREPROCESSING
# ============================================================

class ALEPreprocessor(nn.Module):
    def __init__(self, kernel_size=63):
        super().__init__()
        self.conv = nn.Conv1d(
            in_channels=1,
            out_channels=1,
            kernel_size=kernel_size,
            padding=kernel_size // 2,
            bias=False
        )
        nn.init.dirac_(self.conv.weight)

    def forward(self, x):
        # x: [B, 1, F, T]
        B, _, F, T = x.shape
        x = x.view(B, 1, F * T)
        x = self.conv(x)
        return x.view(B, 1, F, T)

# ============================================================
# 2. ATTENTION MODULES
# ============================================================

class ChannelAttention(nn.Module):
    def __init__(self, channels, reduction=16):
        super().__init__()
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, channels // reduction),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super().__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size, padding=kernel_size // 2)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_map = torch.mean(x, dim=1, keepdim=True)
        max_map, _ = torch.max(x, dim=1, keepdim=True)
        attn = torch.cat([avg_map, max_map], dim=1)
        attn = self.sigmoid(self.conv(attn))
        return x * attn


class DualAttention(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.ca = ChannelAttention(channels)
        self.sa = SpatialAttention()

    def forward(self, x):
        return self.sa(self.ca(x))

# ============================================================
# 3. FISTA BLOCK
# ============================================================

class FISTABlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1)
        self.theta = nn.Parameter(torch.tensor(0.05))

    def soft_threshold(self, x):
        return torch.sign(x) * F.relu(torch.abs(x) - self.theta)

    def forward(self, x):
        y = F.relu(self.conv1(x))
        y = self.conv2(y)
        return self.soft_threshold(x + y)

# ============================================================
# 4. WAVELET TRANSFORM UTILITIES
# ============================================================

def wavelet_decompose(x, wavelet='db4', level=2):
    coeffs = []
    for sample in x:
        c = pywt.wavedec2(sample.squeeze().cpu().numpy(),
                          wavelet, level=level)
        coeffs.append(c)
    return coeffs


def wavelet_reconstruct(coeffs, wavelet='db4'):
    rec = []
    for c in coeffs:
        r = pywt.waverec2(c, wavelet)
        rec.append(r)
    return torch.tensor(np.array(rec)).unsqueeze(1)

# ============================================================
# 5. FULL MODEL
# ============================================================

class WaveletALEFISTANet(nn.Module):
    def __init__(self, num_blocks=6):
        super().__init__()

        self.ale = ALEPreprocessor()
        self.encoder = nn.Conv2d(1, 32, 3, padding=1)

        self.blocks = nn.ModuleList([
            nn.Sequential(
                FISTABlock(32),
                DualAttention(32)
            ) for _ in range(num_blocks)
        ])

        self.decoder = nn.Conv2d(32, 1, 3, padding=1)

    def forward(self, x):
        # ALE preprocessing
        x = self.ale(x)

        # Encode
        z = self.encoder(x)

        # Iterative FISTA + attention
        for blk in self.blocks:
            z = blk(z)

        # Decode
        return self.decoder(z)

# ============================================================
# 6. TRAINING FUNCTION
# ============================================================

def train(model, loader, epochs=100, lr=1e-4):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.L1Loss()

    model.train()
    for ep in range(epochs):
        total_loss = 0
        for noisy, target in loader:
            noisy, target = noisy.to(device), target.to(device)

            optimizer.zero_grad()
            out = model(noisy)
            loss = loss_fn(out, target)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        print(f"Epoch {ep+1:03d} | Loss: {total_loss:.4f}")

# ============================================================
# 7. INFERENCE FUNCTION
# ============================================================

def inference(model, input_npy, output_npy):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    model.eval()

    x = torch.from_numpy(np.load(input_npy)).float()
    x = x.unsqueeze(0).unsqueeze(0).to(device)

    with torch.no_grad():
        y = model(x)

    np.save(output_npy, y.squeeze().cpu().numpy())

# ============================================================
# 8. MAIN (EXAMPLE USAGE)
# ============================================================

if __name__ == "__main__":
    print("Wavelet-ALE Dual-Attention FISTA-Net initialized.")
