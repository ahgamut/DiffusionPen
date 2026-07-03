import torch.nn as nn


class WordUpsampler(nn.Module):
    """Tiny grayscale super-resolution net (ESPCN-style) for word crops.

    A few conv layers extract features at the low resolution, then a
    ``PixelShuffle`` upsamples by ``scale`` (default 2). Single-channel and
    fast; applied as an optional post-process to sharpen generated word crops
    before placement/rescale, with Lanczos as the guaranteed fallback when no
    checkpoint is available.
    """

    def __init__(self, scale=2, channels=1, features=32):
        super().__init__()
        self.scale = scale
        self.body = nn.Sequential(
            nn.Conv2d(channels, features, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(features, features, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(features, channels * scale * scale, kernel_size=3, padding=1),
        )
        self.shuffle = nn.PixelShuffle(scale)

    def forward(self, x):
        x = self.body(x)
        x = self.shuffle(x)
        return x
