import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.layers import trunc_normal_, DropPath

class Block(nn.Module):
    """ Each ConvNeXt Block implementation includes following layers:
    1. Depth wise Convolution Layer
    2. Normalization
    3. 2 Linear layers with GELU activation function between
    4. Permute back
    We use (2) as we find it slightly faster in PyTorch
    """
    def __init__(self, dim, drop_path=0., layer_scale_init_value=1e-6):
        super().__init__()

        # Depthwise convolution (keeps number of channels)
        self.dwconv = nn.Conv2d(dim, dim, kernel_size=7, padding=3, groups=dim)

        # Simple LayerNorm (PyTorch built-in, applied after permute)
        self.norm = nn.LayerNorm(dim, eps=1e-6)

        # Pointwise (1x1) convolutions to mix between channels
        self.pwconv1 = nn.Linear(dim, 4 * dim)
        self.activation_function = nn.GELU()
        self.pwconv2 = nn.Linear(4 * dim, dim)

        # Layer scaling (small learnable scale for residual branch)
        self.layer_scale = nn.Parameter(layer_scale_init_value * torch.ones(dim)) \
            if layer_scale_init_value > 0 else None

        # Stochastic depth (drop path)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x):
        input = x  # Save input for residual connection

        # Depthwise convolution in (N, C, H, W)
        x = self.dwconv(x)

        # Permute for LayerNorm: (N, C, H, W) → (N, H, W, C)
        x = x.permute(0, 2, 3, 1)
        x = self.norm(x)

        # MLP layers
        x = self.pwconv1(x)
        x = self.activation_function(x)
        x = self.pwconv2(x)

        # Optional scaling
        if self.layer_scale is not None:
            x = self.layer_scale * x

        # Back to (N, C, H, W)
        x = x.permute(0, 3, 1, 2)

        # Residual connection
        x = input + self.drop_path(x)
        return x



class DownsampleLayer(nn.Module):
    """ Downsamples spatial resolution and increases channel depth """
    def __init__(self, in_dim, out_dim, kernel_size=2, stride=2):
        super().__init__()
        self.conv = nn.Conv2d(in_dim, out_dim, kernel_size=kernel_size, stride=stride)
        self.norm = nn.LayerNorm(out_dim, eps=1e-6)

    def forward(self, x):
        x = self.conv(x)
        x = x.permute(0, 2, 3, 1)  # (N, C, H, W) → (N, H, W, C)
        x = self.norm(x)
        x = x.permute(0, 3, 1, 2)  # back to (N, C, H, W)
        return x



class Stage(nn.Module):
    """One ConvNeXt Stage includes `depth` blocks as specified """
    def __init__(self, dim, depth, drop_path_rates, layer_scale_init_value=1e-6):
        super().__init__()

        #
        self.blocks = nn.ModuleList()
        for j in range(depth):
            block = Block(
                dim=dim,
                drop_path=drop_path_rates[j],
                layer_scale_init_value=layer_scale_init_value
            )
            self.blocks.append(block)

    def forward(self, x):
        for block in self.blocks:
            x = block(x)
        return x

class ConvNeXt(nn.Module):
    """Simplified ConvNeXt"""
    def __init__(self, in_chans=1, num_classes=2,
                 depths=[3, 3, 9, 3], dims=[96, 192, 384, 768],
                 drop_path_rate=0., layer_scale_init_value=1e-6):

        super().__init__()

        # Stem
        stem = DownsampleLayer(in_chans, dims[0], kernel_size=4, stride=4)
        # Downsample layers
        self.downsample_layers = nn.ModuleList()
        self.downsample_layers.append(stem)

        # Schedule DropPath rates
        dp_rates = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]
        cur = 0


        self.stages = nn.ModuleList()
        # 4 DownSample Layers including stem
        for i in range(3):
            downsample_layer = DownsampleLayer(dims[i], dims[i+1])
            self.downsample_layers.append(downsample_layer)

        # 4 Stages
        for i in range(4):
            # Stage i
            stage = Stage(
                dim=dims[i],
                depth=depths[i],
                drop_path_rates=dp_rates[cur:cur + depths[i]],
                layer_scale_init_value=layer_scale_init_value
            )
            self.stages.append(stage)
            cur += depths[i]

        # Final classifier
        self.head = nn.Sequential(
            nn.LayerNorm(dims[-1]),
            nn.Dropout(0.3),
            nn.Linear(dims[-1], num_classes)
            )

    def forward(self, x):
        # Reduce spatial size and learn at each resolution
        for i in range(4):
            x = self.downsample_layers[i](x)
            x = self.stages[i](x)

        x = x.mean([-2, -1]) # global average pooling, (N, C, H, W) -> (N, C)

        x = self.head(x)
        return x