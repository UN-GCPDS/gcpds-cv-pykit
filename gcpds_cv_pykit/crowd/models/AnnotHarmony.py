"""
Annotator Harmony Segmentation Model with ResNet34 encoder.

Clean, modular PyTorch implementation inspired by U-Net / ResUNet design patterns.
Supports multiple annotators with FiLM-based reliability estimation.
"""

from typing import List, Optional, Tuple, Union, Callable
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models


# -------------------------------------------------------------------------
# Utilities
# -------------------------------------------------------------------------

def kernel_initializer(seed: int) -> Callable:
    """Xavier uniform initializer with reproducibility."""
    def init_weights(m: nn.Module) -> None:
        if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d, nn.Linear)):
            torch.manual_seed(seed)
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
    return init_weights


def get_activation(activation: Optional[Union[str, nn.Module]], alpha: float = 0.5) -> Optional[nn.Module]:
    """Return activation function by name or module."""
    if activation is None:
        return None
    if isinstance(activation, str):
        activation = activation.lower()
        if activation == "sigmoid":
            return nn.Sigmoid()
        elif activation == "softmax":
            return nn.Softmax(dim=1)
        elif activation == "tanh":
            return nn.Tanh()
        elif activation == "relu":
            return nn.ReLU(inplace=True)
        elif activation == "sparse_softmax":
            class SparseSoftmax(nn.Module):
                def forward(self, x):
                    return torch.softmax(x, dim=1).clamp(min=1e-7)
            return SparseSoftmax()
        elif activation == "alpha_activation":
            class AlphaActivation(nn.Module):
                def __init__(self, alpha=alpha):
                    super().__init__()
                    self.alpha = alpha
                def forward(self, x):
                    return (1 - self.alpha) * torch.softmax(x, dim=1) + self.alpha * torch.sigmoid(x)
            return AlphaActivation()
        else:
            raise ValueError(f"Unsupported activation: {activation}")
    elif isinstance(activation, nn.Module):
        return activation
    else:
        raise ValueError("activation must be None, str, or nn.Module")


# -------------------------------------------------------------------------
# Core Blocks
# -------------------------------------------------------------------------

class ConvBlock(nn.Module):
    """Conv → BN → ReLU"""
    def __init__(self, in_ch: int, out_ch: int, activation: bool = True):
        super().__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False)
        self.bn = nn.BatchNorm2d(out_ch)
        self.relu = nn.ReLU(inplace=True) if activation else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.relu(self.bn(self.conv(x)))


class ResidualBlock(nn.Module):
    """Residual block with optional projection."""
    def __init__(self, in_ch: int, out_ch: int, seed: int = 42):
        super().__init__()
        self.conv1 = ConvBlock(in_ch, out_ch, activation=True)
        self.conv2 = ConvBlock(out_ch, out_ch, activation=False)
        self.skip = (
            nn.Conv2d(in_ch, out_ch, kernel_size=1, bias=False)
            if in_ch != out_ch else nn.Identity()
        )
        self.relu = nn.ReLU(inplace=True)
        self.apply(kernel_initializer(seed))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = self.skip(x)
        out = self.conv1(x)
        out = self.conv2(out)
        return self.relu(out + identity)


# -------------------------------------------------------------------------
# Encoder / Decoder
# -------------------------------------------------------------------------

class ResNet34Encoder(nn.Module):
    """ResNet34 encoder with frozen weights (except BN)."""
    def __init__(self, pretrained: bool = True):
        super().__init__()
        resnet = models.resnet34(weights=models.ResNet34_Weights.IMAGENET1K_V1 if pretrained else None)
        for name, param in resnet.named_parameters():
            if "bn" not in name:
                param.requires_grad = False

        self.layer0 = nn.Sequential(resnet.conv1, resnet.bn1, resnet.relu)
        self.maxpool = resnet.maxpool
        self.layer1 = resnet.layer1
        self.layer2 = resnet.layer2
        self.layer3 = resnet.layer3
        self.layer4 = resnet.layer4

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        skips = []
        x0 = self.layer0(x)        # H/2
        skips.append(x0)
        x = self.maxpool(x0)       # H/4
        x1 = self.layer1(x)
        skips.append(x1)
        x2 = self.layer2(x1)       # H/8
        skips.append(x2)
        x3 = self.layer3(x2)       # H/16
        skips.append(x3)
        x4 = self.layer4(x3)       # H/32 bottleneck
        return x4, skips


class UNetDecoder(nn.Module):
    """U-Net style decoder with residual refinement."""
    def __init__(self, encoder_ch: List[int], decoder_ch: List[int] = [256, 128, 64, 32, 16]):
        super().__init__()
        skips = encoder_ch[:-1][::-1] + [3]  # add input image channels
        self.blocks = nn.ModuleList()
        in_ch = encoder_ch[-1]
        for out_ch, skip_ch in zip(decoder_ch, skips):
            self.blocks.append(nn.Sequential(
                ConvBlock(in_ch + skip_ch, out_ch),
                ResidualBlock(out_ch, out_ch)
            ))
            in_ch = out_ch

    def forward(self, bottleneck: torch.Tensor, skips: List[torch.Tensor], x_in: torch.Tensor) -> torch.Tensor:
        skips = skips[::-1] + [x_in]  # reverse order and add input
        x = bottleneck
        for block, skip in zip(self.blocks, skips):
            x = F.interpolate(x, scale_factor=2, mode="bilinear", align_corners=False)
            if x.shape[2:] != skip.shape[2:]:
                skip = F.interpolate(skip, size=x.shape[2:], mode="bilinear", align_corners=False)
            x = torch.cat([skip, x], dim=1)
            x = block(x)
        return x


# -------------------------------------------------------------------------
# Heads
# -------------------------------------------------------------------------

class SegmentationHead(nn.Module):
    """Final segmentation conv + upsample + activation."""
    def __init__(self, in_ch: int, out_ch: int, activation: Optional[Union[str, nn.Module]] = None):
        super().__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, kernel_size=1)
        self.activation = get_activation(activation)

    def forward(self, x: torch.Tensor, target_size: Tuple[int, int]) -> torch.Tensor:
        x = self.conv(x)
        x = F.interpolate(x, size=target_size, mode="bilinear", align_corners=False)
        if self.activation is not None:
            x = self.activation(x)
        return x


class AttentionAnnRel(nn.Module):
    """Annotator-specific FiLM modulation + reliability head."""
    def __init__(self, in_ch: int, num_annotators: int, activation: Optional[Union[str, nn.Module]] = "sigmoid"):
        super().__init__()
        self.attention_gen = nn.Sequential(
            nn.Linear(num_annotators, 128),
            nn.ReLU(),
            nn.Linear(128, in_ch * 2)
        )
        self.confidence_head = nn.Sequential(
            nn.Conv2d(in_ch, 32, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, num_annotators, 1)
        )
        self.activation = get_activation(activation)

    def forward(self, x: torch.Tensor, onehot: torch.Tensor) -> torch.Tensor:
        params = self.attention_gen(onehot)  # B × 2C
        gamma, beta = torch.chunk(params, 2, dim=1)
        B, C = gamma.shape
        modulated = x * gamma.view(B, C, 1, 1) + beta.view(B, C, 1, 1)
        out = self.confidence_head(modulated)
        return self.activation(out) if self.activation else out


# -------------------------------------------------------------------------
# Full Model
# -------------------------------------------------------------------------

class AnnotHarmonyModel(nn.Module):
    """Annotator Harmony Segmentation Model with ResNet34 encoder."""
    def __init__(self, in_ch: int = 3, out_ch: int = 1, n_annotators: int = 1,
                 activation_seg: Optional[str] = "sparse_softmax",
                 activation_rel: Optional[str] = "sigmoid") -> None:
        super().__init__()
        self.encoder = ResNet34Encoder(pretrained=True)
        self.decoder = UNetDecoder([64, 64, 128, 256, 512])
        self.seg_head = SegmentationHead(16, out_ch, activation=activation_seg)
        self.ann_rel = AttentionAnnRel(16, n_annotators, activation=activation_rel)

    def forward(self, x: torch.Tensor, onehot: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        input_size = x.shape[2:]
        bottleneck, skips = self.encoder(x)
        decoded = self.decoder(bottleneck, skips, x)
        seg_out = self.seg_head(decoded, input_size)
        rel_out = self.ann_rel(decoded, onehot) if onehot is not None else None
        return seg_out, rel_out
