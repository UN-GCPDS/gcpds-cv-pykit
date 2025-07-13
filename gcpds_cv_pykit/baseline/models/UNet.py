"""
U-Net implementation with ResNet34 backbone for image segmentation.

This module provides a PyTorch implementation of the U-Net architecture with
ResNet34 pre-trained backbone.
"""

from typing import List, Optional, Tuple, Union, Literal
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet34


class DoubleConv(nn.Module):
    """Double convolution block with batch normalization and ReLU activation.

    Args:
        in_channels: Number of input channels.
        out_channels: Number of output channels.
    """

    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the double convolution block.

        Args:
            x: Input tensor.

        Returns:
            Processed tensor after double convolution.
        """
        return self.double_conv(x)


class ResNet34Encoder(nn.Module):
    """ResNet34-based encoder for U-Net.

    Args:
        pretrained: Whether to use pretrained weights.
        in_channels: Number of input channels.
    """

    def __init__(self, pretrained: bool = True, in_channels: int = 3) -> None:
        super().__init__()
        if pretrained:
            resnet = resnet34(weights='IMAGENET1K_V1')
        else:
            resnet = resnet34(weights=None)
            
        # Initial layers
        self.layer0 = nn.Sequential(
            resnet.conv1,
            resnet.bn1,
            resnet.relu,
        )
        
        # Handle different input channels
        if in_channels != 3:
            self.layer0[0] = nn.Conv2d(
                in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False
            )
        
        self.maxpool = resnet.maxpool
        self.layer1 = resnet.layer1  # 64 channels
        self.layer2 = resnet.layer2  # 128 channels
        self.layer3 = resnet.layer3  # 256 channels
        self.layer4 = resnet.layer4  # 512 channels

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        """Forward pass through the ResNet34 encoder.

        Args:
            x: Input tensor.

        Returns:
            Tuple containing:
                - Output tensor after encoding (bottleneck)
                - List of skip connection tensors [skip0, skip1, skip2, skip3]
        """
        skips = []
        
        # Layer 0: Initial conv + bn + relu
        x = self.layer0(x)  # 64 channels
        skips.append(x)
        
        # MaxPool and subsequent layers
        x = self.maxpool(x)
        
        x = self.layer1(x)  # 64 channels
        skips.append(x)
        
        x = self.layer2(x)  # 128 channels
        skips.append(x)
        
        x = self.layer3(x)  # 256 channels
        skips.append(x)
        
        x = self.layer4(x)  # 512 channels (bottleneck)
        
        return x, skips


class DecoderBlock(nn.Module):
    """Single decoder block with upsampling and skip connection.

    Args:
        in_channels: Number of input channels from previous layer.
        skip_channels: Number of channels from skip connection.
        out_channels: Number of output channels.
    """

    def __init__(self, in_channels: int, skip_channels: int, out_channels: int) -> None:
        super().__init__()
        self.upsample = nn.ConvTranspose2d(
            in_channels, out_channels, kernel_size=2, stride=2
        )
        self.conv = DoubleConv(out_channels + skip_channels, out_channels)

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        """Forward pass through decoder block.

        Args:
            x: Input tensor from previous decoder layer.
            skip: Skip connection tensor from encoder.

        Returns:
            Decoded tensor.
        """
        x = self.upsample(x)
        
        # Handle size mismatch between upsampled and skip connection
        if x.shape[2:] != skip.shape[2:]:
            x = F.interpolate(x, size=skip.shape[2:], mode='bilinear', align_corners=False)
        
        x = torch.cat([x, skip], dim=1)
        x = self.conv(x)
        return x


class Decoder(nn.Module):
    """Decoder module for U-Net with skip connections.

    Args:
        encoder_channels: List of encoder output channels [layer0, layer1, layer2, layer3, layer4].
        decoder_channels: List of decoder output channels.
    """

    def __init__(
        self, 
        encoder_channels: List[int] = [64, 64, 128, 256, 512],
        decoder_channels: List[int] = [256, 128, 64, 64]
    ) -> None:
        super().__init__()
        
        # encoder_channels = [64, 64, 128, 256, 512] (layer0, layer1, layer2, layer3, layer4)
        # decoder_channels = [256, 128, 64, 64] (decoder outputs)
        
        self.layer4 = DecoderBlock(encoder_channels[4], encoder_channels[3], decoder_channels[0])  # 512 -> 256, skip: 256
        self.layer3 = DecoderBlock(decoder_channels[0], encoder_channels[2], decoder_channels[1])  # 256 -> 128, skip: 128
        self.layer2 = DecoderBlock(decoder_channels[1], encoder_channels[1], decoder_channels[2])  # 128 -> 64, skip: 64
        self.layer1 = DecoderBlock(decoder_channels[2], encoder_channels[0], decoder_channels[3])  # 64 -> 64, skip: 64

    def forward(self, x: torch.Tensor, skips: List[torch.Tensor]) -> torch.Tensor:
        """Forward pass through the decoder.

        Args:
            x: Input tensor from encoder bottleneck.
            skips: List of skip connection tensors from encoder [skip0, skip1, skip2, skip3].

        Returns:
            Decoded tensor.
        """
        # skips = [skip0, skip1, skip2, skip3] = [64, 64, 128, 256] channels
        x = self.layer4(x, skips[3])  # Use skip3 (256 channels)
        x = self.layer3(x, skips[2])  # Use skip2 (128 channels)
        x = self.layer2(x, skips[1])  # Use skip1 (64 channels)
        x = self.layer1(x, skips[0])  # Use skip0 (64 channels)
        
        return x


class SegmentationHead(nn.Module):
    """Final 1x1 convolution layer to produce segmentation output.

    Args:
        in_channels: Number of input channels.
        out_channels: Number of output channels (classes).
    """

    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the segmentation head.

        Args:
            x: Input tensor.

        Returns:
            Segmentation output tensor.
        """
        return self.conv(x)


def get_activation(activation: Optional[Union[str, nn.Module]]) -> Optional[nn.Module]:
    """Get activation function based on name or module.

    Args:
        activation: Activation function name or module.

    Returns:
        Activation module or None.

    Raises:
        ValueError: If activation is not supported.
    """
    if activation is None:
        return None
    if isinstance(activation, str):
        activation = activation.lower()
        if activation == 'sigmoid':
            return nn.Sigmoid()
        elif activation == 'softmax':
            return nn.Softmax(dim=1)
        elif activation == 'tanh':
            return nn.Tanh()
        elif activation == 'relu':
            return nn.ReLU()
        else:
            raise ValueError(f"Unsupported activation: {activation}")
    elif isinstance(activation, nn.Module):
        return activation
    else:
        raise ValueError("activation must be None, a string, or a nn.Module instance.")


class UNet(nn.Module):
    """U-Net architecture with ResNet34 backbone for image segmentation.

    Args:
        in_channels: Number of input channels.
        out_channels: Number of output channels (classes).
        pretrained: Whether to use pretrained ResNet34 weights.
        encoder_channels: List of encoder output channels.
        decoder_channels: List of decoder output channels.
        final_activation: Optional activation function after segmentation head.
    """

    def __init__(
        self,
        in_channels: int = 3,
        out_channels: int = 1,
        pretrained: bool = True,
        encoder_channels: List[int] = [64, 64, 128, 256, 512],
        decoder_channels: List[int] = [256, 128, 64, 64],
        final_activation: Optional[Union[str, nn.Module]] = None
    ) -> None:
        super().__init__()
        
        # Encoder
        self.encoder = ResNet34Encoder(pretrained=pretrained, in_channels=in_channels)
        
        # Decoder
        self.decoder = Decoder(encoder_channels=encoder_channels, decoder_channels=decoder_channels)
        
        # Segmentation head
        self.segmentation_head = SegmentationHead(decoder_channels[-1], out_channels)
        
        # Optional final activation
        self.final_activation = get_activation(final_activation)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the U-Net.

        Args:
            x: Input tensor of shape (batch_size, in_channels, height, width).

        Returns:
            Segmentation output tensor of shape (batch_size, out_channels, height, width).
        """
        # Encoder
        x, skips = self.encoder(x)
        
        # Decoder
        x = self.decoder(x, skips)
        
        # Segmentation head
        x = self.segmentation_head(x)
        
        # Final activation
        if self.final_activation is not None:
            x = self.final_activation(x)
        
        return x