"""
U-Net implementation with configurable backbone for image segmentation.

This module provides a PyTorch implementation of the U-Net architecture with
support for multiple backbones (ResNet34, MobileNetV3, etc.) following 
segmentation-models-pytorch design patterns.
"""

from typing import List, Optional, Tuple, Union, Dict
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet34, mobilenet_v3_large


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
        
        # Store output channels for each layer
        self.out_channels = [64, 64, 128, 256, 512]

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
        
        # Layer 0: Initial conv + bn + relu (stride=2, H/2, W/2)
        x = self.layer0(x)  # 64 channels, H/2, W/2
        skips.append(x)
        
        # MaxPool (stride=2, H/4, W/4)
        x = self.maxpool(x)  # H/4, W/4
        
        x = self.layer1(x)  # 64 channels, H/4, W/4
        skips.append(x)
        
        x = self.layer2(x)  # 128 channels, H/8, W/8
        skips.append(x)
        
        x = self.layer3(x)  # 256 channels, H/16, W/16
        skips.append(x)
        
        x = self.layer4(x)  # 512 channels, H/32, W/32 (bottleneck)
        
        return x, skips


class MobileNetV3Encoder(nn.Module):
    """MobileNetV3-based encoder for U-Net.
    
    Extracts features at 5 different scales:
    - layer0: stride=2, channels=16, size=H/2
    - layer1: stride=1, channels=24, size=H/2
    - layer2: stride=2, channels=40, size=H/4
    - layer3: stride=2, channels=112, size=H/8
    - layer4: stride=2, channels=960, size=H/16
    
    Args:
        pretrained: Whether to use pretrained weights.
        in_channels: Number of input channels.
    """

    def __init__(self, pretrained: bool = True, in_channels: int = 3) -> None:
        super().__init__()
        
        if pretrained:
            mobilenet = mobilenet_v3_large(weights='IMAGENET1K_V1')
        else:
            mobilenet = mobilenet_v3_large(weights=None)
        
        features = mobilenet.features
        
        # Handle different input channels
        if in_channels != 3:
            features[0][0] = nn.Conv2d(
                in_channels, 16, kernel_size=3, stride=2, padding=1, bias=False
            )
        
        # Layer grouping based on stride changes
        self.layer0 = nn.Sequential(features[0], features[1])  # 16 channels, H/2
        self.layer1 = nn.Sequential(features[2], features[3])  # 24 channels, H/2
        self.layer2 = nn.Sequential(features[4], features[5], features[6])  # 40 channels, H/4
        self.layer3 = nn.Sequential(*features[7:13])  # 112 channels, H/8
        self.layer4 = nn.Sequential(*features[13:17])  # 960 channels, H/16
        
        # Store output channels for each layer
        self.out_channels = [16, 24, 40, 112, 960]

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        """Forward pass through the MobileNetV3 encoder.

        Args:
            x: Input tensor.

        Returns:
            Tuple containing:
                - Output tensor after encoding (bottleneck)
                - List of skip connection tensors [skip0, skip1, skip2, skip3]
        """
        skips = []
        
        x = self.layer0(x)  # 16 channels, H/2, W/2
        skips.append(x)
        
        x = self.layer1(x)  # 24 channels, H/2, W/2
        skips.append(x)
        
        x = self.layer2(x)  # 40 channels, H/4, W/4
        skips.append(x)
        
        x = self.layer3(x)  # 112 channels, H/8, W/8
        skips.append(x)
        
        x = self.layer4(x)  # 960 channels, H/16, W/16 (bottleneck)
        
        return x, skips


class DecoderBlock(nn.Module):
    """Single decoder block with upsampling and skip connection.
    
    Following segmentation-models-pytorch design: uses F.interpolate for precise size matching.

    Args:
        in_channels: Number of input channels from previous layer.
        skip_channels: Number of channels from skip connection.
        out_channels: Number of output channels.
    """

    def __init__(self, in_channels: int, skip_channels: int, out_channels: int) -> None:
        super().__init__()
        self.conv = DoubleConv(in_channels + skip_channels, out_channels)

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        """Forward pass through decoder block.

        Args:
            x: Input tensor from previous decoder layer.
            skip: Skip connection tensor from encoder.

        Returns:
            Decoded tensor with same spatial size as skip connection.
        """
        # Upsample to match skip connection size exactly (like SMP does)
        target_size = skip.shape[2:]
        x = F.interpolate(x, size=target_size, mode='bilinear', align_corners=False)
        
        # Concatenate with skip connection
        x = torch.cat([x, skip], dim=1)
        
        # Apply convolutions
        x = self.conv(x)
        return x


class Decoder(nn.Module):
    """Decoder module for U-Net with skip connections.
    
    Following segmentation-models-pytorch design: no learnable upsampling layers,
    uses F.interpolate in decoder blocks for precise size matching.

    Args:
        encoder_channels: List of encoder output channels [layer0, layer1, layer2, layer3, layer4].
        decoder_channels: List of decoder output channels.
    """

    def __init__(
        self, 
        encoder_channels: List[int],
        decoder_channels: List[int] = [256, 128, 64, 64]
    ) -> None:
        super().__init__()
        
        # encoder_channels = [ch0, ch1, ch2, ch3, ch4] (layer0, layer1, layer2, layer3, layer4)
        # decoder_channels = [256, 128, 64, 64] (decoder outputs)
        
        self.layer4 = DecoderBlock(encoder_channels[4], encoder_channels[3], decoder_channels[0])
        self.layer3 = DecoderBlock(decoder_channels[0], encoder_channels[2], decoder_channels[1])
        self.layer2 = DecoderBlock(decoder_channels[1], encoder_channels[1], decoder_channels[2])
        self.layer1 = DecoderBlock(decoder_channels[2], encoder_channels[0], decoder_channels[3])

    def forward(self, x: torch.Tensor, skips: List[torch.Tensor]) -> torch.Tensor:
        """Forward pass through the decoder.

        Args:
            x: Input tensor from encoder bottleneck.
            skips: List of skip connection tensors from encoder [skip0, skip1, skip2, skip3].

        Returns:
            Decoded tensor with size H/2, W/2 (matching skip0).
        """
        # Process from deepest to shallowest
        x = self.layer4(x, skips[3])  # bottleneck -> match skip3
        x = self.layer3(x, skips[2])  # -> match skip2
        x = self.layer2(x, skips[1])  # -> match skip1
        x = self.layer1(x, skips[0])  # -> match skip0
        
        return x  # Output: H/2, W/2


class SegmentationHead(nn.Module):
    """Final segmentation head that upsamples to original resolution.
    
    Args:
        in_channels: Number of input channels.
        out_channels: Number of output channels (classes).
    """

    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x: torch.Tensor, target_size: Tuple[int, int]) -> torch.Tensor:
        """Forward pass through the segmentation head.

        Args:
            x: Input tensor (H/2, W/2).
            target_size: Target output size (H, W).

        Returns:
            Segmentation output tensor (H, W).
        """
        # Apply 1x1 conv
        x = self.conv(x)
        
        # Upsample to original input size
        x = F.interpolate(x, size=target_size, mode='bilinear', align_corners=False)
        
        return x


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


# Backbone registry
BACKBONE_REGISTRY: Dict[str, type] = {
    'resnet34': ResNet34Encoder,
    'mobilenetv3': MobileNetV3Encoder,
}


class UNet(nn.Module):
    """U-Net architecture with configurable backbone for image segmentation.
    
    Following segmentation-models-pytorch design principles:
    - No interpolation in main forward pass
    - Precise size matching through decoder block design
    - Final upsampling in segmentation head
    
    Supports multiple backbones:
    - 'resnet34': ResNet34 backbone (default)
    - 'mobilenetv3': MobileNetV3-Large backbone

    Args:
        backbone: Backbone architecture name ('resnet34' or 'mobilenetv3').
        in_channels: Number of input channels.
        out_channels: Number of output channels (classes).
        pretrained: Whether to use pretrained backbone weights.
        decoder_channels: List of decoder output channels.
        final_activation: Optional activation function after segmentation head.
        
    Raises:
        ValueError: If backbone is not supported.
        
    Example:
        >>> # ResNet34 backbone
        >>> model = UNet(backbone='resnet34', in_channels=3, out_channels=1)
        >>> 
        >>> # MobileNetV3 backbone
        >>> model = UNet(backbone='mobilenetv3', in_channels=3, out_channels=1)
        >>> 
        >>> # Custom decoder channels
        >>> model = UNet(
        ...     backbone='resnet34',
        ...     decoder_channels=[512, 256, 128, 64],
        ...     final_activation='sigmoid'
        ... )
    """

    def __init__(
        self,
        backbone: str = 'resnet34',
        in_channels: int = 3,
        out_channels: int = 1,
        pretrained: bool = True,
        decoder_channels: List[int] = [256, 128, 64, 64],
        final_activation: Optional[Union[str, nn.Module]] = None
    ) -> None:
        super().__init__()
        
        # Validate backbone
        if backbone not in BACKBONE_REGISTRY:
            raise ValueError(
                f"Unsupported backbone: {backbone}. "
                f"Available backbones: {list(BACKBONE_REGISTRY.keys())}"
            )
        
        # Create encoder based on backbone
        encoder_class = BACKBONE_REGISTRY[backbone]
        self.encoder = encoder_class(
            pretrained=pretrained, 
            in_channels=in_channels
        )
        
        # Get encoder output channels
        encoder_channels = self.encoder.out_channels
        
        # Decoder
        self.decoder = Decoder(
            encoder_channels=encoder_channels, 
            decoder_channels=decoder_channels
        )
        
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
        input_size = x.shape[2:]  # Store original input size (H, W)
        
        # Encoder: H,W -> bottleneck
        encoded, skips = self.encoder(x)
        
        # Decoder: bottleneck -> H/2,W/2 (using skip connections)
        decoded = self.decoder(encoded, skips)
        
        # Segmentation head: H/2,W/2 -> H,W
        output = self.segmentation_head(decoded, input_size)
        
        # Final activation
        if self.final_activation is not None:
            output = self.final_activation(output)
        
        return output 