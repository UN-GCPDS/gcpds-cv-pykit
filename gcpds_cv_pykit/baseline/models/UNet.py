"""
U-Net implementation with various backbone options for image segmentation.

This module provides a PyTorch implementation of the U-Net architecture with
optional pre-trained backbones like ResNet34 and EfficientNet-B0.
"""

from typing import List, Optional, Tuple, Union, Literal
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet34, efficientnet_b0


class DoubleConv(nn.Module):
    """Double convolution block with batch normalization and ReLU activation.
    
    Args:
        in_channels: Number of input channels.
        out_channels: Number of output channels.
    """
    
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1),
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


class Encoder(nn.Module):
    """Basic encoder with double convolution blocks and max pooling.
    
    Args:
        in_channels: Number of input channels.
        features: List of feature dimensions for each encoder level.
    """
    
    def __init__(self, in_channels: int, features: List[int]) -> None:
        super().__init__()
        layers = []
        for feature in features:
            layers.append(DoubleConv(in_channels, feature))
            in_channels = feature
        self.layers = nn.ModuleList(layers)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        """Forward pass through the encoder.
        
        Args:
            x: Input tensor.
            
        Returns:
            Tuple containing:
                - Output tensor after encoding
                - List of skip connection tensors
        """
        skips = []
        for layer in self.layers:
            x = layer(x)
            skips.append(x)
            x = self.pool(x)
        return x, skips


class ResNet34Encoder(nn.Module):
    """ResNet34-based encoder for U-Net.
    
    Args:
        pretrained: Whether to use pretrained weights.
    """
    
    def __init__(self, pretrained: bool = True) -> None:
        super().__init__()
        resnet = resnet34(weights='IMAGENET1K_V1' if pretrained else None)
        self.initial = nn.Sequential(
            resnet.conv1,
            resnet.bn1,
            resnet.relu,
        )
        self.maxpool = resnet.maxpool
        self.encoder1 = resnet.layer1  # 64
        self.encoder2 = resnet.layer2  # 128
        self.encoder3 = resnet.layer3  # 256
        self.encoder4 = resnet.layer4  # 512

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        """Forward pass through the ResNet34 encoder.
        
        Args:
            x: Input tensor.
            
        Returns:
            Tuple containing:
                - Output tensor after encoding
                - List of skip connection tensors
        """
        skips = []
        x = self.initial(x)
        skips.append(x)  # 64
        x = self.maxpool(x)
        x = self.encoder1(x)
        skips.append(x)  # 64
        x = self.encoder2(x)
        skips.append(x)  # 128
        x = self.encoder3(x)
        skips.append(x)  # 256
        x = self.encoder4(x)
        return x, skips


class EfficientNetEncoder(nn.Module):
    """EfficientNet-B0 based encoder for U-Net.
    
    Args:
        pretrained: Whether to use pretrained weights.
    """
    
    def __init__(self, pretrained: bool = True) -> None:
        super().__init__()
        eff = efficientnet_b0(weights='IMAGENET1K_V1' if pretrained else None)
        self.stem = nn.Sequential(
            eff.features[0],  # Conv2d + BN + SiLU
        )
        self.block1 = eff.features[1]  # 16
        self.block2 = eff.features[2]  # 24
        self.block3 = eff.features[3]  # 40
        self.block4 = eff.features[4]  # 80
        self.block5 = eff.features[5]  # 112
        self.block6 = eff.features[6]  # 192
        self.block7 = eff.features[7]  # 320

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        """Forward pass through the EfficientNet encoder.
        
        Args:
            x: Input tensor.
            
        Returns:
            Tuple containing:
                - Output tensor after encoding
                - List of skip connection tensors
        """
        skips = []
        x = self.stem(x)
        skips.append(x)  # after stem (24 channels)
        x = self.block1(x)
        skips.append(x)  # after block1 (16 channels)
        x = self.block2(x)
        skips.append(x)  # after block2 (24 channels)
        x = self.block3(x)
        skips.append(x)  # after block3 (40 channels)
        x = self.block4(x)
        skips.append(x)  # after block4 (80 channels)
        x = self.block5(x)
        skips.append(x)  # after block5 (112 channels)
        x = self.block6(x)
        skips.append(x)  # after block6 (192 channels)
        x = self.block7(x)  # bottleneck (320 channels)
        return x, skips


class Decoder(nn.Module):
    """Decoder module for U-Net with skip connections.
    
    Args:
        encoder_channels: List of channel dimensions from encoder (bottleneck first).
        decoder_channels: List of channel dimensions for decoder blocks.
    """
    
    def __init__(self, encoder_channels: List[int], decoder_channels: List[int]) -> None:
        super().__init__()
        self.upconvs = nn.ModuleList()
        self.dec_blocks = nn.ModuleList()
        for in_ch, out_ch, skip_ch in zip(encoder_channels, decoder_channels, decoder_channels[1:]+[0]):
            self.upconvs.append(
                nn.ConvTranspose2d(in_ch, out_ch, kernel_size=2, stride=2)
            )
            if skip_ch > 0:
                self.dec_blocks.append(DoubleConv(out_ch + skip_ch, out_ch))
            else:
                self.dec_blocks.append(DoubleConv(out_ch, out_ch))
                
    def forward(self, x: torch.Tensor, skips: List[torch.Tensor]) -> torch.Tensor:
        """Forward pass through the decoder.
        
        Args:
            x: Input tensor from bottleneck.
            skips: List of skip connection tensors from encoder.
            
        Returns:
            Decoded tensor.
        """
        for i in range(len(self.upconvs)):
            x = self.upconvs[i](x)
            skip = skips[-(i+1)]
            if x.shape != skip.shape:
                x = F.interpolate(x, size=skip.shape[2:])
            x = torch.cat([x, skip], dim=1)
            x = self.dec_blocks[i](x)
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
        else:
            raise ValueError(f"Unsupported activation: {activation}")
    elif isinstance(activation, nn.Module):
        return activation
    else:
        raise ValueError("activation must be None, a string, or a nn.Module instance.")


class UNet(nn.Module):
    """U-Net architecture for image segmentation with optional backbones.
    
    Args:
        in_channels: Number of input channels.
        out_channels: Number of output channels (classes).
        features: List of feature dimensions for each encoder level (basic U-Net).
        backbone: Optional backbone name ('resnet34', 'efficientnet_b0', or None).
        pretrained: Whether to use pretrained weights for backbone.
        final_activation: Optional activation function after segmentation head.
    """
    
    def __init__(
        self,
        in_channels: int = 3,
        out_channels: int = 1,
        features: List[int] = [64, 128, 256, 512],
        backbone: Optional[Literal['resnet34', 'efficientnet_b0']] = None,
        pretrained: bool = True,
        final_activation: Optional[Union[str, nn.Module]] = None
    ) -> None:
        super().__init__()
        self.backbone = backbone
        self.final_activation = get_activation(final_activation)
        
        if backbone == 'resnet34':
            self.encoder = ResNet34Encoder(pretrained=pretrained)
            encoder_channels = [512, 256, 128, 64, 64]
            decoder_channels = [256, 128, 64, 64]
            self.decoder = Decoder(encoder_channels, decoder_channels)
            self.segmentation_head = SegmentationHead(decoder_channels[-1], out_channels)
            
        elif backbone == 'efficientnet_b0':
            self.encoder = EfficientNetEncoder(pretrained=pretrained)
            # EfficientNetB0 skip channels: [24, 16, 24, 40, 80, 112, 192], bottleneck: 320
            encoder_channels = [320, 192, 112, 80, 40, 24, 16, 24]  # bottleneck + skips
            decoder_channels = [192, 112, 80, 40, 24, 16, 24]
            self.decoder = Decoder(encoder_channels, decoder_channels)
            self.segmentation_head = SegmentationHead(decoder_channels[-1], out_channels)
            
        else:
            # For the case without backbone
            self.encoder = Encoder(in_channels, features)
            # Create the bottleneck as a separate module
            self.bottleneck = DoubleConv(features[-1], features[-1]*2)
            # Configure channels for the decoder
            encoder_channels = [features[-1]*2] + features
            decoder_channels = features[::-1]
            self.decoder = Decoder(encoder_channels, decoder_channels)
            self.segmentation_head = SegmentationHead(decoder_channels[-1], out_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the U-Net.
        
        Args:
            x: Input tensor.
            
        Returns:
            Segmentation output tensor.
        """
        # Get features from the encoder
        x, skips = self.encoder(x)
        
        # Apply bottleneck if there's no backbone
        if self.backbone is None:
            x = self.bottleneck(x)
        
        # Apply decoder and segmentation head
        x = self.decoder(x, skips)
        x = self.segmentation_head(x)
        
        if self.final_activation is not None:
            x = self.final_activation(x)
            
        return x