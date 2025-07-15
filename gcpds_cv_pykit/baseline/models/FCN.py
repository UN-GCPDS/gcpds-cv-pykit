"""
FCN (Fully Convolutional Network) implementation with ResNet34 backbone for image segmentation.

This module provides a PyTorch implementation of the FCN architecture with
ResNet34 pre-trained backbone following segmentation-models-pytorch design patterns.
Features skip connections for multi-scale feature fusion.
"""

from typing import List, Optional, Tuple, Union
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet34


class ConvBlock(nn.Module):
    """Basic convolution block with BatchNorm and ReLU.
    
    Args:
        in_channels: Number of input channels.
        out_channels: Number of output channels.
        kernel_size: Convolution kernel size.
        stride: Convolution stride.
        padding: Convolution padding.
        bias: Whether to use bias.
    """
    
    def __init__(
        self, 
        in_channels: int, 
        out_channels: int, 
        kernel_size: int = 3,
        stride: int = 1, 
        padding: int = 1, 
        bias: bool = False
    ) -> None:
        super().__init__()
        self.conv = nn.Conv2d(
            in_channels, out_channels, kernel_size, stride, padding, bias=bias
        )
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through convolution block.
        
        Args:
            x: Input tensor.
            
        Returns:
            Output tensor after convolution, batch norm, and ReLU.
        """
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x


class ResNet34Encoder(nn.Module):
    """ResNet34-based encoder for FCN.
    
    Args:
        pretrained: Whether to use pretrained weights.
        in_channels: Number of input channels.
    """
    
    def __init__(
        self, 
        pretrained: bool = True, 
        in_channels: int = 3
    ) -> None:
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
        self.layer1 = resnet.layer1  # 64 channels, stride 1
        self.layer2 = resnet.layer2  # 128 channels, stride 2
        self.layer3 = resnet.layer3  # 256 channels, stride 2
        self.layer4 = resnet.layer4  # 512 channels, stride 2

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Forward pass through the ResNet34 encoder.
        
        Args:
            x: Input tensor.
            
        Returns:
            Tuple containing features from different layers for skip connections:
                - layer1 features (64 channels)
                - layer2 features (128 channels)
                - layer3 features (256 channels)
                - layer4 features (512 channels)
        """
        # Layer 0: Initial conv + bn + relu (stride=2, H/2, W/2)
        x = self.layer0(x)  # 64 channels, H/2, W/2
        
        # MaxPool (stride=2, H/4, W/4)
        x = self.maxpool(x)  # H/4, W/4
        
        x1 = self.layer1(x)  # 64 channels, H/4, W/4
        x2 = self.layer2(x1)  # 128 channels, H/8, W/8
        x3 = self.layer3(x2)  # 256 channels, H/16, W/16
        x4 = self.layer4(x3)  # 512 channels, H/32, W/32
        
        return x1, x2, x3, x4


class FCNDecoder(nn.Module):
    """FCN decoder with skip connections for multi-scale feature fusion.
    
    Args:
        feature_channels: List of feature channels from encoder layers.
        decoder_channels: Number of decoder channels.
        out_channels: Number of output channels.
    """
    
    def __init__(
        self, 
        feature_channels: List[int] = [64, 128, 256, 512],
        decoder_channels: int = 256,
        out_channels: int = 1
    ) -> None:
        super().__init__()
        
        # Classifier for layer4 (deepest features)
        self.classifier4 = nn.Conv2d(feature_channels[3], out_channels, 1)
        
        # Upsampling and fusion layers
        self.upsample4to3 = nn.ConvTranspose2d(
            out_channels, out_channels, kernel_size=4, stride=2, padding=1
        )
        self.classifier3 = nn.Conv2d(feature_channels[2], out_channels, 1)
        
        self.upsample3to2 = nn.ConvTranspose2d(
            out_channels, out_channels, kernel_size=4, stride=2, padding=1
        )
        self.classifier2 = nn.Conv2d(feature_channels[1], out_channels, 1)
        
        self.upsample2to1 = nn.ConvTranspose2d(
            out_channels, out_channels, kernel_size=4, stride=2, padding=1
        )
        self.classifier1 = nn.Conv2d(feature_channels[0], out_channels, 1)
        
        # Final upsampling to original resolution
        self.final_upsample = nn.ConvTranspose2d(
            out_channels, out_channels, kernel_size=16, stride=8, padding=4
        )
        
        # Optional refinement layers
        self.refine = nn.Sequential(
            ConvBlock(out_channels, decoder_channels, 3, 1, 1),
            nn.Conv2d(decoder_channels, out_channels, 1)
        )

    def forward(
        self, 
        x1: torch.Tensor, 
        x2: torch.Tensor, 
        x3: torch.Tensor, 
        x4: torch.Tensor
    ) -> torch.Tensor:
        """Forward pass through the FCN decoder with skip connections.
        
        Args:
            x1: Features from layer1 (64 channels, H/4, W/4).
            x2: Features from layer2 (128 channels, H/8, W/8).
            x3: Features from layer3 (256 channels, H/16, W/16).
            x4: Features from layer4 (512 channels, H/32, W/32).
            
        Returns:
            Decoded features with skip connections.
        """
        # Start from deepest features
        score4 = self.classifier4(x4)  # H/32, W/32
        
        # Upsample and fuse with layer3
        up4 = self.upsample4to3(score4)  # H/16, W/16
        score3 = self.classifier3(x3)    # H/16, W/16
        fuse3 = up4 + score3
        
        # Upsample and fuse with layer2
        up3 = self.upsample3to2(fuse3)   # H/8, W/8
        score2 = self.classifier2(x2)    # H/8, W/8
        fuse2 = up3 + score2
        
        # Upsample and fuse with layer1
        up2 = self.upsample2to1(fuse2)   # H/4, W/4
        score1 = self.classifier1(x1)    # H/4, W/4
        fuse1 = up2 + score1
        
        # Final upsampling to original resolution
        output = self.final_upsample(fuse1)  # H, W
        
        # Optional refinement
        output = self.refine(output)
        
        return output


class SegmentationHead(nn.Module):
    """Final segmentation head that outputs class predictions.
    
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
            x: Input tensor.
            target_size: Target output size (H, W).
            
        Returns:
            Segmentation output tensor (H, W).
        """
        # Apply final classification conv
        x = self.conv(x)
        
        # Ensure exact target size with bilinear interpolation
        if x.shape[2:] != target_size:
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


class FCN(nn.Module):
    """FCN (Fully Convolutional Network) architecture with ResNet34 backbone for image segmentation.
    
    Clean pipeline: x → encoder → decoder (with skip connections) → segmentation_head → final_activation
    
    Features multi-scale skip connections for better boundary preservation and detail recovery.
    
    Args:
        in_channels: Number of input channels.
        out_channels: Number of output channels (classes).
        pretrained: Whether to use pretrained ResNet34 weights.
        decoder_channels: Number of decoder channels.
        final_activation: Optional activation function after segmentation head.
    """
    
    def __init__(
        self,
        in_channels: int = 3,
        out_channels: int = 1,
        pretrained: bool = True,
        decoder_channels: int = 256,
        final_activation: Optional[Union[str, nn.Module]] = None
    ) -> None:
        super().__init__()
        
        # Encoder
        self.encoder = ResNet34Encoder(
            pretrained=pretrained, 
            in_channels=in_channels
        )
        
        # Decoder with skip connections
        self.decoder = FCNDecoder(
            feature_channels=[64, 128, 256, 512],  # ResNet34 layer channels
            decoder_channels=decoder_channels,
            out_channels=out_channels
        )
        
        # Segmentation head (optional, decoder already outputs final classes)
        self.segmentation_head = SegmentationHead(out_channels, out_channels)
        
        # Optional final activation
        self.final_activation = get_activation(final_activation)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through FCN.
        
        Clean pipeline: x → encoder → decoder → segmentation_head → final_activation
        
        Args:
            x: Input tensor of shape (batch_size, in_channels, height, width).
            
        Returns:
            Segmentation output tensor of shape (batch_size, out_channels, height, width).
        """
        input_size = x.shape[2:]  # Store original input size (H, W)
        
        # x → encoder (with skip connections)
        x1, x2, x3, x4 = self.encoder(x)
        
        # encoder → decoder (skip connections integrated inside)
        decoded = self.decoder(x1, x2, x3, x4)
        
        # decoder → segmentation_head (ensure exact size match)
        output = self.segmentation_head(decoded, input_size)
        
        # segmentation_head → final_activation
        if self.final_activation is not None:
            output = self.final_activation(output)
        
        return output