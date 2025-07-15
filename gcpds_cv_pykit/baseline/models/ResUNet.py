"""
ResUNet implementation with ResNet34 backbone for image segmentation.

This module provides a PyTorch implementation of the ResUNet architecture with
ResNet34 pre-trained backbone following segmentation-models-pytorch design patterns.
Features skip connections between encoder and decoder for precise localization.
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
            Output tensor after conv-bn-relu.
        """
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x


class ResidualBlock(nn.Module):
    """Residual block for decoder with skip connection.
    
    Args:
        in_channels: Number of input channels.
        out_channels: Number of output channels.
    """
    
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.conv1 = ConvBlock(in_channels, out_channels)
        self.conv2 = ConvBlock(out_channels, out_channels)
        
        # Skip connection projection if dimensions don't match
        self.skip = None
        if in_channels != out_channels:
            self.skip = nn.Conv2d(in_channels, out_channels, 1, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through residual block.
        
        Args:
            x: Input tensor.
            
        Returns:
            Output tensor with residual connection.
        """
        identity = x
        
        x = self.conv1(x)
        x = self.conv2(x)
        
        # Apply skip projection if needed
        if self.skip is not None:
            identity = self.skip(identity)
        
        x = x + identity
        return F.relu(x, inplace=True)


class ResNet34Encoder(nn.Module):
    """ResNet34-based encoder for ResUNet.
    
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

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        """Forward pass through the ResNet34 encoder.
        
        Args:
            x: Input tensor.
            
        Returns:
            Tuple containing:
                - Bottleneck features from layer4
                - List of skip connection features [layer0, layer1, layer2, layer3]
        """
        skip_connections = []
        
        # Layer 0: Initial conv + bn + relu (stride=2, H/2, W/2)
        x = self.layer0(x)  # 64 channels, H/2, W/2
        skip_connections.append(x)
        
        # MaxPool (stride=2, H/4, W/4)
        x = self.maxpool(x)  # H/4, W/4
        
        x = self.layer1(x)  # 64 channels, H/4, W/4
        skip_connections.append(x)
        
        x = self.layer2(x)  # 128 channels, H/8, W/8
        skip_connections.append(x)
        
        x = self.layer3(x)  # 256 channels, H/16, W/16
        skip_connections.append(x)
        
        x = self.layer4(x)  # 512 channels, H/32, W/32
        
        return x, skip_connections


class ResUNetDecoder(nn.Module):
    """ResUNet decoder with skip connections and residual blocks.
    
    Args:
        encoder_channels: List of encoder channel dimensions [64, 64, 128, 256, 512].
        decoder_channels: List of decoder channel dimensions.
        use_residual: Whether to use residual blocks in decoder.
    """
    
    def __init__(
        self, 
        encoder_channels: List[int] = [64, 64, 128, 256, 512],
        decoder_channels: List[int] = [256, 128, 64, 32],
        use_residual: bool = True
    ) -> None:
        super().__init__()
        
        self.use_residual = use_residual
        
        # Reverse encoder channels for decoder (skip bottleneck)
        skip_channels = encoder_channels[:-1][::-1]  # [256, 128, 64, 64]
        
        # Decoder blocks
        self.decoder_blocks = nn.ModuleList()
        
        # First decoder block (from bottleneck)
        in_channels = encoder_channels[-1]  # 512
        out_channels = decoder_channels[0]  # 256
        
        self.decoder_blocks.append(
            self._make_decoder_block(in_channels, out_channels)
        )
        
        # Subsequent decoder blocks with skip connections
        for i in range(1, len(decoder_channels)):
            in_channels = decoder_channels[i-1] + skip_channels[i-1]  # decoder + skip
            out_channels = decoder_channels[i]
            
            self.decoder_blocks.append(
                self._make_decoder_block(in_channels, out_channels)
            )

    def _make_decoder_block(self, in_channels: int, out_channels: int) -> nn.Module:
        """Create a decoder block.
        
        Args:
            in_channels: Number of input channels.
            out_channels: Number of output channels.
            
        Returns:
            Decoder block module.
        """
        if self.use_residual:
            return nn.Sequential(
                ResidualBlock(in_channels, out_channels),
                ConvBlock(out_channels, out_channels)
            )
        else:
            return nn.Sequential(
                ConvBlock(in_channels, out_channels),
                ConvBlock(out_channels, out_channels)
            )

    def forward(
        self, 
        bottleneck: torch.Tensor, 
        skip_connections: List[torch.Tensor]
    ) -> torch.Tensor:
        """Forward pass through the decoder.
        
        Args:
            bottleneck: Bottleneck features from encoder.
            skip_connections: List of skip connection features.
            
        Returns:
            Decoded features.
        """
        x = bottleneck
        
        # Reverse skip connections to match decoder order
        skip_connections = skip_connections[::-1]
        
        # First decoder block (no skip connection)
        x = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=False)
        x = self.decoder_blocks[0](x)
        
        # Subsequent decoder blocks with skip connections
        for i in range(1, len(self.decoder_blocks)):
            # Upsample
            x = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=False)
            
            # Get corresponding skip connection
            skip = skip_connections[i-1]
            
            # Resize skip connection to match upsampled features if needed
            if x.shape[2:] != skip.shape[2:]:
                skip = F.interpolate(
                    skip, size=x.shape[2:], mode='bilinear', align_corners=False
                )
            
            # Concatenate with skip connection
            x = torch.cat([x, skip], dim=1)
            
            # Apply decoder block
            x = self.decoder_blocks[i](x)
        
        return x


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


class ResUNet(nn.Module):
    """ResUNet architecture with ResNet34 backbone for image segmentation.
    
    Clean pipeline: x → encoder → decoder (with skip connections) → segmentation_head → final_activation
    
    Args:
        in_channels: Number of input channels.
        out_channels: Number of output channels (classes).
        pretrained: Whether to use pretrained ResNet34 weights.
        decoder_channels: List of decoder channel dimensions.
        use_residual: Whether to use residual blocks in decoder.
        final_activation: Optional activation function after segmentation head.
    """
    
    def __init__(
        self,
        in_channels: int = 3,
        out_channels: int = 1,
        pretrained: bool = True,
        decoder_channels: List[int] = [256, 128, 64, 32],
        use_residual: bool = True,
        final_activation: Optional[Union[str, nn.Module]] = None
    ) -> None:
        super().__init__()
        
        # Encoder
        self.encoder = ResNet34Encoder(
            pretrained=pretrained, 
            in_channels=in_channels
        )
        
        # Decoder with skip connections
        self.decoder = ResUNetDecoder(
            encoder_channels=[64, 64, 128, 256, 512],  # ResNet34 channel dims
            decoder_channels=decoder_channels,
            use_residual=use_residual
        )
        
        # Segmentation head
        self.segmentation_head = SegmentationHead(decoder_channels[-1], out_channels)
        
        # Optional final activation
        self.final_activation = get_activation(final_activation)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through ResUNet.
        
        Clean pipeline: x → encoder → decoder → segmentation_head → final_activation
        
        Args:
            x: Input tensor of shape (batch_size, in_channels, height, width).
            
        Returns:
            Segmentation output tensor of shape (batch_size, out_channels, height, width).
        """
        input_size = x.shape[2:]  # Store original input size (H, W)
        
        # x → encoder
        bottleneck, skip_connections = self.encoder(x)
        
        # encoder → decoder (with skip connections)
        decoded = self.decoder(bottleneck, skip_connections)
        
        # decoder → segmentation_head
        output = self.segmentation_head(decoded, input_size)
        
        # segmentation_head → final_activation
        if self.final_activation is not None:
            output = self.final_activation(output)
        
        return output