"""
DeepLab v3+ implementation with configurable backbone for image segmentation.

This module provides a PyTorch implementation of the DeepLab v3+ architecture with
support for multiple backbones (ResNet34, MobileNetV3, etc.) following 
segmentation-models-pytorch design patterns.
"""

from typing import List, Optional, Tuple, Union, Dict
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet34, mobilenet_v3_large


class SeparableConv2d(nn.Module):
    """Separable convolution implementation.
    
    Args:
        in_channels: Number of input channels.
        out_channels: Number of output channels.
        kernel_size: Convolution kernel size.
        stride: Convolution stride.
        padding: Convolution padding.
        dilation: Convolution dilation.
        bias: Whether to use bias.
    """
    
    def __init__(
        self, 
        in_channels: int, 
        out_channels: int, 
        kernel_size: int = 3,
        stride: int = 1, 
        padding: int = 0, 
        dilation: int = 1, 
        bias: bool = False
    ) -> None:
        super().__init__()
        self.depthwise = nn.Conv2d(
            in_channels, in_channels, kernel_size, stride, padding, 
            dilation, groups=in_channels, bias=bias
        )
        self.pointwise = nn.Conv2d(in_channels, out_channels, 1, bias=bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through separable convolution.
        
        Args:
            x: Input tensor.
            
        Returns:
            Output tensor after separable convolution.
        """
        x = self.depthwise(x)
        x = self.pointwise(x)
        return x


class ASPPConv(nn.Module):
    """ASPP convolution block with atrous convolution.
    
    Args:
        in_channels: Number of input channels.
        out_channels: Number of output channels.
        dilation: Dilation rate for atrous convolution.
    """
    
    def __init__(self, in_channels: int, out_channels: int, dilation: int) -> None:
        super().__init__()
        self.conv = nn.Conv2d(
            in_channels, out_channels, 3, padding=dilation, 
            dilation=dilation, bias=False
        )
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through ASPP convolution block.
        
        Args:
            x: Input tensor.
            
        Returns:
            Output tensor after ASPP convolution.
        """
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x


class ASPPPooling(nn.Module):
    """ASPP pooling block with global average pooling.
    
    Args:
        in_channels: Number of input channels.
        out_channels: Number of output channels.
    """
    
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Conv2d(in_channels, out_channels, 1, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through ASPP pooling block.
        
        Args:
            x: Input tensor.
            
        Returns:
            Output tensor after global pooling and upsampling.
        """
        size = x.shape[2:]
        x = self.gap(x)
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        x = F.interpolate(x, size=size, mode='bilinear', align_corners=False)
        return x


class ASPP(nn.Module):
    """Atrous Spatial Pyramid Pooling module.
    
    Args:
        in_channels: Number of input channels.
        out_channels: Number of output channels.
        atrous_rates: List of dilation rates for atrous convolutions.
    """
    
    def __init__(
        self, 
        in_channels: int, 
        out_channels: int = 256,
        atrous_rates: List[int] = [6, 12, 18]
    ) -> None:
        super().__init__()
        
        # 1x1 convolution
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
        
        # Atrous convolutions
        self.conv2 = ASPPConv(in_channels, out_channels, atrous_rates[0])
        self.conv3 = ASPPConv(in_channels, out_channels, atrous_rates[1])
        self.conv4 = ASPPConv(in_channels, out_channels, atrous_rates[2])
        
        # Global pooling
        self.pool = ASPPPooling(in_channels, out_channels)
        
        # Final projection
        self.project = nn.Sequential(
            nn.Conv2d(5 * out_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through ASPP module.
        
        Args:
            x: Input tensor.
            
        Returns:
            Output tensor after ASPP processing.
        """
        x1 = self.conv1(x)
        x2 = self.conv2(x)
        x3 = self.conv3(x)
        x4 = self.conv4(x)
        x5 = self.pool(x)
        
        # Concatenate all branches
        x = torch.cat([x1, x2, x3, x4, x5], dim=1)
        
        # Final projection
        x = self.project(x)
        return x


class ResNet34Encoder(nn.Module):
    """ResNet34-based encoder for DeepLab v3+.
    
    Args:
        pretrained: Whether to use pretrained weights.
        in_channels: Number of input channels.
        output_stride: Output stride (8 or 16).
    """
    
    def __init__(
        self, 
        pretrained: bool = True, 
        in_channels: int = 3,
        output_stride: int = 16
    ) -> None:
        super().__init__()
        
        if pretrained:
            resnet = resnet34(weights='IMAGENET1K_V1')
        else:
            resnet = resnet34(weights=None)
        
        self.output_stride = output_stride
        
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
        
        # Store output channels
        self.high_level_channels = 512  # layer4 output
        self.low_level_channels = 64    # layer0 output
        
        # Modify layers for different output strides
        if output_stride == 8:
            # Make layer3 and layer4 use dilation instead of stride
            self._make_layer_dilated(self.layer3, dilation=2, stride=1)
            self._make_layer_dilated(self.layer4, dilation=4, stride=1)
        elif output_stride == 16:
            # Make layer4 use dilation instead of stride
            self._make_layer_dilated(self.layer4, dilation=2, stride=1)

    def _make_layer_dilated(self, layer: nn.Module, dilation: int, stride: int) -> None:
        """Convert layer to use dilated convolutions.
        
        Args:
            layer: Layer to modify.
            dilation: Dilation rate.
            stride: New stride (usually 1).
        """
        for module in layer.modules():
            if isinstance(module, nn.Conv2d) and module.kernel_size[0] == 3:
                module.dilation = (dilation, dilation)
                module.padding = (dilation, dilation)
            # Set stride of first block to 1 to prevent downsampling
            if hasattr(module, 'downsample') and module.downsample is not None:
                module.downsample[0].stride = (stride, stride)
        
        # Set stride of first conv in layer
        if hasattr(layer[0], 'conv1'):
            layer[0].conv1.stride = (stride, stride)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass through the ResNet34 encoder.
        
        Args:
            x: Input tensor.
            
        Returns:
            Tuple containing:
                - High-level features from layer4 (raw, before ASPP)
                - Low-level features from layer0 for decoder
        """
        # Layer 0: Initial conv + bn + relu (stride=2, H/2, W/2)
        x = self.layer0(x)  # 64 channels, H/2, W/2
        low_level_features = x  # Save for decoder
        
        # MaxPool (stride=2, H/4, W/4)
        x = self.maxpool(x)  # H/4, W/4
        
        x = self.layer1(x)  # 64 channels, H/4, W/4
        x = self.layer2(x)  # 128 channels, H/8, W/8
        x = self.layer3(x)  # 256 channels, H/8 or H/16, W/8 or W/16
        x = self.layer4(x)  # 512 channels, final features (raw)
        
        return x, low_level_features


class MobileNetV3Encoder(nn.Module):
    """MobileNetV3-based encoder for DeepLab v3+.
    
    Extracts high-level features (bottleneck) and low-level features for decoder.
    
    Args:
        pretrained: Whether to use pretrained weights.
        in_channels: Number of input channels.
        output_stride: Output stride (8 or 16). Note: MobileNetV3 naturally has OS=16.
    """
    
    def __init__(
        self, 
        pretrained: bool = True, 
        in_channels: int = 3,
        output_stride: int = 16
    ) -> None:
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
        # Low-level features: early layers with fine spatial details
        self.layer0 = nn.Sequential(features[0], features[1])  # 16 channels, H/2
        self.layer1 = nn.Sequential(features[2], features[3])  # 24 channels, H/2
        
        # High-level features: deeper layers
        self.layer2 = nn.Sequential(features[4], features[5], features[6])  # 40 channels, H/4
        self.layer3 = nn.Sequential(*features[7:13])  # 112 channels, H/8
        self.layer4 = nn.Sequential(*features[13:17])  # 960 channels, H/16
        
        # Store output channels
        self.high_level_channels = 960  # layer4 output (bottleneck)
        self.low_level_channels = 24    # layer1 output (low-level features)
        
        self.output_stride = output_stride
        
        # Note: For output_stride=8, would need to modify layer3 and layer4 with dilation
        # This is more complex for MobileNetV3 due to inverted residuals
        if output_stride == 8:
            print("Warning: output_stride=8 not fully implemented for MobileNetV3. Using OS=16.")

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass through the MobileNetV3 encoder.
        
        Args:
            x: Input tensor.
            
        Returns:
            Tuple containing:
                - High-level features from layer4 (bottleneck)
                - Low-level features from layer1 for decoder
        """
        x = self.layer0(x)  # 16 channels, H/2, W/2
        x = self.layer1(x)  # 24 channels, H/2, W/2
        low_level_features = x  # Save for decoder (fine spatial details)
        
        x = self.layer2(x)  # 40 channels, H/4, W/4
        x = self.layer3(x)  # 112 channels, H/8, W/8
        x = self.layer4(x)  # 960 channels, H/16, W/16 (bottleneck)
        
        return x, low_level_features


class Decoder(nn.Module):
    """DeepLab v3+ decoder with integrated ASPP module.
    
    Perfect for training strategy: layer3+layer4+decoder (including ASPP) are trainable,
    while layer0+layer1+layer2 can be frozen to preserve pretrained features.
    
    Args:
        high_level_channels: Number of channels from encoder bottleneck output.
        low_level_channels: Number of channels from encoder low-level output.
        out_channels: Number of decoder output channels.
        aspp_out_channels: Number of output channels from ASPP.
        atrous_rates: List of dilation rates for ASPP.
    """
    
    def __init__(
        self, 
        high_level_channels: int,
        low_level_channels: int,
        out_channels: int = 256,
        aspp_out_channels: int = 256,
        atrous_rates: List[int] = [6, 12, 18]
    ) -> None:
        super().__init__()
        
        # ASPP processes raw high-level features from bottleneck
        self.aspp = ASPP(
            in_channels=high_level_channels,
            out_channels=aspp_out_channels,
            atrous_rates=atrous_rates
        )
        
        # Low-level feature projection
        self.low_level_conv = nn.Sequential(
            nn.Conv2d(low_level_channels, 48, 1, bias=False),
            nn.BatchNorm2d(48),
            nn.ReLU(inplace=True)
        )
        
        # Final convolutions
        self.conv = nn.Sequential(
            SeparableConv2d(
                aspp_out_channels + 48, out_channels, 
                kernel_size=3, padding=1, bias=False
            ),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            SeparableConv2d(
                out_channels, out_channels, 
                kernel_size=3, padding=1, bias=False
            ),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(
        self, 
        high_level_features: torch.Tensor, 
        low_level_features: torch.Tensor
    ) -> torch.Tensor:
        """Forward pass through the decoder with integrated ASPP.
        
        Args:
            high_level_features: Raw features from encoder bottleneck.
            low_level_features: Features from encoder low-level layer.
            
        Returns:
            Decoded features ready for segmentation head.
        """
        # Apply ASPP to high-level features first
        aspp_features = self.aspp(high_level_features)
        
        # Process low-level features
        low_level = self.low_level_conv(low_level_features)
        
        # Upsample ASPP features to match low-level size
        target_size = low_level.shape[2:]
        high_level = F.interpolate(
            aspp_features, size=target_size, 
            mode='bilinear', align_corners=False
        )
        
        # Concatenate and process
        x = torch.cat([high_level, low_level], dim=1)
        x = self.conv(x)
        
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


# Backbone registry
BACKBONE_REGISTRY: Dict[str, type] = {
    'resnet34': ResNet34Encoder,
    'mobilenetv3': MobileNetV3Encoder,
}


class DeepLabV3Plus(nn.Module):
    """DeepLab v3+ architecture with configurable backbone for image segmentation.
    
    Clean pipeline: x → encoder → decoder (with ASPP) → segmentation_head → final_activation
    
    Supports multiple backbones:
    - 'resnet34': ResNet34 backbone (default)
    - 'mobilenetv3': MobileNetV3-Large backbone
    
    Designed for training strategy where early layers are frozen (pretrained)
    and deeper layers + decoder (including ASPP) are trainable.
    
    Args:
        backbone: Backbone architecture name ('resnet34' or 'mobilenetv3').
        in_channels: Number of input channels.
        out_channels: Number of output channels (classes).
        pretrained: Whether to use pretrained backbone weights.
        output_stride: Output stride for encoder (8 or 16).
        decoder_channels: Number of decoder output channels.
        aspp_out_channels: Number of output channels from ASPP.
        atrous_rates: List of dilation rates for ASPP.
        final_activation: Optional activation function after segmentation head.
        
    Raises:
        ValueError: If backbone is not supported.
        
    Example:
        >>> # ResNet34 backbone
        >>> model = DeepLabV3Plus(backbone='resnet34', in_channels=3, out_channels=1)
        >>> 
        >>> # MobileNetV3 backbone
        >>> model = DeepLabV3Plus(backbone='mobilenetv3', in_channels=3, out_channels=1)
        >>> 
        >>> # Custom configuration
        >>> model = DeepLabV3Plus(
        ...     backbone='resnet34',
        ...     output_stride=8,
        ...     atrous_rates=[12, 24, 36],
        ...     final_activation='sigmoid'
        ... )
    """
    
    def __init__(
        self,
        backbone: str = 'resnet34',
        in_channels: int = 3,
        out_channels: int = 1,
        pretrained: bool = True,
        output_stride: int = 16,
        decoder_channels: int = 256,
        aspp_out_channels: int = 256,
        atrous_rates: List[int] = [6, 12, 18],
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
            in_channels=in_channels,
            output_stride=output_stride
        )
        
        # Get encoder output channels
        high_level_channels = self.encoder.high_level_channels
        low_level_channels = self.encoder.low_level_channels
        
        # Decoder (with integrated ASPP)
        self.decoder = Decoder(
            high_level_channels=high_level_channels,
            low_level_channels=low_level_channels,
            out_channels=decoder_channels,
            aspp_out_channels=aspp_out_channels,
            atrous_rates=atrous_rates
        )
        
        # Segmentation head
        self.segmentation_head = SegmentationHead(decoder_channels, out_channels)
        
        # Optional final activation
        self.final_activation = get_activation(final_activation)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through DeepLab v3+.
        
        Clean pipeline: x → encoder → decoder → segmentation_head → final_activation
        
        Args:
            x: Input tensor of shape (batch_size, in_channels, height, width).
            
        Returns:
            Segmentation output tensor of shape (batch_size, out_channels, height, width).
        """
        input_size = x.shape[2:]  # Store original input size (H, W)
        
        # x → encoder
        high_level_features, low_level_features = self.encoder(x)
        
        # encoder → decoder (ASPP integrated inside)
        decoded = self.decoder(high_level_features, low_level_features)
        
        # decoder → segmentation_head
        output = self.segmentation_head(decoded, input_size)
        
        # segmentation_head → final_activation
        if self.final_activation is not None:
            output = self.final_activation(output)
        
        return output